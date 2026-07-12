"""Feature selection: which SAE features count as 'conversational', plus controls.

The paper screened all 32,768 features with an LLM judge over ~50 top-activating
contexts each, keeping features whose activations occurred in conversational
settings >50% of the time ("conversation ratio"), then hand-picked one: feature
30939, "a discourse marker for surprise, realization, or acknowledgment".

We do not re-run that judge. Neuronpedia publishes, for every feature of this
exact SAE, a GPT-4o-mini explanation (the same autointerp model the paper cites),
a max-activation estimate, and a firing rate. We use those directly, via two
independent selectors:

  neighbors -- cosine similarity to feature 30939 in Neuronpedia's own explanation
               embedding space. "Features whose meaning is like the paper's
               conversational feature."
  lexicon   -- keyword scoring of the explanation text against a dialogue lexicon.
               No embedding, different failure modes.

Agreement between two selectors that share no machinery is worth more than either
alone. Features they both rank highly are the strongest candidates.

CONTROLS. The paper compared its feature against random conversational and random
non-conversational features. That leaves a confound: SAE features differ wildly in
firing rate and activation magnitude, and steering a rare, high-magnitude feature
is simply a bigger perturbation than steering a common, low-magnitude one --
regardless of what it means. Feature 30939 is very sparse (fires on 0.016% of
tokens). So our controls are sampled to MATCH the candidate on sparsity and max
activation, and differ only in meaning. If matched controls reproduce the effect,
the "society of thought" reading is dead and it was really about perturbation size.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import urllib.request

S3 = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
MODEL_ID = "deepseek-r1-distill-llama-8b"

ANCHOR_FEATURE = 30939  # the paper's "conversational surprise" feature (layer 15, slimpj)

# Dialogue / turn-taking / interpersonal-stance vocabulary. Deliberately about the
# SOCIAL form of the token's context, not its topic.
CONVERSATIONAL_LEXICON = {
    "conversation", "conversational", "dialogue", "dialog", "discourse", "speaker",
    "turn-taking", "turn", "interjection", "exclamation", "greeting", "reply",
    "response", "question", "asking", "answer", "acknowledgment", "acknowledgement",
    "agreement", "disagreement", "surprise", "realization", "interpersonal",
    "social", "politeness", "apology", "thanking", "addressing", "quoted", "quotation",
    "speech", "utterance", "informal", "colloquial", "interruption", "reaction",
    "affirmation", "negation", "hedging", "uncertainty", "emphasis", "exclamatory",
    "first-person", "second-person", "you", "we", "listener", "audience",
}


@dataclass
class Feature:
    index: int
    description: str = ""
    max_act: float = 0.0
    frac_nonzero: float = 0.0
    embedding: list[float] = field(default_factory=list)
    conv_score: float = 0.0
    role: str = ""  # candidate | control_matched | control_random | anchor


def _download_shards(source_id: str, kind: str, cache: Path) -> list[dict]:
    """kind is 'explanations' or 'features'. Shards are batch-N.jsonl.gz."""
    out_path = cache / f"{MODEL_ID}__{source_id}__{kind}.jsonl"
    if out_path.exists():
        return [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]

    cache.mkdir(parents=True, exist_ok=True)
    prefix = f"v1/{MODEL_ID}/{source_id}/{kind}/"
    keys = _list_keys(prefix)
    if not keys:
        raise RuntimeError(f"no {kind} shards on S3 for {source_id!r}")

    rows: list[dict] = []
    for k in keys:
        if not k.endswith(".jsonl.gz"):
            continue
        raw = urllib.request.urlopen(f"{S3}/{k}").read()
        with gzip.open(io.BytesIO(raw), "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def _list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{S3}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        xml = urllib.request.urlopen(url).read().decode()
        keys += re.findall(r"<Key>(.*?)</Key>", xml)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    return keys


def load_features(source_id: str, cache: Path) -> dict[int, Feature]:
    feats: dict[int, Feature] = {}
    for r in _download_shards(source_id, "features", cache):
        i = int(r["index"])
        feats[i] = Feature(
            index=i,
            max_act=float(r.get("maxActApprox") or 0.0),
            frac_nonzero=float(r.get("frac_nonzero") or 0.0),
        )
    for r in _download_shards(source_id, "explanations", cache):
        i = int(r["index"])
        if i in feats:
            feats[i].description = (r.get("description") or "").strip()
            feats[i].embedding = r.get("embedding") or []
    return feats


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def score_conversationality(feats: dict[int, Feature], method: str, anchor: int = ANCHOR_FEATURE) -> None:
    """Fill Feature.conv_score in place."""
    if method == "neighbors":
        ref = feats[anchor].embedding
        if not ref:
            raise RuntimeError(f"anchor feature {anchor} has no embedding on Neuronpedia")
        for f in feats.values():
            f.conv_score = _cos(f.embedding, ref)
    elif method == "lexicon":
        for f in feats.values():
            words = set(re.findall(r"[a-z\-']+", f.description.lower()))
            hits = len(words & CONVERSATIONAL_LEXICON)
            f.conv_score = hits / math.sqrt(len(words) or 1)
    else:
        raise ValueError(f"unknown method {method!r}")


def select(
    feats: dict[int, Feature],
    n_candidates: int = 6,
    n_controls: int = 6,
    method: str = "neighbors",
    anchor: int = ANCHOR_FEATURE,
    seed: int = 0,
    min_max_act: float = 1.0,
) -> list[Feature]:
    """Return anchor + top conversational candidates + sparsity/magnitude-matched controls."""
    score_conversationality(feats, method, anchor)

    usable = [
        f for f in feats.values()
        if f.description and f.max_act >= min_max_act and f.frac_nonzero > 0
    ]
    ranked = sorted(usable, key=lambda f: -f.conv_score)

    chosen: list[Feature] = []
    a = feats[anchor]
    a.role = "anchor"
    chosen.append(a)

    for f in ranked:
        if len(chosen) > n_candidates:
            break
        if f.index == anchor:
            continue
        f.role = "candidate"
        chosen.append(f)

    # Matched controls: bottom half of the conversationality ranking, but drawn to
    # match the anchor's firing rate and max activation within a tolerance, so the
    # only thing that differs from the candidates is meaning -- not how big or how
    # rare the intervention is.
    rng = random.Random(seed)
    nonconv = ranked[len(ranked) // 2 :]

    def matched(f: Feature) -> bool:
        if f.frac_nonzero <= 0 or a.frac_nonzero <= 0:
            return False
        sparsity_ok = abs(math.log10(f.frac_nonzero) - math.log10(a.frac_nonzero)) < 0.5
        act_ok = 0.6 <= (f.max_act / a.max_act if a.max_act else 0) <= 1.7
        return sparsity_ok and act_ok

    pool = [f for f in nonconv if matched(f)]
    rng.shuffle(pool)
    for f in pool[:n_controls]:
        f.role = "control_matched"
        chosen.append(f)

    if len(pool) < n_controls:
        # Fall back to unmatched random non-conversational features, and say so.
        extra = [f for f in nonconv if f.role == ""]
        rng.shuffle(extra)
        for f in extra[: n_controls - len(pool)]:
            f.role = "control_random"
            chosen.append(f)

    return chosen
