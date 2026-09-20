"""The paper's LLM-as-judge for conversational behaviours.

Fig. 4b is measured by an LLM judge that reads a reasoning trace and counts instances of
four behaviours. The definitions below are the paper's own words; the counting convention
("integer counts, 0 if none are present") is theirs too.

DECLARED DEVIATION. The paper judges with Gemini-2.5-Pro. We judge with whatever backend
is configured -- by default an Anthropic model -- because that is the capable judge this
project has credentials for. The paper's own cross-judge agreement (ICC(3,1) ~ .85
between Gemini-2.5-Pro and GPT-5.2, ~ .76 against human raters) is the reason this is a
declarable deviation rather than a different experiment, but it IS a deviation and the
judge identity is recorded in every cache entry and every result file.

WHY A PARSE FAILURE IS NOT A ZERO. A judge that reads malformed output as "no behaviours
present" fails open. Traces get longer and more complex as RL proceeds, so parse failures
would concentrate in late training, and the curve would show a decline that is entirely
an artifact of the instrument. `parse_verdict` raises instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

BEHAVIOURS = (
    "question_answering",
    "perspective_shift",
    "conflict_of_perspectives",
    "reconciliation",
)

# Bump when the prompt changes. It is part of the cache key, so an edited prompt cannot
# silently mix two instruments inside one curve.
PROMPT_VERSION = "v1"

DEFINITIONS = {
    "question_answering":
        "sequences where a question is posed and later answered",
    "perspective_shift":
        "transition to a different idea, viewpoint, assumption, or approach",
    "conflict_of_perspectives":
        "expressions of disagreement, correction, or tension",
    "reconciliation":
        "conflicting views are integrated or resolved into coherent synthesis",
}

_TEMPLATE = """You are annotating a language model's reasoning trace for conversational \
structure.

Count the number of distinct instances of each of the following behaviours. Return an \
integer count for each, using 0 if none are present.

{definitions}

Also report n_personas: the number of distinct perspectives present in the trace. A trace \
written in a single undifferentiated voice has n_personas = 1.

Judge only what is in the trace. Do not reward or penalise whether the reasoning is \
correct.

Return ONLY a JSON object with exactly these keys and integer values:
{{"question_answering": <int>, "perspective_shift": <int>, \
"conflict_of_perspectives": <int>, "reconciliation": <int>, "n_personas": <int>}}

--- BEGIN TRACE ---
{trace}
--- END TRACE ---
"""


def build_prompt(trace: str) -> str:
    defs = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in DEFINITIONS.items())
    return _TEMPLATE.format(definitions=defs, trace=trace)


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        return brace.group(0)
    raise ValueError(f"no JSON object in judge output: {text[:200]!r}")


def _as_count(value, field: str) -> int:
    # bool is a subclass of int; True would otherwise silently become 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} is not an integer: {value!r}")
    if value < 0:
        raise ValueError(f"{field} is negative: {value}")
    return value


def parse_verdict(text: str) -> dict:
    """Parse a judge response. Raises ValueError rather than defaulting anything to 0."""
    try:
        obj = json.loads(_extract_json(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"judge output is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"judge returned {type(obj).__name__}, not an object")

    out = {}
    for field in (*BEHAVIOURS, "n_personas"):
        if field not in obj:
            raise ValueError(f"judge omitted {field!r} -- refusing to read that as 0")
        out[field] = _as_count(obj[field], field)
    return out


def cache_key(trace: str, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    h = hashlib.sha256()
    h.update(prompt_version.encode())
    h.update(b"\x00")
    h.update(model.encode())
    h.update(b"\x00")
    h.update(trace.encode())
    return h.hexdigest()


def _load_cache(path) -> dict:
    if not path or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            out[rec["key"]] = rec["verdict"]
        except (json.JSONDecodeError, KeyError):
            continue      # a half-written line from a killed run, not a reason to stop
    return out


def judge_trace(trace: str, backend, cache=None, model: str = "unknown") -> dict:
    """Judge one trace, reading through a content-addressed cache.

    The cache is append-only JSONL so a killed run loses at most the line in flight, and
    the key includes both the judge model and the prompt version -- mixing either inside
    one curve would make a change of instrument look like a change in the model.
    """
    key = cache_key(trace, model)
    hits = _load_cache(cache)
    if key in hits:
        return hits[key]

    verdict = parse_verdict(backend(build_prompt(trace)))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "a") as fh:
            fh.write(json.dumps({"key": key, "model": model,
                                 "prompt_version": PROMPT_VERSION,
                                 "verdict": verdict}) + "\n")
    return verdict


def anthropic_backend(model: str = "claude-opus-5", max_tokens: int = 300):
    """A judge backend backed by the Anthropic API. Requires ANTHROPIC_API_KEY."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(prompt: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    return call
