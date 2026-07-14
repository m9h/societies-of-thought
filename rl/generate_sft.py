"""Generate the two SFT datasets: multi-agent DIALOGUE vs single-voice MONOLOGUE.

The experiment's whole validity rests on one property:

    Both arms must solve the IDENTICAL problems with the IDENTICAL correct answers,
    so the only thing that differs between them is the FORM of the reasoning.

If the dialogue set were filtered for correctness and the monologue set were not (or
they covered different problems), we would be comparing "correct reasoning" against
"any reasoning" and the result would be worthless. So we do not filter the arms
independently. We generate both, keep a problem only if BOTH arms solved it correctly,
and assert the resulting problem sets are equal.

A previous automated attempt produced 500 "dialogues" in which 98% of the arithmetic was
wrong ("11 + 59 = 10"). Every example here is verified with the same tested grader the
RL reward uses: if the stated equation does not actually evaluate to the target using
each number exactly once, it is discarded. No exceptions, no templates, no synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import http.client
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from rl.reward import accuracy_reward  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
_FAILURES: list = []
# DEVIATION FROM THE PAPER, RECORDED: the paper generated its SFT dialogues with
# Qwen-2.5-32B-Instruct. OpenRouter does not host that model ("not a valid model ID").
# The nearest available member of the same family is the 72B instruct model -- same
# generation and post-training recipe, larger. It is used for BOTH arms, so it cannot
# bias the dialogue-vs-monologue comparison; it only makes both sets of traces somewhat
# better than the paper's.
GEN_MODEL = "qwen/qwen-2.5-72b-instruct"

PERSONAS = [
    "an extrovert mathematician focused on arithmetic heuristics",
    "an analytical engineer emphasising step efficiency",
    "a cautious verifier who double-checks every calculation",
    "a creative lateral thinker who tries unusual groupings",
]

DIALOGUE_PROMPT = """You are writing a training example: {k} experts collaborating out loud on a Countdown puzzle.

Numbers: {nums}
Target: {target}

Rules: combine the numbers with + - * / and parentheses. Each number must be used EXACTLY ONCE. The expression must evaluate EXACTLY to {target}.

Write the collaboration in this format, with no other text:
{persona_block}
<think1> ... </think1>
<think2> ... </think2>
(alternate between the personas, letting them question and correct each other)
<group_consensus> THE_FINAL_EXPRESSION </group_consensus>

The content of <group_consensus> must be ONLY the arithmetic expression (no prose, no "=").
Solve it correctly. Verify your arithmetic before writing the consensus."""

MONOLOGUE_PROMPT = """You are writing a training example: one person solving a Countdown puzzle, thinking step by step.

Numbers: {nums}
Target: {target}

Rules: combine the numbers with + - * / and parentheses. Each number must be used EXACTLY ONCE. The expression must evaluate EXACTLY to {target}.

Write it in this format, with no other text:
<think> ... your step-by-step reasoning ... </think>
<answer> THE_FINAL_EXPRESSION </answer>

The content of <answer> must be ONLY the arithmetic expression (no prose, no "=").
Solve it correctly. Verify your arithmetic before writing the answer."""


def call(prompt: str, key: str, temperature: float = 0.7, retries: int = 3) -> str | None:
    body = json.dumps({
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 900,
        # Pin to providers that accept every parameter we send. OpenRouter otherwise
        # load-balances across backends, and one that rejects temperature/max_tokens
        # returns a 400 that looks permanent but is only that provider's problem.
        "provider": {"require_parameters": True},
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            permanent = e.code in (401, 403, 404) or "not a valid model" in last
            if permanent:
                raise SystemExit(f"API rejected the request -- fix this, do not retry:\n  {last}")
            time.sleep(2 * (attempt + 1))  # 400s from a bad provider are transient
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError,
                ConnectionError, json.JSONDecodeError, KeyError, OSError) as e:
            last = repr(e)  # truncated responses, resets, malformed JSON: all retryable
            time.sleep(2 * (attempt + 1))
    _FAILURES.append(last)
    return None


def make_dialogue(problem: dict, key: str, rng: random.Random) -> str | None:
    k = rng.choice([2, 3, 4])
    chosen = rng.sample(PERSONAS, k)
    persona_block = "\n".join(
        f"<persona{i+1}> {p} </persona{i+1}>" for i, p in enumerate(chosen)
    )
    text = call(DIALOGUE_PROMPT.format(
        k=k, nums=problem["nums"], target=problem["target"], persona_block=persona_block
    ), key)
    return _keep_if_correct(text, problem)


def make_monologue(problem: dict, key: str) -> str | None:
    text = call(MONOLOGUE_PROMPT.format(nums=problem["nums"], target=problem["target"]), key)
    return _keep_if_correct(text, problem)


def _keep_if_correct(text: str | None, problem: dict) -> str | None:
    """The only filter. The stated equation must actually solve the problem."""
    if not text:
        return None
    text = re.sub(r"^```[a-z]*\n?|```$", "", text.strip(), flags=re.M)
    if accuracy_reward(text, problem["target"], list(problem["nums"])) != 1.0:
        return None
    return text


def pair_arms(
    problems: list[dict],
    dialogues: list[str | None],
    monologues: list[str | None],
) -> list[tuple[dict, str, str]]:
    """Keep a problem only if BOTH arms solved it correctly. Re-verifies both traces.

    This is the one function that decides whether the experiment means anything. Filtering
    the arms independently would compare "correct reasoning" against "any reasoning";
    letting them cover different problems would compare different tasks. So the pairing is
    an intersection, and every surviving trace is re-checked against ITS OWN problem --
    which also catches an alignment bug, where each arm trains on solutions to the wrong
    problems while every count and assertion still looks healthy.

    Raises rather than returning empty: two empty sets are equal, so an equality assertion
    over zero data reports success. It did exactly that once.
    """
    paired: list[tuple[dict, str, str]] = []
    for problem, d, m in zip(problems, dialogues, monologues, strict=True):
        if d is None or m is None:
            continue
        t, nums = problem["target"], list(problem["nums"])
        if accuracy_reward(d, t, nums) != 1.0:
            raise ValueError(f"pid {problem['pid']}: dialogue does not solve its own problem")
        if accuracy_reward(m, t, nums) != 1.0:
            raise ValueError(f"pid {problem['pid']}: monologue does not solve its own problem")
        paired.append((problem, d, m))

    if not paired:
        raise ValueError(
            "no problem was solved correctly by both arms -- the dataset is empty. "
            "Zero verified examples is a broken generator, not a passing gate."
        )
    return paired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=1200,
                    help="problems to ATTEMPT; both arms must solve one for it to be kept")
    ap.add_argument("--n-keep", type=int, default=600, help="the paper kept 600")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("rl/data"))
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see ~/.openrouter.env)")

    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    problems = [{"nums": list(ds[i]["nums"]), "target": ds[i]["target"], "pid": i}
                for i in idx[: args.n_problems]]

    rng = random.Random(args.seed)
    print(f"attempting {len(problems)} problems with {GEN_MODEL}")

    with ThreadPoolExecutor(args.workers) as ex:
        dialogues = list(ex.map(lambda p: make_dialogue(p, key, rng), problems))
    ok_d = sum(d is not None for d in dialogues)
    print(f"  dialogue:  {ok_d}/{len(problems)} solved correctly")

    with ThreadPoolExecutor(args.workers) as ex:
        monologues = list(ex.map(lambda p: make_monologue(p, key), problems))
    ok_m = sum(m is not None for m in monologues)
    print(f"  monologue: {ok_m}/{len(problems)} solved correctly")

    # THE MATCHING STEP -- see rl/generate_sft.pair_arms, covered by tests/test_sft_pairing.py.
    # It raises on an empty pairing rather than returning one, because two empty sets are
    # equal and an equality assertion over zero data reports success.
    paired = pair_arms(problems, dialogues, monologues)
    print(f"  both arms correct: {len(paired)} problems")
    if len(paired) < args.n_keep:
        print(f"  WARNING: wanted {args.n_keep}, got {len(paired)}. Raise --n-problems.")
    paired = paired[: args.n_keep]

    n_val = max(1, len(paired) // 6)  # paper: 500 train / 100 val
    splits = {"train": paired[n_val:], "val": paired[:n_val]}

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        for arm, i in (("dialogue", 1), ("monologue", 2)):
            data = [
                {"pid": p["pid"], "numbers": p["nums"], "target": p["target"],
                 arm: (d if arm == "dialogue" else m)}
                for p, d, m in rows
            ]
            (args.out / f"{arm}_{split}.json").write_text(json.dumps(data, indent=1))

    # GATE 1. Note the emptiness check FIRST: two empty sets are equal, so without it a
    # run that generated nothing at all reports "GATE 1 PASSED". It did exactly that once.
    if _FAILURES:
        print(f"\n  {len(_FAILURES)} API calls failed; last: {_FAILURES[-1]}")
    for split in splits:
        a = {r["pid"] for r in json.loads((args.out / f"dialogue_{split}.json").read_text())}
        b = {r["pid"] for r in json.loads((args.out / f"monologue_{split}.json").read_text())}
        if not a:
            raise SystemExit(
                f"GATE 1 FAILED: {split} split is EMPTY. Zero verified examples is not a "
                "passing gate -- it is a broken generator."
            )
        assert a == b, f"{split}: arms cover different problems -- experiment is confounded"
    n_train = len(json.loads((args.out / "dialogue_train.json").read_text()))
    if n_train < 100:
        raise SystemExit(f"GATE 1 FAILED: only {n_train} training examples; too few to prime on.")
    print(f"\nGATE 1 PASSED: both arms cover identical problem sets, {n_train} train examples")
    print(f"wrote {args.out}/  ({len(splits['train'])} train / {len(splits['val'])} val)")


if __name__ == "__main__":
    main()
