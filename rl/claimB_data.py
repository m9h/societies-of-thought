"""Claim B data: turn the paired dialogue/monologue traces into SFT + PPO datasets.

Claim B (the paper's main event) asks whether priming a base model on multi-agent
DIALOGUE vs single-voice MONOLOGUE -- over identical problems with identical
answers -- makes the dialogue-primed model learn faster under the SAME PPO. The
paired traces already exist and are verified (rl/generate_sft.py, both arms solve
identical problems). This module converts them into what verl needs, holding one
invariant above all:

    THE PROMPT IS SHARED. Every arm -- baseline, dialogue, monologue -- sees the
    same Countdown prompt, and that prompt does NOT pre-open a <think> tag.

Why not stock TinyZero. Tier-0/Claim A used TinyZero's 'base' template, which ends
"Assistant: Let me solve this step by step.\n<think>". That trailing "<think>"
would force the dialogue arm's "<persona1> ..." opening out-of-distribution the
instant PPO starts, quietly erasing the priming this experiment is meant to test.
So Claim B ends the prompt at "Assistant:" and lets each arm open its own tag. The
baseline arm is re-run on this same prompt so all three are strictly comparable --
a documented deviation from the stock recipe, made once, applied to every arm.

Gradability. The dialogue traces state their answer in <group_consensus>; the PPO
scorer reads <answer>. rl.reward._normalise_answer_container already rewrites the
former into the latter (it exists because this bug inverted a result once). Every
SFT response here is passed through it, so both arms are graded by the identical
scorer PPO will use. tests/test_claimB_data.py pins that every response scores 1.0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

from rl.reward import _normalise_answer_container

# Stock TinyZero 'base' Countdown instructions, verbatim, EXCEPT the trailing
# "Let me solve this step by step.\n<think>" is dropped so the prompt ends neutrally
# at "Assistant:" -- see the module docstring for why that matters.
COUNTDOWN_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer.\n"
    "User: Using the numbers {numbers}, create an equation that equals {target}. You "
    "can use basic arithmetic operations (+, -, *, /) and each number can only be used "
    "once. Show your work in <think> </think> tags. And return the final answer in "
    "<answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\n"
    "Assistant:"
)

_END_ANSWER = re.compile(r"</answer>", re.I)


def make_prompt(numbers, target) -> str:
    """The shared prompt. Identical for every arm; ends at 'Assistant:'."""
    return COUNTDOWN_PROMPT.format(numbers=list(numbers), target=int(target))


def to_response(trace: str) -> str:
    """Make a raw trace into a gradable SFT target.

    Normalise <group_consensus> -> <answer> (so the stock scorer can read it), then
    trim anything after the final </answer> so the model learns to stop once it has
    committed an answer.
    """
    norm = _normalise_answer_container(trace.strip())
    ends = list(_END_ANSWER.finditer(norm))
    if ends:
        norm = norm[: ends[-1].end()]
    return norm.rstrip()


def sft_records(json_path, arm: str) -> list[dict]:
    """[{pid, prompt, response}] for one arm. prompt is shared; response is the trace."""
    rows = json.loads(Path(json_path).read_text())
    out = []
    for r in rows:
        out.append({
            "pid": r["pid"],
            "prompt": make_prompt(r["numbers"], r["target"]),
            "response": to_response(r[arm]),
        })
    return out


def ppo_records(rows: list[dict], split: str) -> list[dict]:
    """verl RL parquet schema, matching TinyZero's countdown.py output columns so the
    stock 'countdown' reward function grades it -- but with our shared prompt."""
    out = []
    for i, r in enumerate(rows):
        nums, target = list(r["numbers"]), int(r["target"])
        out.append({
            "data_source": "countdown",
            "prompt": [{"role": "user", "content": make_prompt(nums, target)}],
            "ability": "math",
            "reward_model": {"style": "rule",
                             "ground_truth": {"target": target, "numbers": nums}},
            "extra_info": {"split": split, "index": r.get("pid", i)},
        })
    return out


def length_stats(texts: list[str]) -> dict:
    """Character/word length summary -- for recording the dialogue>monologue length
    confound (more priming tokens), not for gating anything."""
    chars = sorted(len(t) for t in texts)
    words = sorted(len(t.split()) for t in texts)
    return {
        "n": len(texts),
        "median_chars": int(median(chars)) if chars else 0,
        "median_words": int(median(words)) if words else 0,
        "max_chars": chars[-1] if chars else 0,
    }


# ---- CLI: write the parquets verl consumes -------------------------------------

def _write_parquet(records: list[dict], out_path: Path) -> None:
    import pandas as pd  # lazy: tests exercise the pure functions without pandas

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Claim B SFT + PPO parquets.")
    ap.add_argument("--data", type=Path, default=Path("rl/data"))
    ap.add_argument("--out", type=Path, default=Path("rl/data/claimB"))
    ap.add_argument("--countdown-parquet", type=Path, default=None,
                    help="optional: a countdown train.parquet whose (numbers,target) "
                         "rows become the PPO set with OUR shared prompt")
    args = ap.parse_args()

    # SFT: one {prompt,response} parquet per arm.
    for arm in ("dialogue", "monologue"):
        for split in ("train", "val"):
            recs = sft_records(args.data / f"{arm}_{split}.json", arm)
            _write_parquet(recs, args.out / f"sft_{arm}_{split}.parquet")
            stats = length_stats([r["response"] for r in recs])
            print(f"  sft_{arm}_{split}: {len(recs)} rows, "
                  f"median {stats['median_words']}w / {stats['median_chars']}c")

    print("SFT parquets written. Length confound (median words):")
    for arm in ("dialogue", "monologue"):
        recs = sft_records(args.data / f"{arm}_train.json", arm)
        print(f"    {arm:10} {length_stats([r['response'] for r in recs])['median_words']}")


if __name__ == "__main__":
    main()
