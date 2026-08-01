"""LIAR2 misinformation task for SoT Claim C6 (cross-domain transfer).

The paper's transfer experiment:

    We further test whether conversational scaffolding transfers across domains. Models
    fine-tuned on multi-agent dialogues for the Countdown task are evaluated on a
    qualitatively different task: political misinformation detection ... from 23,299
    fact-checked claims from PolitiFact.

    ... six PolitiFact labels -- True, Mostly True, Half True, Mostly False, False, and
    Pants on Fire -- ... which we recode into three categories

**The corpus.** `chengxuphd/liar2` is the extended PolitiFact benchmark: 22,962 claims
(18,369 / 2,297 / 2,296), within 1.5% of the paper's 23,299, and its per-speaker count
columns are named for exactly the paper's six labels. The paper does not name its source
file, so this is the closest public match rather than a certainty.

**The int -> name mapping was verified empirically, not assumed.** The card does not
document it. Reading statements and PolitiFact justifications: label 0 covers "COVID-19
vaccines are weapons of mass destruction" and a fabricated paid-protester story, while
label 5 covers claims whose justification supports them ("We found support for that policy
at 94 percent"). So the scale ascends in truthfulness, 0 = pants-on-fire, 5 = true. Getting
this backwards would invert the entire task silently.

**Priming for C6 is Countdown, not the out-of-domain pool.** The paper is explicit: "SFT on
correct multi-agent dialogues using 'Countdown task', not misinformation detection task".
That differs from C5, whose priming is drawn from the 8,262-problem general pool. So C6
uses `rl/data/*.json` (the Countdown dialogues) rather than `rl/data/ood/`.

**What we add.** The paper compares only two conditions here -- baseline and
conversation-primed. With no monologue arm it cannot separate "conversational structure
transfers" from "any priming transfers". We run all three.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# LIAR2 integer labels, ascending truthfulness. Verified against corpus content.
LABEL_NAMES = ["pants-on-fire", "false", "mostly-false", "half-true", "mostly-true", "true"]

# The paper's 6 -> 3 recoding, verbatim:
#   True      = {True, Mostly True}
#   Half True = {Half True}
#   False     = {False, Mostly False, Pants on Fire}
_RECODE = {5: "true", 4: "true", 3: "half-true", 2: "false", 1: "false", 0: "false"}

VERDICTS = ("true", "half-true", "false")

# The scaffold ends at "Assistant:" for the same reason as Countdown: verl's scorer
# locates the response by splitting on that marker and returns None (score 0) without it.
# It also must NOT pre-open <think>, which would force a dialogue-primed model's
# <persona1> opening out of distribution.
PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer.\n"
    "User: A fact-checker rated the following political claim. Decide the rating.\n"
    "Claim: {statement}\n"
    "Answer exactly one of: true, half-true, false. Show your work in <think> </think> "
    "tags. And return the final answer in <answer> </answer> tags, for example "
    "<answer> false </answer>.\n"
    "Assistant:"
)


def recode_label(raw: int) -> str:
    """Map a LIAR2 six-way label onto the paper's three categories."""
    if raw not in _RECODE:
        raise ValueError(f"label must be 0-5, got {raw!r}")
    return _RECODE[raw]


def make_prompt(statement: str) -> str:
    return PROMPT.format(statement=statement.strip())


def build_records(rows, split: str) -> list[dict]:
    """verl RL parquet schema, with a distinct data_source so our scorer is dispatched."""
    out = []
    for i, r in enumerate(rows):
        verdict = recode_label(int(r["label"]))
        out.append({
            "data_source": "liar2",
            "prompt": [{"role": "user", "content": make_prompt(r["statement"])}],
            "ability": "fact-checking",
            "reward_model": {"style": "rule", "ground_truth": {"verdict": verdict}},
            "extra_info": {"split": split, "index": i, "raw_label": int(r["label"])},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the LIAR2 PPO parquets for C6.")
    ap.add_argument("--out", type=Path, default=Path("rl/data/liar2"))
    ap.add_argument("--val-size", type=int, default=1024,
                    help="held-out claims per eval, matching the paper's Countdown 1,024")
    args = ap.parse_args()

    from datasets import load_dataset

    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, hf_split in (("train", "train"), ("test", "validation")):
        ds = load_dataset("chengxuphd/liar2", split=hf_split)
        rows = [{"label": r["label"], "statement": r["statement"]} for r in ds]
        if split == "test":
            rows = rows[: args.val_size]
        recs = build_records(rows, split)
        pd.DataFrame(recs).to_parquet(args.out / f"{split}.parquet", index=False)
        dist: dict[str, int] = {}
        for r in recs:
            v = r["reward_model"]["ground_truth"]["verdict"]
            dist[v] = dist.get(v, 0) + 1
        stats[split] = {"n": len(recs), "verdicts": dist}
        print(f"  {split}: {len(recs)} claims, verdicts {dist}")

    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
