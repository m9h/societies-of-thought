"""Build the MATH-Hard discriminating dataset for the pre-registered DDM fit.

`NEXT.md` §1 pre-registers this test and states the MATH traces "live on the
pod/workstation, not in this repo". They are in the repo: `results/steering/main_rg.jsonl`
carries 800 math_hard rows, including the baseline (feature -1) and the paper's feature
30939 at alpha=1.0 — exactly the two conditions the prediction discriminates on.

Two improvements over `build_dataset.py`, both because this source is the raw sweep rather
than the measured HSE output:

  * **pid is real, not reconstructed.** `main_rg.jsonl` carries `pid` per row, so the
    position=problem assumption that file has to validate statistically is retired here.
  * grading is the re-graded (`_rg`) file, matching the corrected grader.

Same conventions as the Countdown builder so the fits are comparable: rt = kilowords of
trace, response = +1 correct / -1 error, near_ceiling flags likely budget truncation.

THE PRE-REGISTERED PREDICTION (NEXT.md §1, written before these traces were touched):
    exhaust story  -> boundary UP, drift DOWN at alpha=1.0
    paper's story  -> drift UP on MATH too
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "results" / "steering" / "main_rg.jsonl"
OUT = HERE / "math_pair.csv"

TASK = "math_hard"
PAPER_FEATURE = 30939
CEILING_WORDS = 2400          # same threshold as the Countdown ladder


def load_pair(path: Path = SOURCE):
    """Return (baseline_rows, steered_rows) for the paper's feature at alpha=1.0."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    math = [r for r in rows if r.get("task") == TASK]
    base = [r for r in math if r.get("feature") == -1]
    steer = [r for r in math if r.get("feature") == PAPER_FEATURE
             and float(r.get("alpha", 0)) == 1.0]
    return base, steer


def to_records(base, steer) -> list[dict]:
    out = []
    for cond, rows in (("0.0", base), ("1.0", steer)):
        for r in rows:
            words = len((r.get("trace") or "").split())
            out.append({
                "pid": r["pid"],
                "alpha": cond,
                "correct": int(bool(r.get("correct"))),
                "words": words,
                "rt": round(words / 1000.0, 3),
                "response": 1 if r.get("correct") else -1,
                "near_ceiling": int(words > CEILING_WORDS),
            })
    return out


def main() -> None:
    base, steer = load_pair()
    if not base or not steer:
        raise SystemExit(f"missing a condition: baseline={len(base)} steered={len(steer)}")

    # The two conditions must cover the SAME problems, or this is not a paired contrast.
    pb, ps = {r["pid"] for r in base}, {r["pid"] for r in steer}
    shared = pb & ps
    print(f"baseline {len(base)} rows / {len(pb)} pids; "
          f"steered {len(steer)} rows / {len(ps)} pids; shared pids {len(shared)}")
    if len(shared) < 0.9 * min(len(pb), len(ps)):
        raise SystemExit("conditions do not cover the same problems -- not a paired design")

    recs = to_records(base, steer)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    for cond in ("0.0", "1.0"):
        s = [r for r in recs if r["alpha"] == cond]
        acc = sum(r["correct"] for r in s) / len(s)
        ceil = sum(r["near_ceiling"] for r in s) / len(s)
        rt = sorted(r["rt"] for r in s)[len(s) // 2]
        print(f"  alpha={cond}: n={len(s)} acc={acc:.1%} median_rt={rt:.2f}kw "
              f"near_ceiling={ceil:.0%}")
    print(f"wrote {OUT.relative_to(HERE.parent.parent)}: {len(recs)} rows")


if __name__ == "__main__":
    main()
