"""Build the DDM pilot dataset from the committed HSE recheck results.

Source: results/steering/hse_recheck_rg.json `per_trace` (the RE-GRADED traces —
its 24% baseline matches the corrected grader; hse_recheck.json is the tag-only
grading with the known 5.5% artifact).

per_trace has no pid, but run_sweep iterates the same seeded problem list for
every condition, so the file is blocked 6 conditions x 200 problems with
position = problem. That alignment is validated below, not assumed: pid-aligned
correctness must correlate across conditions well above a shuffled-alignment
floor. If the raw gate_dose.jsonl is ever re-exported with pid/sample fields
(commit 911c4fb did this for later runs), prefer those over this reconstruction.

RT proxy = kilowords of trace. near_ceiling flags likely budget-truncated traces
(>2,400 words); censoring is dose-dependent (7% at baseline, 46% at alpha=1.0).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "results" / "steering" / "hse_recheck_rg.json"
OUT = HERE / "countdown_ladder.csv"
ALPHAS = [0.0, 0.25, 0.5, 0.678, 1.0, 1.693]
BLOCK = 200
CEILING_WORDS = 2400


def phi(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    pt = json.loads(SOURCE.read_text())["per_trace"]
    assert len(pt) == len(ALPHAS) * BLOCK, len(pt)
    blocks = {a: pt[i * BLOCK:(i + 1) * BLOCK] for i, a in enumerate(ALPHAS)}
    for a, rows in blocks.items():
        assert all(r["alpha"] == a for r in rows), f"block order broken at alpha={a}"

    # Validate the position=pid assumption: aligned correctness must beat a
    # shuffled floor. Observed on this file: phi ~ +0.15..+0.39 vs ~0.03 shuffled.
    c0 = [r["correct"] for r in blocks[0.0]]
    aligned = [phi(c0, [r["correct"] for r in blocks[a]]) for a in ALPHAS[1:]]
    rng = np.random.default_rng(0)
    sh = rng.permutation(c0)
    floor = abs(phi(sh, [r["correct"] for r in blocks[0.5]]))
    assert min(aligned) > max(0.10, 2 * floor), (aligned, floor)
    print(f"pid alignment OK: phi = {[f'{v:+.3f}' for v in aligned]} (shuffled {floor:+.3f})")

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "alpha", "correct", "words", "rt", "response", "near_ceiling"])
        for i, r in enumerate(pt):
            blk, pos = divmod(i, BLOCK)
            w.writerow([pos, ALPHAS[blk], int(r["correct"]), r["words"],
                        f"{r['words'] / 1000.0:.3f}", 1 if r["correct"] else -1,
                        int(r["words"] > CEILING_WORDS)])
    print(f"wrote {OUT.relative_to(HERE.parent.parent)}: {len(pt)} rows")


if __name__ == "__main__":
    main()
