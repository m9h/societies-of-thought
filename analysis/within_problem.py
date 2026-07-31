"""Compare diversity between a problem's OWN correct and incorrect traces.

`results/qwq/hse_domains.json` found that on GPQA, correct traces are more diverse than
length-matched incorrect ones (+0.0133, p=0.0003, survives Bonferroni over four domains).
That is the first result in this project pointing the paper's way, so it deserves the
harshest available test.

The weakness of the length-matched estimate is that the two traces in a pair come from
DIFFERENT problems. Problem difficulty is uncontrolled, and difficulty plausibly drives
both error rate and trace structure. Sampling one problem k times and comparing its correct
traces against its own incorrect traces removes that entirely: the problem is held fixed by
construction.

Two things this design must be honest about, both enforced by tests:

  * **Only problems yielding both outcomes are informative**, which selects for middling
    difficulty. Problems the model always gets right or always gets wrong contribute
    nothing, and their counts are reported.
  * **Each problem contributes once.** Averaging over traces rather than problems would
    let a problem sampled twenty times outweigh one sampled twice.

Length is still worth watching: within a problem, incorrect traces may run longer, so the
achieved balance is reported rather than assumed away.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def pair_within_problem(recs, metric: str = "hse_norm",
                        min_problems: int = 30) -> dict:
    """Mean of (per-problem correct mean − per-problem incorrect mean).

    Returns counts always; returns an estimate only when enough problems yield both
    outcomes, since a handful of informative problems cannot support a CI.
    """
    by = defaultdict(lambda: {"cor": [], "inc": [], "w_cor": [], "w_inc": []})
    for r in recs:
        b = by[r["pid"]]
        key, wkey = ("cor", "w_cor") if r["correct"] else ("inc", "w_inc")
        b[key].append(float(r[metric]))
        b[wkey].append(float(r.get("words", 0)))

    diffs, wc, wi = [], [], []
    n_all_cor = n_all_inc = 0
    for _pid, b in by.items():
        if b["cor"] and b["inc"]:
            diffs.append(float(np.mean(b["cor"]) - np.mean(b["inc"])))
            wc.append(float(np.mean(b["w_cor"])))
            wi.append(float(np.mean(b["w_inc"])))
        elif b["cor"]:
            n_all_cor += 1
        elif b["inc"]:
            n_all_inc += 1

    out = {
        "metric": metric,
        "n_problems_total": len(by),
        "n_problems_used": len(diffs),
        "n_problems_all_correct": n_all_cor,
        "n_problems_all_incorrect": n_all_inc,
    }
    if len(diffs) < max(min_problems, 2):
        out["note"] = (f"only {len(diffs)} problems yielded both outcomes; "
                       f"need >= {min_problems} to estimate")
        return out

    d = np.array(diffs, dtype=float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    out.update({
        "difference": mean,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "ci_excludes_zero": bool(abs(mean) > 1.96 * se),
        "mean_words_correct": float(np.mean(wc)),
        "mean_words_incorrect": float(np.mean(wi)),
    })
    return out


def report(res: dict) -> None:
    print("=" * 82)
    print(f"WITHIN-PROBLEM DIVERSITY: {res['metric']} (problem held fixed)")
    print("=" * 82)
    print(f"  problems seen         : {res['n_problems_total']}")
    print(f"  informative (both)    : {res['n_problems_used']}")
    print(f"  always correct        : {res['n_problems_all_correct']}")
    print(f"  always incorrect      : {res['n_problems_all_incorrect']}")
    if "difference" not in res:
        print(f"\n  {res.get('note', 'cannot estimate')}")
        return
    star = "*" if res["ci_excludes_zero"] else " "
    print(f"  mean words            : {res['mean_words_correct']:.0f} correct vs "
          f"{res['mean_words_incorrect']:.0f} incorrect")
    print(f"\n  correct - incorrect   : {res['difference']:+.4f} "
          f"[{res['ci95_low']:+.4f},{res['ci95_high']:+.4f}]{star}")
    print()
    if not res["ci_excludes_zero"]:
        print("  VERDICT: holding the problem fixed, diversity does NOT differ. The")
        print("    length-matched effect was carried by differences BETWEEN problems.")
    elif res["difference"] > 0:
        print("  VERDICT: SURVIVES. On the same problem, the model's correct traces are")
        print("    more diverse than its own incorrect ones. Difficulty cannot explain it.")
    else:
        print("  VERDICT: sign REVERSES within problem -- correct traces are LESS diverse.")
    print("\n  Note: only problems the model gets both right and wrong are informative,")
    print("    so this estimate is conditioned on middling difficulty.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", type=Path, required=True,
                    help="hse_domains.json-style output containing per-trace records")
    ap.add_argument("--metric", default="hse_norm")
    ap.add_argument("--source", default=None, help="restrict to one domain, e.g. gpqa")
    ap.add_argument("--min-problems", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("results/qwq/within_problem.json"))
    args = ap.parse_args()

    d = json.loads(args.measured.read_text())
    recs = d.get("per_trace") or d
    if args.source:
        recs = [r for r in recs if r.get("source") == args.source]
    print(f"{len(recs)} traces" + (f" from {args.source}" if args.source else ""))

    res = pair_within_problem(recs, args.metric, args.min_problems)
    res["source"] = args.source
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
