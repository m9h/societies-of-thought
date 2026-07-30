"""Compare QwQ diversity runs across conditions, and state what is robust.

A single run of this measurement has already produced two opposite conclusions from the
same data (see `results/qwq/FINDINGS.md`: dropping single-voice traces vs scoring them
zero). So the result is only worth as much as its stability across the choices we made
arbitrarily: how the sample was drawn, and which embedding model scored it.

This prints one row per run and flags the metrics whose SIGN is consistent everywhere.

    python -m analysis.hse_qwq_compare results/qwq/*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("hse_norm", "mean_dist", "hse")


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    s = d["summary"]
    return {
        "name": path.stem,
        "embedder": (d.get("embedder") or "?").split("/")[-1],
        "shuffle": d.get("shuffle_buffer", "prefix"),
        "handling": s.get("dropped", {}).get("handling", "?"),
        "n": f"{s['n_correct']}/{s['n_incorrect']}",
        "metrics": s["metrics"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    runs = [load(p) for p in args.runs if p.exists()]
    if not runs:
        raise SystemExit("no runs found")

    print(f"{'run':<34}{'embedder':<20}{'shuffle':>9}{'single-voice':>14}{'n':>12}")
    for r in runs:
        print(f"{r['name']:<34}{r['embedder']:<20}{str(r['shuffle']):>9}"
              f"{r['handling']:>14}{r['n']:>12}")

    for m in METRICS:
        print(f"\n--- {m} ---")
        print(f"{'run':<34}{'correct':>10}{'incorrect':>11}{'difference':>26}")
        signs = []
        for r in runs:
            v = r["metrics"].get(m)
            if not v:
                continue
            star = "*" if v["ci_excludes_zero"] else " "
            print(f"{r['name']:<34}{v['correct_mean']:>10.4f}{v['incorrect_mean']:>11.4f}"
                  f"{v['difference']:>+11.4f} [{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}]{star}")
            if v["ci_excludes_zero"]:
                signs.append((r["handling"], 1 if v["difference"] > 0 else -1))

        # Robustness is judged only over runs that follow the paper's zero convention;
        # the `drop` runs are retained above as the cautionary comparison, not as evidence.
        paper = [s for h, s in signs if h == "zero"]
        if len(paper) >= 2:
            if all(s == paper[0] for s in paper):
                d = "correct MORE diverse" if paper[0] > 0 else "correct LESS diverse"
                print(f"  => SIGN STABLE across {len(paper)} paper-convention runs: {d}")
            else:
                print(f"  => SIGN UNSTABLE across paper-convention runs -- "
                      f"do not report a direction for {m}")
        elif len(paper) == 1:
            print("  => only one paper-convention run with a CI excluding zero; "
                  "not yet a robustness claim")
        else:
            print("  => no paper-convention run separates the groups on this metric")
    print("\n* = 95% CI on the difference excludes zero")


if __name__ == "__main__":
    main()
