"""Is the diversity result just length? Stratified comparison at matched trace length.

`results/qwq/FINDINGS.md` reports that within QwQ, correct traces are LESS diverse than
incorrect ones. But incorrect traces are also 2.3x longer (2,100 vs 924 median words), and
length is not innocent: more text means more perspective-shift cues, more segments, and
more opportunity for embeddings to spread out. `hse_norm` divides out segment *count* via
log2(N), but that is not the same as controlling for length.

Two rival readings, and the pooled number cannot tell them apart:

  A. diversity anti-predicts correctness
  B. long traces are both more diverse and more often wrong, and diversity carries no
     independent information

This stratifies on trace length and asks the question *within* length bins, where the two
groups are directly comparable. If the effect survives at matched length, reading A holds.
If it vanishes, reading B does, and the headline must be restated.

Stratification rather than regression on purpose: it makes the overlap visible. Where the
two groups barely coexist -- very short traces are nearly all correct, very long ones
nearly all wrong -- no amount of modelling recovers a comparison, and a bin table shows
that plainly while a single coefficient hides it.

    python -m analysis.hse_qwq_length --run results/qwq/hse_shuffled_minilm.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRIC_DEFAULT = "hse_norm"


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(difference, ci_low, ci_high) for mean(a) - mean(b)."""
    d = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return d, d - 1.96 * se, d + 1.96 * se


def match_on_length(recs: list[dict], metric: str = METRIC_DEFAULT,
                    caliper: float = 0.05) -> dict:
    """1:1 nearest-neighbour matching on trace length, within a relative caliper.

    Coarse stratification is not enough on its own: inside a wide bin the correct traces
    still sit at the short end and the incorrect ones at the long end, so a length
    gradient survives the binning. Our own test with a PURE length confound (diversity a
    function of words only) was not dissolved by 8 quantile bins -- the residual gradient
    leaked straight through.

    Matching removes most of it. Each correct trace is paired with the unused incorrect
    trace closest in length, and the pair is kept only if the lengths agree within
    `caliper` (relative, so 0.05 = within 5%). The comparison is then a paired difference
    over traces of essentially identical length.

    What matching CANNOT do: drive a length confound to exactly zero. Any imbalance
    surviving inside the caliper is real bias, and at n in the thousands it is detectable.
    So read the **shrinkage** -- how much smaller the matched estimate is than the
    unadjusted one -- rather than treating a nominally significant matched estimate as
    proof of an independent effect. A confound that is purely length shrinks by most of
    its size; a genuine effect barely shrinks at all.
    """
    cor = sorted((r for r in recs if r["correct"]), key=lambda r: r["words"])
    inc = sorted((r for r in recs if not r["correct"]), key=lambda r: r["words"])
    if not cor or not inc:
        return {"metric": metric, "caliper": caliper, "n_pairs": 0}

    inc_w = np.array([r["words"] for r in inc], dtype=float)
    used = np.zeros(len(inc), dtype=bool)
    diffs, pair_words = [], []
    for c in cor:
        w = float(c["words"])
        # nearest unused incorrect trace by length
        cand = np.where(~used)[0]
        if not len(cand):
            break
        j = cand[int(np.argmin(np.abs(inc_w[cand] - w)))]
        if abs(inc_w[j] - w) > caliper * max(w, 1.0):
            continue                          # no acceptable match within the caliper
        used[j] = True
        diffs.append(float(c[metric]) - float(inc[j][metric]))
        pair_words.append((w, float(inc_w[j])))

    if len(diffs) < 30:
        return {"metric": metric, "caliper": caliper, "n_pairs": len(diffs),
                "note": "too few matched pairs to estimate"}
    d = np.array(diffs, dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    mean = float(d.mean())
    cw = np.array(pair_words)
    return {
        "metric": metric, "caliper": caliper, "n_pairs": len(d),
        "difference": mean, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
        "ci_excludes_zero": bool(abs(mean) > 1.96 * se),
        "mean_words_correct": float(cw[:, 0].mean()),
        "mean_words_incorrect": float(cw[:, 1].mean()),
        "max_relative_length_gap": float(np.abs(cw[:, 0] - cw[:, 1]).max()
                                        / max(cw[:, 0].mean(), 1.0)),
    }


def stratify(recs: list[dict], metric: str = METRIC_DEFAULT, n_bins: int = 8,
             min_per_cell: int = 30) -> dict:
    """Compare correct vs incorrect within equal-count bins of trace length.

    Bins are quantiles of `words` over the pooled sample, so each holds a similar number
    of traces. A bin is only usable if BOTH classes have >= min_per_cell members; the rest
    are reported but excluded from the pooled estimate.
    """
    words = np.array([r["words"] for r in recs], dtype=float)
    edges = np.quantile(words, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf                       # keep the top bin closed on the right

    bins, usable = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in recs if lo <= r["words"] < hi]
        a = np.array([r[metric] for r in sel if r["correct"]], dtype=float)
        b = np.array([r[metric] for r in sel if not r["correct"]], dtype=float)
        row = {"lo": float(lo), "hi": (None if np.isinf(hi) else float(hi)),
               "n_correct": len(a), "n_incorrect": len(b)}
        if len(a) >= min_per_cell and len(b) >= min_per_cell:
            d, cl, ch = _welch(a, b)
            row.update({"correct_mean": float(a.mean()), "incorrect_mean": float(b.mean()),
                        "difference": d, "ci95_low": cl, "ci95_high": ch,
                        "ci_excludes_zero": bool(cl > 0 or ch < 0), "usable": True})
            usable.append((row, len(a) + len(b)))
        else:
            row["usable"] = False
        bins.append(row)

    out = {"metric": metric, "n_bins": n_bins, "min_per_cell": min_per_cell, "bins": bins}
    if usable:
        # Pool by inverse variance of each bin's difference -- the standard fixed-effects
        # combination, which weights precisely-estimated bins more heavily.
        diffs = np.array([r["difference"] for r, _ in usable])
        ses = np.array([(r["ci95_high"] - r["difference"]) / 1.96 for r, _ in usable])
        w = 1.0 / np.maximum(ses ** 2, 1e-12)
        pooled = float((w * diffs).sum() / w.sum())
        se_p = float(np.sqrt(1.0 / w.sum()))
        out["pooled"] = {
            "difference": pooled, "ci95_low": pooled - 1.96 * se_p,
            "ci95_high": pooled + 1.96 * se_p,
            "ci_excludes_zero": bool(abs(pooled) > 1.96 * se_p),
            "n_usable_bins": len(usable),
            "n_traces": int(sum(n for _, n in usable)),
        }
    return out


def report_matched(m: dict, unadjusted: float | None = None) -> None:
    print("=" * 82)
    print(f"{m['metric'].upper()} AT MATCHED LENGTH  (1:1 nearest-neighbour, "
          f"caliper {m['caliper']:.0%})")
    print("=" * 82)
    if m.get("n_pairs", 0) < 30:
        print(f"  only {m.get('n_pairs', 0)} matched pairs -- "
              f"{m.get('note', 'cannot estimate')}")
        return
    star = "*" if m["ci_excludes_zero"] else " "
    print(f"  matched pairs      : {m['n_pairs']}")
    print(f"  mean words         : {m['mean_words_correct']:.0f} correct vs "
          f"{m['mean_words_incorrect']:.0f} incorrect")
    print(f"  paired difference  : {m['difference']:+.4f} "
          f"[{m['ci95_low']:+.4f},{m['ci95_high']:+.4f}]{star}")
    if unadjusted is not None:
        print(f"  unadjusted         : {unadjusted:+.4f}")
        print(f"  => matching changes the effect by "
              f"{abs(m['difference']) - abs(unadjusted):+.4f} "
              f"({1 - abs(m['difference'])/max(abs(unadjusted),1e-12):.0%} shrinkage)")
    print()
    if not m["ci_excludes_zero"]:
        print("  VERDICT: at matched length the difference is NOT distinguishable from")
        print("    zero. Length explains the pooled result; diversity carries no")
        print("    independent signal.")
    elif m["difference"] < 0:
        print("  VERDICT: SURVIVES matching -- among traces of the same length, the")
        print("    correct ones are less diverse. Diversity carries information beyond")
        print("    length.")
    else:
        print("  VERDICT: sign REVERSES under matching -- at equal length the correct")
        print("    traces are MORE diverse.")


def report(res: dict, pooled_unadjusted: float | None = None) -> None:
    m = res["metric"]
    print("=" * 82)
    print(f"IS THE {m.upper()} RESULT JUST LENGTH?  (stratified on words per trace)")
    print("=" * 82)
    print(f"{'words':>18}{'n_cor':>7}{'n_inc':>7}{'correct':>10}{'incorrect':>11}"
          f"{'difference':>26}")
    for b in res["bins"]:
        rng = f"{int(b['lo'])}-{'inf' if b['hi'] is None else int(b['hi'])}"
        if not b["usable"]:
            print(f"{rng:>18}{b['n_correct']:>7}{b['n_incorrect']:>7}"
                  f"{'--':>10}{'--':>11}{'(too few in one class)':>26}")
            continue
        star = "*" if b["ci_excludes_zero"] else " "
        print(f"{rng:>18}{b['n_correct']:>7}{b['n_incorrect']:>7}"
              f"{b['correct_mean']:>10.4f}{b['incorrect_mean']:>11.4f}"
              f"{b['difference']:>+11.4f} [{b['ci95_low']:+.4f},{b['ci95_high']:+.4f}]{star}")

    p = res.get("pooled")
    if not p:
        print("\nno bin had both classes represented -- length and outcome are too "
              "collinear here to compare at matched length")
        return
    star = "*" if p["ci_excludes_zero"] else " "
    print(f"\nPOOLED over {p['n_usable_bins']} usable bins ({p['n_traces']} traces): "
          f"{p['difference']:+.4f} [{p['ci95_low']:+.4f},{p['ci95_high']:+.4f}]{star}")
    if pooled_unadjusted is not None:
        print(f"unadjusted (whole sample):                        "
              f"{pooled_unadjusted:+.4f}")
        shrink = 1.0 - abs(p["difference"]) / max(abs(pooled_unadjusted), 1e-12)
        print(f"=> controlling for length shrinks the effect by {shrink:.0%}")
    print()
    if not p["ci_excludes_zero"]:
        print("VERDICT: at matched length the difference is NOT distinguishable from zero.")
        print("  The pooled result is explained by length, not by diversity. The headline")
        print("  must be restated as: longer traces are both more diverse and more often")
        print("  wrong, and diversity carries no independent signal.")
    elif p["difference"] < 0:
        print("VERDICT: the effect SURVIVES at matched length -- correct traces are less")
        print("  diverse than incorrect traces of the same length. Diversity carries")
        print("  information beyond length.")
    else:
        print("VERDICT: at matched length the sign REVERSES -- correct traces are MORE")
        print("  diverse. The pooled negative was a length artifact and the paper's")
        print("  predicted direction holds within length strata.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=Path("results/qwq/hse_shuffled_minilm.json"))
    ap.add_argument("--metric", default=METRIC_DEFAULT)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--min-per-cell", type=int, default=30)
    ap.add_argument("--caliper", type=float, default=0.05,
                    help="relative length tolerance for 1:1 matching (0.05 = within 5%%)")
    ap.add_argument("--out", type=Path, default=Path("results/qwq/hse_length_matched.json"))
    args = ap.parse_args()

    d = json.loads(args.run.read_text())
    recs = d["per_trace"]
    unadj = d["summary"]["metrics"].get(args.metric, {}).get("difference")
    matched = match_on_length(recs, args.metric, args.caliper)
    report_matched(matched, unadj)
    print()
    res = stratify(recs, args.metric, args.bins, args.min_per_cell)
    res["source_run"] = args.run.name
    res["unadjusted_difference"] = unadj
    res["matched"] = matched
    report(res, unadj)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
