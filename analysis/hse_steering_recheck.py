"""Re-audit the steering diversity result for the two errors found in the QwQ analysis.

`results/steering/FINDINGS.md` reports that steering the conversational feature makes the
induced society BIGGER but proportionally MORE REDUNDANT: segments per trace rise 21.4 ->
54.7 while normalised diversity falls 0.236 -> 0.190. That "redundant society" reading is
load-bearing for the whole project.

The QwQ analysis (`results/qwq/FINDINGS.md`) found two errors that both apply here:

  1. **Dropped single-voice traces.** `analysis/hse.py` skips traces with fewer than
     MIN_SEGMENTS segments. Steering changes how many perspective shifts a trace contains,
     so the drop RATE differs by condition -- the same asymmetric filtering that inverted
     the QwQ result. The paper's convention is to score a single voice as zero, not to
     exclude it.
  2. **Length.** Steering changes trace length as well as trace content. `hse_norm`
     divides out segment COUNT, not length, and in QwQ a 99% "effect" turned out to be
     length alone.

This recomputes the steering numbers under the paper's zero convention and then asks
whether any remaining alpha effect survives matching on trace length.

    python -m analysis.hse_steering_recheck --results results/steering/gate_dose.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.hse import MIN_SEGMENTS, hierarchic_social_entropy, segment
from analysis.hse_qwq_length import match_on_length


def per_trace(rows, model_name: str, degenerate: str = "zero", encoder=None):
    """Per-trace diversity for every steering condition, keeping single-voice traces."""
    if encoder is None:
        from sentence_transformers import SentenceTransformer

    enc = encoder if encoder is not None else SentenceTransformer(model_name)
    out, drops = [], defaultdict(int)
    for i, r in enumerate(rows):
        alpha = 0.0 if r.get("feature") == -1 else float(r.get("alpha", 0.0))
        trace = r.get("trace", "") or ""
        words = len(trace.split())
        segs = segment(trace)
        rec = {"alpha": alpha, "correct": bool(r.get("correct")), "words": words,
               "n_segments": len(segs), "single_voice": False}
        if len(segs) < MIN_SEGMENTS:
            drops[alpha] += 1
            if degenerate == "drop":
                continue
            rec.update({"hse": 0.0, "hse_norm": 0.0, "mean_dist": 0.0,
                        "single_voice": True})
            out.append(rec)
            continue
        E = enc.encode(segs, normalize_embeddings=True, show_progress_bar=False)
        D = np.clip(1.0 - (E @ E.T), 0.0, None)
        np.fill_diagonal(D, 0.0)
        hse, hse_n, md = hierarchic_social_entropy(D)
        if not np.isfinite(hse):
            drops[alpha] += 1
            if degenerate == "drop":
                continue
            hse, hse_n, md = 0.0, 0.0, 0.0
            rec["single_voice"] = True
        rec.update({"hse": float(hse), "hse_norm": float(hse_n), "mean_dist": float(md)})
        out.append(rec)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return out, dict(drops)


def summarise(recs) -> dict:
    by = defaultdict(list)
    for r in recs:
        by[r["alpha"]].append(r)
    rows = {}
    for a in sorted(by):
        v = by[a]
        rows[a] = {
            "n": len(v),
            "single_voice": sum(r["single_voice"] for r in v),
            "words": float(np.mean([r["words"] for r in v])),
            "segments": float(np.mean([r["n_segments"] for r in v])),
            "hse_norm": float(np.mean([r["hse_norm"] for r in v])),
            "mean_dist": float(np.mean([r["mean_dist"] for r in v])),
            "accuracy": float(np.mean([r["correct"] for r in v])),
        }
    return rows


def matched_vs_baseline(recs, alpha: float, caliper: float = 0.10) -> dict:
    """Compare one steering dose against alpha=0 at matched trace length.

    `match_on_length` pairs on a boolean `correct` field, so the condition label is mapped
    onto it: baseline traces play the role of "correct". The returned difference is
    therefore (baseline - steered).
    """
    sub = [{**r, "correct": (r["alpha"] == 0.0)}
           for r in recs if r["alpha"] in (0.0, alpha)]
    return match_on_length(sub, "hse_norm", caliper=caliper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("results/steering/gate_dose.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--caliper", type=float, default=0.10)
    ap.add_argument("--out", type=Path,
                    default=Path("results/steering/hse_recheck.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    print(f"{len(rows)} traces from {args.results.name}")

    recs, drops = per_trace(rows, args.model, degenerate="zero")
    summ = summarise(recs)

    print("\n" + "=" * 88)
    print("STEERING, RECOMPUTED WITH SINGLE-VOICE TRACES SCORED ZERO (the paper's rule)")
    print("=" * 88)
    print(f"{'alpha':>7}{'n':>6}{'1-voice':>9}{'words':>9}{'segments':>10}"
          f"{'hse_norm':>10}{'mean_dist':>11}{'accuracy':>10}")
    base = summ.get(0.0)
    for a, s in summ.items():
        print(f"{a:>7.3f}{s['n']:>6}{s['single_voice']:>9}{s['words']:>9.0f}"
              f"{s['segments']:>10.1f}{s['hse_norm']:>10.4f}{s['mean_dist']:>11.4f}"
              f"{s['accuracy']:>9.1%}")

    if base:
        print("\nchange vs alpha=0:")
        for a, s in summ.items():
            if a == 0.0:
                continue
            print(f"  alpha {a:<6.3f} words {s['words']-base['words']:+8.0f}  "
                  f"segments {s['segments']-base['segments']:+6.1f}  "
                  f"hse_norm {s['hse_norm']-base['hse_norm']:+.4f}  "
                  f"accuracy {s['accuracy']-base['accuracy']:+.1%}")

    print("\n" + "=" * 88)
    print("DOES THE hse_norm CHANGE SURVIVE MATCHING ON TRACE LENGTH?")
    print("=" * 88)
    matched = {}
    for a in sorted(summ):
        if a == 0.0:
            continue
        m = matched_vs_baseline(recs, a, args.caliper)
        matched[a] = m
        if m.get("n_pairs", 0) < 30:
            print(f"  alpha {a:<6.3f} only {m.get('n_pairs',0)} matched pairs -- "
                  f"lengths do not overlap enough to compare")
            continue
        unadj = summ[a]["hse_norm"] - base["hse_norm"]
        star = "*" if m["ci_excludes_zero"] else " "
        shrink = 1 - abs(m["difference"]) / max(abs(unadj), 1e-12)
        print(f"  alpha {a:<6.3f} pairs {m['n_pairs']:>4}  "
              f"unadjusted {-unadj:+.4f}  matched {m['difference']:+.4f} "
              f"[{m['ci95_low']:+.4f},{m['ci95_high']:+.4f}]{star}  "
              f"shrinkage {shrink:.0%}")
    print("\n  (differences are baseline - steered, so POSITIVE = steering lowers "
          "diversity)")
    print("  * = 95% CI excludes zero")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"source": args.results.name, "embedder": args.model, "caliper": args.caliper,
         "summary": {str(k): v for k, v in summ.items()},
         "matched_vs_baseline": {str(k): v for k, v in matched.items()},
         "per_trace": recs}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
