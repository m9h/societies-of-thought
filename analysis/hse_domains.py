"""Diversity vs correctness across NON-MATH domains, with the confound controls built in.

`results/qwq/FINDINGS.md` found a null within QwQ on math: at matched trace length,
perspective diversity does not predict correctness. This runs the same question over
chemistry, physics, biology, law, deductive reasoning and the rest of our pool, on traces
from the same model (`modal_qwq_domains.py`).

Written test-first. Every control here exists because its absence already cost this project
a published-in-repo result:

  * **truncated traces excluded by default.** A trace cut off at the token cap never
    reached an answer, so it is graded incorrect *and* is maximum-length by construction.
    Including them manufactures the length/correctness confound in its purest form.
  * **length matching is part of the result, not a follow-up.** The unadjusted number and
    the matched number are computed together and reported together, so nobody can quote
    the first without seeing the second.
  * **single-voice traces score zero**, the paper's own convention, never dropped —
    dropping them inverted both the QwQ and the steering results.
  * **per-domain breakdown**, because BBH, GPQA, MuSR and MMLU-Pro differ enormously in
    accuracy and trace length and a pooled figure over them is not interpretable.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.hse import MIN_SEGMENTS, hierarchic_social_entropy, segment
from analysis.hse_qwq_length import match_on_length

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def load_traces(rows, exclude_truncated: bool = True):
    """Return (kept_rows, drop_counts).

    Truncation is excluded by default and the count is always reported: a truncated trace
    is not a wrong answer, it is a missing one, and it sits at the maximum of the length
    distribution.
    """
    dropped = {"truncated": 0, "empty": 0}
    kept = []
    for r in rows:
        if exclude_truncated and r.get("truncated"):
            dropped["truncated"] += 1
            continue
        if not (r.get("response") or "").strip():
            dropped["empty"] += 1
            continue
        kept.append(r)
    return kept, dropped


def _measure(rows, encoder):
    """Per-trace diversity, keeping single-voice traces as zero (the paper's rule)."""
    recs, n_single = [], 0
    for r in rows:
        trace = r.get("response") or ""
        segs = segment(trace)
        # pid and sample are carried through so `analysis.within_problem` can group by
        # problem -- the control that holds difficulty fixed. Without them the two
        # modules cannot compose and the strongest test of the GPQA effect is unrunnable.
        rec = {"pid": r.get("pid"), "sample": r.get("sample", 0),
               "correct": bool(r.get("correct")), "source": r.get("source", "?"),
               "words": len(trace.split()), "n_segments": len(segs),
               "single_voice": False}
        if len(segs) < MIN_SEGMENTS:
            n_single += 1
            rec.update({"hse": 0.0, "hse_norm": 0.0, "mean_dist": 0.0,
                        "single_voice": True})
            recs.append(rec)
            continue
        E = encoder.encode(segs, normalize_embeddings=True, show_progress_bar=False)
        E = np.asarray(E)
        D = np.clip(1.0 - (E @ E.T), 0.0, None)
        np.fill_diagonal(D, 0.0)
        hse, hse_n, md = hierarchic_social_entropy(D)
        if not np.isfinite(hse):
            n_single += 1
            hse, hse_n, md = 0.0, 0.0, 0.0
            rec["single_voice"] = True
        rec.update({"hse": float(hse), "hse_norm": float(hse_n), "mean_dist": float(md)})
        recs.append(rec)
    return recs, n_single


def _unadjusted(recs, metric: str):
    a = np.array([r[metric] for r in recs if r["correct"]], dtype=float)
    b = np.array([r[metric] for r in recs if not r["correct"]], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    d = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return {"correct_mean": float(a.mean()), "incorrect_mean": float(b.mean()),
            "difference": d, "ci95_low": d - 1.96 * se, "ci95_high": d + 1.96 * se,
            "ci_excludes_zero": bool(abs(d) > 1.96 * se),
            "n_correct": int(len(a)), "n_incorrect": int(len(b))}


def analyse(rows, metric: str = "hse_norm", encoder=None,
            embedder: str = DEFAULT_EMBEDDER, exclude_truncated: bool = True,
            caliper: float = 0.05) -> dict:
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedder)

    kept, dropped = load_traces(rows, exclude_truncated)
    recs, n_single = _measure(kept, encoder)

    out = {"metric": metric, "n_input": len(rows), "n_measured": len(recs),
           "n_single_voice": n_single, "dropped": dropped,
           "unadjusted": _unadjusted(recs, metric) if recs else None,
           "matched": match_on_length(recs, metric, caliper) if recs else {"n_pairs": 0},
           "by_domain": {}, "per_trace": recs}

    by = defaultdict(list)
    for r in recs:
        by[r["source"]].append(r)
    for src, v in by.items():
        m = match_on_length(v, metric, caliper)
        out["by_domain"][src] = {
            "n": len(v),
            "n_correct": sum(r["correct"] for r in v),
            "n_incorrect": sum(not r["correct"] for r in v),
            "words_correct": float(np.mean([r["words"] for r in v if r["correct"]]))
            if any(r["correct"] for r in v) else None,
            "words_incorrect": float(np.mean([r["words"] for r in v if not r["correct"]]))
            if any(not r["correct"] for r in v) else None,
            "unadjusted": _unadjusted(v, metric),
            "matched": m if m.get("n_pairs", 0) >= 30 else None,
        }
    return out


def report(res: dict) -> None:
    print("=" * 88)
    print("DIVERSITY vs CORRECTNESS ACROSS DOMAINS  (QwQ, non-math)")
    print("=" * 88)
    print(f"  input {res['n_input']}  measured {res['n_measured']}  "
          f"single-voice {res['n_single_voice']}  dropped {res['dropped']}")

    u, m = res.get("unadjusted"), res.get("matched") or {}
    if u:
        print(f"\n  POOLED unadjusted : {u['difference']:+.4f} "
              f"[{u['ci95_low']:+.4f},{u['ci95_high']:+.4f}]"
              f"{'*' if u['ci_excludes_zero'] else ' '}  "
              f"(n {u['n_correct']}/{u['n_incorrect']})")
    if m.get("n_pairs", 0) >= 30:
        print(f"  POOLED matched    : {m['difference']:+.4f} "
              f"[{m['ci95_low']:+.4f},{m['ci95_high']:+.4f}]"
              f"{'*' if m['ci_excludes_zero'] else ' '}  ({m['n_pairs']} pairs, "
              f"{m['mean_words_correct']:.0f} vs {m['mean_words_incorrect']:.0f} words)")
        if u:
            shrink = 1 - abs(m["difference"]) / max(abs(u["difference"]), 1e-12)
            print(f"  => length control changes the effect by {shrink:.0%}")
    else:
        print("  POOLED matched    : too few length-matched pairs to estimate")

    print(f"\n{'domain':>10}{'n':>7}{'cor':>6}{'inc':>6}{'w_cor':>8}{'w_inc':>8}"
          f"{'unadjusted':>22}{'matched':>22}")
    for src, v in sorted(res["by_domain"].items()):
        uu, mm = v["unadjusted"], v["matched"]
        us = (f"{uu['difference']:+.4f}{'*' if uu['ci_excludes_zero'] else ' '}"
              if uu else "--")
        ms = (f"{mm['difference']:+.4f}{'*' if mm['ci_excludes_zero'] else ' '} "
              f"({mm['n_pairs']})" if mm else "--")
        wc = f"{v['words_correct']:.0f}" if v["words_correct"] is not None else "--"
        wi = f"{v['words_incorrect']:.0f}" if v["words_incorrect"] is not None else "--"
        print(f"{src:>10}{v['n']:>7}{v['n_correct']:>6}{v['n_incorrect']:>6}"
              f"{wc:>8}{wi:>8}{us:>22}{ms:>22}")
    print("\n* = 95% CI excludes zero;  w_cor/w_inc = mean words per trace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path,
                    default=Path("results/qwq/qwq_domains.json"))
    ap.add_argument("--metric", default="hse_norm")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--caliper", type=float, default=0.05)
    ap.add_argument("--include-truncated", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/qwq/hse_domains.json"))
    args = ap.parse_args()

    rows = json.loads(args.traces.read_text())
    res = analyse(rows, args.metric, embedder=args.embedder,
                  exclude_truncated=not args.include_truncated, caliper=args.caliper)
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
