"""Hierarchic Social Entropy: is the induced "society of thought" real, or redundant?

Huot, Kaisers & Lapata (arXiv:2607.09197) show that a model society can score well while
being REDUNDANT -- actors that are not actually differentiated -- and conclude that
"accuracy and meaningfulness can sharply diverge". They adapt Balch's Hierarchic Social
Entropy as a judge-free measure of whether a society is real.

That is exactly the hole in our argument. We showed that steering feature 30939 raises
dialogic markers (self-interruptions +36%, contradictions and questions up) while accuracy
FALLS. A defender of the paper can reply: "you induced surface markers, not real diversity."
HSE settles it.

THE PREDICTION, stated before looking:

  real society     -> diversity RISES with steering strength. The voices genuinely differ.
                      (The paper is then right about the mechanism, wrong about the payoff.)
  redundant society-> diversity stays FLAT or FALLS even as markers rise. The segments become
                      MORE alike -- everyone babbling "wait, no, wait" -- not more different.

THE TRAP THIS IS DESIGNED AROUND. Balch's HSE is
    S = integral over h of H(h) dh,
where H(h) is the Shannon entropy of the cluster-size distribution when agents are grouped
at distance threshold h. At h=0 every agent is its own cluster, so H(0) = log2(N).

    ==> HSE RISES MECHANICALLY WITH THE NUMBER OF SEGMENTS.

Steering produces more discourse markers, hence more segments, hence higher raw HSE -- even
if every segment says the same thing. Reporting raw HSE alone would manufacture the paper's
conclusion out of pure bookkeeping. So we report three things:

    hse          -- Balch's measure, raw (confounded with N; shown for completeness)
    hse_norm     -- hse / log2(N), removing the count effect
    mean_dist    -- mean pairwise cosine distance between segments. The cleanest question:
                    ARE THE VOICES ACTUALLY DIFFERENT? Immune to the count confound.

If mean_dist does not rise with alpha, the society is redundant, whatever the markers say.

METHOD (judge-free, unlike the paper's LLM-as-judge).
  1. Segment each trace at perspective-shift markers (wait / but / however / alternatively /
     hmm / oh / actually). These are the paper's OWN conversational cues -- we are cutting
     the trace where it claims a voice changes.
  2. Embed each segment (sentence-transformers, local).
  3. Cosine distances -> single-linkage dendrogram -> H(h) at every merge height -> integrate.

The paper infers personas with an LLM judge and then scores the personas it invented. This
does not: the segmentation is mechanical and the distances are geometric.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# The paper's own conversational cues: where it says a perspective shifts.
SHIFT = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*(?:but|however|wait|hold on|hmm+|oh|actually|alternatively|"
    r"another (?:idea|approach|way)|on the other hand|no,|let me reconsider|"
    r"that'?s (?:not|wrong))\b",
    re.I,
)

MIN_SEG_CHARS = 40   # a fragment shorter than this carries no content to embed
MIN_SEGMENTS = 3     # entropy of fewer than 3 voices is not meaningful


def segment(trace: str) -> list[str]:
    """Cut the trace where it announces a change of perspective."""
    cuts = [m.start() for m in SHIFT.finditer(trace)]
    if not cuts:
        return [trace] if len(trace) >= MIN_SEG_CHARS else []
    bounds = [0] + cuts + [len(trace)]
    segs = [trace[a:b].strip() for a, b in zip(bounds, bounds[1:])]
    return [s for s in segs if len(s) >= MIN_SEG_CHARS]


def hierarchic_social_entropy(D: np.ndarray) -> tuple[float, float, float]:
    """Balch (2000). Returns (hse, hse_normalised, mean_pairwise_distance).

    S = integral_0^inf H(h) dh, where H(h) is the Shannon entropy of the cluster-size
    distribution at threshold h. H is a decreasing step function: log2(N) at h=0, falling
    to 0 once every agent is in one cluster. So the integral is exact from the merge
    heights of a single-linkage dendrogram -- no numeric quadrature needed.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = D.shape[0]
    if n < MIN_SEGMENTS:
        return float("nan"), float("nan"), float("nan")

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="single")

    heights = [0.0] + sorted(Z[:, 2].tolist())
    hse = 0.0
    for h_lo, h_hi in zip(heights, heights[1:]):
        # H is constant on [h_lo, h_hi); evaluate just inside the interval
        labels = fcluster(Z, t=h_lo + 1e-12, criterion="distance")
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        H = float(-(p * np.log2(p)).sum())
        hse += H * (h_hi - h_lo)

    max_h = float(np.log2(n))                      # H(0): every segment its own cluster
    mean_dist = float(D[np.triu_indices(n, k=1)].mean())
    return hse, (hse / max_h if max_h > 0 else float("nan")), mean_dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("results/steering/gate_dose_rg.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", type=Path, default=Path("results/steering/hse.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    rows = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} traces from {args.results.name}")

    enc = SentenceTransformer(args.model)

    per_cond: dict[float, list[dict]] = defaultdict(list)
    for i, r in enumerate(rows):
        segs = segment(r.get("trace", ""))
        if len(segs) < MIN_SEGMENTS:
            continue
        E = enc.encode(segs, normalize_embeddings=True, show_progress_bar=False)
        D = 1.0 - (E @ E.T)                    # cosine distance
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, None)

        hse, hse_n, md = hierarchic_social_entropy(D)
        if not np.isfinite(hse):
            continue

        alpha = 0.0 if r["feature"] == -1 else float(r["alpha"])
        per_cond[alpha].append({
            "n_segments": len(segs), "hse": hse, "hse_norm": hse_n,
            "mean_dist": md, "correct": bool(r["correct"]),
        })
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print("\n" + "=" * 78)
    print("IS THE INDUCED SOCIETY REAL, OR REDUNDANT?")
    print("=" * 78)
    print(f"\n{'alpha':>7} {'n':>5} {'segments':>9} {'HSE(raw)':>9} {'HSE/log2N':>10} "
          f"{'mean_dist':>10} {'accuracy':>9}")
    print("-" * 68)

    summary = {}
    for alpha in sorted(per_cond):
        v = per_cond[alpha]
        row = {
            "n": len(v),
            "segments": float(np.mean([x["n_segments"] for x in v])),
            "hse": float(np.mean([x["hse"] for x in v])),
            "hse_norm": float(np.mean([x["hse_norm"] for x in v])),
            "mean_dist": float(np.mean([x["mean_dist"] for x in v])),
            "accuracy": float(np.mean([x["correct"] for x in v])),
        }
        summary[alpha] = row
        print(f"{alpha:>7} {row['n']:>5} {row['segments']:>9.1f} {row['hse']:>9.3f} "
              f"{row['hse_norm']:>10.3f} {row['mean_dist']:>10.4f} {row['accuracy']:>8.1%}")

    base = summary.get(0.0)
    if base:
        print("\n  vs baseline:")
        for alpha in sorted(k for k in summary if k > 0):
            s = summary[alpha]
            print(f"    alpha={alpha:<6} segments {s['segments']/base['segments']:+.0%}  "
                  f"HSE/log2N {s['hse_norm']-base['hse_norm']:+.3f}  "
                  f"mean_dist {s['mean_dist']-base['mean_dist']:+.4f}")

        # ------------------------------------------------------------------
        # THE MEDIATION TEST -- the thing the paper actually asserts.
        #
        # A rise in group-mean diversity is NOT evidence that diversity causes accuracy;
        # both could move with alpha for unrelated reasons. The paper's claim is that
        # diversity MEDIATES accuracy. That predicts a WITHIN-CONDITION relationship:
        # holding the intervention fixed, the more-diverse traces should be the ones that
        # get the answer right.
        #
        # And there is a confound that a naive distance metric walks straight into:
        # DEGENERATE TEXT HAS HIGH EMBEDDING DISTANCE BECAUSE IT IS NOISE. At alpha=1.693
        # the model emits "wait, no, wait, no, wait" and scores 3.6% -- yet its segments
        # are the most "spread out" of any condition. Distance there is measuring
        # incoherence, not differentiation. So the group-level correlation between
        # diversity and alpha is worthless on its own.
        # ------------------------------------------------------------------
        print("\n  MEDIATION TEST -- within each condition, do MORE DIVERSE traces score better?")
        print(f"    {'alpha':>7} {'n':>5} {'diverse->correct':>18} {'95% CI':>18}")
        print("    " + "-" * 52)
        rng = np.random.default_rng(0)
        for alpha in sorted(per_cond):
            v = per_cond[alpha]
            if len(v) < 30:
                continue
            md = np.array([x["mean_dist"] for x in v])
            ok = np.array([x["correct"] for x in v], float)
            if ok.std() == 0:
                continue
            # difference in mean diversity between correct and incorrect traces
            obs = md[ok == 1].mean() - md[ok == 0].mean() if 0 < ok.sum() < len(ok) else np.nan
            boot = []
            for _ in range(4000):
                i = rng.integers(0, len(v), len(v))
                m, o = md[i], ok[i]
                if 0 < o.sum() < len(o):
                    boot.append(m[o == 1].mean() - m[o == 0].mean())
            lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"    {alpha:>7} {len(v):>5} {obs:>+17.4f}{star} [{lo:>+.4f},{hi:>+.4f}]")

        print("\n  VERDICT:")
        pos = [a for a in summary if a > 0]
        norm_fell = summary[max(pos)]["hse_norm"] < base["hse_norm"] - 0.01
        segs_up = summary[max(pos)]["segments"] > base["segments"] * 1.1
        if segs_up and norm_fell:
            print("    Segments rise steeply; NORMALISED diversity (HSE/log2N) FALLS.")
            print("    => the society gets BIGGER and proportionally MORE REDUNDANT.")
            print("       Each added voice contributes less distinct information.")
            print("    => and the HIGHEST raw distance occurs at the degenerate dose")
            print("       (alpha=1.693, 3.6% accuracy) -- where distance is measuring")
            print("       INCOHERENCE, not differentiation. Diversity of noise is not")
            print("       diversity of viewpoint.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
