"""Does the RL 'society of thought' EMERGE -- and is it diverse or redundant?

The Societies-of-Thought paper's Claim A: under PPO on Countdown rewarding only
correctness+format, dialogic / multi-persona structure arises anyway. Our Tier-0
run reproduced the LEARNING (val 24%->56%); this asks the emergence question of
the SAME run, and connects it to our steering-side §7 finding.

Method. The verl training log prints the model's generated traces in
chronological order across ~232 PPO steps. We bin them:

    EARLY  = the first traces  (~near-baseline model, pre-learning)
    LATE   = the last  traces  (~converged, post-RL)

For each trace we segment at perspective-shift cues (the paper's own markers),
embed the segments locally, and compute Balch's Hierarchic Social Entropy --
exactly the judge-free instrument from analysis/hse.py / FINDINGS §7.

Two questions:
  1. EMERGENCE: do segments/markers RISE from early to late? (the descriptive
     claim -- does dialogic structure appear under RL?)
  2. REALITY:   does normalised diversity (HSE/log2N) rise, or stay flat/fall?
     §7 found that STEERING produces a bigger-but-more-REDUNDANT society. If RL
     emergence shows the same pattern, the redundancy result extends from an
     artificial intervention to genuine RL training -- a stronger claim.

No GPU. Reuses analysis/hse.py's segment() and hierarchic_social_entropy().
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from analysis.hse import MIN_SEGMENTS, hierarchic_social_entropy, segment

# Dialogue / self-audit markers, same family the paper and §7 use, for a simple
# marker-density measure independent of the segmentation.
MARKERS = re.compile(r"\b(wait|hmm|but|however|actually|oh|let'?s|no,|alternatively|"
                     r"what if|hold on|recheck|reconsider)\b", re.I)


def extract_traces(log_path: Path) -> list[str]:
    """Pull the model's <think>..</answer> traces from the verl log, in order,
    stripping Ray's interleaved worker prefixes."""
    t = log_path.read_text(errors="ignore")
    t = re.sub(r"\x1b\[\d+m", "", t)                          # ANSI colour
    t = re.sub(r"\((?:main_task|WorkerDict) pid=\d+\)", "", t)  # Ray prefixes
    return re.findall(r"<think>.*?</answer>", t, re.S)


def bin_stats(traces: list[str], enc) -> dict:
    seg_counts, hse_norms, mean_dists, marker_rates, kept = [], [], [], [], 0
    for tr in traces:
        segs = segment(tr)
        marker_rates.append(len(MARKERS.findall(tr)) / max(len(tr.split()), 1))
        if len(segs) < MIN_SEGMENTS:
            continue
        E = enc.encode(segs, normalize_embeddings=True, show_progress_bar=False)
        D = np.clip(1.0 - (E @ E.T), 0.0, None)
        np.fill_diagonal(D, 0.0)
        hse, hse_n, md = hierarchic_social_entropy(D)
        if not np.isfinite(hse_n):
            continue
        seg_counts.append(len(segs))
        hse_norms.append(hse_n)
        mean_dists.append(md)
        kept += 1
    return {
        "n": kept,
        "segments": float(np.mean(seg_counts)) if seg_counts else float("nan"),
        "hse_norm": float(np.mean(hse_norms)) if hse_norms else float("nan"),
        "mean_dist": float(np.mean(mean_dists)) if mean_dists else float("nan"),
        "marker_rate": float(np.mean(marker_rates)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("/tmp/tz_train.log"))
    ap.add_argument("--bin", type=int, default=300, help="traces per early/late bin")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    traces = extract_traces(args.log)
    print(f"{len(traces)} traces extracted from {args.log.name}")
    if len(traces) < 2 * args.bin:
        args.bin = len(traces) // 3
    early, late = traces[: args.bin], traces[-args.bin:]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.model)

    print(f"\n  binning: EARLY = first {len(early)}, LATE = last {len(late)}")
    e = bin_stats(early, enc)
    l = bin_stats(late, enc)

    print("\n" + "=" * 72)
    print("CLAIM A EMERGENCE: does the RL society appear, and is it diverse?")
    print("=" * 72)
    print(f"\n  {'':10} {'n':>5} {'markers/wd':>11} {'segments':>9} {'HSE/log2N':>10} {'mean_dist':>10}")
    print("  " + "-" * 60)
    for name, s in (("EARLY", e), ("LATE", l)):
        print(f"  {name:10} {s['n']:>5} {s['marker_rate']:>11.4f} {s['segments']:>9.2f} "
              f"{s['hse_norm']:>10.3f} {s['mean_dist']:>10.4f}")

    d_mark = l["marker_rate"] - e["marker_rate"]
    d_seg = l["segments"] - e["segments"]
    d_div = l["hse_norm"] - e["hse_norm"]
    print("\n  EARLY -> LATE:")
    print(f"    marker rate  {d_mark:+.4f}  ({d_mark / e['marker_rate']:+.0%})")
    print(f"    segments     {d_seg:+.2f}  ({d_seg / e['segments']:+.0%})")
    print(f"    HSE/log2N    {d_div:+.3f}  ({d_div / e['hse_norm']:+.0%})")

    print("\n  READING:")
    emerged = d_mark > 0 or d_seg > 0
    if emerged and d_div <= 0.01:
        print("    Dialogic structure EMERGES (more markers/segments) while normalised")
        print("    diversity stays FLAT or FALLS -> a REDUNDANT society, the same pattern")
        print("    §7 found for steering, now under genuine RL emergence.")
    elif emerged and d_div > 0.01:
        print("    Dialogic structure emerges AND diversity rises -> a genuinely more")
        print("    differentiated society. This would NOT match the §7 redundancy result.")
    else:
        print("    No clear rise in dialogic structure early->late in this run.")


if __name__ == "__main__":
    main()
