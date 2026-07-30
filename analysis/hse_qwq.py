"""Does perspective diversity separate CORRECT from INCORRECT reasoning traces?

The claim under test (SoT C2/C3): the internal "society of thought" is genuinely diverse,
and that diversity *accounts for* the accuracy advantage of reasoning models. The paper
measures diversity with an LLM judge that first infers personas and then scores their
spread, over 8,262 traces.

This runs the same question judge-free, within a single model, at ~1000x the scale, on a
model the paper itself studies. `PrimeIntellect/NuminaMath-QwQ-CoT-5M` (MIT) is 5,138,102
**QwQ** traces carrying a per-trace `correct` boolean. QwQ-32B is one of the two reasoning
models SoT builds claims D and C3 on.

The test is deliberately within-model and within-task, which removes the confound that
sinks the cross-model version: reasoning models differ from instruction-tuned models in
many ways besides diversity, so "R1 is more diverse and more accurate" cannot isolate
diversity. Here the model, the decoding, and the task distribution are held fixed and only
the outcome varies. If diversity mediates accuracy, correct traces must be more diverse
than incorrect ones.

Instrument reused unchanged from `analysis/hse.py`: segmentation at the paper's own
perspective-shift cues, then Balch's Hierarchic Social Entropy over the segment
dendrogram. `hse_norm` divides out log2(N) so a longer trace is not scored as more diverse
merely for having more segments -- which matters here, because incorrect traces on math
tend to be longer.

    python -m analysis.hse_qwq --n-per-class 3000 --out results/qwq/hse_qwq.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from analysis.hse import MIN_SEGMENTS, hierarchic_social_entropy, segment

DATASET = "PrimeIntellect/NuminaMath-QwQ-CoT-5M"


def load_balanced(n_per_class: int, seed: int = 0, scan_cap: int = 400_000):
    """Stream the dataset and take a balanced sample of correct/incorrect traces.

    Streaming rather than downloading: the corpus is 43.8GB and we need thousands of
    traces, not millions. Balanced by outcome because the comparison is within-model
    across outcome, so unequal group sizes buy nothing.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    pos, neg, scanned = [], [], 0
    for r in ds:
        scanned += 1
        resp = r.get("response") or ""
        bucket = pos if r.get("correct") else neg
        if len(bucket) < n_per_class and resp:
            bucket.append({"response": resp, "correct": bool(r.get("correct")),
                           "problem_id": r.get("problem_id")})
        if len(pos) >= n_per_class and len(neg) >= n_per_class:
            break
        if scanned >= scan_cap:
            break
    print(f"scanned {scanned} rows -> {len(pos)} correct, {len(neg)} incorrect")
    rows = pos + neg
    random.Random(seed).shuffle(rows)
    return rows


def measure(rows, model_name: str, batch: int = 256) -> tuple[list[dict], dict]:
    """Returns (per-trace records, drop counts by class).

    The MIN_SEGMENTS filter is a selection effect that runs AGAINST the null: traces with
    too few perspective shifts to score are dropped, and correct traces have far fewer
    shifts, so they are dropped more often. Reporting the drop counts by class is
    mandatory -- silently filtering the low-diversity end of one group would manufacture
    whatever result the filter favours.
    """
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model_name)
    out = []
    dropped = {"correct_too_few_segments": 0, "incorrect_too_few_segments": 0,
               "correct_degenerate": 0, "incorrect_degenerate": 0}
    for i, r in enumerate(rows):
        key = "correct" if r["correct"] else "incorrect"
        segs = segment(r["response"])
        if len(segs) < MIN_SEGMENTS:
            dropped[f"{key}_too_few_segments"] += 1
            continue
        E = enc.encode(segs, normalize_embeddings=True, show_progress_bar=False,
                       batch_size=batch)
        D = 1.0 - (E @ E.T)
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, None)
        hse, hse_n, md = hierarchic_social_entropy(D)
        if not np.isfinite(hse):
            dropped[f"{key}_degenerate"] += 1
            continue
        out.append({"correct": r["correct"], "n_segments": len(segs),
                    "hse": float(hse), "hse_norm": float(hse_n),
                    "mean_dist": float(md), "words": len(r["response"].split())})
        if i % 250 == 0:
            print(f"  {i}/{len(rows)}  kept {len(out)}", flush=True)
    return out, dropped


def _ci95(x: np.ndarray) -> float:
    return 1.96 * float(x.std(ddof=1)) / max(np.sqrt(len(x)), 1.0)


def report(recs: list[dict], dropped: dict | None = None) -> dict:
    cor = [r for r in recs if r["correct"]]
    inc = [r for r in recs if not r["correct"]]
    summary = {"n_correct": len(cor), "n_incorrect": len(inc), "metrics": {},
               "dropped": dropped or {}}
    if dropped:
        print(f"\ndropped as unscorable: {dropped}")
        dc = dropped.get("correct_too_few_segments", 0)
        di = dropped.get("incorrect_too_few_segments", 0)
        if dc + di:
            print(f"  NOTE: {dc} correct vs {di} incorrect traces had <{MIN_SEGMENTS} "
                  f"segments. This filter removes the LOW-diversity tail, and it removes "
                  f"more from the group with fewer shifts -- it biases toward finding "
                  f"correct traces MORE diverse than they are.")

    print("\n" + "=" * 78)
    print("DOES DIVERSITY SEPARATE CORRECT FROM INCORRECT TRACES? (within QwQ)")
    print("=" * 78)
    print(f"{'metric':>12} {'correct':>18} {'incorrect':>18} {'difference':>18}")
    for m in ("hse_norm", "mean_dist", "hse", "n_segments", "words"):
        a = np.array([r[m] for r in cor], dtype=float)
        b = np.array([r[m] for r in inc], dtype=float)
        if not len(a) or not len(b):
            continue
        d = float(a.mean() - b.mean())
        # Welch standard error on the difference of means
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        lo, hi = d - 1.96 * se, d + 1.96 * se
        excl = (lo > 0) or (hi < 0)
        summary["metrics"][m] = {
            "correct_mean": float(a.mean()), "correct_ci95": _ci95(a),
            "incorrect_mean": float(b.mean()), "incorrect_ci95": _ci95(b),
            "difference": d, "ci95_low": lo, "ci95_high": hi,
            "ci_excludes_zero": bool(excl),
        }
        star = "*" if excl else " "
        print(f"{m:>12} {a.mean():>10.4f}±{_ci95(a):<7.4f} "
              f"{b.mean():>10.4f}±{_ci95(b):<7.4f} "
              f"{d:>+10.4f} [{lo:+.4f},{hi:+.4f}]{star}")
    print("\n* = 95% CI on the difference excludes zero")

    hn = summary["metrics"].get("hse_norm", {})
    md = summary["metrics"].get("mean_dist", {})
    verdict = []
    if hn and not hn["ci_excludes_zero"]:
        verdict.append("normalised diversity does NOT separate correct from incorrect")
    elif hn and hn["difference"] > 0:
        verdict.append("correct traces ARE more diverse (normalised)")
    elif hn:
        verdict.append("correct traces are LESS diverse (normalised)")
    if md and md["ci_excludes_zero"]:
        verdict.append(f"mean pairwise distance differs by {md['difference']:+.4f}")
    summary["verdict"] = "; ".join(verdict)
    print("\nVERDICT:", summary["verdict"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", type=Path, default=Path("results/qwq/hse_qwq.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_balanced(args.n_per_class, args.seed)
    recs, dropped = measure(rows, args.model)
    print(f"\nmeasured {len(recs)} traces with >= {MIN_SEGMENTS} segments")
    summary = report(recs, dropped)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dataset": DATASET, "embedder": args.model, "seed": args.seed,
         "summary": summary, "per_trace": recs}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
