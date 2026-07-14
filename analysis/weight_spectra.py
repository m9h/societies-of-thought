"""Where, in weight space, does "reasoning" get written?

DeepSeek-R1-Distill-Llama-8B IS Llama-3.1-8B with reasoning fine-tuned into it: same
architecture, same tensor names, weights differing only by the distillation. The two are
a matched pair, so we can localise the change.

Two measurements per weight matrix:

  1. RELATIVE WEIGHT DELTA   ||W_r1 - W_base||_F / ||W_base||_F
     Theory-free -- just linear algebra. Which layers, and which projections, did the
     reasoning behaviour actually get installed into?

  2. HTSR ALPHA (via m9h/wwj) -- power-law exponent of the eigenvalue tail of W^T W.
     We use wwj rather than the reference WeightWatcher because it gives a calibrated
     POSTERIOR over alpha (closed-form Gamma-Pareto conjugacy) and Vuong-style model
     comparison for whether the tail is a power law AT ALL. An alpha you cannot justify
     fitting is a number you should not interpret -- the usual failure mode of HTSR work.

WHY THIS BEARS ON THE PAPER. The society-of-thought mechanism is claimed to live at
LAYER 15. Prediction, stated before looking:

  - if reasoning-distillation barely touched layer 15, that is independent structural
    evidence against layer 15 being where reasoning lives, and corroborates our steering
    result (steering there hurt on MATH-Hard).
  - if layer 15 IS a hotspot, that is a point FOR the paper, and we say so.

DISK. The Spark has ~21GB free and the two models are ~32GB together, so we never hold
both. Tensors are grouped by (base_shard, distill_shard); each pair is downloaded, used,
and deleted. Peak footprint ~12GB.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

BASE = "NousResearch/Meta-Llama-3.1-8B"          # ungated mirror; meta-llama/* is gated
DISTILL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def weight_map(repo: str) -> dict[str, str]:
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
    return json.load(urllib.request.urlopen(url))["weight_map"]


def fetch(repo: str, shard: str, tmp: Path) -> Path:
    dest = tmp / f"{repo.replace('/', '__')}__{shard}"
    if dest.exists():
        return dest
    url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
    print(f"    fetching {repo.split('/')[-1]}/{shard} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest


@torch.no_grad()
def eigenvalues(W: torch.Tensor, device: str) -> np.ndarray:
    """Eigenvalues of W^T W == squared singular values. svdvals, not a full SVD."""
    s = torch.linalg.svdvals(W.to(device=device, dtype=torch.float32))
    return (s**2).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/wspec"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/weight_spectra.json"))
    ap.add_argument("--skip-alpha", action="store_true", help="deltas only (no SVDs)")
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    bmap, dmap = weight_map(BASE), weight_map(DISTILL)
    keys = sorted(
        k for k in (set(bmap) & set(dmap))
        if k.startswith("model.layers.") and k.endswith(".weight") and any(p in k for p in PROJ)
    )
    print(f"{len(keys)} matched weight matrices; device={args.device}")

    # Group by shard pair so each pair is downloaded exactly once.
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for k in keys:
        groups[(bmap[k], dmap[k])].append(k)
    print(f"{len(groups)} shard-pairs to stream\n")

    alpha_fn = None
    if not args.skip_alpha:
        import jax.numpy as jnp
        from wwj.bayes import alpha_posterior, model_posterior

        def alpha_fn(eigs: np.ndarray):
            # wwj returns plain dicts. (Reaching for post.mean silently produced NaN for
            # all 224 matrices on the first run -- a reminder to check the return type
            # rather than assume it.)
            e = jnp.asarray(np.sort(eigs)[-2000:])          # the tail; wwj picks xmin in it
            post = alpha_posterior(e)
            mp = model_posterior(e)   # log Bayes factor: is a power law even justified?
            return {
                "alpha": float(post["alpha_mean"]),
                "ci_low": float(post["ci_low"]),
                "ci_high": float(post["ci_high"]),
                # >0 favours power-law over exponential. If this is negative, the alpha
                # for that layer is a fit to a distribution that isn't there -- report it,
                # don't interpret it.
                "logbf_pl_vs_exp": float(mp["logbf_pl_vs_exp"]),
            }

    rows = []
    eig_store: dict[str, dict] = {}
    for (bshard, dshard), ks in groups.items():
        bpath, dpath = fetch(BASE, bshard, args.tmp), fetch(DISTILL, dshard, args.tmp)
        with safe_open(bpath, framework="pt") as fb, safe_open(dpath, framework="pt") as fd:
            for k in ks:
                Wb, Wd = fb.get_tensor(k), fd.get_tensor(k)
                delta = float(torch.linalg.matrix_norm(Wd.float() - Wb.float())
                              / torch.linalg.matrix_norm(Wb.float()))
                layer = int(k.split(".")[2])
                rec = {"layer": layer,
                       "proj": k.replace(f"model.layers.{layer}.", "").replace(".weight", ""),
                       "rel_delta": delta}
                if alpha_fn is not None:
                    eb, ed = eigenvalues(Wb, args.device), eigenvalues(Wd, args.device)
                    ab, ad = alpha_fn(eb), alpha_fn(ed)
                    rec |= {
                        "alpha_base": ab["alpha"], "alpha_distill": ad["alpha"],
                        "d_alpha": ad["alpha"] - ab["alpha"],
                        "ci_base": [ab["ci_low"], ab["ci_high"]],
                        "ci_distill": [ad["ci_low"], ad["ci_high"]],
                        "logbf_base": ab["logbf_pl_vs_exp"],
                        "logbf_distill": ad["logbf_pl_vs_exp"],
                    }
                    # Save the tails: the download is the expensive part, and never
                    # having to repeat it is worth 4MB.
                    eig_store[k] = {"base": np.sort(eb)[-2000:].tolist(),
                                    "distill": np.sort(ed)[-2000:].tolist()}
                rows.append(rec)
                del Wb, Wd
        # free the pair before the next one -- this is what keeps us inside 21GB
        os.remove(bpath)
        os.remove(dpath)
        print(f"  done shard-pair ({len(ks)} matrices); {len(rows)}/{len(keys)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    if eig_store:
        np.savez_compressed(args.out.with_suffix(".eigs.npz"),
                            **{k: np.array([v["base"], v["distill"]]) for k, v in eig_store.items()})
        print(f"  saved eigenvalue tails -> {args.out.with_suffix('.eigs.npz')}")

    # ---- the question ----
    by_layer: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r["rel_delta"])
    means = {l: float(np.mean(v)) for l, v in by_layer.items()}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])

    print("\n" + "=" * 68)
    print("WHERE DID REASONING-DISTILLATION CHANGE THE WEIGHTS?")
    print("=" * 68)
    print("\n  mean relative weight change, by layer (top 10):")
    for l, m in ranked[:10]:
        print(f"    L{l:02d}  {m:.4f}  {'#' * int(300 * m)}")

    rank15 = [l for l, _ in ranked].index(15) + 1
    print(f"\n  >>> LAYER 15 (the paper's claimed mechanism) ranks {rank15} of {len(ranked)}")
    print(f"      layer 15:  {means[15]:.4f}")
    print(f"      max:       {ranked[0][1]:.4f}  (L{ranked[0][0]})")
    print(f"      median:    {float(np.median(list(means.values()))):.4f}")

    by_proj: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_proj[r["proj"]].append(r["rel_delta"])
    print("\n  mean relative change, by projection:")
    for p, v in sorted(by_proj.items(), key=lambda kv: -float(np.mean(kv[1]))):
        print(f"    {p:20s} {float(np.mean(v)):.4f}")

    if not args.skip_alpha:
        ok = [r for r in rows if np.isfinite(r.get("d_alpha", np.nan))]
        if ok:
            # Only interpret alpha where a power law actually beats an exponential.
            just = [r for r in ok if r["logbf_base"] > 0 and r["logbf_distill"] > 0]
            print(f"\n  HTSR alpha (wwj, Bayesian):")
            print(f"    power-law justified (logBF>0 vs exponential): {len(just)}/{len(ok)} matrices")
            if just:
                da = [r["d_alpha"] for r in just]
                print(f"    mean alpha shift (distill - base): {float(np.mean(da)):+.3f}")
                by_l: dict[int, list[float]] = defaultdict(list)
                for r in just:
                    by_l[r["layer"]].append(r["d_alpha"])
                rk = sorted(((l, float(np.mean(v))) for l, v in by_l.items()), key=lambda kv: -abs(kv[1]))
                print("    largest |alpha shift| by layer:")
                for l, v in rk[:6]:
                    print(f"      L{l:02d}  {v:+.3f}")
                if 15 in by_l:
                    r15 = [l for l, _ in rk].index(15) + 1
                    print(f"    LAYER 15 alpha shift: {float(np.mean(by_l[15])):+.3f}  (rank {r15}/{len(rk)})")

    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
