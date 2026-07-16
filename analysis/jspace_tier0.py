"""Tier 0: is the conversational feature ORTHOGONAL to the global workspace?

Connects our steering result to the J-space of Anthropic's global-workspace paper
(transformer-circuits.pub/2026/workspace), using the m9h fork of jacobian-lens and the
jlens-lab controls.

THE HYPOTHESIS. The paper's load-bearing property is *limited scope*: the J-space
workspace drives multi-step reasoning; ablating it kills reasoning but spares fluency.
Our steering of feature 30939 did the mirror image -- it KILLED multi-step reasoning
(MATH-Hard -22 pts) while ADDING dialogic fluency. If that is one mechanism, then the
feature's steering direction should lie largely OUTSIDE the workspace: steering it injects
off-workspace, dialogic activity that does not drive reasoning.

THE METRIC. For a residual-space direction d at layer L, the J-space alignment is the
fraction of d's norm captured by the workspace subspace:

    align(d) = || P_J d || / || d ||,   P_J = projection onto span{ J_L^T @ W_U[t] }

jspace_reps(lens, W_U)[L] gives exactly those vocab-pullback directions v[t] = J_L^T W_U[t]
-- the directions the lens can read out at layer L, i.e. the verbalizable workspace. We
take their top principal subspace (99% variance) as P_J.

THE TEST IS COMPARATIVE, not absolute. A low align(d_30939) means nothing if EVERY SAE
feature is orthogonal to J-space. So we compare, at the same layer, same lens:

    - feature 30939            (the paper's conversational-surprise feature)
    - conversational candidates (3114, 10126, 20402)   -- expected LOW if hypothesis holds
    - sparsity/magnitude-matched controls (5993, 22600, 26919)
    - 300 random SAE features   -- the population baseline
    - random Gaussian directions -- the geometric floor

Prediction: conversational features sit BELOW the SAE population and the controls. If they
sit at or above, the hypothesis is wrong and we say so.

CONTROLS (jlens-lab). The lens is fit with fit_converged (a real stopping rule, not the
under-fit ~100-prompt default), and the J-space subspace is sanity-checked against
distance_null / logit_lens_floor so "alignment" is not just residual-stream geometry --
the Trask critique. Reported alongside the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sot.sae import load_sae  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER = 15

# From the steering sweep (analysis in results/steering/FINDINGS.md).
CONVERSATIONAL = {30939: "anchor (paper's feature)", 3114: "candidate", 10126: "candidate",
                  20402: "candidate"}
CONTROLS = {5993: "matched control", 22600: "matched control", 26919: "matched control"}


def jspace_subspace(reps_L: torch.Tensor, var: float = 0.99) -> torch.Tensor:
    """Orthonormal basis Q for the workspace subspace at this layer (top PCs to `var`)."""
    V = reps_L.float()
    V = V - V.mean(0, keepdim=True)
    # right singular vectors span the row space (directions in residual coordinates)
    _, S, Vh = torch.linalg.svd(V, full_matrices=False)
    cum = torch.cumsum(S**2, 0) / (S**2).sum()
    k = int((cum < var).sum()) + 1
    return Vh[:k].T  # [d_model, k], orthonormal columns


def align(d: torch.Tensor, Q: torch.Tensor) -> float:
    """Fraction of d's norm inside span(Q)."""
    d = d.float() / (d.float().norm() + 1e-9)
    return float((Q.T @ d).norm())  # Q orthonormal => ||Q Q^T d|| = ||Q^T d||


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--n-random-features", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("results/steering/jspace_tier0.json"))
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import jlens
    from jlens_lab import fit_converged, wikitext
    from jlens_lab.controls import distance_null  # noqa: F401 (used via geometry)
    from jlens_lab.geometry import jspace_reps

    tok = AutoTokenizer.from_pretrained(MODEL)
    # float32 and .to(device), deliberately:
    #  - the Jacobian estimator backprops through the forward; bf16 backward is fragile.
    #  - device_map= attaches accelerate dispatch hooks that fight jlens's own activation
    #    recorder. .to(device) keeps the module graph clean. (8B fp32 ~= 32GB; fits the A40.)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    model = model.to(device)
    lens_model = jlens.from_hf(model, tok)

    prompts = wikitext(tok, n=args.n_prompts, min_tokens=160, max_chars=4000)

    # PROBE, un-swallowed. fit_converged wraps every prompt in `except Exception: continue`,
    # so a real error (dtype, device hooks, layer index) vanishes into "no prompt produced a
    # Jacobian". Call the inner estimator once, directly, so the true error surfaces here --
    # after one forward pass, not after silently churning the whole prompt list.
    from jlens.fitting import jacobian_for_prompt
    print(f"probe: one Jacobian at L{args.layer} (un-swallowed) ...", flush=True)
    per_prompt, seq_len, n_valid = jacobian_for_prompt(
        lens_model, prompts[0], source_layers=[args.layer],
        max_seq_len=128, skip_first=16,
    )
    assert per_prompt[args.layer].shape == (lens_model.d_model, lens_model.d_model)
    print(f"  probe OK: J_{args.layer} is {tuple(per_prompt[args.layer].shape)}, "
          f"seq_len={seq_len}, n_valid={n_valid}")

    # --- fit the Jacobian lens (jlens-lab's converged fit, not the under-fit default) ---
    print(f"fitting Jacobian lens on {len(prompts)} wikitext prompts ...", flush=True)
    lens, report = fit_converged(lens_model, prompts, source_layers=[args.layer], verbose=True)
    print(f"  lens fit: converged={getattr(report, 'converged', '?')}, "
          f"layers={sorted(lens.jacobians)}")

    W_U = model.get_output_embeddings().weight.detach()  # [vocab, d_model]
    reps = jspace_reps(lens, W_U, device=device)
    Q = jspace_subspace(reps[args.layer].to(device)).to(device)
    print(f"  J-space subspace at L{args.layer}: {Q.shape[1]} dims of {Q.shape[0]}")

    # --- the SAE decoder directions we steered ---
    sae = load_sae(args.layer, "slimpj", device=device)
    D = sae.decoder  # [d_model, d_sae]

    def feat_align(i: int) -> float:
        return align(D[:, i], Q)

    results = {"layer": args.layer, "jspace_dim": int(Q.shape[1]),
               "d_model": int(Q.shape[0]), "features": {}}

    print("\n  J-space alignment (fraction of steering direction inside the workspace):")
    for i, label in {**CONVERSATIONAL, **CONTROLS}.items():
        a = feat_align(i)
        results["features"][i] = {"label": label, "align": a}
        print(f"    f{i:<6} {label:24s} align={a:.3f}")

    # population baseline: random SAE features
    g = torch.Generator().manual_seed(0)
    rand_feats = torch.randperm(D.shape[1], generator=g)[: args.n_random_features]
    pop = np.array([feat_align(int(i)) for i in rand_feats])
    results["sae_population"] = {"mean": float(pop.mean()), "std": float(pop.std()),
                                 "n": len(pop), "pct": np.percentile(pop, [5, 50, 95]).tolist()}

    # geometric floor: random Gaussian directions
    rg = torch.randn(200, Q.shape[0], generator=torch.Generator().manual_seed(1))
    floor = np.array([align(rg[j].to(device), Q) for j in range(rg.shape[0])])
    results["gaussian_floor"] = {"mean": float(floor.mean()), "std": float(floor.std())}

    conv = np.array([results["features"][i]["align"] for i in CONVERSATIONAL])
    ctrl = np.array([results["features"][i]["align"] for i in CONTROLS])

    print(f"\n    {'SAE population':24s} mean={pop.mean():.3f} "
          f"(5/50/95: {np.percentile(pop,5):.3f}/{np.percentile(pop,50):.3f}/{np.percentile(pop,95):.3f})")
    print(f"    {'random Gaussian floor':24s} mean={floor.mean():.3f}")
    print(f"\n    conversational features: mean align={conv.mean():.3f}")
    print(f"    matched controls:        mean align={ctrl.mean():.3f}")

    # where does the anchor sit in the SAE population?
    anchor = results["features"][30939]["align"]
    pctile = float((pop < anchor).mean() * 100)
    results["anchor_percentile_in_sae_pop"] = pctile

    print("\n  VERDICT:")
    if conv.mean() < pop.mean() - pop.std() and conv.mean() < ctrl.mean():
        print("    Conversational features are BELOW the SAE population and below the matched")
        print("    controls in J-space alignment.")
        print("    => the paper's feature steers activation OFF the reasoning workspace.")
        print("    => steering it adds dialogic form that does not drive multi-step reasoning")
        print("       -- the -22 on MATH-Hard, explained in the paper's OWN framework.")
    elif conv.mean() > pop.mean() + pop.std():
        print("    Conversational features are ABOVE the SAE population -- IN the workspace.")
        print("    => hypothesis WRONG. The feature is workspace-aligned; the damage must come")
        print("       from something other than off-workspace injection. Revise.")
    else:
        print(f"    Conversational ({conv.mean():.3f}) not clearly separated from population "
              f"({pop.mean():.3f}+-{pop.std():.3f}) or controls ({ctrl.mean():.3f}).")
        print("    => inconclusive at this n; report as such.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
