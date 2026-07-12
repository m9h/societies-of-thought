"""Preflight: resolve WHERE the SAE actually attaches, and prove our scaling is right.

Two published facts disagree. The SAE's own config.json says it was trained on
`blocks.15.hook_resid_post`; Neuronpedia's feature metadata for the same SAE says
`blocks.15.hook_resid_pre`. Those are different tensors -- resid_pre of layer 15 is
resid_post of layer 14 -- so one of them is off by a layer. Steering the wrong one
would still produce plausible-looking numbers, and we would have no way to tell
from the results alone.

So we settle it with the model in hand. Feature 30939 is documented as firing on
surprise/acknowledgment markers ("Oh!"), with a max activation of ~5.9. We encode
both candidate hook points and check which one (a) fires on the right tokens and
(b) reproduces the published magnitude. The winner is what the sweep will steer,
and it is written to hook_point.json rather than assumed.

In HF, `output_hidden_states=True` gives hidden_states[i] = input to layer i, so:
    resid_pre(L)  == hidden_states[L]        == output of layer L-1
    resid_post(L) == hidden_states[L + 1]    == output of layer L
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sae import load_sae

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Contexts in the style Neuronpedia shows as top activations for feature 30939:
# a surprise/realization marker inside an interpersonal exchange.
PROBES = [
    "[Nice] to meet you. Oh! I still have your Angel Hello Kitty doll!",
    "to tour the mine - lovely and detailed! I did tour McCarthy and don't think you missed much of anything. Oh, thanks, it's always hard to find the balance of what to see.",
    "how often do you wash your walls? You have to wash the walls often. Oh, well... back to training the cat.",
]
# Matched controls with no surprise marker: the feature should be near-silent here.
NEGATIVES = [
    "The derivative of x squared with respect to x is two x, by the power rule.",
    "The mitochondrion is the organelle responsible for oxidative phosphorylation.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--mixture", default="slimpj")
    ap.add_argument("--feature", type=int, default=30939)
    ap.add_argument("--expected-max-act", type=float, default=5.906)  # Neuronpedia
    ap.add_argument("--out", type=Path, default=Path("results/hook_point.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    sae = load_sae(args.layer, args.mixture, device=device)

    print(f"SAE   layer={args.layer} mixture={args.mixture}")
    print(f"      dataset_avg_norm={sae.dataset_avg_norm:.3f}  d_model={sae.d_model}")
    print(f"      sae->real scale={sae.sae_to_real:.4f}")
    print(f"      ||decoder[:, {args.feature}]||={sae.decoder_norm(args.feature):.4f}")
    print()

    results = {}
    for name, offset in (("resid_post", args.layer + 1), ("resid_pre", args.layer)):
        peak_pos, peak_neg = 0.0, 0.0
        top_tokens = []
        for text in PROBES:
            acts, toks = _feature_acts(model, tok, sae, text, offset, args.feature, device)
            j = int(acts.argmax())
            peak_pos = max(peak_pos, float(acts[j]))
            top_tokens.append((repr(toks[j]), round(float(acts[j]), 2)))
        for text in NEGATIVES:
            acts, _ = _feature_acts(model, tok, sae, text, offset, args.feature, device)
            peak_neg = max(peak_neg, float(acts.max()))

        err = abs(peak_pos - args.expected_max_act) / args.expected_max_act
        results[name] = {
            "peak_on_conversational": round(peak_pos, 3),
            "peak_on_control": round(peak_neg, 3),
            "rel_err_vs_neuronpedia": round(err, 3),
            "top_tokens": top_tokens,
        }
        print(f"{name:10s} peak={peak_pos:6.3f} (neuronpedia {args.expected_max_act})"
              f"  rel_err={err:5.1%}  peak_on_nonconversational={peak_neg:6.3f}")
        print(f"           top-activating tokens: {top_tokens}")

    # Neuronpedia's magnitudes cannot arbitrate the hook point: its own metadata
    # contradicts the SAE config, so we have no reason to trust its scale either.
    # The SAE's TRAINING OBJECTIVE can. An SAE only reconstructs the tensor it was
    # trained on, in the space it was trained in. Feed it the wrong layer, or the
    # wrong normalization, and reconstruction collapses. So we sweep both hook
    # points x both plausible conventions and keep whatever actually reconstructs.
    print("\n--- reconstruction fidelity (the SAE's own objective; ground truth) ---")
    recon = {}
    for hook_name, offset in (("resid_post", args.layer + 1), ("resid_pre", args.layer)):
        for conv in ("dataset-wise", "none"):
            ev = _explained_variance(model, tok, sae, PROBES + NEGATIVES, offset, conv, device)
            recon[(hook_name, conv)] = ev
            print(f"  {hook_name:10s} norm={conv:12s} explained variance = {ev:6.1%}")

    (best_hook, best_conv), best_ev = max(recon.items(), key=lambda kv: kv[1])
    print(f"\n=> reconstruction picks: hook={best_hook}  norm={best_conv}  (EV={best_ev:.1%})")

    winner = best_hook
    sep = results[winner]["peak_on_conversational"] - results[winner]["peak_on_control"]
    print(f"=> feature 30939 at {winner}: fires {sep:.2f} on conversational vs "
          f"{results[winner]['peak_on_control']:.2f} on control text")

    if best_ev < 0.4:
        print("\n!! STOP: nothing reconstructs above 40% explained variance.")
        print("!! The SAE is not attaching where we think it is. Do not run the sweep.")
    if best_conv != "dataset-wise":
        print("\n!! NOTE: reconstruction prefers NO dataset-wise rescaling, contradicting")
        print("!! the SAE config. sae.py's sae_to_real must be set to 1.0 before steering.")
    if sep <= 0:
        print("\n!! WARNING: feature does not separate conversational from control text.")

    # Neuronpedia's activation scale disagrees with ours by a consistent factor
    # (~3.2x on the paper's own Fig. 2a contexts). Reconstruction says our scaling
    # is the right one, so Neuronpedia's number must not be used to size steering
    # strengths -- calibrate.py measures max activations in our units instead.
    ratio = results[winner]["peak_on_conversational"] / args.expected_max_act
    if ratio > 1.3 or ratio < 0.77:
        print(f"\n!! Neuronpedia's activation scale differs from ours by {ratio:.2f}x.")
        print("!! Do NOT size alpha off Neuronpedia's maxAct. Run:")
        print(f"!!   python -m sot.calibrate --layer {args.layer} --mixture {args.mixture}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "layer": args.layer,
        "mixture": args.mixture,
        "feature": args.feature,
        "hook_point": winner,
        # The layer whose OUTPUT we hook. resid_post(L) is layer L's output;
        # resid_pre(L) is layer L-1's output.
        "hook_layer": args.layer if winner == "resid_post" else args.layer - 1,
        "dataset_avg_norm": sae.dataset_avg_norm,
        "sae_to_real": sae.sae_to_real,
        "detail": results,
    }, indent=2))
    print(f"\nwrote {args.out}")


@torch.no_grad()
def _explained_variance(model, tok, sae, texts, hs_offset, convention, device) -> float:
    """Fraction of residual-stream variance the SAE reconstructs at this hook point.

    An SAE is trained to satisfy x ~= W_dec @ relu(W_enc @ x + b_enc) + b_dec in
    whatever space it was trained in. That identity only holds for the right tensor
    in the right space, so explained variance is a sharp, self-contained test of
    both the hook point and the normalization convention -- no external metadata,
    no trust in Neuronpedia.
    """
    num, den = 0.0, 0.0
    for text in texts:
        enc = tok(text, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        # Drop BOS. Its residual norm is ~466 against ~11 for ordinary tokens: it is
        # an attention sink the SAE was never trained to reconstruct, and leaving it
        # in makes the error term ~25x the variance regardless of hook or scaling.
        x_real = out.hidden_states[hs_offset][0, 1:].float()  # [seq-1, d_model]

        scale = sae.sae_to_real if convention == "dataset-wise" else 1.0
        x = x_real / scale

        acts = sae.encode_sae_space(x)
        x_hat = sae.decode_sae_space(acts)

        num += float(((x - x_hat) ** 2).sum())
        den += float(((x - x.mean(0, keepdim=True)) ** 2).sum())
    return 1.0 - num / den if den else float("nan")


@torch.no_grad()
def _feature_acts(model, tok, sae, text, hs_offset, feature, device):
    enc = tok(text, return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    resid = out.hidden_states[hs_offset][0]  # [seq, d_model]
    acts = sae.encode(resid.float())[:, feature]
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    return acts.cpu(), toks


if __name__ == "__main__":
    main()
