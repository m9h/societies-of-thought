"""Diagnostic: why doesn't the SAE reconstruct?

Measures the geometry directly instead of guessing at conventions.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sae import load_sae

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TEXT = (
    "The quick brown fox jumps over the lazy dog. In organic chemistry, a Diels-Alder "
    "reaction forms a six-membered ring. Oh! I didn't expect that at all, thanks."
)


@torch.no_grad()
def main() -> None:
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
    sae = load_sae(15, "slimpj", device=device)

    enc = tok(TEXT, return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)

    print(f"SAE dataset_avg_norm = {sae.dataset_avg_norm:.4f}")
    print(f"sqrt(d_model)        = {sae.d_model ** 0.5:.1f}")
    print(f"implied sae_to_real  = {sae.sae_to_real:.4f}\n")

    for name, off in (("resid_pre(15)", 15), ("resid_post(15)", 16)):
        x = out.hidden_states[off][0].float()
        norms = x.norm(dim=-1)
        print(f"{name}:  ||x|| bos={norms[0]:8.1f}  "
              f"mean(no-bos)={norms[1:].mean():7.2f}  median={norms[1:].median():7.2f}  "
              f"max={norms[1:].max():7.2f}")
    print()

    # Which scaling makes the SAE's input look like what it expects
    # (average norm sqrt(d_model) = 64)?
    print("If the SAE expects avg input norm == sqrt(d_model) == 64.0, then the")
    print("correct scale s satisfies  mean||x_real||/s == 64:\n")
    for name, off in (("resid_pre(15)", 15), ("resid_post(15)", 16)):
        x = out.hidden_states[off][0].float()[1:]  # drop BOS
        implied = float(x.norm(dim=-1).mean()) / 64.0
        print(f"  {name:16s} implied scale = {implied:.4f}   "
              f"(sae.sae_to_real = {sae.sae_to_real:.4f})")
    print()

    # Reconstruction, with and without BOS, under both conventions.
    print(f"{'hook':16s} {'norm':14s} {'incl BOS':>10s} {'excl BOS':>10s} {'L0':>7s} {'||xhat||/||x||':>15s}")
    print("-" * 78)
    for name, off in (("resid_pre(15)", 15), ("resid_post(15)", 16)):
        x_real = out.hidden_states[off][0].float()
        for conv, scale in (("dataset-wise", sae.sae_to_real), ("none", 1.0)):
            x = x_real / scale
            acts = sae.encode_sae_space(x)
            xh = sae.decode_sae_space(acts)
            ev_all = _ev(x, xh)
            ev_nobos = _ev(x[1:], xh[1:])
            l0 = float((acts[1:] > 0).sum(-1).float().mean())
            ratio = float(xh[1:].norm(dim=-1).mean() / x[1:].norm(dim=-1).mean())
            print(f"{name:16s} {conv:14s} {ev_all:9.1%} {ev_nobos:9.1%} {l0:7.1f} {ratio:14.3f}")


def _ev(x, xh) -> float:
    num = ((x - xh) ** 2).sum()
    den = ((x - x.mean(0, keepdim=True)) ** 2).sum()
    return float(1 - num / den)


if __name__ == "__main__":
    main()
