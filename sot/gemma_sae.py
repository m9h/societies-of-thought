"""Gemma Scope 2 SAE loading (Gemma 2 / Gemma 3).

Sibling to sot/sae.py, which loads Llama Scope. They are both JumpReLU SAEs but the
stored format differs in three ways that all fail SILENTLY, so they get separate loaders
rather than one with flags:

    thing            Llama Scope                      Gemma Scope 2
    encoder          encoder.weight [d_sae, d_model]  w_enc  [d_model, d_sae]  TRANSPOSED
    decoder          decoder.weight [d_model, d_sae]  w_dec  [d_sae, d_model]  TRANSPOSED
    threshold        log_jumprelu_threshold (log!)    threshold (raw!)
    normalisation    dataset_average_activation_norm  absent -> scale is 1.0

Getting the transpose wrong gives silent garbage at square-ish shapes. exp()-ing Gemma's
raw threshold moves every feature's gate. Reusing Llama's dataset-wise rescale would
mis-size every steering vector -- the same class of bug that made our first interventions
2.5x too weak. tests/test_gemma_sae.py pins all three.

Hook point is unambiguous here: config says `model.layers.N.output`, i.e. the layer's
output = resid_post. No resid_pre/resid_post guessing (Neuronpedia's Llama metadata was
wrong about exactly this).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sot.sources import gemma_sae_location

GEMMA_SCOPE_REPOS = {
    # model id -> HF repo. Which LAYERS exist (and at which width/l0) is not a
    # property of the repo alone: the flagship resid_post/ directory holds four
    # layers with rich options, and resid_post_all/ holds all 62 with a narrower
    # grid. sot.sources.gemma_sae_location owns that, so it lives in one place.
    "google/gemma-3-27b-it": "google/gemma-scope-2-27b-it",
    "google/gemma-3-27b-pt": "google/gemma-scope-2-27b-pt",
    "google/gemma-3-12b-it": "google/gemma-scope-2-12b-it",
}


@dataclass
class GemmaScopeSAE:
    """Same interface as sot.sae.LlamaScopeSAE so the sweep code is model-agnostic."""

    layer: int
    d_model: int
    d_sae: int
    encoder: torch.Tensor       # [d_sae, d_model]  (OUR convention)
    decoder: torch.Tensor       # [d_model, d_sae]  (OUR convention)
    encoder_bias: torch.Tensor  # [d_sae]
    decoder_bias: torch.Tensor  # [d_model]
    threshold: torch.Tensor     # [d_sae], RAW JumpReLU threshold
    hook_point: str

    @property
    def sae_to_real(self) -> float:
        """Gemma Scope has no dataset-wise rescaling: SAE space IS residual space."""
        return 1.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """JumpReLU: a_f = z_f * 1[z_f > theta_f], theta used RAW (not exp'd)."""
        z = x.to(self.encoder.dtype) @ self.encoder.T + self.encoder_bias
        return z * (z > self.threshold)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.decoder.T + self.decoder_bias

    def steering_vector(self, feature: int, strength: float) -> torch.Tensor:
        """Vector to add to the residual stream. sae_to_real is 1.0 here, but keep the
        multiplication explicit so this stays correct if that ever changes."""
        return self.decoder[:, feature] * strength * self.sae_to_real


def load_gemma_sae(
    path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> GemmaScopeSAE:
    """Load from a local directory containing params.safetensors + config.json."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())

    arch = cfg.get("architecture")
    if arch != "jump_relu":
        raise RuntimeError(f"architecture is {arch!r}; this loader implements JumpReLU")

    w = load_file(str(path / "params.safetensors"))

    # Transpose into OUR convention. Gemma stores w_enc [d_model, d_sae] and
    # w_dec [d_sae, d_model] -- both the opposite way round from Llama Scope.
    w_enc, w_dec = w["w_enc"], w["w_dec"]
    d_model, d_sae = w_enc.shape
    if tuple(w_dec.shape) != (d_sae, d_model):
        raise RuntimeError(
            f"w_dec is {tuple(w_dec.shape)}, expected {(d_sae, d_model)} -- "
            "Gemma Scope layout assumption violated"
        )

    return GemmaScopeSAE(
        layer=_layer_from_hook(cfg.get("hf_hook_point_in", "")),
        d_model=d_model,
        d_sae=d_sae,
        encoder=w_enc.T.contiguous().to(device=device, dtype=dtype),   # -> [d_sae, d_model]
        decoder=w_dec.T.contiguous().to(device=device, dtype=dtype),   # -> [d_model, d_sae]
        encoder_bias=w["b_enc"].to(device=device, dtype=dtype),
        decoder_bias=w["b_dec"].to(device=device, dtype=dtype),
        threshold=w["threshold"].to(device=device, dtype=dtype),       # RAW
        hook_point=cfg.get("hf_hook_point_in", ""),
    )


def download_gemma_sae(
    model: str, layer: int, width: str = "16k", l0: str = "medium",
    cache_dir: str | Path | None = None, device: str = "cuda",
) -> GemmaScopeSAE:
    """Fetch a resid_post SAE from the Gemma Scope repo and load it.

    All 62 layers are available. Note that l0="medium" exists only for the four
    flagship layers (16/31/40/53); every other layer needs "small" or "big".
    gemma_sae_location raises on an impossible combination rather than letting
    hf_hub_download 404 on a path we claimed was valid.
    """
    if model not in GEMMA_SCOPE_REPOS:
        raise ValueError(f"no Gemma Scope repo registered for {model!r}")
    repo = GEMMA_SCOPE_REPOS[model]
    sub = gemma_sae_location(layer, width=width, l0=l0)
    files = {}
    for name in ("config.json", "params.safetensors"):
        files[name] = hf_hub_download(repo, f"{sub}/{name}", cache_dir=cache_dir)
    return load_gemma_sae(Path(files["config.json"]).parent, device=device)


def _layer_from_hook(hook: str) -> int:
    """'model.layers.16.output' -> 16"""
    parts = hook.split(".")
    for a, b in zip(parts, parts[1:]):
        if a == "layers" and b.isdigit():
            return int(b)
    return -1
