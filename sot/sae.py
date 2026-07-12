"""Llama-Scope SAE loading for DeepSeek-R1-Distill-Llama-8B.

The SAEs published as OpenMOSS-Team/Llama-Scope-R1-Distill are trained with
`norm_activation: "dataset-wise"`. That means the SAE never saw the model's raw
residual stream: it saw activations rescaled so that the dataset-average L2 norm
maps to sqrt(d_model).

    x_sae = x_real * sqrt(d_model) / dataset_avg_norm

Every quantity that lives on the SAE side of that transform -- decoder columns,
feature activations, and the max-activation values reported by Neuronpedia -- is
therefore in *SAE space*, not in the model's residual space. Steering strengths
quoted in the paper (s = +-10 on feature 30939, whose max activation is ~5.9) are
SAE-space numbers. So we choose the strength in SAE space and convert the
resulting vector back to real space exactly once, in `steering_vector()`.

Getting this backwards silently rescales every intervention by
dataset_avg_norm / sqrt(d_model) (~an order of magnitude here), which would still
"work" -- it would just be a different experiment than the one we mean to run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

SAE_REPO = "OpenMOSS-Team/Llama-Scope-R1-Distill"

# Training-mixture directories inside the SAE repo. Layer 15 is the only layer
# published for all three mixtures; the paper used the pure-SlimPajama one
# ("slimpj" in its Neuronpedia id, 15-llamascope-slimpj-res-32k).
MIXTURES = {
    "slimpj": "800M-Slimpajama-0-OpenR1-Math-220k",
    "mixed": "400M-Slimpajama-400M-OpenR1-Math-220k",
    "openr1": "0-Slimpajama-800M-OpenR1-Math-220k",
}

# Neuronpedia hosts explanations under these source ids. Only layer 15 exists for
# the pure-slimpj SAE; every layer exists for the mixed SAE.
def neuronpedia_source_id(layer: int, mixture: str) -> str:
    if mixture == "slimpj":
        return f"{layer}-llamascope-slimpj-res-32k"
    if mixture == "mixed":
        return f"{layer}-llamascope-slimpj-openr1-res-32k"
    if mixture == "openr1":
        return f"{layer}-llamascope-openr1-res-32k"
    raise ValueError(f"unknown mixture {mixture!r}")


@dataclass
class LlamaScopeSAE:
    layer: int
    mixture: str
    d_model: int
    d_sae: int
    decoder: torch.Tensor  # [d_model, d_sae]
    encoder: torch.Tensor  # [d_sae, d_model]
    encoder_bias: torch.Tensor  # [d_sae]
    decoder_bias: torch.Tensor  # [d_model]
    log_jumprelu_threshold: torch.Tensor  # [d_sae]
    dataset_avg_norm: float

    @property
    def sae_to_real(self) -> float:
        """Multiplier taking an SAE-space vector into the real residual stream."""
        return self.dataset_avg_norm / (self.d_model**0.5)

    def steering_vector(self, feature: int, strength_sae: float) -> torch.Tensor:
        """Vector to ADD to the real residual stream.

        `strength_sae` is in SAE activation units -- the same units as the
        paper's s in h' = h + s * d_f, and the same units as Neuronpedia's
        maxActApprox. The returned vector is in real residual-stream units.
        """
        return self.decoder[:, feature] * strength_sae * self.sae_to_real

    def encode_sae_space(self, x_sae: torch.Tensor) -> torch.Tensor:
        """Feature activations for activations already in SAE space.

        act_fn is JumpReLU, not ReLU: a feature fires at its pre-activation value
        only once that value clears a per-feature threshold, and is hard-zeroed
        below it.

            a_f = z_f * 1[z_f > theta_f],   theta_f = exp(log_jumprelu_threshold_f)

        Using a plain ReLU instead leaves thousands of small sub-threshold
        activations alive. They barely register individually, but they all get
        multiplied by decoder columns and summed, so the reconstruction acquires a
        large amount of spurious mass -- which is why plain ReLU produced an
        explained variance of -2600% here rather than a merely mediocre one.
        """
        z = x_sae.to(self.encoder.dtype) @ self.encoder.T + self.encoder_bias
        theta = torch.exp(self.log_jumprelu_threshold)
        return z * (z > theta)

    def decode_sae_space(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.decoder.T + self.decoder_bias

    def encode(self, x_real: torch.Tensor) -> torch.Tensor:
        """Feature activations for real-space activations x_real [..., d_model]."""
        return self.encode_sae_space(x_real / self.sae_to_real)

    def decoder_norm(self, feature: int) -> float:
        return float(self.decoder[:, feature].norm())


def load_sae(
    layer: int,
    mixture: str = "slimpj",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    cache_dir: str | Path | None = None,
) -> LlamaScopeSAE:
    if mixture not in MIXTURES:
        raise ValueError(f"mixture must be one of {list(MIXTURES)}")
    subdir = f"{MIXTURES[mixture]}/L{layer}R"

    cfg_path = hf_hub_download(SAE_REPO, f"{subdir}/config.json", cache_dir=cache_dir)
    w_path = hf_hub_download(SAE_REPO, f"{subdir}/sae_weights.safetensors", cache_dir=cache_dir)

    cfg = json.loads(Path(cfg_path).read_text())
    w = load_file(w_path)

    hook = cfg["hook_point_in"]
    if hook != f"blocks.{layer}.hook_resid_post":
        raise RuntimeError(f"unexpected hook point {hook!r} for layer {layer}")
    if cfg.get("act_fn") != "jumprelu":
        raise RuntimeError(
            f"act_fn is {cfg.get('act_fn')!r}, but encode() implements JumpReLU."
        )
    if cfg.get("norm_activation") != "dataset-wise":
        raise RuntimeError(
            f"norm_activation is {cfg.get('norm_activation')!r}, not 'dataset-wise'. "
            "The SAE-space <-> real-space rescaling in this module assumes dataset-wise."
        )

    norm_key = f"dataset_average_activation_norm.{hook}"
    dataset_avg_norm = float(w[norm_key].item())

    d_model = int(cfg["d_model"])
    d_sae = d_model * int(cfg["expansion_factor"])
    decoder = w["decoder.weight"].to(device=device, dtype=dtype)
    if tuple(decoder.shape) != (d_model, d_sae):
        raise RuntimeError(f"decoder shape {tuple(decoder.shape)} != {(d_model, d_sae)}")

    return LlamaScopeSAE(
        layer=layer,
        mixture=mixture,
        d_model=d_model,
        d_sae=d_sae,
        decoder=decoder,
        encoder=w["encoder.weight"].to(device=device, dtype=dtype),
        encoder_bias=w["encoder.bias"].to(device=device, dtype=dtype),
        decoder_bias=w["decoder.bias"].to(device=device, dtype=dtype),
        log_jumprelu_threshold=w["log_jumprelu_threshold"].to(device=device, dtype=dtype),
        dataset_avg_norm=dataset_avg_norm,
    )
