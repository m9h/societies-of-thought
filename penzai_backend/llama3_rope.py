"""Llama-3.1 RoPE frequency rescaling as a penzai layer.

WHY THIS EXISTS
---------------
penzai's Llama converter refuses Llama-3.1 checkpoints. `variants/llama.py`
validates the HF config against `LlamaConfig()` defaults and raises on any
unhandled key; Llama-3.1 carries

    rope_scaling = {"rope_type": "llama3", "factor": 8.0,
                    "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192}

which penzai (last release 2025-04) never implemented. The refusal is correct
behaviour, and is exactly what the HF Flax path fails to do -- Flax hardcodes
the rotary base and silently computes different math (docs/flax_rope_bug.md).

Only the frequency rescale is missing. penzai's `ApplyRoPE` already takes
`max_wavelength`, and `llamalike_common` passes `rope_wavelength=
hf_config.rope_theta`, so the base itself is handled.

WHAT LLAMA3 SCALING DOES
------------------------
Standard RoPE gives each embedding pair an inverse frequency

    inv_freq[i] = 1 / theta**(2i/head_dim)         wavelen = 2*pi / inv_freq

Llama-3.1 rescales these piecewise by wavelength, against a reference context
length (8192) and two band factors:

    wavelen < old_ctx/high_freq_factor   ->  unchanged
        High-frequency components encode fine local position. Leave them alone
        or short-range behaviour degrades.

    wavelen > old_ctx/low_freq_factor    ->  inv_freq / factor
        Low-frequency components encode long-range position. Dividing by the
        factor is what stretches the usable context window.

    in between                           ->  smooth interpolation
        A hard cutoff would put a discontinuity in the middle of the frequency
        spectrum.

USAGE
-----
Inject into a penzai model by TYPE, without forking penzai:

    from penzai import pz
    from penzai.nn.embeddings import ApplyRoPE
    from penzai_backend.llama3_rope import Llama3RoPE

    model = pz.select(model).at_instances_of(ApplyRoPE).apply(
        lambda r: Llama3RoPE.from_apply_rope(r, **cfg.rope_scaling))

Correctness is pinned against transformers' own
`ROPE_INIT_FUNCTIONS["llama3"]` in tests/test_llama3_rope.py -- the numbers the
model was actually trained under, not a second copy of the formula.

STATUS: the frequency math is verified against the HF oracle. End-to-end
numerical equivalence of a converted 8B model against HF PyTorch is NOT yet
established and is the gate before trusting anything downstream.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.nn.embeddings import ApplyRoPE


def llama3_inv_freq(
    head_dim: int,
    rope_theta: float,
    *,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> jax.Array:
    """Inverse frequencies with Llama-3.1 rescaling applied.

    Mirrors `transformers.modeling_rope_utils._compute_llama3_parameters`.
    Returns an array of length head_dim//2.
    """
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

    old_ctx = float(original_max_position_embeddings)
    low_wavelen = old_ctx / low_freq_factor
    high_wavelen = old_ctx / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # Long wavelengths get divided by the factor; everything else passes for now.
    inv_freq_llama = jnp.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)

    # Smoothly interpolate across the middle band. Guard the denominator so a
    # degenerate config (low == high) cannot produce NaN -- is_medium_freq is
    # empty in that case anyway, but NaN would propagate through jnp.where.
    denom = high_freq_factor - low_freq_factor
    smooth = (old_ctx / wavelen - low_freq_factor) / (denom if denom != 0 else 1.0)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama

    is_medium = jnp.logical_and(wavelen >= high_wavelen, wavelen <= low_wavelen)
    return jnp.where(is_medium, smoothed, inv_freq_llama)


@pz.pytree_dataclass
class Llama3RoPE(pz.nn.Layer):
    """Drop-in replacement for penzai's ApplyRoPE with llama3 rescaling.

    Mirrors ApplyRoPE's interface (same embedding_axis / positions_input_name /
    max_wavelength semantics) so `pz.select(...).at_instances_of(ApplyRoPE)` can
    swap it in without touching the surrounding model.
    """

    embedding_axis: str = dataclasses.field(metadata={"pytree_node": False})
    max_wavelength: float = dataclasses.field(metadata={"pytree_node": False})
    positions_input_name: str = dataclasses.field(metadata={"pytree_node": False})
    factor: float = dataclasses.field(metadata={"pytree_node": False})
    low_freq_factor: float = dataclasses.field(metadata={"pytree_node": False})
    high_freq_factor: float = dataclasses.field(metadata={"pytree_node": False})
    original_max_position_embeddings: int = dataclasses.field(
        metadata={"pytree_node": False})

    @classmethod
    def from_apply_rope(cls, rope: ApplyRoPE, *, factor: float,
                        low_freq_factor: float, high_freq_factor: float,
                        original_max_position_embeddings: int,
                        rope_type: str = "llama3") -> "Llama3RoPE":
        """Build from an existing ApplyRoPE, keeping its axis names and base.

        Accepts `rope_type` so a config's rope_scaling dict can be splatted in
        directly, and rejects anything other than llama3 rather than silently
        applying the wrong scaling -- the failure mode this whole module exists
        to avoid.
        """
        if rope_type != "llama3":
            raise ValueError(
                f"rope_type={rope_type!r} is not llama3. This layer implements "
                "Llama-3.1 scaling only; applying it to another rope_type would "
                "silently compute the wrong frequencies."
            )
        return cls(
            embedding_axis=rope.embedding_axis,
            max_wavelength=rope.max_wavelength,
            positions_input_name=rope.positions_input_name,
            factor=factor,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            original_max_position_embeddings=original_max_position_embeddings,
        )

    def _apply_1d(self, input_slice: jax.Array, position: jax.Array) -> jax.Array:
        assert input_slice.ndim == 1
        assert position.ndim == 0
        [head_dim] = input_slice.shape
        inv_freq = llama3_inv_freq(
            head_dim, self.max_wavelength,
            factor=self.factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
        )
        sinusoid_inp = position * inv_freq
        sin, cos = jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
        first_half, second_half = jnp.split(input_slice, 2)
        return jnp.concatenate([
            first_half * cos - second_half * sin,
            second_half * cos + first_half * sin,
        ])

    def __call__(self, inputs, **side_inputs):
        positions = side_inputs[self.positions_input_name]
        if self.embedding_axis in positions.named_shape:
            raise ValueError(
                f"Embedding axis {self.embedding_axis} should not already be part "
                "of the positions side input."
            )
        out = pz.nx.nmap(self._apply_1d)(inputs.untag(self.embedding_axis), positions)
        return out.tag(self.embedding_axis).astype(inputs.dtype)
