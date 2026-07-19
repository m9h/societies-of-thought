"""Repair HF Flax Llama's RoPE, which hardcodes the rotary base.

`transformers.models.llama.modeling_flax_llama.create_sinusoidal_positions`
builds its frequencies as

    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

with the base **hardcoded**, reading neither `config.rope_theta` nor
`config.rope_scaling`. The implementation predates Llama 3 and was never
updated. Llama-3.1 uses rope_theta=500000 plus llama3 frequency rescaling, so
every Flax forward pass on a Llama-3 model runs with positional math the weights
were never trained under -- silently, with no warning. See docs/flax_rope_bug.md.

WHY PATCH RATHER THAN MIGRATE
-----------------------------
penzai_backend/ is the strategic answer (not deprecated, reaches Gemma 3, TPU
capable). But jlens-jax already runs on Flax, and a migration invalidates
nothing while a patch RESCUES existing work: results can be recomputed correctly
in place instead of being discarded pending a backend rewrite.

Measured on a tiny Llama-3.1-config model at 512 tokens, relative max logit
error against HF PyTorch:

    Flax unpatched                     5.7e-03
    Flax + correct rope_theta          8.6e-04
    Flax + rope_theta + llama3 scaling 3.7e-07     <- matches penzai exactly

The decomposition is worth noting: the hardcoded base is the bulk of the error,
and the missing llama3 scaling accounts for the remaining ~3 orders of
magnitude. Fixing only the base gets you most of the way and is still wrong.

LIMITS -- this is a rescue, not a future
----------------------------------------
  * Requires transformers < 5. v5 deleted the Flax classes entirely, and v4
    already emits "TensorFlow and JAX classes are deprecated and will be removed
    in Transformers v5" on import.
  * HF Flax has FlaxLlamaForCausalLM and FlaxGemmaForCausalLM (Gemma 1) and
    NOTHING for Gemma 2 or Gemma 3. The Gemma 3 27B arm is unreachable this way
    regardless of RoPE.
  * It monkeypatches a module-level function, so it is process-global while
    active. Use the context manager and load inside it.

USAGE
-----
    from flax_compat.rope_fix import patched_llama_rope

    with patched_llama_rope.from_config(cfg):
        model = FlaxLlamaForCausalLM.from_pretrained(path, from_pt=True)

The frequencies are baked in at model-construction time (`setup()` calls
`create_sinusoidal_positions` once), so the patch only needs to be active while
the model is being BUILT, not while it runs.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Iterator

import numpy as np


def corrected_inv_freq(dim: int, rope_theta: float, scaling: dict | None) -> np.ndarray:
    """Inverse frequencies with the real base and llama3 rescaling if requested.

    Mirrors transformers' `_compute_llama3_parameters`; kept in numpy because
    `create_sinusoidal_positions` is a numpy function.
    """
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, 2) / dim))
    if not scaling:
        return inv_freq

    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_ctx = float(scaling["original_max_position_embeddings"])

    wavelen = 2 * math.pi / inv_freq
    out = np.where(wavelen > old_ctx / low_freq_factor, inv_freq / factor, inv_freq)

    denom = high_freq_factor - low_freq_factor
    smooth = (old_ctx / wavelen - low_freq_factor) / (denom if denom else 1.0)
    smoothed = (1 - smooth) * out / factor + smooth * out

    is_medium = (wavelen >= old_ctx / high_freq_factor) & (
        wavelen <= old_ctx / low_freq_factor)
    return np.where(is_medium, smoothed, out)


class _PatchedLlamaRope:
    """Context manager that swaps in a correct frequency builder."""

    @contextlib.contextmanager
    def __call__(self, rope_theta: float,
                 scaling: dict | None = None) -> Iterator[None]:
        import jax.numpy as jnp
        from transformers.models.llama import modeling_flax_llama as flax_llama

        original = flax_llama.create_sinusoidal_positions

        def fixed(num_pos: int, dim: int):
            inv_freq = corrected_inv_freq(dim, rope_theta, scaling)
            freqs = np.einsum("i,j->ij", np.arange(num_pos), inv_freq).astype("float32")
            emb = np.concatenate((freqs, freqs), axis=-1)
            out = np.concatenate(
                (np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
            return jnp.array(out[:, :, :num_pos])

        flax_llama.create_sinusoidal_positions = fixed
        try:
            yield
        finally:
            flax_llama.create_sinusoidal_positions = original

    @contextlib.contextmanager
    def from_config(self, hf_config: Any) -> Iterator[None]:
        """Read rope_theta / rope_scaling off a config and patch accordingly.

        Raises on a rope_type we do not implement, rather than silently applying
        no scaling -- which is the original bug wearing a different hat.
        """
        rope_theta, scaling = rope_settings(hf_config)
        with self(rope_theta, scaling):
            yield


patched_llama_rope = _PatchedLlamaRope()


def rope_settings(hf_config: Any) -> tuple[float, dict | None]:
    """Extract (rope_theta, llama3_scaling_or_None) from a HF config.

    Handles both the transformers 4.x layout (rope_theta + rope_scaling as
    separate attributes) and the 5.x layout (a single rope_parameters dict),
    for the same reason penzai_backend/loader.py does: reading only the old key
    under v5 silently yields "no scaling".
    """
    params = getattr(hf_config, "rope_parameters", None)
    if isinstance(params, dict) and params:                     # transformers 5.x
        bag = dict(params)
        rope_theta = bag.get("rope_theta", 10000.0)
    else:                                                       # transformers 4.x
        bag = dict(getattr(hf_config, "rope_scaling", None) or {})
        rope_theta = getattr(hf_config, "rope_theta", 10000.0)

    rope_type = bag.get("rope_type") or bag.get("type")
    if rope_type in (None, "default"):
        return rope_theta, None
    if rope_type != "llama3":
        raise ValueError(
            f"rope_type={rope_type!r} is not implemented here. Only 'llama3' "
            "(and unscaled) are handled. Patching with the wrong scaling would "
            "reproduce the very failure this module fixes."
        )
    return rope_theta, {
        "factor": bag["factor"],
        "low_freq_factor": bag["low_freq_factor"],
        "high_freq_factor": bag["high_freq_factor"],
        "original_max_position_embeddings": bag["original_max_position_embeddings"],
    }
