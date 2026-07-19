"""Load a Llama-3.1 checkpoint into penzai, which otherwise refuses it.

penzai ships two entry points for Llama:

    variants.llama.llama_from_huggingface_model         validates config, then delegates
    variants.llamalike_common.llamalike_from_huggingface_model    does the actual work

The validating wrapper raises on Llama-3.1 because `rope_scaling` is not in its
handled set (`variants/llama.py:96`). That guard is right in spirit -- converting
without llama3 RoPE would be silently wrong, which is exactly what the HF Flax
path does (docs/flax_rope_bug.md) -- but it is stricter than necessary once the
scaling is actually implemented.

So we call the inner converter directly and then inject the RoPE we do
implement, via `pz.select(...).at_instances_of(ApplyRoPE)`. No fork of penzai,
no monkeypatch: the injection uses penzai's own model-surgery API.

Crucially this keeps the guard's INTENT. An unrecognised rope_type still raises,
because the failure this module exists to prevent is a model that loads happily
while computing different math than it was trained with.

Verified end to end against HF PyTorch in tests/test_penzai_loader.py.

TRANSFORMERS VERSION. `rope_settings` below reads both the 4.x and 5.x config
layouts, because v5 moved rope_theta/rope_scaling into a single
`rope_parameters` dict and reading only the old key would silently drop llama3
scaling.

But note penzai itself is NOT v5-ready: `llamalike_common.py:599` does
`rope_wavelength=hf_config.rope_theta`, and that attribute no longer exists in
v5. So conversion currently requires transformers 4.x. It fails with a plain
AttributeError rather than doing something subtly wrong, which is the acceptable
kind of failure -- but it does mean the penzai path and the steering sweep
cannot share one environment today (societies-of-thought/.venv is on 5.x).
"""

from __future__ import annotations

from typing import Any

from penzai import pz
from penzai.models.transformer import model_parts
from penzai.models.transformer.variants import llamalike_common
from penzai.nn.embeddings import ApplyRoPE

from penzai_backend.llama3_rope import Llama3RoPE

# rope_types penzai handles natively once the base (rope_theta) is passed
# through, which llamalike_common already does via rope_wavelength.
_NO_SCALING_NEEDED = {None, "default"}


def rope_settings(hf_config: Any) -> tuple[str | None, dict]:
    """Return (rope_type, scaling_params) across transformers 4.x and 5.x.

    The layout moved between major versions:

        v4   config.rope_theta   = 500000.0
             config.rope_scaling = {"rope_type": "llama3", "factor": 8.0, ...}

        v5   config.rope_parameters = {"rope_type": "llama3",
                                       "rope_theta": 500000.0,
                                       "factor": 8.0, ...}
             and rope_scaling / rope_theta no longer exist

    Reading only `rope_scaling` means that under v5 rope_type comes back None,
    the caller concludes there is no scaling to apply, and a Llama-3.1 model
    converts WITHOUT llama3 frequencies -- silently. That is the same class of
    bug as the HF Flax hardcoded base (docs/flax_rope_bug.md), reintroduced by a
    dependency bump rather than by anyone writing wrong math.

    Verified against the real DeepSeek-R1-Distill-Llama-8B config under
    transformers 5.13.1 on 2026-07-19.
    """
    params = getattr(hf_config, "rope_parameters", None)
    if isinstance(params, dict) and params:                 # transformers 5.x
        scaling = dict(params)
    else:                                                   # transformers 4.x
        scaling = dict(getattr(hf_config, "rope_scaling", None) or {})

    rope_type = scaling.get("rope_type") or scaling.get("type")
    return rope_type, scaling


def llama3_to_penzai(
    hf_model: Any,
    *,
    upcast_activations_to_float32: bool = True,
    use_layer_stack: bool = False,
) -> model_parts.TransformerLM:
    """Convert a HF LlamaForCausalLM (incl. Llama-3.1) to a penzai TransformerLM.

    Args:
      hf_model: a `transformers.LlamaForCausalLM`, already loaded.
      upcast_activations_to_float32: default True. Checkpoints are usually bf16,
        and comparing a bf16 forward pass against a reference is dominated by
        rounding rather than by conversion correctness. Upcasting makes a
        numerical mismatch mean something.
      use_layer_stack: pass through to penzai; stacks blocks into one scanned
        layer. Leave False when you intend to tap individual blocks, since a
        stacked model has one block object rather than N.

    Raises:
      ValueError: if the config carries a rope_type we do not implement. Loading
        it anyway would produce a model that runs and is wrong.
    """
    rope_type, scaling = rope_settings(hf_model.config)

    if rope_type not in _NO_SCALING_NEEDED and rope_type != "llama3":
        raise ValueError(
            f"rope_type={rope_type!r} is not implemented by this loader. Only "
            "'llama3' (and unscaled) are supported. Converting anyway would give "
            "a model that loads cleanly and computes different positional math "
            "than the weights were trained with -- see docs/flax_rope_bug.md for "
            "what that failure looks like in practice."
        )

    # Call the INNER converter, skipping variants/llama.py's config guard, which
    # rejects rope_scaling wholesale. We re-impose the part of that guard that
    # still matters (above) rather than dropping it.
    model = llamalike_common.llamalike_from_huggingface_model(
        hf_model,
        upcast_activations_to_float32=upcast_activations_to_float32,
        use_layer_stack=use_layer_stack,
    )

    if rope_type == "llama3":
        model = pz.select(model).at_instances_of(ApplyRoPE).apply(
            lambda rope: Llama3RoPE.from_apply_rope(
                rope,
                rope_type="llama3",
                factor=scaling["factor"],
                low_freq_factor=scaling["low_freq_factor"],
                high_freq_factor=scaling["high_freq_factor"],
                original_max_position_embeddings=scaling[
                    "original_max_position_embeddings"],
            )
        )
        if not pz.select(model).at_instances_of(Llama3RoPE).selected_by_path:
            raise RuntimeError(
                "config requested llama3 RoPE but no ApplyRoPE layer was found to "
                "replace. penzai's internal structure has changed; refusing to "
                "return a model with unscaled frequencies."
            )

    return model
