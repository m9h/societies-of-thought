"""Activation-addition steering on a Llama decoder layer's residual stream.

TransformerLens `blocks.{L}.hook_resid_post` -- the hook point the Llama-Scope
SAEs were trained on -- is the output hidden state of HF decoder layer L. So a
forward hook on `model.model.layers[L]` intervenes at exactly the right place.
"""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def steer(model, layer: int, delta: torch.Tensor | None, scope: str = "all"):
    """Add `delta` (real residual-stream units) to layer L's output.

    scope="all"        -- steer prompt tokens and generated tokens (default).
    scope="generated"  -- steer only during incremental decoding, leaving the
                          prompt's own representations untouched. With a KV cache
                          the prefill pass is the one call whose sequence length
                          exceeds 1, so we skip it.

    delta=None is a no-op, so the baseline condition runs through the identical
    code path as the steered ones (same batching, same kernels, same RNG draws).
    """
    if delta is None:
        yield
        return
    if scope not in ("all", "generated"):
        raise ValueError(f"scope must be 'all' or 'generated', got {scope!r}")

    block = model.model.layers[layer]
    state = {"prefill_done": False}

    def hook(_module, _args, output):
        # Llama decoder layers return a tuple whose first element is the hidden state.
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        prefill = hidden.shape[1] > 1
        apply = True
        if scope == "generated":
            if prefill and not state["prefill_done"]:
                apply = False
            if prefill:
                state["prefill_done"] = True

        if apply:
            hidden = hidden + delta.to(dtype=hidden.dtype, device=hidden.device)

        if is_tuple:
            return (hidden,) + output[1:]
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
