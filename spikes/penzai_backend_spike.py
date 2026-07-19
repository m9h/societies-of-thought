# SPDX-License-Identifier: Apache-2.0
"""SPIKE: can penzai back jlens-jax now that HF Flax is gone?

WHY THIS EXISTS
---------------
jlens-jax (~/Workspace/jlens-jax) gets activations through HuggingFace Flax:
`FlaxAutoModelForCausalLM` + `output_hidden_states=True`, wrapped in models.py's
FlaxLayout. Its pyproject pins `transformers>=4.38,<5.0`.

That pin is load-bearing and terminal. Verified 2026-07-18 against the
transformers 5.14.1 installed in societies-of-thought/.venv:

    FlaxAutoModelForCausalLM   GONE
    FlaxLlamaForCausalLM       GONE
    FlaxGemmaForCausalLM       GONE

HF removed Flax outright in v5. So the JAX port's backend is not merely dated,
it is pinned to a branch that receives no new model support -- meaning no
Gemma 3, and no future model, ever, through that path.

penzai is the obvious replacement: a JAX-native transformer implementation that
reads HF *PyTorch* checkpoints and needs no Flax. It is dormant (last release
0.2.5, 2025-04-08) but dormancy turns out to be the wrong thing to judge it on
-- see FINDINGS below.

WHAT THIS SPIKE PROVES
----------------------
Run it and it demonstrates, on a toy 4-block llamalike model:

  1. penzai imports and runs on jax 0.11.0 (released 2026-07-16, two days before
     this was written -- and NEWER than the 0.10.2 in .venv-test). The 15-month
     dormancy has not broken it.
  2. `pz.select(model).at_instances_of(ApplyRoPE)` finds and replaces every RoPE
     layer BY TYPE, with no string paths. This matters because it is how we would
     add llama3 frequency rescaling, which is penzai's actual blocker (below).
  3. `.at_instances_of(TransformerBlock).apply_with_selected_index(...)` taps every
     block and captures its residual -- i.e. it can supply exactly the
     `forward_with_intermediates(ids) -> (logits, hidden_states)` contract that
     jlens_jax/protocol.py already defines as its seam.

FINDINGS -- the real blocker is coverage, not maintenance
---------------------------------------------------------
penzai's Llama converter (variants/llama.py:55) does a strict type check that
DeepSeek-R1-Distill-Llama-8B passes (it IS a LlamaForCausalLM). It then validates
config against LlamaConfig() defaults and rejects anything unhandled. Llama-3.1
carries:

    rope_scaling = {"rope_type": "llama3", "factor": 8.0,
                    "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192}

which is NOT in its handled set -- so our model is rejected. The guard is correct
to refuse: converting without llama3 RoPE would silently give wrong long-context
behaviour. penzai predates Llama-3.1.

Separately, variants/gemma.py tops out at `gemma2_27b`. There is no Gemma 3.

Both gaps are additive, and point 2 above is why they are cheap: llama3 RoPE is a
piecewise rescale of the inverse frequencies (~40 lines), and it can be injected
via pz.select WITHOUT forking penzai. Gemma 3 would be a larger lift.

WHAT THIS SPIKE DOES NOT PROVE
------------------------------
  - That real converted weights produce numerics matching HF. Nothing here loads
    a real checkpoint; the RoPE class below is a pass-through STAND-IN, not an
    implementation. Numerical equivalence against HF is the actual gate before
    anything downstream can be trusted, and it is not attempted here.
  - That penzai should back the PyTorch pipeline. It should not -- see below.

SCOPE: this is about the JAX side ONLY
--------------------------------------
analysis/jspace_tier1.py is PyTorch (torch.nn.Module, torch.no_grad,
jlens.from_hf) and its `import jax` at line 8 is dead. penzai is irrelevant there.
Likewise the steering sweep stays PyTorch: it is generation-bound (run_sweep runs
to max_new_tokens=8192 because steered traces often never emit EOS), and giving up
vLLM to gain a hook abstraction is the wrong trade.

Run:  /tmp/claude-1000/pz/venv/bin/python spikes/penzai_backend_spike.py
      (or any env with `pip install "jax[cpu]" penzai`)
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer.model_parts import TransformerBlock
from penzai.models.transformer.variants import llamalike_common as L
from penzai.nn.embeddings import ApplyRoPE


@pz.pytree_dataclass
class Llama3RoPE(pz.nn.Layer):
    """STAND-IN for llama3 frequency rescaling -- the gap penzai has.

    A real implementation rescales the inverse frequencies piecewise by
    wavelength against low_freq_factor / high_freq_factor / factor, per the
    Llama-3.1 RoPE. This just delegates, so the spike demonstrates the INJECTION
    MECHANISM without pretending to be numerically correct. Do not use downstream.
    """

    inner: ApplyRoPE

    def __call__(self, x, **kw):
        return self.inner(x, **kw)


@pz.pytree_dataclass
class BlockTap(pz.nn.Layer):
    """Captures a block's output residual, the jlens `hidden_states[l+1]` contract."""

    inner: TransformerBlock
    idx: int = dataclasses.field(metadata={"pytree_node": False})
    sink: list = dataclasses.field(metadata={"pytree_node": False})

    def __call__(self, x, **kw):
        out = self.inner(x, **kw)
        self.sink.append((self.idx, out))
        return out


def main() -> None:
    cfg = L.LlamalikeTransformerConfig(
        num_kv_heads=2, query_head_multiplier=1, embedding_dim=32,
        projection_dim=8, mlp_hidden_dim=64, num_decoder_blocks=4,
        vocab_size=128, mlp_variant="geglu_approx", tie_embedder_and_logits=False,
    )
    model = L.build_llamalike_transformer(cfg, init_base_rng=jax.random.key(0))
    print(f"jax {jax.__version__} | built {cfg.num_decoder_blocks}-block llamalike")

    # 1. Replace RoPE by TYPE -- no string paths, no fork of penzai.
    patched = pz.select(model).at_instances_of(ApplyRoPE).apply(
        lambda r: Llama3RoPE(inner=r))
    n_rope = len(pz.select(patched).at_instances_of(Llama3RoPE).selected_by_path)
    print(f"  RoPE layers replaced by type : {n_rope}   (q and k per block)")

    # 2. Tap every block -> the forward_with_intermediates contract.
    sink: list = []
    probed = pz.select(patched).at_instances_of(
        TransformerBlock).apply_with_selected_index(
            lambda i, b: BlockTap(inner=b, idx=i, sink=sink))

    seq = pz.nx.wrap(jnp.arange(6)).tag("seq")
    logits = probed(seq, token_positions=pz.nx.wrap(jnp.arange(6)).tag("seq"))

    print(f"  blocks tapped                : {len(sink)}  {[i for i, _ in sink]}")
    print(f"  residual named_shape         : {sink[0][1].named_shape}")
    print(f"  logits named_shape           : {logits.named_shape}")
    assert len(sink) == cfg.num_decoder_blocks, "must capture one residual per block"
    assert n_rope == 2 * cfg.num_decoder_blocks, "q and k RoPE per block"
    print("\nOK: penzai can supply jlens-jax's protocol without HF Flax.")
    print("GATE: numerical equivalence vs HF on real weights is NOT tested here.")


if __name__ == "__main__":
    main()
