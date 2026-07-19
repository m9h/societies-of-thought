# SPDX-License-Identifier: Apache-2.0
"""Mutation check for tests/test_penzai_loader.py.

"The test could fail" and "the test does fail" are different claims. This breaks
the loader on purpose -- converts WITHOUT injecting llama3 RoPE -- and confirms
the agreement threshold notices. Result on 2026-07-19:

    correct loader   rel = 3.7e-07   PASS   (270x inside the 1e-4 tolerance)
    mutant, no swap  rel = 8.6e-04   FAIL   (correctly caught)

Run: /tmp/claude-1000/ropetest/venv/bin/python spikes/mutation_check_loader.py
(or any env with jax + penzai + torch + transformers)
"""
import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, torch, transformers
from penzai import pz
from penzai.models.transformer.variants import llamalike_common
from penzai_backend.loader import llama3_to_penzai

S = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
     "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
TOL = 1e-4

cfg = transformers.LlamaConfig(
    vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=131072,
    rope_theta=500000.0, rope_scaling=S, torch_dtype="float32")
torch.manual_seed(0)
hf = transformers.LlamaForCausalLM(cfg).eval()
ids = [i % 64 for i in range(512)]
with torch.no_grad():
    ref = hf(torch.tensor(ids)[None, :]).logits[0].numpy()

def pz_logits(m):
    out = m(pz.nx.wrap(jnp.asarray(ids)).tag("seq"),
            token_positions=pz.nx.wrap(jnp.arange(len(ids))).tag("seq"))
    return jax.device_get(out.unwrap("seq", "vocabulary"))

mutant = llamalike_common.llamalike_from_huggingface_model(
    hf, upcast_activations_to_float32=True)
rel_mut = abs(ref - pz_logits(mutant)).max() / abs(ref).max()
rel_ok = abs(ref - pz_logits(llama3_to_penzai(hf))).max() / abs(ref).max()

print(f"  correct loader  : rel={rel_ok:.3e}  {'PASS' if rel_ok < TOL else 'FAIL'} (want PASS)")
print(f"  MUTANT (no swap): rel={rel_mut:.3e}  {'PASS' if rel_mut < TOL else 'FAIL'} (want FAIL)")
print("\n  suite is meaningful" if rel_ok < TOL < rel_mut
      else "\n  SUITE IS DECORATIVE -- it cannot catch the bug")
