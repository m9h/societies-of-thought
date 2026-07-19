"""Does HF's Flax Llama actually ignore rope_theta / rope_scaling?

docs/flax_rope_bug.md claims it does, based on reading the source:

    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))   # base hardcoded

This tests the claim behaviourally instead, on a tiny random-init model -- no
8B checkpoint, no GPU, no HF cache needed.

DESIGN. Load the SAME weights into PyTorch LlamaForCausalLM and
FlaxLlamaForCausalLM and diff the logits, at two rotary bases:

    rope_theta = 10000    -> CONTROL. This is the value Flax hardcodes, so the
                             two must AGREE. If they disagree here, the harness
                             is broken (dtype, weight conversion, masking) and
                             the 500000 result would mean nothing.

    rope_theta = 500000   -> The real Llama-3.1 / DeepSeek-R1-Distill value.
                             If Flax ignores the config, the two must DIVERGE.

The control is the point. A bare "they differ at 500000" is consistent with any
number of conversion bugs; "they agree at 10000 and differ at 500000" isolates
the cause to the rotary base specifically.

Third case adds rope_scaling={"rope_type":"llama3",...} on top, which is what
our model actually carries.
"""

import numpy as np
import torch
from transformers import (
    FlaxLlamaForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

SEQ = 12
SEED = 0


def build(rope_theta: float, rope_scaling=None, tmp="/tmp/claude-1000/ropetest/tiny"):
    """Random-init a tiny Llama in PyTorch, save it, reload it in Flax."""
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, rope_theta=rope_theta,
        rope_scaling=rope_scaling, torch_dtype="float32",
    )
    torch.manual_seed(SEED)
    pt = LlamaForCausalLM(cfg).eval()
    pt.save_pretrained(tmp, safe_serialization=True)
    fx = FlaxLlamaForCausalLM.from_pretrained(tmp, from_pt=True, dtype=np.float32)
    return pt, fx


def compare(rope_theta, rope_scaling=None):
    pt, fx = build(rope_theta, rope_scaling)
    ids = np.arange(SEQ, dtype=np.int32)[None, :]

    with torch.no_grad():
        pt_logits = pt(torch.tensor(ids, dtype=torch.long)).logits.numpy()
    fx_logits = np.asarray(fx(ids).logits)

    d = np.abs(pt_logits - fx_logits)
    denom = np.abs(pt_logits).max() or 1.0
    return d.max(), d.max() / denom


def main():
    print(f"{'case':<44} {'max|Δlogit|':>12} {'relative':>10}   verdict")
    print("-" * 84)

    cases = [
        ("CONTROL  rope_theta=10000 (Flax's hardcoded)", 10000.0, None),
        ("rope_theta=500000 (Llama-3.1 real value)", 500000.0, None),
        ("rope_theta=500000 + rope_scaling=llama3", 500000.0, {
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}),
    ]

    results = {}
    for label, theta, scaling in cases:
        try:
            absd, reld = compare(theta, scaling)
            agree = reld < 1e-3
            results[label] = (absd, reld, agree)
            print(f"{label:<44} {absd:12.6f} {reld:10.2e}   "
                  f"{'AGREE' if agree else 'DIVERGE'}")
        except Exception as e:
            print(f"{label:<44} {'ERROR':>12}  {type(e).__name__}: {str(e)[:60]}")
            results[label] = None

    print()
    ctrl = results.get(cases[0][0])
    real = results.get(cases[1][0])
    if not ctrl or not real:
        print("INCONCLUSIVE: a case failed to run.")
        return
    if not ctrl[2]:
        print("HARNESS BROKEN: control disagrees at rope_theta=10000, so the")
        print("divergence at 500000 cannot be attributed to the rotary base.")
        return
    if real[2]:
        print("REFUTED: Flax and PyTorch agree at rope_theta=500000, so Flax is")
        print("NOT ignoring the config. docs/flax_rope_bug.md is wrong.")
    else:
        print("CONFIRMED: control agrees at 10000, diverges at 500000.")
        print("Flax hardcodes the rotary base and ignores config.rope_theta.")
        print("Any jlens-jax result on a Llama-3 model via Flax is invalid.")


if __name__ == "__main__":
    main()
