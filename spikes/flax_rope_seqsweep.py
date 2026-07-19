"""How does the Flax RoPE error scale with sequence length?
A 2.3e-3 relative diff at seq=12 is easy to dismiss. RoPE phase error grows with
position, so the number that matters is the one at reasoning-trace lengths."""
import numpy as np, torch
from transformers import FlaxLlamaForCausalLM, LlamaConfig, LlamaForCausalLM

def run(seq, theta):
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=8192, rope_theta=theta, torch_dtype="float32")
    torch.manual_seed(0)
    pt = LlamaForCausalLM(cfg).eval()
    pt.save_pretrained("/tmp/claude-1000/ropetest/t2", safe_serialization=True)
    fx = FlaxLlamaForCausalLM.from_pretrained("/tmp/claude-1000/ropetest/t2", from_pt=True, dtype=np.float32)
    ids = np.arange(seq, dtype=np.int32)[None, :] % 128
    with torch.no_grad():
        a = pt(torch.tensor(ids, dtype=torch.long)).logits.numpy()
    b = np.asarray(fx(ids).logits)
    return np.abs(a-b).max()/ (np.abs(a).max() or 1.0)

print(f"{'seq_len':>8} {'control θ=10k':>15} {'real θ=500k':>14}   ratio")
print("-"*56)
for seq in [16, 64, 256, 1024, 4096]:
    c = run(seq, 10000.0); r = run(seq, 500000.0)
    print(f"{seq:>8} {c:>15.2e} {r:>14.2e}   {r/c:>8.0f}x")
