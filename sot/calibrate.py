"""Measure each feature's activation scale in OUR convention.

Neuronpedia reports max activation 5.906 for feature 30939. At the hook point and
scaling that reconstruction proves correct, we measure ~18.4 on the very contexts
the paper prints in Fig. 2a (where it shows 5.78 / 5.75 / 4.75). The offset is
consistent (~3.2x), so Neuronpedia's displayed activations simply live on a
different scale than the SAE's own.

That does not affect the paper's conclusions, and it does not affect which
features are conversational. But it *would* wreck the sweep: steering strength is
defined relative to a feature's max activation, so sizing alpha off Neuronpedia's
number would make every intervention ~3x weaker than intended -- silently, and
with no error to notice.

So we calibrate ourselves. One pass over SlimPajama (the SAE's own training
corpus, and the corpus Neuronpedia sampled) gives, per feature, the max activation
and the firing rate in exactly the units the steering hook operates in.

BOS is excluded. Its residual norm here is ~466 against ~11 for ordinary tokens --
an attention sink the SAE was never trained to model, and including it corrupts
every statistic it touches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sae import load_sae

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CORPUS = "DKYoon/SlimPajama-6B"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--mixture", default="slimpj")
    ap.add_argument("--n-docs", type=int, default=2000)
    ap.add_argument("--ctx", type=int, default=128)  # Neuronpedia's context length
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hook-point", type=Path, default=Path("results/hook_point.json"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    info = json.loads(args.hook_point.read_text())
    hook_layer = info["hook_layer"]
    hs_offset = hook_layer + 1  # hidden_states[L+1] == output of layer L
    print(f"hook: {info['hook_point']} -> output of layer {hook_layer}")

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=device
    ).eval()
    sae = load_sae(args.layer, args.mixture, device=device)

    ds = load_dataset(CORPUS, split="train", streaming=True)
    texts = []
    for r in ds:
        t = (r.get("text") or "").strip()
        if len(t) > 200:
            texts.append(t)
        if len(texts) >= args.n_docs:
            break
    print(f"calibrating on {len(texts)} SlimPajama docs, ctx={args.ctx}")

    max_act = torch.zeros(sae.d_sae, device=device)
    n_fire = torch.zeros(sae.d_sae, device=device, dtype=torch.long)
    n_tok = 0

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=args.ctx, padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        x = out.hidden_states[hs_offset].float()  # [B, T, d_model]

        acts = sae.encode(x)  # [B, T, d_sae]

        # Mask BOS (position 0) and padding. Both would poison the statistics:
        # BOS is an unmodeled attention sink, padding isn't real text.
        mask = enc["attention_mask"].bool().clone()
        mask[:, 0] = False
        acts = acts[mask]  # [n_real_tokens, d_sae]

        max_act = torch.maximum(max_act, acts.max(0).values)
        n_fire += (acts > 0).sum(0)
        n_tok += acts.shape[0]

        if (i // args.batch_size) % 10 == 0:
            print(f"  {i + len(batch):5d}/{len(texts)} docs, {n_tok:8d} tokens", flush=True)

    stats = {
        "layer": args.layer,
        "mixture": args.mixture,
        "hook_layer": hook_layer,
        "n_tokens": n_tok,
        "max_act": max_act.cpu().tolist(),
        "frac_nonzero": (n_fire.float() / max(n_tok, 1)).cpu().tolist(),
    }
    out = args.out or Path(f"results/feature_stats_L{args.layer}_{args.mixture}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats))

    anchor = 30939
    print(f"\nfeature {anchor}: max_act={max_act[anchor]:.3f}  "
          f"frac_nonzero={n_fire[anchor].item() / max(n_tok,1):.5f}")
    print(f"  (Neuronpedia reports max_act=5.906, frac_nonzero=0.00016)")
    print(f"\nwrote {out}  ({n_tok} tokens)")


if __name__ == "__main__":
    main()
