"""Prime a base model on one arm's SFT traces -- full-FT, prompt-masked, single-GPU.

Claim B primes the base model, then RLs it. This does the priming. It is deliberately
small and self-contained rather than verl's FSDP SFT trainer: the priming set is 500
examples, it fits full-fine-tuning of a 3B on one 80GB A100 (bf16 + grad checkpointing +
8-bit Adam), and owning the loop removes a dependence on whichever verl-SFT CLI the pinned
TinyZero image happens to ship. The output is a plain HuggingFace checkpoint dir, which
verl PPO loads directly via model.path.

The one thing that must be right is loss masking: train on the RESPONSE only, never the
prompt (tests/test_sft_prime.py). Everything else here is standard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IGNORE_INDEX = -100


def mask_prompt_labels(input_ids: list[int], prompt_len: int,
                       ignore_index: int = IGNORE_INDEX) -> list[int]:
    """Labels = input_ids with the first `prompt_len` positions masked out."""
    n = len(input_ids)
    k = min(max(prompt_len, 0), n)
    return [ignore_index] * k + list(input_ids[k:])


def encode_example(tokenizer, prompt: str, response: str, max_len: int) -> dict:
    """Tokenise prompt+response into a single causal-LM example with the prompt masked
    and an EOS appended to the response so the model learns to stop."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        response_ids = response_ids + [eos]

    input_ids = (prompt_ids + response_ids)[:max_len]
    labels = mask_prompt_labels(input_ids, len(prompt_ids))
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---- training (imports torch/transformers lazily; not exercised by the unit tests) --

def _collate(batch, pad_id):
    import torch

    m = max(len(b["input_ids"]) for b in batch)
    def pad(seq, v): return seq + [v] * (m - len(seq))
    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], IGNORE_INDEX) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-FT SFT priming for one Claim B arm.")
    ap.add_argument("--train", type=Path, required=True, help="sft_<arm>_train.parquet")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--out", type=Path, required=True, help="checkpoint dir for PPO to load")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=1536)  # dialogue median ~220w; headroom
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    df = pd.read_parquet(args.train)
    examples = [encode_example(tok, r["prompt"], r["response"], args.max_len)
                for _, r in df.iterrows()]
    print(f"{len(examples)} SFT examples from {args.train.name}; "
          f"median len {sorted(len(e['input_ids']) for e in examples)[len(examples)//2]}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    targs = TrainingArguments(
        output_dir=str(args.out / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        optim="adamw_bnb_8bit",
        report_to=[],
        seed=args.seed,
    )
    Trainer(model=model, args=targs, train_dataset=examples,
            data_collator=lambda b: _collate(b, tok.pad_token_id)).train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"primed checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
