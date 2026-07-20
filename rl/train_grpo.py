"""SFT-prime, then GRPO on Countdown. The paper's Claim B, run for real.

Three arms, identical in every respect except how the model was primed:

    baseline    -- no SFT at all
    dialogue    -- SFT on multi-agent dialogue traces
    monologue   -- SFT on single-voice traces, SAME problems, SAME correct answers

Then identical RL on all three. Any difference is attributable to the FORM of the
primed reasoning, because the content is held fixed by construction (see generate_sft.py).

WHY GRPO AND NOT PPO. The paper used PPO. TRL 1.8 has REMOVED PPOTrainer -- it does not
exist, and an earlier automated attempt at this destroyed itself writing code against
that dead API. The paper itself reports that its preliminary analyses found no
significant difference between PPO and GRPO, and DeepSeek-R1 used GRPO. GRPO also drops
the critic, which roughly halves memory. It is used for EVERY arm, so the comparison is
unaffected.

WHAT WE ADD OVER THE PAPER. The paper appears to report single runs, and its headline is
an early-training gap (step 40 of 250) -- exactly where seed noise is largest. We run
>= 3 seeds per arm and report the spread. If between-seed variance swamps the
between-arm gap, that is the finding.

Everything is logged to trackio, including parse_rate and mean_completion_tokens: an arm
can gain or lose apparent accuracy purely by changing its formatting or its verbosity,
and without those series you cannot tell that apart from reasoning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from peft import LoraConfig  # noqa: E402
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

from rl.reward import countdown_reward, format_reward, grade_completion  # noqa: E402
from rl.grpo_config import check_grpo_config
from sot.grade import grade  # noqa: E402

PROMPT = (
    "Using the numbers {nums}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only "
    "be used once. Show your work in <think> </think> tags. And return the final "
    "answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


def countdown_split(n: int, seed: int, skip: int = 0) -> Dataset:
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    import random

    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[skip : skip + n]
    rows = [
        {
            "prompt": PROMPT.format(nums=list(ds[i]["nums"]), target=ds[i]["target"]),
            "target": ds[i]["target"],
            "nums": list(ds[i]["nums"]),
        }
        for i in idx
    ]
    return Dataset.from_list(rows)


def sft_prime(arm: str, base_model: str, out: Path, seed: int, epochs: int = 3):
    """Fine-tune on dialogue or monologue traces. Returns the primed model path."""
    data_path = Path("rl/data") / f"{arm}_train.json"
    rows = json.loads(data_path.read_text())
    text_key = arm

    examples = [
        {"prompt": PROMPT.format(nums=r["numbers"], target=r["target"]),
         "completion": r[text_key]}
        for r in rows
    ]
    print(f"[{arm}] SFT on {len(examples)} verified examples")

    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,      # unified memory: keep it small
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=seed,
        max_length=1024,
    )
    trainer = SFTTrainer(model=base_model, args=cfg, train_dataset=Dataset.from_list(examples))
    trainer.train()
    trainer.save_model(str(out))
    del trainer
    torch.cuda.empty_cache()
    return str(out)


class EvalAndLog:
    """Held-out accuracy every N steps, plus the two series that expose confounds."""

    def __init__(self, tok, eval_ds, every, run):
        self.tok, self.eval_ds, self.every, self.run = tok, eval_ds, every, run
        self.history = []

    @torch.no_grad()
    def __call__(self, model, step: int):
        if step % self.every:
            return
        model.eval()
        correct = parsed = 0
        toks = 0
        for row in self.eval_ds:
            enc = self.tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=400, do_sample=False,
                                 pad_token_id=self.tok.pad_token_id)
            text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            # grade_completion, NOT grade: the eval must normalise dialogue answer
            # containers exactly as the reward does, or the dialogue arm reports 0%
            # while actually scoring 0.4 reward. See rl/reward.py.
            g = grade_completion(text, row["target"], row["nums"])
            correct += g.correct
            parsed += g.parsed
            toks += int((out[0] != self.tok.pad_token_id).sum()) - enc["input_ids"].shape[1]
        n = len(self.eval_ds)
        rec = {
            "step": step,
            "val_accuracy": correct / n,
            "parse_rate": parsed / n,
            "mean_completion_tokens": toks / n,
        }
        self.history.append(rec)
        print(f"    step {step:4d}  acc={rec['val_accuracy']:.1%}  "
              f"parse={rec['parse_rate']:.1%}  tok={rec['mean_completion_tokens']:.0f}")
        try:
            import trackio
            trackio.log(rec)
        except Exception:
            pass
        model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "dialogue", "monologue"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--steps", type=int, default=250)     # the paper's horizon
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=128)
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--num-generations", type=int, default=8)  # completions per prompt
    ap.add_argument("--batch-size", type=int, default=16,
                    help="COMPLETIONS per step, not prompts. TRL divides by "
                         "num_generations to get prompts/step. batch=8 with "
                         "num_generations=8 gives ONE prompt per step -- GRPO's advantage "
                         "is computed within a prompt's group, so that is a comparison "
                         "among 8 attempts at a single puzzle and it does not learn.")
    ap.add_argument("--grad-accum", type=int, default=24)
    ap.add_argument("--no-vllm", action="store_true",
                    help="use HF generate for rollouts instead of vLLM. GRPO generates "
                         "num_generations completions per prompt per step (384/step here); "
                         "HF generate holds the batch until the SLOWEST completion ends, so "
                         "one long rollout stalls 383 others. vLLM's continuous batching "
                         "frees each slot as it finishes -- roughly an order of magnitude, "
                         "and the difference between a 1-day and a 10-day A/B (3 arms x 3 "
                         "seeds).")
    ap.add_argument("--full-finetune", action="store_true",
                    help="full FT instead of LoRA. OOMs a 80GB A100 on a 3B model: "
                         "policy + frozen reference + fp32 AdamW states is ~42GB before "
                         "a single rollout.")
    ap.add_argument("--strict-format", action="store_true",
                    help="score format strictly; shows whether the result is an artifact "
                         "of NOT penalising dialogue scaffolding")
    ap.add_argument("--out", type=Path, default=Path("rl/runs"))
    args = ap.parse_args()

    # Fail on a configuration that would train without learning BEFORE loading a
    # model or touching a GPU. GRPO's advantage is computed within a prompt's
    # group, so num_generations=1 gives a zero gradient and one-prompt-per-step
    # gives a within-puzzle comparison -- neither raises on its own, and both
    # look like a learning-rate problem after eight hours. See rl/grpo_config.py.
    n_prompts_per_step = check_grpo_config(
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        num_generations=args.num_generations)
    print(f"  config OK: {n_prompts_per_step} prompts per optimizer step "
          f"({args.batch_size} completions x {args.grad_accum} accum / "
          f"{args.num_generations} generations)")

    run_name = f"{args.arm}_seed{args.seed}"
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import trackio
        trackio.init(project="societies-of-thought-rl", name=run_name,
                     config={"arm": args.arm, "seed": args.seed, "model": args.model,
                             "trainer": "GRPO", "steps": args.steps,
                             "strict_format": args.strict_format})
    except Exception as e:
        print(f"(trackio unavailable: {e})")

    model_path = args.model
    if args.arm != "baseline":
        model_path = sft_prime(args.arm, args.model, run_dir / "sft", args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = countdown_split(args.train_n, seed=args.seed)
    eval_ds = countdown_split(args.eval_n, seed=12345, skip=50_000)  # disjoint held-out

    cfg = GRPOConfig(
        output_dir=str(run_dir),
        learning_rate=1e-6,
        per_device_train_batch_size=args.batch_size,     # completions
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        # 400 tokens truncates Countdown reasoning; TinyZero-style runs use ~1024.
        max_completion_length=1024,
        max_steps=args.steps,
        logging_steps=1,
        # Each checkpoint is ~35GB (weights + optimizer state) and filled a 120GB disk.
        # Provenance needs SOME real checkpoints, not many -- keep the last two.
        save_steps=max(args.steps // 3, 1),
        save_total_limit=2,
        bf16=True,
        report_to=[],
        seed=args.seed,
        beta=0.04,
        # vLLM for rollouts, colocated in-process (no separate server to babysit).
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,   # leave room for policy + LoRA + grads
        # 96 completions x 1024 tokens of activations OOM'd an 80GB A100 (policy +
        # reference model + fp32 AdamW states leave little room). What matters for GRPO
        # is PROMPTS PER OPTIMIZER STEP, not the micro-batch -- so keep the micro-batch
        # small and recover the prompt count through gradient accumulation.
        gradient_checkpointing=True,
    )

    def reward_fn(completions, target, nums, **kw):
        return countdown_reward(completions, target, nums, strict_format=args.strict_format)

    # LoRA rather than full fine-tuning. Full FT of a 3B model needs policy (6GB) +
    # frozen reference (6GB) + fp32 AdamW states (~24GB) + grads before any rollout is
    # generated, and OOMs an 80GB A100. LoRA drops the optimizer state to a few hundred
    # MB. DEVIATION FROM THE PAPER, recorded: it is applied IDENTICALLY to every arm, so
    # the dialogue-vs-monologue comparison is unaffected -- only the absolute learning
    # rate of all three arms is.
    peft_cfg = None if args.full_finetune else LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    trainer = GRPOTrainer(
        model=model_path,
        args=cfg,
        train_dataset=train_ds,
        reward_funcs=reward_fn,
        processing_class=tok,
        peft_config=peft_cfg,
    )

    evaluator = EvalAndLog(tok, eval_ds, args.eval_every, run_name)
    evaluator(trainer.model, 0)  # step 0: the paper's baseline starts near zero

    # Must subclass TrainerCallback: HF's Trainer calls the full hook interface
    # (on_train_begin, on_log, ...), so a bare class with only on_step_end dies
    # immediately with AttributeError.
    class _Cb(TrainerCallback):
        def on_step_end(self, cfg_, state, control, **kw):
            evaluator(trainer.model, state.global_step)

    trainer.add_callback(_Cb())
    trainer.train()

    (run_dir / "curve.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "model": args.model, "trainer": "GRPO",
        "strict_format": args.strict_format,
        "history": evaluator.history,
    }, indent=1))
    print(f"wrote {run_dir/'curve.json'}")


if __name__ == "__main__":
    main()
