"""GRPO batch arithmetic, and the configurations that silently do not learn.

Separate from train_grpo.py on purpose: importing the trainer drags in torch,
trl, peft and datasets, and validation that needs a GPU stack to import is
validation nobody runs before submitting a job. This module imports nothing.

THE PROBLEM IT SOLVES. GRPO computes each completion's advantage *within its
prompt's group* -- the group being the `num_generations` completions sampled for
that one prompt. Two consequences that are easy to get wrong and impossible to
see from a log:

  num_generations = 1     The group has one member, so its advantage is
                          identically zero. No gradient. Trains, logs, saves,
                          learns nothing.

  one prompt per step     Every comparison is among attempts at a SINGLE puzzle.
                          There is a gradient, but no signal about which PROBLEMS
                          are being solved better. train_grpo.py's own comment:
                          "that is a comparison among 8 attempts at a single
                          puzzle and it does not learn."

Neither condition raises. Both produce a flat reward curve after hours of GPU
time, which is then usually blamed on the learning rate.

THE ARITHMETIC. TRL's `per_device_train_batch_size` counts COMPLETIONS, not
prompts, so:

    prompts per optimiser step = batch_size * grad_accum / num_generations

The micro-batch has to stay small (96 completions x 1024 tokens of activations
OOM'd an 80GB A100), so the prompt count is recovered through gradient
accumulation rather than a bigger batch. That is a legitimate configuration and
must not be rejected -- only the inverted ratio is a bug.

tests/test_grpo_config.py pins all of it.
"""

from __future__ import annotations

# Below this many prompts per optimiser step, the gradient carries essentially no
# information about which problems are getting easier -- it is a within-puzzle
# comparison. Not a hard mathematical threshold; a floor chosen so the degenerate
# cases the comments describe are rejected and the legitimate small-micro-batch
# case is not.
MIN_PROMPTS_PER_STEP = 2
MIN_GENERATIONS = 2


def prompts_per_step(*, batch_size: int, grad_accum: int, num_generations: int) -> int:
    """Distinct prompts contributing to one optimiser step.

    batch_size is COMPLETIONS per device, which is the part that trips people up.
    """
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    return (batch_size * grad_accum) // num_generations


def check_grpo_config(*, batch_size: int, grad_accum: int, num_generations: int) -> int:
    """Raise if this configuration would train without learning. Returns the
    prompts-per-step it validated, so callers can log it.

    Every message names the knob to turn: a startup failure is only cheaper than
    an hour-eight failure if it says what to change.
    """
    if num_generations < MIN_GENERATIONS:
        raise ValueError(
            f"num_generations={num_generations} gives a group of "
            f"{num_generations}, so every completion's advantage is identically "
            f"zero and there is no gradient. Use num_generations >= "
            f"{MIN_GENERATIONS} (the harness default is 8)."
        )

    if batch_size % num_generations != 0:
        raise ValueError(
            f"batch_size={batch_size} is not a multiple of "
            f"num_generations={num_generations}. TRL requires this and otherwise "
            f"fails deep in the trainer with a shape error that does not name the "
            f"cause. Set batch_size to a multiple of num_generations."
        )

    n = prompts_per_step(batch_size=batch_size, grad_accum=grad_accum,
                         num_generations=num_generations)
    if n < MIN_PROMPTS_PER_STEP:
        raise ValueError(
            f"batch_size={batch_size} x grad_accum={grad_accum} / "
            f"num_generations={num_generations} = {n} prompt(s) per optimiser "
            f"step. GRPO compares completions within a prompt's group, so this is "
            f"a comparison among attempts at a single puzzle: it will train, log "
            f"and checkpoint without learning. Raise grad_accum (cheapest -- keeps "
            f"the micro-batch small, which is what avoids the OOM), or raise "
            f"batch_size, or lower num_generations. Need >= "
            f"{MIN_PROMPTS_PER_STEP}."
        )
    return n
