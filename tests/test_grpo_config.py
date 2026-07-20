"""GRPO config invariants that fail SILENTLY if violated.

train_grpo.py is the one module in rl/ with no tests, and the one that has never
run end to end. Its own comments record two config traps, both of which produce a
run that trains, logs, saves checkpoints, and learns nothing:

    "batch=8 with num_generations=8 gives ONE prompt per step -- GRPO's advantage
     is computed within a prompt's group, so that is a comparison among 8 attempts
     at a single puzzle and it does not learn."

    "96 completions x 1024 tokens of activations OOM'd an 80GB A100 ... What
     matters for GRPO is PROMPTS PER OPTIMIZER STEP, not the micro-batch."

Nothing enforced either. A bad --batch-size/--num-generations pair was accepted
and would have burned a day of GPU time before anyone noticed the reward curve
was flat. That is the expensive way to discover arithmetic.

These tests pin the arithmetic, and drive extraction of a `prompts_per_step` /
`check_grpo_config` pair so the harness refuses a useless configuration at
startup instead of at hour eight.

No GPU, no model download, no TRL trainer -- pure arithmetic on the numbers the
argument parser produces. That is why these live in rl/grpo_config.py rather than
in train_grpo.py: importing the trainer drags in torch, trl and peft, and config
validation that needs a GPU stack to import is validation nobody runs.
"""

from __future__ import annotations

import pytest

from rl.grpo_config import check_grpo_config, prompts_per_step


# --- the arithmetic ------------------------------------------------------------

def test_prompts_per_step_is_batch_times_accum_over_generations():
    # the harness defaults: 16 completions x 24 accum / 8 generations = 48 prompts
    assert prompts_per_step(batch_size=16, grad_accum=24, num_generations=8) == 48


def test_defaults_are_sane():
    """The committed defaults must pass their own check, or the harness ships
    broken out of the box."""
    check_grpo_config(batch_size=16, grad_accum=24, num_generations=8)


# --- the trap the comments describe --------------------------------------------

def test_batch_equal_to_generations_is_rejected():
    """THE documented failure: batch=8, num_generations=8 -> one prompt per
    micro-batch. GRPO's advantage is computed within a prompt's group, so every
    comparison is among attempts at a SINGLE puzzle. It trains and does not learn."""
    with pytest.raises(ValueError, match="(?i)prompt"):
        check_grpo_config(batch_size=8, grad_accum=1, num_generations=8)


def test_batch_not_divisible_by_generations_is_rejected():
    """TRL requires per_device_train_batch_size to be a multiple of
    num_generations. Getting this wrong fails deep inside the trainer with a
    shape error that does not name the cause."""
    with pytest.raises(ValueError, match="(?i)divisib|multiple"):
        check_grpo_config(batch_size=12, grad_accum=4, num_generations=8)


def test_single_generation_is_rejected():
    """num_generations=1 makes the within-group advantage identically zero --
    there is nothing to compare a completion against. No gradient, no error."""
    with pytest.raises(ValueError, match="(?i)generation"):
        check_grpo_config(batch_size=8, grad_accum=4, num_generations=1)


def test_too_few_prompts_per_step_is_rejected_even_with_accumulation():
    """Accumulation can rescue a small micro-batch, but not an inverted ratio.
    2 completions x 1 accum / 2 generations = 1 prompt per optimiser step."""
    with pytest.raises(ValueError, match="(?i)prompt"):
        check_grpo_config(batch_size=2, grad_accum=1, num_generations=2)


def test_accumulation_can_rescue_a_small_micro_batch():
    """The legitimate case the OOM comment describes: keep the micro-batch small,
    recover prompt count through gradient accumulation. Must NOT be rejected."""
    check_grpo_config(batch_size=8, grad_accum=8, num_generations=4)   # 16 prompts
    assert prompts_per_step(batch_size=8, grad_accum=8, num_generations=4) == 16


# --- the smoke config I nearly ran ---------------------------------------------

def test_the_smoke_config_is_checked_not_assumed():
    """scripts/sbatch_rl_smoke.sh uses batch 8 / accum 2 / generations 4 -> 4
    prompts per step. That is legal but thin, and it was chosen by eye rather
    than checked. Pin it so a future edit to the smoke script cannot quietly
    drop it below the floor."""
    assert prompts_per_step(batch_size=8, grad_accum=2, num_generations=4) == 4
    check_grpo_config(batch_size=8, grad_accum=2, num_generations=4)


# --- the error must say what to do ---------------------------------------------

def test_rejection_message_names_the_fix():
    """A config error at startup is only cheaper than one at hour eight if it
    tells you which knob to turn."""
    with pytest.raises(ValueError) as exc:
        check_grpo_config(batch_size=8, grad_accum=1, num_generations=8)
    msg = str(exc.value)
    assert "num_generations" in msg or "num-generations" in msg
    assert any(k in msg for k in ("grad_accum", "grad-accum", "batch"))
