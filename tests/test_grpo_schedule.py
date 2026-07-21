"""The learning-rate schedule that produced a flat reward curve must be fixable.

The first Claim A run drifted +0.014 reward over 150 steps -- noise. Two causes,
both in the schedule, both invisible without reading the trajectory:

  peak LR 1e-6   ~10x below typical GRPO-on-3B (2-5e-6); the policy barely moves
                 (KL topped out at 0.0014).
  no scheduler   TRL defaults to linear decay to ZERO, so LR fell to 6.7e-9 by
                 step 150 -- the back half of training did almost nothing.

`resolve_schedule` makes both explicit and refuses the decay-to-zero default
silently. Pure arithmetic, no trl, no GPU.
"""
from __future__ import annotations
import pytest
from rl.grpo_config import resolve_schedule


def test_default_does_not_decay_to_zero():
    """The trap. The default schedule must hold the LR up, not anneal it away
    over a short run."""
    lr, sched, warmup = resolve_schedule()
    assert sched in ("constant", "constant_with_warmup"), (
        f"default scheduler {sched!r} decays the LR; the first run died this way"
    )


def test_default_peak_is_not_the_starved_value():
    """1e-6 barely moved a 3B policy. The default peak should be in the range
    that actually learns."""
    lr, _, _ = resolve_schedule()
    assert lr >= 2e-6, f"default peak LR {lr} is in the range that produced flat reward"


def test_explicit_lr_and_schedule_pass_through():
    lr, sched, warmup = resolve_schedule(lr=3e-6, schedule="cosine", warmup_ratio=0.1)
    assert lr == 3e-6 and sched == "cosine" and warmup == 0.1


def test_linear_decay_is_allowed_when_asked_for_explicitly():
    """The point is not to ban decay -- it is to not get it by accident."""
    lr, sched, _ = resolve_schedule(schedule="linear")
    assert sched == "linear"


def test_rejects_a_clearly_wrong_lr():
    with pytest.raises(ValueError, match="(?i)learning rate|lr"):
        resolve_schedule(lr=5.0)
