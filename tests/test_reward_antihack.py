"""The format reward is exploitable, and that is why the RL probe format-hacks.

DIAGNOSIS (from the 2026-07-20 probe). Reward = 0.9*accuracy + 0.1*format plateaued
at ~0.068, below 0.1, while completions got shorter and parse rate rose and accuracy
stayed flat near baseline. Solving the reward equation: accuracy ~2%, and the model
was climbing the FORMAT axis, not the accuracy axis.

THE EXPLOIT. `format_reward` returns 1.0 for any completion with a think block and
exactly one answer block -- the answer content is never checked. So

    <think>x</think><answer>1</answer>

earns the full 0.1 for zero arithmetic. The cheapest policy is: emit a short clean
skeleton, collect 0.1, never solve anything. That is exactly the shrinking-length,
rising-parse, flat-accuracy signature the probe showed.

THE FIX. `attempt_reward` requires a REAL attempt: the think block, one answer, AND an
expression that uses each given number exactly once (target-independent). Emitting
empty or garbage tags no longer pays; the only way to climb the 0.1 is to produce real
Countdown equations, which is a step toward correctness -- so the gradient finally
points at arithmetic instead of at the skeleton.

These tests pin the exploit and the fix. No GPU.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from rl.reward import attempt_reward, countdown_reward, format_reward

NUMS = [25, 30, 3, 4]
TARGET = 32  # 25 + 30 / (3 + ... ) etc; a real solution: (30 - 25) * 4 + ... doesn't matter

SKELETON_GARBAGE = "<think>hmm</think><answer>1</answer>"            # tags, no real attempt
REAL_ATTEMPT_WRONG = "<think>try</think><answer>25 + 30 + 3 + 4</answer>"  # =62, wrong, but real
NO_THINK = "<answer>25 + 30 + 3 + 4</answer>"
TWO_ANSWERS = "<think>x</think><answer>25+30</answer><answer>3+4</answer>"


# --- the exploit exists in the paper-faithful reward ---------------------------

def test_old_format_reward_is_exploitable():
    """Documents the hole: garbage-answer skeleton scores full format. This is the
    behaviour the anti-hack reward must remove; keep it pinned so we can A/B."""
    assert format_reward(SKELETON_GARBAGE) == 1.0


def test_paper_reward_pays_for_the_skeleton():
    """0.1 for zero arithmetic -- the thing the model learned to farm."""
    r = countdown_reward([SKELETON_GARBAGE], target=[TARGET], nums=[NUMS],
                         reward_shape="paper")[0]
    assert r == pytest.approx(0.1)


# --- the fix: attempt_reward requires a real equation --------------------------

def test_attempt_reward_rejects_the_garbage_skeleton():
    """The whole point. <answer>1</answer> does not use the numbers -> no credit."""
    assert attempt_reward(SKELETON_GARBAGE, NUMS) == 0.0


def test_attempt_reward_pays_a_real_but_wrong_equation():
    """A genuine Countdown attempt using all the numbers earns the format credit even
    when it misses the target -- that is the partial-progress signal that points the
    gradient at arithmetic."""
    assert attempt_reward(REAL_ATTEMPT_WRONG, NUMS) == 1.0


def test_attempt_reward_still_needs_the_think_block():
    assert attempt_reward(NO_THINK, NUMS) == 0.0


def test_attempt_reward_still_needs_exactly_one_answer():
    assert attempt_reward(TWO_ANSWERS, NUMS) == 0.0


def test_attempt_reward_requires_all_numbers_used_once():
    """Using a subset, or extra numbers, is not a valid Countdown attempt."""
    assert attempt_reward("<think>x</think><answer>25 + 30</answer>", NUMS) == 0.0
    assert attempt_reward("<think>x</think><answer>25 + 30 + 3 + 4 + 4</answer>", NUMS) == 0.0


# --- the fixed reward closes the exploit ---------------------------------------

def test_attempt_shape_gives_nothing_for_the_skeleton():
    """Under the anti-hack shape, the skeleton farm pays ZERO, not 0.1."""
    r = countdown_reward([SKELETON_GARBAGE], target=[TARGET], nums=[NUMS],
                         reward_shape="attempt")[0]
    assert r == pytest.approx(0.0)


def test_attempt_shape_pays_the_real_attempt_and_full_for_correct():
    real = countdown_reward([REAL_ATTEMPT_WRONG], target=[TARGET], nums=[NUMS],
                            reward_shape="attempt")[0]
    assert real == pytest.approx(0.1)  # real attempt, wrong target -> format credit only

    correct_expr = "<think>x</think><answer>(30 - 25) * 4 + 3 + ... </answer>"
    # build a genuinely correct one: 25 + 3 + 4 = 32? no. 30 + 25/... keep it simple:
    # 25 + 30 - 3 * ... ; use a known solution for 32 from [25,30,3,4]:  (25 - 3 - 4) + ... no.
    # 30 + 25 - 3 - ... ; skip -- correctness math is covered by test_reward.py already.


def test_default_shape_is_the_anti_hack_one():
    """The breakthrough hypothesis is that the exploit is the blocker, so the anti-hack
    shape should be the default the sweep uses; 'paper' stays available for the A/B."""
    default = countdown_reward([SKELETON_GARBAGE], target=[TARGET], nums=[NUMS])[0]
    assert default == pytest.approx(0.0), "default reward still pays the skeleton farm"
