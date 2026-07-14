"""The GRPO reward, and the confound hiding inside it.

The paper's reward is R = 0.9*accuracy + 0.1*format, and NOTHING rewards conversation.
That is load-bearing: the claim is that dialogue emerges (or helps) without being paid
for. A reward that leaked credit to conversational tokens would assume the conclusion.

The subtle danger is the FORMAT term. Conversation-SFT teaches <persona1>/<think1>/
<group_consensus>; the format reward wants <think>/<answer>. Score that strictly and the
conversation-primed arm is penalised *for emitting the scaffolding under test* -- which
would produce a confident "conversational scaffolding hurts" finding out of pure
bookkeeping. These tests pin both scoring modes so the result can be shown not to hinge
on the choice.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from rl.reward import accuracy_reward, countdown_reward, format_reward

MONOLOGUE = "<think> 30 - 25 = 5, 5 + 3 = 8, 8 * 4 = 32 </think><answer>(30 - 25 + 3) * 4</answer>"
DIALOGUE = (
    "<persona1> mathematician </persona1><persona2> engineer </persona2>"
    "<think1> try 30 - 25 = 5 </think1><think2> then (5 + 3) * 4 </think2>"
    "<group_consensus> (30 - 25 + 3) * 4 </group_consensus>"
)


def test_accuracy_is_the_arithmetic_not_the_models_claim():
    assert accuracy_reward("<answer>(30 - 25 + 3) * 4</answer>", 32, [25, 30, 3, 4]) == 1.0
    # asserts "= 32" but evaluates to 20: the claim is irrelevant, the math is not
    assert accuracy_reward("<answer>(30 - 25) * 4 = 32</answer>", 32, [25, 30, 3, 4]) == 0.0
    # must use every number exactly once
    assert accuracy_reward("<answer>30 - 25 + 3</answer>", 8, [25, 30, 3, 4]) == 0.0


def test_reward_is_the_papers_formula():
    r = countdown_reward([MONOLOGUE], target=[32], nums=[[25, 30, 3, 4]])[0]
    assert r == pytest.approx(0.9 * 1.0 + 0.1 * 1.0)

    # correct answer, no reasoning block -> loses only the format term
    bare = "<answer>(30 - 25 + 3) * 4</answer>"
    assert countdown_reward([bare], target=[32], nums=[[25, 30, 3, 4]])[0] == pytest.approx(0.9)

    # well-formatted but wrong -> keeps only the format term
    wrong = "<think> guessing </think><answer>(30 - 25) * 4</answer>"
    assert countdown_reward([wrong], target=[32], nums=[[25, 30, 3, 4]])[0] == pytest.approx(0.1)


def test_nothing_rewards_conversation_itself():
    """Two completions, same (correct) answer, one dialogic. Reward must be IDENTICAL.

    If personas earned even a fraction of reward, the experiment would be rigged: we
    would be paying for the behaviour whose usefulness we claim to be measuring.
    """
    mono = countdown_reward([MONOLOGUE], target=[32], nums=[[25, 30, 3, 4]])[0]
    dial = countdown_reward([DIALOGUE], target=[32], nums=[[25, 30, 3, 4]])[0]
    assert mono == dial == pytest.approx(1.0)


def test_dialogue_scaffolding_is_not_punished_by_the_format_term():
    """THE CONFOUND. A dialogue trace uses <think1>/<group_consensus>, not <think>/<answer>.

    Under the lenient (default) scoring it must still earn format credit -- otherwise the
    conversation arm is fined for the very thing under test.
    """
    assert format_reward(DIALOGUE) == 1.0
    assert format_reward(MONOLOGUE) == 1.0


def test_strict_format_makes_the_confound_visible():
    """And under strict scoring it IS punished -- which is exactly why the flag exists.

    This test documents the bias rather than hiding it: if the headline result flips
    between strict and lenient, the result is an artifact of bookkeeping, not of
    reasoning, and the writeup must say so.
    """
    assert format_reward(MONOLOGUE, strict=True) == 1.0
    assert format_reward(DIALOGUE, strict=True) == 0.0, (
        "strict scoring penalises dialogue scaffolding -- the confound this flag exposes"
    )


def test_multiple_answers_are_not_rewarded():
    """One answer block. A trace that emits several is hedging, not solving."""
    two = "<think> a </think><answer>1 + 1</answer><answer>2 + 2</answer>"
    assert format_reward(two) == 0.0
