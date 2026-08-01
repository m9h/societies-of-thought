"""LIAR2 misinformation task (SoT Claim C6) — tests written BEFORE the module.

The paper: "we use 23,299 fact-checked claims from the PolitiFact corpus ... six
PolitiFact labels -- True, Mostly True, Half True, Mostly False, False, and Pants on
Fire -- ... which we recode into three categories":

    True      = {True, Mostly True}
    Half True = {Half True}
    False     = {False, Mostly False, Pants on Fire}

Getting the 6->3 recoding or the int->name mapping backwards would invert the task
silently, so both are pinned here. The reward mirrors the paper's Countdown reward
(0.9*accuracy + 0.1*format), and must carry the "Assistant:" marker that verl's scorer
splits on -- omitting it zeroed every rollout in Claim B and cost a 170-step run.
"""
from __future__ import annotations

import pytest

from rl.misinfo_data import LABEL_NAMES, make_prompt, recode_label
from rl.misinfo_reward import compute_score, extract_verdict


# --- the 6 -> 3 recoding ------------------------------------------------------


@pytest.mark.parametrize("raw,expect", [
    (5, "true"),          # True
    (4, "true"),          # Mostly True
    (3, "half-true"),     # Half True
    (2, "false"),         # Mostly False
    (1, "false"),         # False
    (0, "false"),         # Pants on Fire
])
def test_recoding_matches_the_paper(raw, expect):
    assert recode_label(raw) == expect


def test_label_names_are_ascending_truthfulness():
    """Empirically verified against LIAR2 content: 0 is pants-on-fire, 5 is true."""
    assert LABEL_NAMES[0] == "pants-on-fire"
    assert LABEL_NAMES[5] == "true"
    assert len(LABEL_NAMES) == 6


def test_recoding_rejects_an_out_of_range_label():
    with pytest.raises(ValueError):
        recode_label(9)


# --- the prompt ---------------------------------------------------------------


def test_prompt_carries_the_scorer_marker():
    """verl's reward splits on 'Assistant:' and returns None otherwise, scoring 0
    unconditionally. This exact omission zeroed a 170-step Claim B run."""
    p = make_prompt("Taxes went up 90 percent.")
    assert "Assistant:" in p


def test_prompt_states_the_three_options_and_the_answer_format():
    p = make_prompt("Taxes went up 90 percent.")
    for token in ("true", "half-true", "false", "<answer>", "<think>"):
        assert token in p
    assert "Taxes went up 90 percent." in p


def test_prompt_does_not_pre_open_the_reasoning_tag():
    """A pre-opened <think> forces a dialogue-primed model's <persona1> opening out of
    distribution -- the trap documented in rl/claimB_data.py."""
    assert not make_prompt("x").rstrip().endswith("<think>")


# --- verdict extraction -------------------------------------------------------


@pytest.mark.parametrize("text,expect", [
    ("Assistant: <think>hm</think><answer>true</answer>", "true"),
    ("Assistant: <answer>False</answer>", "false"),
    ("Assistant: <answer> Half-True </answer>", "half-true"),
    ("Assistant: <answer>half true</answer>", "half-true"),
    ("Assistant: <answer>**false**</answer>", "false"),
    ("Assistant: <answer>The claim is true</answer>", "true"),
])
def test_extract_verdict_handles_real_formats(text, expect):
    assert extract_verdict(text) == expect


def test_extract_returns_none_without_the_marker():
    assert extract_verdict("<answer>true</answer>") is None


def test_extract_returns_none_without_an_answer_tag():
    assert extract_verdict("Assistant: I think it is true") is None


def test_ambiguous_answer_is_rejected_rather_than_guessed():
    assert extract_verdict("Assistant: <answer>true or false</answer>") is None


# --- the reward ---------------------------------------------------------------


def test_correct_verdict_scores_one():
    s = compute_score("Assistant: <think>x</think><answer>true</answer>", "true")
    assert s == pytest.approx(1.0)


def test_wrong_verdict_with_valid_format_scores_format_only():
    """Mirrors the paper's 0.9*accuracy + 0.1*format."""
    s = compute_score("Assistant: <think>x</think><answer>false</answer>", "true")
    assert s == pytest.approx(0.1)


def test_unparseable_answer_scores_zero():
    assert compute_score("Assistant: no idea", "true") == 0.0


def test_missing_marker_scores_zero():
    assert compute_score("<answer>true</answer>", "true") == 0.0


def test_three_way_task_is_not_silently_binary():
    """half-true must be its own class; collapsing it would make the task easier and
    would not be the paper's task."""
    assert compute_score("Assistant: <answer>half-true</answer>", "half-true") == 1.0
    assert compute_score("Assistant: <answer>true</answer>", "half-true") == pytest.approx(0.1)
