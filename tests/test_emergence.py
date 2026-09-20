"""Fig. 4 emergence: do conversational behaviours rise under accuracy-only RL?

Tests written BEFORE the implementation. The paper's Fig. 4(b) reports the frequency of
two named behaviours -- "Question & Answering" and "Conflict of Perspectives" -- rising
across PPO steps on Qwen-2.5-3B trained on Countdown with reward = 0.9*accuracy +
0.1*format. Their instrument is an LLM judge; this module builds the judge-free proxy and
the step alignment that the judge also needs.

The step alignment is the part most likely to break silently. verl prints a batch's
rollouts BEFORE the metric line for the step that consumed them, so a naive "nearest
step:N above" rule assigns every trace to the wrong step, and an emergence curve computed
that way would still look perfectly plausible.
"""
from __future__ import annotations

import pytest

from analysis.emergence import (
    behaviours_per_step,
    conflict_of_perspectives,
    parse_rollouts,
    question_answering,
    strip_log_noise,
)

# Verbatim from results/rl_ab/tz_train_claimA.log, ANSI codes and Ray prefixes included.
RAW = (
    "\x1b[36m(main_task pid=3491)\x1b[0m User: Using the numbers [2, 31, 80, 1], create an "
    "equation that equals 51.\n"
    "\x1b[36m(main_task pid=3491)\x1b[0m <think> We can get something close to 51. </think>\n"
    "\x1b[36m(main_task pid=3491)\x1b[0m <answer> (31 * 1) + (80 / 2) </answer>\n"
    "\x1b[36m(main_task pid=3491)\x1b[0m step:1 - critic/kl:0.000 - "
    "critic/score/mean:0.047\n"
    "\x1b[36m(main_task pid=3491)\x1b[0m User: Using the numbers [43, 7, 77], create an "
    "equation that equals 54.\n"
    "\x1b[36m(main_task pid=3491)\x1b[0m <think> (77 / 7) - 43 </think>\n"
    "\x1b[36m(main_task pid=3491)\x1b[0m <answer> (77 / 7) - 43 </answer>\n"
    "\x1b[36m(main_task pid=3491)\x1b[0m step:2 - critic/score/mean:0.061\n"
)


def test_strip_log_noise_removes_ansi_and_ray_prefixes():
    out = strip_log_noise(RAW)
    assert "\x1b[" not in out
    assert "main_task pid=" not in out
    assert "Using the numbers [2, 31, 80, 1]" in out


def test_parse_rollouts_recovers_both_samples():
    rs = parse_rollouts(RAW)
    assert len(rs) == 2
    assert "80, 1]" in rs[0]["prompt"]
    assert "(31 * 1) + (80 / 2)" in rs[0]["response"]


def test_rollouts_are_attributed_to_the_step_that_consumed_them():
    """verl prints rollouts BEFORE the step line they belong to. Assigning them to the
    preceding step line instead would shift the whole emergence curve by one batch and
    still look plausible."""
    rs = parse_rollouts(RAW)
    assert [r["step"] for r in rs] == [1, 2]


def test_trailing_rollouts_with_no_following_step_line_are_dropped():
    """A run killed mid-batch leaves rollouts that were never trained on. They have no
    step and must not be silently folded into the last one."""
    rs = parse_rollouts(RAW + "(main_task pid=3491) User: Using the numbers [1, 2], "
                              "create an equation that equals 3.\n"
                              "(main_task pid=3491) <think> 1 + 2 </think>\n")
    assert len(rs) == 2
    assert [r["step"] for r in rs] == [1, 2]


def test_a_response_keeps_all_of_its_think_blocks():
    """The multi-block structure IS the phenomenon under test -- one sample emitting
    several <think> turns is what the paper counts as conversational. Splitting on
    <think> rather than on the prompt boundary would turn one conversational trace into
    several monologic ones and destroy the effect being measured."""
    raw = ("(main_task pid=1) User: Using the numbers [50, 57], create an equation.\n"
           "(main_task pid=1) <think>First, let's start with 18.</think>\n"
           "(main_task pid=1) <think>Now that didn't work, let's try another way.</think>\n"
           "(main_task pid=1) <think>Time to switch things around.</think>\n"
           "(main_task pid=1) step:7 - critic/score/mean:0.1\n")
    rs = parse_rollouts(raw)
    assert len(rs) == 1
    assert rs[0]["response"].count("<think>") == 3


def test_question_answering_counts_posed_and_answered_questions():
    qa = question_answering("Are either of the numbers divisible by 4? Yes, 80 is.")
    assert qa > 0
    assert question_answering("We can get something close to 51.") == 0


def test_conflict_of_perspectives_needs_a_reversal_not_just_a_conjunction():
    """'but' as a plain conjunction is not a conflict of perspectives. Counting every
    'but' would make the measure track sentence length instead of disagreement."""
    assert conflict_of_perspectives("Wait, no, that's wrong -- let's reconsider.") > 0
    assert conflict_of_perspectives("Take 80 but not 31, and add 2.") == 0


def test_behaviours_per_step_is_a_rate_not_a_count():
    """Response length grows under RL. A per-trace COUNT would rise with length alone;
    the emergence claim needs a rate per unit of text."""
    rs = [{"step": 1, "prompt": "p", "response": "Wait, no, that's wrong."},
          {"step": 2, "prompt": "p", "response": "Wait, no, that's wrong. " + "filler " * 100}]
    out = behaviours_per_step(rs)
    assert out[1]["conflict_rate"] > out[2]["conflict_rate"]


def test_behaviours_per_step_reports_n_so_thin_bins_are_visible():
    rs = [{"step": 1, "prompt": "p", "response": "a"},
          {"step": 1, "prompt": "p", "response": "b"},
          {"step": 5, "prompt": "p", "response": "c"}]
    out = behaviours_per_step(rs)
    assert out[1]["n"] == 2 and out[5]["n"] == 1


def test_empty_log_yields_no_rollouts_rather_than_crashing():
    assert parse_rollouts("nothing here\n") == []


# --- the other two behaviours the paper names -------------------------------


def test_all_four_paper_behaviours_have_a_proxy():
    """The paper names four. Measuring two and reporting on Fig. 4b would be reporting
    on half their instrument."""
    from analysis.emergence import BEHAVIOUR_FNS
    from rl.judge import BEHAVIOURS
    assert set(BEHAVIOUR_FNS) == set(BEHAVIOURS)


def test_perspective_shift_counts_transitions_not_segments():
    """N segments means N-1 transitions. Reporting the segment count would give a
    single-voice trace a shift count of one."""
    from analysis.emergence import perspective_shift
    assert perspective_shift("A single steady line of reasoning that never turns.") == 0


def test_perspective_shift_rises_with_added_turns():
    from analysis.emergence import perspective_shift
    one = ("Let me try adding the two largest numbers together first and see how close "
           "that gets us to the target value we need.")
    two = one + (" However, that overshoots the target by quite a lot, so a different "
                 "combination is needed here instead.")
    assert perspective_shift(two) > perspective_shift(one)


def test_reconciliation_needs_an_integration_not_just_a_conclusion():
    from analysis.emergence import reconciliation
    assert reconciliation("Combining both approaches, the answer is 51.") > 0
    assert reconciliation("The answer is 51.") == 0


def test_behaviours_per_step_reports_a_rate_for_each_of_the_four():
    rs = [{"step": 1, "prompt": "p", "response": "Wait, no. Combining both, it works."}]
    out = behaviours_per_step(rs)[1]
    for b in ("question_answering", "perspective_shift",
              "conflict_of_perspectives", "reconciliation"):
        assert f"{b}_rate" in out, f"no rate reported for {b}"


# --- instrument health -------------------------------------------------------


def test_pattern_dominance_flags_a_single_template():
    """A surface-marker measure can be driven entirely by one repeated string. On the
    claimA log, 99.8% of late 'conflict of perspectives' matches were the single
    parenthetical "doesn't equal" inside a numbered enumeration -- which scores as a 12x
    rise in dialogic behaviour while every genuinely dialogic marker went to zero.
    Any reported marker rate needs this alongside it."""
    from analysis.emergence import pattern_dominance
    d = pattern_dominance(["x doesn't equal 1. y doesn't equal 2. z doesn't equal 3."],
                          "conflict_of_perspectives")
    assert d["top_share"] > 0.9
    assert d["top"] == "doesn't equal"


def test_pattern_dominance_is_low_for_varied_language():
    from analysis.emergence import pattern_dominance
    d = pattern_dominance(["Wait. However, nope. Alternatively, let's try another way."],
                          "conflict_of_perspectives")
    assert d["top_share"] < 0.5


def test_pattern_dominance_on_no_matches_does_not_divide_by_zero():
    from analysis.emergence import pattern_dominance
    d = pattern_dominance(["plain arithmetic with no markers at all"],
                          "conflict_of_perspectives")
    assert d["total"] == 0 and d["top_share"] == 0.0


# --- the scorer's debug output is not part of the response -------------------


def test_response_stops_at_the_scorer_debug_block():
    """verl's Countdown scorer echoes each graded rollout back to stdout as
    `Target: ... / Extracted equation: ... / Solution string: <the whole prompt>`, and
    that block lands AFTER the response and BEFORE the next prompt. Swallowing it put
    ~45 junk words into 78% of parsed traces -- diluting every rate -- and fed an LLM
    judge the literal phrase "A conversation between User and Assistant", which biases
    a persona count in the one direction we are trying to rule out."""
    raw = ("(main_task pid=1) User: Using the numbers [7, 20, 4], create an equation.\n"
           "(main_task pid=1) Assistant: <think> 7 * 20 - 4 </think>\n"
           "(main_task pid=1) --------------------------------\n"
           "(main_task pid=1) Target: 18 | Numbers: [50 57 73 52]\n"
           "(main_task pid=1) Extracted equation: None\n"
           "(main_task pid=1) Solution string: A conversation between User and Assistant.\n"
           "(main_task pid=1) step:4 - critic/score/mean:0.1\n")
    r = parse_rollouts(raw)[0]
    assert "7 * 20 - 4" in r["response"]
    for junk in ("Target:", "Extracted equation:", "Solution string:",
                 "A conversation between User and Assistant"):
        assert junk not in r["response"], f"scorer debug leaked into the response: {junk}"


def test_the_separator_rule_does_not_truncate_a_real_response():
    raw = ("(main_task pid=1) User: p\n"
           "(main_task pid=1) Assistant: <think> first I target 18, then extract it </think>\n"
           "(main_task pid=1) step:1 - critic/score/mean:0.1\n")
    r = parse_rollouts(raw)[0]
    assert "first I target 18, then extract it" in r["response"], \
        "a response that merely contains the words must not be cut"
