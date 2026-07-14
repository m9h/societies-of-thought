"""Grading tests.

Written after a grader bug destroyed a whole run: the Countdown grader accepted only
<answer> tags, but R1-distill answers in LaTeX \\boxed{}. That marked 74% of baseline
traces "unparseable" -- correct ones included -- and put the baseline at 5.5% against
the paper's 27.1%. It looked exactly like "the paper does not reproduce."

The invariant that keeps this honest:

    Widening the grader may only change what counts as FOUND, never what counts as
    RIGHT. An answer we cannot find is WRONG (never dropped), and an answer we can
    find is judged on its arithmetic, never on the model's own claim about it.

Every extraction format the models actually emit gets a regression test here.
"""

from __future__ import annotations

import sys
import types

# grade.py imports extract_boxed from .data, which imports `datasets` at module import.
# Stub it: grading is pure text processing and must be testable without the ML stack.
sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from sot.grade import grade


# --------------------------------------------------------------------------
# Countdown
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "completion, gold, correct, parsed, why",
    [
        # the format the paper's prompt asks for
        ("<answer> (30 - 25 + 3) * 4 </answer>", "32|25,30,3,4", True, True,
         "answer tag, the paper's own worked example"),
        ("<answer>75 + 73 - 52 = 96</answer>", "96|75,73,52", True, True,
         "answer tag with trailing = target"),

        # the format the model ACTUALLY emits most of the time
        (r"\[\boxed{\frac{98}{7} + 34 - 27 + 9 = 30}\]", "30|98,7,34,27,9", True, True,
         "latex boxed with \\frac -- the bug that cost a whole run"),
        (r"\boxed{32 = (30 - 25 + 3) \times 4}", "32|25,30,3,4", True, True,
         "target=expr orientation, \\times"),

        # the model's CLAIM is never trusted: 98/7 + (34-27) + 7 = 28, not 30
        (r"\boxed{\frac{98}{7} + (34 - 27) + 7 = 30}", "30|98,7,34,27,7", False, True,
         "model asserts =30 but its arithmetic gives 28; must be WRONG"),

        # constraint violations
        ("<answer>(30 - 25) * 4</answer>", "32|25,30,3,4", False, True,
         "did not use every number"),
        ("<answer>(30 - 25 + 3) * 4 * 1</answer>", "32|25,30,3,4", False, True,
         "invented a number not in the set"),
        ("<answer>25 / (30 - 30)</answer>", "32|25,30,3,4", False, True,
         "division by zero must not crash"),

        # a number legitimately repeated in the input is a multiset, not a set
        ("<answer>7 * 7 - 3</answer>", "46|7,7,3", True, True,
         "duplicate input numbers both used"),

        # not found at all -> WRONG and unparsed, never dropped
        ("I thought about it and gave up.", "32|25,30,3,4", False, False,
         "no answer in any format"),
        ("<answer>__import__('os')</answer>", "32|25,30,3,4", False, False,
         "non-arithmetic must never be evaluated"),
    ],
)
def test_countdown(completion, gold, correct, parsed, why):
    g = grade("countdown", completion, gold)
    assert g.correct is correct, f"correct: {why} (pred={g.pred!r})"
    assert g.parsed is parsed, f"parsed: {why} (pred={g.pred!r})"


# --------------------------------------------------------------------------
# GPQA (multiple choice)
#
# This is the next sweep's grader, and it is the SAME failure class as the Countdown
# bug: it must accept every way a reasoning model actually states a final letter, or
# it will silently score correct answers as unparseable and manufacture a null result.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "completion, gold, correct, parsed, why",
    [
        (r"\boxed{C}", "C", True, True, "bare letter"),
        (r"\boxed{(B)}", "B", True, True, "parenthesised letter"),
        ("<answer>D</answer>", "D", True, True, "answer tag"),
        ("The answer is A", "A", True, True, "prose"),
        (r"\boxed{A}", "C", False, True, "wrong letter is wrong"),
        ("ran out of tokens mid-thought", "C", False, False, "no answer -> wrong+unparsed"),

        # R1-distill emits these constantly. Each was a silent zero before.
        (r"\boxed{\text{C}}", "C", True, True, "latex \\text wrapper"),
        (r"\boxed{\textbf{B}}", "B", True, True, "latex \\textbf wrapper"),
        (r"\boxed{D)}", "D", True, True, "trailing paren"),
        (r"\boxed{\text{(A)}}", "A", True, True, "\\text with parens"),
        ("**Answer:** C", "C", True, True, "markdown bold answer"),
        ("Final Answer: **D**", "D", True, True, "bolded letter"),
        (r"\boxed{C. The planet is denser}", "C", True, True, "letter followed by the option text"),

        # must NOT be fooled by an incidental capital letter in prose
        ("Option B looks plausible, but actually the answer is D", "D", True, True,
         "take the FINAL stated answer, not the first capital letter seen"),
    ],
)
def test_gpqa(completion, gold, correct, parsed, why):
    g = grade("gpqa", completion, gold)
    assert g.correct is correct, f"correct: {why} (pred={g.pred!r})"
    assert g.parsed is parsed, f"parsed: {why} (pred={g.pred!r})"


# --------------------------------------------------------------------------
# MATH
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "completion, gold, correct, why",
    [
        (r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}", True, "dfrac vs frac"),
        (r"\boxed{ 12 }", "12", True, "whitespace"),
        (r"\boxed{\text{blue}}", "blue", True, "text answer"),
        (r"\boxed{7}", "8", False, "wrong is wrong"),
        ("no box at all", "8", False, "unparseable -> wrong"),
    ],
)
def test_math(completion, gold, correct, why):
    assert grade("math_hard", completion, gold).correct is correct, why


def test_unparseable_is_wrong_never_dropped():
    """The load-bearing invariant.

    Steering degrades formatting before it degrades reasoning: a steered model that
    babbles emits no answer at all. If an unparseable trace were dropped instead of
    scored wrong, steering would 'improve' accuracy purely by shrinking the
    denominator -- manufacturing exactly the effect this project is testing for.
    """
    for task, gold in [("countdown", "32|25,30,3,4"), ("gpqa", "C"), ("math_hard", "8")]:
        g = grade(task, "oh wait no wait oh wait no", gold)
        assert g.parsed is False
        assert g.correct is False, f"{task}: unparseable must be WRONG, not dropped"
