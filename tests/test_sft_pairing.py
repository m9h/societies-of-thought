"""The arm-matching logic — the single point where this experiment can be silently voided.

Claim B is only meaningful if the dialogue and monologue arms solve the IDENTICAL
problems with the IDENTICAL correct answers, so that the sole difference between them is
the FORM of the reasoning. Every way of getting that wrong produces a confident,
publishable-looking, meaningless number:

  * filter dialogues for correctness but not monologues -> you are comparing "correct
    reasoning" against "any reasoning"
  * let the arms cover different problem sets -> you are comparing different tasks
  * accept an empty dataset -> two empty sets ARE equal, so the matching assertion passes
    vacuously and reports success on zero data (this actually happened)

These tests were written before the code they test. `pair_arms` does not exist yet.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from rl.generate_sft import pair_arms

P1 = {"pid": 1, "nums": [25, 30, 3, 4], "target": 32}
P2 = {"pid": 2, "nums": [7, 7, 3], "target": 46}
P3 = {"pid": 3, "nums": [1, 2, 3], "target": 6}

GOOD_D = "<think1> x </think1><group_consensus> (30 - 25 + 3) * 4 </group_consensus>"
GOOD_M = "<think> y </think><answer> (30 - 25 + 3) * 4 </answer>"


def test_keeps_only_problems_both_arms_solved():
    """A problem survives only if BOTH arms produced a correct solution for it."""
    problems = [P1, P2, P3]
    dialogues = [GOOD_D, None, GOOD_D]   # arm failed on P2
    monologues = [GOOD_M, GOOD_M, None]  # arm failed on P3

    paired = pair_arms(problems, dialogues, monologues)

    assert [p["pid"] for p, _, _ in paired] == [1], (
        "only P1 was solved by both arms; keeping P2 or P3 would mean one arm trained on "
        "a problem the other never saw"
    )


def test_arms_end_up_with_identical_problem_sets():
    # NOTE: each trace must solve ITS OWN problem. Written first with P1's solution
    # reused for P2, this test failed -- correctly. The alignment check is the point.
    problems = [P1, P2, P3]
    d2 = "<group_consensus> 7 * 7 - 3 </group_consensus>"  # solves P2 (46)
    m2 = "<answer> 7 * 7 - 3 </answer>"
    paired = pair_arms(problems, [GOOD_D, d2, None], [GOOD_M, m2, None])
    d_pids = {p["pid"] for p, d, _ in paired if d}
    m_pids = {p["pid"] for p, _, m in paired if m}
    assert d_pids == m_pids == {1, 2}


def test_empty_result_is_a_failure_not_a_pass():
    """THE VACUOUS-GATE BUG. Two empty sets are equal, so a naive equality assertion
    reports success on zero data. It did. `pair_arms` must refuse to return an empty
    pairing silently."""
    with pytest.raises(ValueError, match="(?i)no problem.*both arms|empty"):
        pair_arms([P1, P2], [None, None], [None, None])


def test_a_correct_looking_but_wrong_solution_is_rejected():
    """Verification is by arithmetic, never by the model's say-so."""
    wrong_d = "<group_consensus> (30 - 25) * 4 </group_consensus>"  # = 20, not 32
    with pytest.raises(ValueError):
        pair_arms([P1], [wrong_d], [GOOD_M])


def test_pairing_preserves_alignment_between_problem_and_its_traces():
    """An off-by-one here would train each arm on solutions to the WRONG problems, while
    every count and every assertion still looked perfectly healthy."""
    good_d2 = "<group_consensus> 7 * 7 - 3 </group_consensus>"
    good_m2 = "<answer> 7 * 7 - 3 </answer>"
    paired = pair_arms([P1, P2], [GOOD_D, good_d2], [GOOD_M, good_m2])
    for problem, d, m in paired:
        assert str(problem["target"]) in ("32", "46")
        # each trace must actually solve ITS OWN problem
        from rl.reward import accuracy_reward
        assert accuracy_reward(d, problem["target"], problem["nums"]) == 1.0
        assert accuracy_reward(m, problem["target"], problem["nums"]) == 1.0
