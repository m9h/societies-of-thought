"""Within-problem diversity comparison — tests written BEFORE the implementation.

The GPQA result (matched +0.0133, p=0.0003) compares correct traces to incorrect traces of
similar LENGTH, but from DIFFERENT problems. Problem difficulty is therefore uncontrolled:
if hard problems produce both more errors and different trace structure, that alone could
generate the effect.

Sampling the same problem k times and comparing its correct traces against its own
incorrect traces removes difficulty entirely -- the problem is literally held fixed. Only
problems that yield BOTH outcomes are informative, and that selection must be reported,
since it keeps exactly the problems of middling difficulty.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.within_problem import pair_within_problem


def _rec(pid, correct, hse_norm, words=1000):
    return {"pid": pid, "correct": correct, "hse_norm": hse_norm, "words": words,
            "source": "gpqa"}


def test_only_problems_with_both_outcomes_are_used():
    rows = ([_rec("p1", True, 0.30), _rec("p1", False, 0.20)]      # informative
            + [_rec("p2", True, 0.30), _rec("p2", True, 0.31)]     # all correct
            + [_rec("p3", False, 0.20), _rec("p3", False, 0.21)])  # all incorrect
    res = pair_within_problem(rows, "hse_norm")
    assert res["n_problems_used"] == 1
    assert res["n_problems_all_correct"] == 1
    assert res["n_problems_all_incorrect"] == 1


def test_difference_is_mean_correct_minus_mean_incorrect_within_problem():
    """Two problems, not one: a standard error needs n >= 2, so the function refuses to
    estimate from a single problem no matter what min_problems says."""
    rows = [_rec("p1", True, 0.40), _rec("p1", True, 0.40),
            _rec("p1", False, 0.20), _rec("p1", False, 0.20),
            _rec("p2", True, 0.50), _rec("p2", False, 0.30)]
    res = pair_within_problem(rows, "hse_norm", min_problems=1)
    assert res["n_problems_used"] == 2
    assert abs(res["difference"] - 0.20) < 1e-9


def test_single_problem_cannot_yield_a_standard_error():
    rows = [_rec("p1", True, 0.40), _rec("p1", False, 0.20)]
    res = pair_within_problem(rows, "hse_norm", min_problems=1)
    assert res["n_problems_used"] == 1
    assert "difference" not in res


def test_each_problem_contributes_once_regardless_of_sample_count():
    """A problem sampled 20 times must not outweigh one sampled twice."""
    heavy = ([_rec("big", True, 1.0)] * 19) + [_rec("big", False, 0.0)]
    light = [_rec("small", True, 0.0), _rec("small", False, 1.0)]
    res = pair_within_problem(heavy + light, "hse_norm", min_problems=1)
    assert res["n_problems_used"] == 2
    assert abs(res["difference"] - 0.0) < 1e-9, "per-problem means must be averaged evenly"


def test_reports_ci_and_significance():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        rows.append(_rec(f"p{i}", True, float(rng.normal(0.32, 0.03))))
        rows.append(_rec(f"p{i}", False, float(rng.normal(0.28, 0.03))))
    res = pair_within_problem(rows, "hse_norm")
    assert res["ci_excludes_zero"] is True
    assert res["difference"] > 0.02
    assert res["ci95_low"] < res["difference"] < res["ci95_high"]


def test_null_is_reported_as_null():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(200):
        rows.append(_rec(f"p{i}", True, float(rng.normal(0.30, 0.03))))
        rows.append(_rec(f"p{i}", False, float(rng.normal(0.30, 0.03))))
    res = pair_within_problem(rows, "hse_norm")
    assert res["ci_excludes_zero"] is False


def test_length_balance_within_pairs_is_reported():
    """Within a problem, correct and incorrect traces can still differ in length; the
    achieved balance must be visible rather than assumed away."""
    rows = [_rec("p1", True, 0.3, words=500), _rec("p1", False, 0.2, words=2500)]
    rows += [_rec("p2", True, 0.3, words=600), _rec("p2", False, 0.2, words=2400)]
    res = pair_within_problem(rows, "hse_norm", min_problems=1)
    assert res["mean_words_correct"] < res["mean_words_incorrect"]


def test_too_few_informative_problems_refuses_to_estimate():
    rows = [_rec("p1", True, 0.3), _rec("p1", False, 0.2)]
    res = pair_within_problem(rows, "hse_norm", min_problems=30)
    assert res["n_problems_used"] == 1
    assert "difference" not in res or res.get("note")


def test_empty_input_does_not_crash():
    res = pair_within_problem([], "hse_norm")
    assert res["n_problems_used"] == 0
