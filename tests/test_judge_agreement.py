"""Chance-corrected agreement between judges — tests BEFORE implementation.

Norman et al. report 33-41 point kappa deflation between exact match and Cohen's kappa
across 541k judgments. Exact match on a near-constant label (our n_personas is 1 on
every trace) is 100% for any two judges that both say 1, which says nothing. So the
report must carry both, and must say "undefined" rather than 1.0 when there is no
variance to correct for.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.judge_agreement import (
    bin_curves,
    cohen_kappa,
    icc31,
    pairwise_agreement,
)


def test_kappa_identical_is_one():
    assert cohen_kappa([0, 1, 2, 1, 0], [0, 1, 2, 1, 0]) == pytest.approx(1.0)


def test_kappa_textbook_two_by_two():
    # confusion [[20, 5], [10, 15]]: po = .7, pe = .5 -> kappa = .4
    a = [0] * 25 + [1] * 25
    b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_chance_level_is_zero():
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_kappa_constant_raters_is_nan_not_one():
    """Both judges say n_personas=1 on every trace. Exact match is 1.0; kappa is undefined."""
    k = cohen_kappa([1] * 30, [1] * 30)
    assert math.isnan(k)


def test_quadratic_kappa_penalises_distance():
    truth = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    near = [1, 2, 3, 4, 3, 1, 0, 1, 2, 3]
    far = [4, 4, 4, 0, 0, 4, 4, 0, 0, 0]
    kn = cohen_kappa(truth, near, weights="quadratic")
    kf = cohen_kappa(truth, far, weights="quadratic")
    assert kn > kf
    assert cohen_kappa(truth, truth, weights="quadratic") == pytest.approx(1.0)


def test_icc31_shrout_fleiss_example():
    # Shrout & Fleiss (1979) Table 2: ICC(3,1) = 0.71
    r = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                  [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    assert icc31(r) == pytest.approx(0.7148, abs=1e-3)


def test_icc31_constant_is_nan():
    assert math.isnan(icc31(np.ones((20, 2))))


def _v(**kw):
    base = {"question_answering": 0, "perspective_shift": 0,
            "conflict_of_perspectives": 0, "reconciliation": 0, "n_personas": 1}
    base.update(kw)
    return base


def test_pairwise_agreement_reports_both_exact_and_chance_corrected():
    A = [_v(question_answering=i % 3) for i in range(30)]
    B = [_v(question_answering=i % 3) for i in range(30)]
    rows = pairwise_agreement({"a": A, "b": B})
    qa = next(r for r in rows if r["field"] == "question_answering")
    assert qa["pair"] == ("a", "b")
    assert qa["n"] == 30
    assert qa["exact"] == pytest.approx(1.0)
    assert qa["kappa"] == pytest.approx(1.0)
    assert qa["icc31"] == pytest.approx(1.0)
    np_ = next(r for r in rows if r["field"] == "n_personas")
    assert np_["exact"] == pytest.approx(1.0)
    assert math.isnan(np_["kappa"])          # constant: undefined, NOT 1.0
    assert "mean_a" in qa and "mean_b" in qa


def test_pairwise_agreement_rejects_misaligned_lengths():
    with pytest.raises(ValueError, match="aligned"):
        pairwise_agreement({"a": [_v()] * 3, "b": [_v()] * 4})


def test_bin_curves_mean_per_bin_per_judge():
    V = {"a": [dict(_v(conflict_of_perspectives=1), bin=0), dict(_v(), bin=0),
               dict(_v(), bin=1), dict(_v(), bin=1)]}
    c = bin_curves(V)
    assert c["a"][0]["conflict_of_perspectives"] == pytest.approx(0.5)
    assert c["a"][1]["conflict_of_perspectives"] == pytest.approx(0.0)
    assert c["a"][0]["n"] == 2
