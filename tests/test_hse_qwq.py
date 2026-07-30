"""Tests for the QwQ diversity analysis.

The instrument itself is tested in test_hse.py; these pin the parts specific to this
experiment -- balanced sampling, the correct/incorrect contrast, and the CI arithmetic --
because the whole point is a difference of means and a wrong SE would invent a finding.
"""
from __future__ import annotations

import numpy as np

from analysis.hse_qwq import report


def _rec(correct, hse_norm, mean_dist=0.2, n_seg=8, words=300):
    return {"correct": correct, "hse_norm": hse_norm, "mean_dist": mean_dist,
            "hse": hse_norm * 3, "n_segments": n_seg, "words": words}


def test_reports_group_sizes():
    recs = [_rec(True, 0.3)] * 5 + [_rec(False, 0.3)] * 7
    s = report(recs)
    assert s["n_correct"] == 5 and s["n_incorrect"] == 7


def test_no_difference_gives_ci_containing_zero():
    rng = np.random.default_rng(0)
    recs = ([_rec(True, float(x)) for x in rng.normal(0.30, 0.05, 400)]
            + [_rec(False, float(x)) for x in rng.normal(0.30, 0.05, 400)])
    s = report(recs)
    m = s["metrics"]["hse_norm"]
    assert not m["ci_excludes_zero"], "a null must not be reported as a difference"
    assert m["ci95_low"] < 0 < m["ci95_high"]


def test_real_difference_is_detected_with_correct_sign():
    rng = np.random.default_rng(1)
    recs = ([_rec(True, float(x)) for x in rng.normal(0.40, 0.05, 400)]
            + [_rec(False, float(x)) for x in rng.normal(0.30, 0.05, 400)])
    s = report(recs)
    m = s["metrics"]["hse_norm"]
    assert m["ci_excludes_zero"]
    assert m["difference"] > 0.08
    assert "more diverse" in s["verdict"]


def test_detects_the_opposite_direction_too():
    """If correct traces are LESS diverse, say so -- do not report an absolute value."""
    rng = np.random.default_rng(2)
    recs = ([_rec(True, float(x)) for x in rng.normal(0.25, 0.05, 400)]
            + [_rec(False, float(x)) for x in rng.normal(0.35, 0.05, 400)])
    s = report(recs)
    assert s["metrics"]["hse_norm"]["difference"] < 0
    assert "LESS diverse" in s["verdict"]


def test_length_is_reported_so_the_confound_is_visible():
    """Incorrect math traces run longer; hse_norm divides out segment count, but the
    raw word count must still be reported so a reader can see the asymmetry."""
    recs = [_rec(True, 0.3, words=200)] * 50 + [_rec(False, 0.3, words=600)] * 50
    s = report(recs)
    assert "words" in s["metrics"]
    assert s["metrics"]["words"]["difference"] < 0
