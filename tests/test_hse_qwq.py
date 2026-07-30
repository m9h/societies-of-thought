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


# --- the single-voice convention, which decides the sign of the result ---------


def _fake_rows():
    """Two 'correct' traces with no shift cues (single voice) and two 'incorrect' ones
    with several, mirroring QwQ: wrong answers flail and shift more."""
    plain = "x" * 200                                   # no SHIFT cue -> 1 segment
    shifty = ("First we try the obvious route and it looks fine so far. "
              "But actually that cannot be right because the parity is off. "
              "Wait, let me reconsider the whole setup from the beginning again. "
              "However the constraint forces the other branch entirely instead.")
    return ([{"response": plain, "correct": True}] * 2
            + [{"response": shifty, "correct": False}] * 2)


def test_zero_handling_keeps_every_trace():
    """The paper: 'If a reasoning trace contained only a single implicit voice, E = 0.'
    Single-voice traces are a measurement of zero diversity, not a missing value."""
    from analysis.hse_qwq import measure

    recs, drop = measure(_fake_rows(), "sentence-transformers/all-MiniLM-L6-v2",
                         degenerate="zero")
    assert len(recs) == 4, "zero handling must not discard any trace"
    assert drop["handling"] == "zero"
    assert drop["correct_single_voice"] == 2
    zs = [r for r in recs if r["single_voice"]]
    assert len(zs) == 2 and all(r["hse_norm"] == 0.0 for r in zs)


def test_drop_handling_discards_and_is_flagged():
    from analysis.hse_qwq import measure

    recs, drop = measure(_fake_rows(), "sentence-transformers/all-MiniLM-L6-v2",
                         degenerate="drop")
    assert len(recs) < 4
    assert drop["handling"] == "drop"


def test_default_is_the_papers_convention():
    """Guards the exact mistake that inverted this result once."""
    import inspect

    from analysis.hse_qwq import measure

    assert inspect.signature(measure).parameters["degenerate"].default == "zero"
