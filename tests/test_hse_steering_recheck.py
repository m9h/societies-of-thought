"""Tests for the steering re-audit.

The original steering analysis dropped traces with too few segments. Steering changes how
many perspective shifts a trace has, so that filter bites unequally across conditions --
the same asymmetry that inverted the QwQ result. These pin the corrected behaviour.
"""
from __future__ import annotations

import numpy as np

from analysis.hse_steering_recheck import matched_vs_baseline, per_trace, summarise


class _FakeEncoder:
    def encode(self, segs, **kw):
        rng = np.random.default_rng(len(segs))
        E = rng.normal(size=(len(segs), 8))
        return E / np.linalg.norm(E, axis=1, keepdims=True)


SHIFTY = ("We start down the obvious path and it seems fine for now. "
          "But actually the parity argument rules that out completely. "
          "Wait, reconsider the second constraint before going further. "
          "However the remaining branch forces a different value here.")
PLAIN = "The answer follows directly from the definition. " * 8


def _rows():
    """alpha=0 traces are single-voice; steered traces have shifts. The original filter
    would drop the baseline condition almost entirely."""
    return ([{"feature": -1, "alpha": 0.0, "trace": PLAIN, "correct": True}] * 40
            + [{"feature": 30939, "alpha": 1.0, "trace": SHIFTY, "correct": False}] * 40)


def test_single_voice_traces_are_kept_and_scored_zero():
    recs, drops = per_trace(_rows(), None, degenerate="zero", encoder=_FakeEncoder())
    assert len(recs) == 80, "no condition may lose traces to the filter"
    base = [r for r in recs if r["alpha"] == 0.0]
    assert len(base) == 40
    assert all(r["single_voice"] and r["hse_norm"] == 0.0 for r in base)


def test_drop_mode_loses_a_whole_condition():
    """Demonstrates the original bug: the filter can erase the baseline condition."""
    recs, _ = per_trace(_rows(), None, degenerate="drop", encoder=_FakeEncoder())
    assert not [r for r in recs if r["alpha"] == 0.0]


def test_summary_reports_single_voice_counts_per_condition():
    recs, _ = per_trace(_rows(), None, degenerate="zero", encoder=_FakeEncoder())
    s = summarise(recs)
    assert s[0.0]["single_voice"] == 40
    assert s[1.0]["single_voice"] == 0
    assert s[0.0]["n"] == s[1.0]["n"] == 40


def test_matched_comparison_returns_baseline_minus_steered():
    """Sign convention must be explicit: positive = steering LOWERS diversity."""
    recs = ([{"alpha": 0.0, "correct": True, "words": 500 + i, "hse_norm": 0.30,
              "n_segments": 9, "single_voice": False} for i in range(300)]
            + [{"alpha": 1.0, "correct": False, "words": 500 + i, "hse_norm": 0.20,
                "n_segments": 9, "single_voice": False} for i in range(300)])
    m = matched_vs_baseline(recs, 1.0, caliper=0.02)
    assert m["n_pairs"] > 100
    assert m["difference"] > 0.05, "baseline higher than steered must read positive"
