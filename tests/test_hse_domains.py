"""Tests for the multi-domain QwQ diversity analysis — written BEFORE the module.

Every requirement here encodes a bug that has already cost this project a result:

  * truncated traces are max-length and always-wrong, so including them manufactures the
    exact length/correctness confound that overturned the QwQ and steering findings
  * length must be controlled by construction, not bolted on after a headline is written
  * single-voice traces score zero (the paper's rule), never dropped
  * per-domain reporting, because a pooled number over BBH+GPQA+MuSR+MMLU-Pro hides that
    those tasks have wildly different accuracy and trace length
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.hse_domains import analyse, load_traces


class _FakeEncoder:
    def encode(self, segs, **kw):
        rng = np.random.default_rng(len(segs))
        E = rng.normal(size=(len(segs), 8))
        return E / np.linalg.norm(E, axis=1, keepdims=True)


SHIFTY = ("We begin with the obvious reading of the clause and it seems fine. "
          "But actually the second constraint rules that out completely here. "
          "Wait, reconsider the ordering before committing to any conclusion. "
          "However the remaining option forces a different answer entirely now.")
PLAIN = "The answer follows directly from the stated definition of the term. " * 6


def _rec(pid, source, correct, truncated, trace):
    return {"pid": pid, "source": source, "subtask": source, "correct": correct,
            "truncated": truncated, "response": trace}


# --- truncation ---------------------------------------------------------------


def test_truncated_traces_are_excluded_by_default():
    """A truncated trace never reached an answer, so grading it 'incorrect' is a
    measurement error, and it is maximum-length by construction."""
    rows = ([_rec(f"a{i}", "bbh", True, False, SHIFTY) for i in range(40)]
            + [_rec(f"b{i}", "bbh", False, True, SHIFTY) for i in range(40)])
    kept, dropped = load_traces(rows)
    assert len(kept) == 40
    assert dropped["truncated"] == 40
    assert all(not r["truncated"] for r in kept)


def test_including_truncated_is_possible_but_must_be_explicit():
    rows = ([_rec(f"a{i}", "bbh", True, False, SHIFTY) for i in range(40)]
            + [_rec(f"b{i}", "bbh", False, True, SHIFTY) for i in range(40)])
    kept, _ = load_traces(rows, exclude_truncated=False)
    assert len(kept) == 80


# --- the paper's single-voice convention --------------------------------------


def test_single_voice_traces_are_scored_zero_not_dropped():
    rows = ([_rec(f"a{i}", "bbh", True, False, PLAIN) for i in range(40)]
            + [_rec(f"b{i}", "bbh", False, False, SHIFTY) for i in range(40)])
    res = analyse(rows, encoder=_FakeEncoder())
    assert res["n_measured"] == 80, "no trace may be lost to the segment filter"
    assert res["n_single_voice"] == 40


# --- length control is built in, not optional ---------------------------------


def test_result_reports_matched_estimate_alongside_unadjusted():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        rows.append(_rec(f"c{i}", "bbh", True, False, SHIFTY * int(rng.integers(1, 4))))
    for i in range(300):
        rows.append(_rec(f"w{i}", "bbh", False, False, SHIFTY * int(rng.integers(1, 4))))
    res = analyse(rows, encoder=_FakeEncoder())
    assert "unadjusted" in res and "matched" in res
    assert "difference" in res["unadjusted"]
    assert "n_pairs" in res["matched"]


def test_pure_length_confound_shrinks_under_matching():
    """Diversity a function of length only; correctness merely SHIFTS the length
    distribution while keeping the ranges overlapping, so pairs can be formed."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(500):                      # correct: skewed short, range 1-5
        rows.append(_rec(f"c{i}", "bbh", True, False,
                         SHIFTY * int(min(5, 1 + rng.geometric(0.55)))))
    for i in range(500):                      # incorrect: skewed long, same range
        rows.append(_rec(f"w{i}", "bbh", False, False,
                         SHIFTY * int(max(1, 6 - rng.geometric(0.55)))))
    res = analyse(rows, encoder=_FakeEncoder())
    assert res["matched"]["n_pairs"] >= 30, "overlapping ranges must yield pairs"
    assert abs(res["matched"]["difference"]) < abs(res["unadjusted"]["difference"])


def test_non_overlapping_lengths_refuse_to_estimate():
    """When correct and incorrect traces share no length range, no comparison at matched
    length exists. Refusing is the correct answer; inventing one is not."""
    rows = ([_rec(f"c{i}", "bbh", True, False, SHIFTY) for i in range(200)]
            + [_rec(f"w{i}", "bbh", False, False, SHIFTY * 6) for i in range(200)])
    res = analyse(rows, encoder=_FakeEncoder())
    assert res["unadjusted"] is not None
    assert res["matched"].get("n_pairs", 0) < 30
    assert res["by_domain"]["bbh"]["matched"] is None


# --- per-domain reporting -----------------------------------------------------


def test_per_domain_breakdown_is_reported():
    """BBH, GPQA, MuSR and MMLU-Pro differ hugely in accuracy and trace length; a pooled
    number over all of them is not interpretable on its own."""
    rows = []
    for src in ("bbh", "gpqa", "musr"):
        rows += [_rec(f"{src}c{i}", src, True, False, SHIFTY) for i in range(30)]
        rows += [_rec(f"{src}w{i}", src, False, False, SHIFTY * 2) for i in range(30)]
    res = analyse(rows, encoder=_FakeEncoder())
    assert set(res["by_domain"]) == {"bbh", "gpqa", "musr"}
    for v in res["by_domain"].values():
        assert "n_correct" in v and "n_incorrect" in v and "unadjusted" in v


def test_domain_with_one_class_only_is_reported_but_not_estimated():
    rows = [_rec(f"g{i}", "gpqa", False, False, SHIFTY) for i in range(40)]
    rows += [_rec(f"b{i}", "bbh", True, False, SHIFTY) for i in range(40)]
    rows += [_rec(f"b{i}w", "bbh", False, False, SHIFTY * 2) for i in range(40)]
    res = analyse(rows, encoder=_FakeEncoder())
    assert res["by_domain"]["gpqa"]["unadjusted"] is None
    assert res["by_domain"]["bbh"]["unadjusted"] is not None


def test_empty_input_does_not_crash():
    res = analyse([], encoder=_FakeEncoder())
    assert res["n_measured"] == 0
    with pytest.raises(KeyError):
        _ = res["by_domain"]["bbh"]


def test_per_trace_records_carry_pid_for_within_problem_grouping():
    """analysis/within_problem.py groups traces by pid to hold the problem fixed. Without
    pid on the measured records the two modules cannot compose, and the within-problem
    control -- the strongest test of the GPQA effect -- is unrunnable."""
    rows = ([_rec("p1", "gpqa", True, False, SHIFTY),
             _rec("p1", "gpqa", False, False, SHIFTY * 2),
             _rec("p2", "gpqa", True, False, SHIFTY)])
    res = analyse(rows, encoder=_FakeEncoder())
    assert "per_trace" in res, "measured records must be returned for downstream analysis"
    assert {r["pid"] for r in res["per_trace"]} == {"p1", "p2"}


def test_per_trace_records_carry_sample_index_when_present():
    rows = [{"pid": "p1", "sample": 0, "source": "gpqa", "subtask": "gpqa",
             "correct": True, "truncated": False, "response": SHIFTY},
            {"pid": "p1", "sample": 1, "source": "gpqa", "subtask": "gpqa",
             "correct": False, "truncated": False, "response": SHIFTY * 2}]
    res = analyse(rows, encoder=_FakeEncoder())
    assert sorted(r["sample"] for r in res["per_trace"]) == [0, 1]
