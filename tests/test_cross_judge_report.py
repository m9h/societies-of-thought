"""Markdown rendering of a cross_judge result — small, but it is what gets quoted."""
from __future__ import annotations

import math

from analysis.cross_judge_report import render


def _v(b, k, n=1):
    return {"idx": 0, "step": 0, "bin": b, "bin_lo": b * 10, "bin_hi": b * 10 + 9, "source": "train",
            "words": 100, "question_answering": k, "perspective_shift": k,
            "conflict_of_perspectives": k, "reconciliation": 0, "n_personas": n}


def test_render_has_curve_and_agreement_tables():
    out = {"n": 4, "judges": ["a", "b"], "primary": "a", "source": "train",
           "prompt_version": "paper-v1", "failures": {"a": 0, "b": 1},
           "verdicts": {"a": [_v(0, 1), _v(0, 3, 2), _v(1, 0), _v(1, 0)],
                        "b": [_v(0, 1), _v(0, 1), _v(1, 0)]},
           "curves": {"a": [{"bin": 0, "bin_lo": 0, "bin_hi": 9, "n": 2, "question_answering": 2.0,
                             "perspective_shift": 2.0, "conflict_of_perspectives": 2.0,
                             "reconciliation": 0.0, "n_personas": 1.5},
                            {"bin": 1, "bin_lo": 10, "bin_hi": 19, "n": 2, "question_answering": 0.0,
                             "perspective_shift": 0.0, "conflict_of_perspectives": 0.0,
                             "reconciliation": 0.0, "n_personas": 1.0}],
                      "b": []},
           "agreement": [{"pair": ["a", "b"], "field": "n_personas", "n": 3, "exact": 0.67,
                          "kappa": float("nan"), "kappa_quadratic": float("nan"), "icc31": 0.1,
                          "spearman": float("nan"), "mean_a": 1.33, "mean_b": 1.0}]}
    md = render(out)
    assert "| 0–9 |" in md and "1.50" in md          # curve row with persona mean
    assert ">1" in md                                  # share of traces with >1 persona
    assert "a vs b" in md and "n/a" in md              # agreement row; NaN rendered as n/a
    assert "failures" in md.lower() and "b: 1" in md
