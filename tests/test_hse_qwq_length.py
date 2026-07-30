"""Tests for the length-stratified comparison.

This analysis exists to tell "diversity anti-predicts correctness" apart from "long traces
are both more diverse and more often wrong". If it cannot detect a pure length confound it
is worse than useless, so that is the first thing pinned.
"""
from __future__ import annotations

import numpy as np

from analysis.hse_qwq_length import match_on_length, stratify


def _rec(correct, words, hse_norm):
    return {"correct": correct, "words": words, "hse_norm": hse_norm}


def test_pure_length_confound_is_largely_removed():
    """Data where diversity depends ONLY on length; correctness merely shifts the length
    distribution. Matching must SHRINK the apparent effect toward zero.

    Not "to exactly zero": with a deterministic length gradient, any residual imbalance
    inside the caliper is a real bias, and at large n it is detectable. The meaningful
    criterion is the shrinkage ratio, which is what the report prints.
    """
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(3000):                       # correct: short traces
        w = float(rng.uniform(200, 1200))
        recs.append(_rec(True, w, 0.2 + w / 10000 + rng.normal(0, 0.01)))
    for _ in range(3000):                       # incorrect: long traces
        w = float(rng.uniform(600, 2500))
        recs.append(_rec(False, w, 0.2 + w / 10000 + rng.normal(0, 0.01)))

    unadj = (np.mean([r["hse_norm"] for r in recs if r["correct"]])
             - np.mean([r["hse_norm"] for r in recs if not r["correct"]]))
    m = match_on_length(recs, "hse_norm", caliper=0.02)
    assert m["n_pairs"] >= 200, "matching must find pairs when lengths overlap"
    # the whole confound is length, so matching must remove most of the apparent effect
    assert abs(m["difference"]) < 0.2 * abs(unadj), (
        f"matched {m['difference']:+.4f} vs unadjusted {unadj:+.4f}: matching failed to "
        "remove a pure length confound"
    )
    res = stratify(recs, "hse_norm", n_bins=8, min_per_cell=20)
    assert res["pooled"]["n_usable_bins"] >= 3


def test_genuine_effect_survives_stratification():
    """Diversity depends on correctness independently of length -> must survive."""
    rng = np.random.default_rng(1)
    recs = []
    for _ in range(3000):
        w = float(rng.uniform(200, 2500))
        recs.append(_rec(True, w, 0.20 + w / 10000 + rng.normal(0, 0.01)))
    for _ in range(3000):
        w = float(rng.uniform(200, 2500))
        recs.append(_rec(False, w, 0.26 + w / 10000 + rng.normal(0, 0.01)))

    unadj = (np.mean([r["hse_norm"] for r in recs if r["correct"]])
             - np.mean([r["hse_norm"] for r in recs if not r["correct"]]))
    m = match_on_length(recs, "hse_norm", caliper=0.02)
    # a genuine effect must survive matching at close to full size
    assert m["ci_excludes_zero"] and m["difference"] < -0.03
    assert abs(m["difference"]) > 0.7 * abs(unadj)


def test_bins_without_both_classes_are_excluded_not_silently_pooled():
    """Where length and outcome are collinear no comparison exists; say so."""
    recs = ([_rec(True, 100 + i, 0.3) for i in range(200)]
            + [_rec(False, 5000 + i, 0.3) for i in range(200)])
    res = stratify(recs, "hse_norm", n_bins=8, min_per_cell=30)
    assert all(not b["usable"] for b in res["bins"])
    assert "pooled" not in res


def test_bin_counts_are_reported_for_every_bin():
    recs = ([_rec(True, 100 + i * 3, 0.3) for i in range(400)]
            + [_rec(False, 200 + i * 3, 0.32) for i in range(400)])
    res = stratify(recs, "hse_norm", n_bins=5, min_per_cell=10)
    assert len(res["bins"]) == 5
    assert all("n_correct" in b and "n_incorrect" in b for b in res["bins"])


def test_matching_reports_the_achieved_length_balance():
    """A matched estimate is only credible if the achieved balance is shown."""
    rng = np.random.default_rng(3)
    recs = ([_rec(True, float(rng.uniform(300, 2000)), 0.3) for _ in range(800)]
            + [_rec(False, float(rng.uniform(300, 2000)), 0.3) for _ in range(800)])
    m = match_on_length(recs, "hse_norm", caliper=0.02)
    assert m["n_pairs"] > 100
    assert abs(m["mean_words_correct"] - m["mean_words_incorrect"]) < 30
    assert m["max_relative_length_gap"] <= 0.05


def test_matching_refuses_when_lengths_do_not_overlap():
    recs = ([_rec(True, 100.0 + i, 0.3) for i in range(200)]
            + [_rec(False, 9000.0 + i, 0.3) for i in range(200)])
    m = match_on_length(recs, "hse_norm", caliper=0.02)
    assert m["n_pairs"] == 0 or "note" in m
