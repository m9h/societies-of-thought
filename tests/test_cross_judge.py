"""Cross-judge runner — tests BEFORE implementation.

The published Fig. 4 numbers are the 8 x 32 stratified sample of the FINAL log, scored
by one judge. A second judge is only a replication if it scores THE SAME traces, so the
runner must reproduce that sample exactly -- and prove it did, by requiring every primary
verdict to come from cache rather than from a fresh call.
"""
from __future__ import annotations

import json

import pytest

from analysis.cross_judge import run_cross_judge
from rl.judge import PROMPT_VERSION, cache_key


def _sample(n=6):
    return [{"step": i * 10, "bin": i // 3, "bin_lo": 0, "bin_hi": 0,
             "prompt": "p", "response": f"trace {i}"} for i in range(n)]


def _verdict(k=0):
    return {"question_answering": k, "perspective_shift": 0,
            "conflict_of_perspectives": 0, "reconciliation": 0, "n_personas": 1}


def _seed_cache(path, sample, model):
    with open(path, "w") as fh:
        for r in sample:
            fh.write(json.dumps({"key": cache_key(r["response"], model),
                                 "model": model, "prompt_version": PROMPT_VERSION,
                                 "verdict": _verdict(1)}) + "\n")


def test_primary_must_be_fully_cached(tmp_path):
    cache = tmp_path / "c.jsonl"
    sample = _sample()
    _seed_cache(cache, sample[:-1], "primary")          # one trace missing

    def never(prompt):
        raise AssertionError("primary judge must not be called")

    with pytest.raises(RuntimeError, match="not in cache"):
        run_cross_judge(sample, {"primary": never, "second": lambda p: json.dumps(_verdict())},
                        cache=cache, primary="primary")


def test_second_judge_is_called_and_aligned(tmp_path):
    cache = tmp_path / "c.jsonl"
    sample = _sample()
    _seed_cache(cache, sample, "primary")
    calls = []

    def second(prompt):
        calls.append(prompt)
        return json.dumps(_verdict(2))

    out = run_cross_judge(sample, {"primary": None, "second": second},
                          cache=cache, primary="primary")
    assert len(calls) == len(sample)
    assert out["n"] == len(sample)
    assert [v["question_answering"] for v in out["verdicts"]["primary"]] == [1] * 6
    assert [v["question_answering"] for v in out["verdicts"]["second"]] == [2] * 6
    assert all(v["step"] == r["step"] and v["bin"] == r["bin"]
               for v, r in zip(out["verdicts"]["second"], sample))
    assert out["failures"] == {"primary": 0, "second": 0}
    assert any(r["pair"] == ("primary", "second") for r in out["agreement"])
    assert "curves" in out and set(out["curves"]) == {"primary", "second"}


def test_failures_are_counted_and_rows_dropped_pairwise(tmp_path):
    """A failed second-judge call must not leave a misaligned pair; the trace is dropped
    from the comparison and the failure is reported, never read as zero."""
    cache = tmp_path / "c.jsonl"
    sample = _sample()
    _seed_cache(cache, sample, "primary")
    n = {"i": 0}

    def flaky(prompt):
        n["i"] += 1
        if n["i"] == 2:
            return "I cannot annotate this."
        return json.dumps(_verdict())

    out = run_cross_judge(sample, {"primary": None, "flaky": flaky},
                          cache=cache, primary="primary")
    assert out["failures"]["flaky"] == 1
    assert len(out["verdicts"]["flaky"]) == 5
    row = next(r for r in out["agreement"] if r["pair"] == ("primary", "flaky")
               and r["field"] == "question_answering")
    assert row["n"] == 5
