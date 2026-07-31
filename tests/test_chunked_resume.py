"""Chunked generation with resume — tests written BEFORE the implementation.

A GPQA run reached 92% of 750 sequences and then hit Modal's function timeout. Because
each shard wrote its output only at the very end, ~16 GPU-hours produced nothing, and
retries=1 restarted the whole job rather than failing fast.

The fix is structural: generate in chunks, persist after every chunk, and on restart skip
what is already on disk. Then a timeout costs the current chunk, not the run.
"""
from __future__ import annotations

import pytest

from rl.chunking import plan_chunks


def _p(pid):
    return {"pid": pid, "task": "q"}


def test_all_work_is_planned_when_nothing_is_done():
    probs = [_p(f"p{i}") for i in range(10)]
    chunks = plan_chunks(probs, done_pids=set(), chunk_size=4)
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert [r["pid"] for c in chunks for r in c] == [f"p{i}" for i in range(10)]


def test_completed_problems_are_skipped():
    probs = [_p(f"p{i}") for i in range(10)]
    chunks = plan_chunks(probs, done_pids={"p0", "p1", "p2"}, chunk_size=4)
    remaining = [r["pid"] for c in chunks for r in c]
    assert remaining == [f"p{i}" for i in range(3, 10)]


def test_nothing_left_returns_no_chunks():
    probs = [_p(f"p{i}") for i in range(5)]
    chunks = plan_chunks(probs, done_pids={f"p{i}" for i in range(5)}, chunk_size=4)
    assert chunks == []


def test_chunk_size_larger_than_work_gives_one_chunk():
    probs = [_p(f"p{i}") for i in range(3)]
    assert len(plan_chunks(probs, set(), chunk_size=100)) == 1


def test_order_is_preserved_so_resume_is_deterministic():
    """Resume must continue the same seeded ordering, not reshuffle."""
    probs = [_p(f"p{i}") for i in range(20)]
    first = plan_chunks(probs, set(), chunk_size=5)
    done = {r["pid"] for r in first[0]}
    rest = plan_chunks(probs, done, chunk_size=5)
    assert [r["pid"] for c in rest for r in c] == [f"p{i}" for i in range(5, 20)]


def test_rejects_a_nonsensical_chunk_size():
    with pytest.raises(ValueError):
        plan_chunks([_p("a")], set(), chunk_size=0)


def test_empty_problem_list_is_safe():
    assert plan_chunks([], set(), chunk_size=4) == []
