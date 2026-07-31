"""Plan generation work in resumable chunks.

Why this exists
---------------
A GPQA generation run reached 92% of 750 sequences and then hit Modal's 7,200s function
timeout. Each shard wrote its output only after finishing everything, so roughly 16
GPU-hours produced **zero** records — and `retries=1` restarted the whole two-hour job
rather than failing fast.

The docstring of that app claimed each shard "writes its own file before returning, so a
worker dying costs one shard rather than the run". That was true for a crash and false for
a timeout, which is the failure that actually happened.

The fix is structural rather than a bigger timeout: split the work into chunks, persist
after each one, and on restart skip whatever is already on disk. Then an interruption costs
the chunk in flight, not the run. A larger timeout is still worth setting, but it only
changes *when* the cliff arrives; chunking removes the cliff.
"""
from __future__ import annotations


def plan_chunks(problems: list[dict], done_pids: set, chunk_size: int) -> list[list[dict]]:
    """Split the not-yet-done problems into chunks, preserving input order.

    Order preservation matters for resume: the caller shuffles with a fixed seed before
    calling, so keeping order means a restart continues the same sequence instead of
    drawing a different sample.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    todo = [p for p in problems if p["pid"] not in done_pids]
    return [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
