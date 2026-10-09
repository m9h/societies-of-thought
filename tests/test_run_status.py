"""`scripts/run_status.sh` — tests BEFORE implementation.

The session monitor for the Fig. 4 run had two bugs: it anchored `^step:` and so read
step 0 on Ray-prefixed lines when the log said 250, and it used `kill -0` alone, which
cannot tell a normal exit from a crash, so it reported a successful 250-step completion
as FAILURE. A watchdog that cries wolf on success is ignored on failure.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_status.sh"

RAY = "\x1b[36m(main_task pid=3491)\x1b[0m "


def _run(log, *extra):
    p = subprocess.run(["bash", str(SCRIPT), str(log), *map(str, extra)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return dict(kv.split("=", 1) for kv in p.stdout.split())


def test_step_is_read_through_ray_prefix(tmp_path):
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:249 - global_seqlen/min:1 - critic/kl:0\n")
    st = _run(log, 250)
    assert st["STEP"] == "249"


def test_completed_by_final_marker(tmp_path):
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:249 - global_seqlen/min:1\n"
                   + RAY + "Final validation metrics: {'val/test_score/countdown': 0.70}\n")
    st = _run(log, 250)
    assert st["STATUS"] == "completed"


def test_completed_by_off_by_one_step_counter(tmp_path):
    """verl writes its last training-step line at N-1; the old monitor demanded N."""
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:249 - global_seqlen/min:1\n")
    assert _run(log, 250)["STATUS"] == "completed"


def test_dead_pid_after_completion_is_still_completed(tmp_path):
    """The exact false alarm: process gone + log complete == success, not FAILURE."""
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:249 - global_seqlen/min:1\n")
    assert _run(log, 250, 999999)["STATUS"] == "completed"


def test_dead_pid_without_completion_is_crashed(tmp_path):
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:40 - global_seqlen/min:1\n")
    st = _run(log, 250, 999999)
    assert st["STATUS"] == "crashed"
    assert st["STEP"] == "40"


def test_live_pid_is_running(tmp_path):
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:40 - global_seqlen/min:1\n")
    p = subprocess.Popen(["sleep", "30"])
    try:
        assert _run(log, 250, p.pid)["STATUS"] == "running"
    finally:
        p.kill()


def test_no_pid_reports_incomplete_with_log_age(tmp_path):
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:40 - global_seqlen/min:1\n")
    st = _run(log, 250)
    assert st["STATUS"] == "incomplete"
    assert int(st["LOG_AGE_S"]) >= 0


def test_missing_log_is_not_started(tmp_path):
    st = _run(tmp_path / "nope.log", 250)
    assert st["STATUS"] == "not_started"
    assert st["STEP"] == "0"


def test_last_val_score_is_reported(tmp_path):
    log = tmp_path / "ppo.log"
    log.write_text(RAY + "step:10 - val/test_score/countdown:0.104 - x:1\n"
                   + RAY + "step:20 - val/test_score/countdown:0.351 - x:1\n"
                   + RAY + "step:40 - global_seqlen/min:1\n")
    assert _run(log, 250)["VAL"] == "0.351"
