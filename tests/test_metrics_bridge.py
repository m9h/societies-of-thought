"""Tests for the metrics bridge.

Pinned against REAL verl output lines taken from our own Claim B logs, so a parser
regression cannot silently produce empty curves -- the failure mode that would leave us
grepping console text again.
"""
from __future__ import annotations

import json

from rl.metrics_bridge import bridge, parse_line

# Verbatim shape of a verl console line, ANSI prefix and all.
REAL = ("\x1b[36m(main_task pid=3982)\x1b[0m step:25 - global_seqlen/min:65311.000 - "
        "critic/kl:0.055 - critic/kl_coeff:0.001 - val/test_score/countdown:0.081 - "
        "mfu/critic:0.12\n")


def test_parses_a_real_verl_line():
    rec = parse_line(REAL)
    assert rec is not None
    assert rec["step"] == 25
    assert rec["val/test_score/countdown"] == 0.081
    assert rec["critic/kl"] == 0.055
    assert rec["critic/kl_coeff"] == 0.001


def test_step_is_not_duplicated_as_a_metric():
    rec = parse_line(REAL)
    assert "step" in rec and rec["step"] == 25
    assert not any(k == "step" for k in rec if k != "step")


def test_ignores_non_metric_lines():
    for line in ("Loading checkpoint shards: 100%|##########|\n",
                 "INFO worker.py: Started a local Ray instance.\n",
                 "\n"):
        assert parse_line(line) is None


def test_handles_scientific_notation_and_negatives():
    rec = parse_line("step:7 - actor/grad_norm:1.2e-05 - actor/advantage:-0.334\n")
    assert rec["actor/grad_norm"] == 1.2e-05
    assert rec["actor/advantage"] == -0.334


def test_bridge_writes_jsonl_without_trackio(tmp_path):
    log = tmp_path / "ppo.log"
    log.write_text(REAL + "step:50 - val/test_score/countdown:0.183\n")
    out = tmp_path / "m.jsonl"
    n = bridge(log, out, "proj", "run", follow=False)
    assert n == 2
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["step"] for r in rows] == [25, 50]
    assert rows[1]["val/test_score/countdown"] == 0.183


def test_bridge_resumes_without_duplicating(tmp_path):
    """Re-running on a grown log must append only the new steps."""
    log = tmp_path / "ppo.log"
    out = tmp_path / "m.jsonl"
    log.write_text(REAL)
    assert bridge(log, out, "p", "r", follow=False) == 1

    log.write_text(REAL + "step:50 - val/test_score/countdown:0.183\n")
    assert bridge(log, out, "p", "r", follow=False) == 1  # only the new one

    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["step"] for r in rows] == [25, 50]


def test_bridge_tolerates_a_missing_log(tmp_path):
    """A run that has not started yet must not crash the bridge."""
    out = tmp_path / "m.jsonl"
    assert bridge(tmp_path / "nope.log", out, "p", "r", follow=False) == 0


def test_jsonl_is_written_even_without_trackio(tmp_path):
    """Trackio must never gate the durable artifact. When trackio.init() blocked, three
    training arms ran with no metrics recorded while the process looked alive."""
    log = tmp_path / "p.log"
    log.write_text(REAL)
    out = tmp_path / "m.jsonl"
    assert bridge(log, out, "p", "r", follow=False) == 1
    assert out.exists() and out.read_text().strip()


def test_output_file_is_created_even_with_an_empty_log(tmp_path):
    log = tmp_path / "p.log"
    log.write_text("no metrics here\n")
    out = tmp_path / "m.jsonl"
    assert bridge(log, out, "p", "r", follow=False) == 0
    assert out.exists(), "artifact must exist even before the first parsable line"


def test_a_hanging_tracker_cannot_block_the_bridge(tmp_path, monkeypatch):
    """Trackio init hung once and three training arms recorded zero metrics while the
    process looked alive. Whatever the cause, a slow tracker must never gate the run."""
    import rl.metrics_bridge as mb

    def _hang(*a, **k):
        import time
        time.sleep(30)

    monkeypatch.setattr(mb, "_open_tracker", _hang)  # init never returns
    log = tmp_path / "p.log"
    log.write_text(REAL)
    out = tmp_path / "m.jsonl"

    import time
    t0 = time.time()
    n = mb.bridge(log, out, "p", "r", follow=False, trackio=True, tracker_timeout=2.0)
    assert n == 1, "metrics must still be written when the tracker hangs"
    assert time.time() - t0 < 15, "bridge must not wait on a hanging tracker"


def test_trackio_is_enabled_by_default_now_that_it_is_guarded():
    import inspect

    from rl.metrics_bridge import bridge

    assert inspect.signature(bridge).parameters["trackio"].default is True
