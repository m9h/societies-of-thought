"""Corpus export — tests BEFORE implementation.

The authors ship no code and no traces. Our complete faithful-config run (2,555 training
rollouts, every one judged or judgeable) is the artifact other replicators need. Rows
must be the parser's clean responses (no scorer-debug tail), and verdicts must join to
rollouts by the same content hash the cache uses, so a reader can verify the join.
"""
from __future__ import annotations

import json

import pytest

from rl.judge import PROMPT_VERSION, cache_key
from scripts.export_fig4_corpus import build_rows, join_verdicts, write_corpus

RAY = "\x1b[36m(main_task pid=1)\x1b[0m "

LOG = (
    "User: Using the numbers [1, 2, 3], create an equation that equals 6.\n"
    "Assistant: Let me think.\n<think> 1+2+3 = 6 </think>\n<answer>1+2+3</answer>\n"
    "--------------------------------\n"
    "Target: 6 | Numbers: [1, 2, 3]\n"
    "Extracted equation: 1+2+3\n"
    "Solution string: A conversation between User and Assistant.\n"
    "User: Using the numbers [4, 5], create an equation that equals 9.\n"
    "Assistant: 4+5\n<answer>4+5</answer>\n"
    + RAY + "step:1 - global_seqlen/min:1 - critic/kl:0\n"
    "User: Using the numbers [7, 8], create an equation that equals 15.\n"
    "Assistant: 7+8\n<answer>7+8</answer>\n"
    + RAY + "step:2 - global_seqlen/min:1\n"
)


def test_build_rows_are_clean_and_stepped():
    rows = build_rows(LOG)
    assert [r["step"] for r in rows] == [1, 1, 2]
    assert [r["idx"] for r in rows] == [0, 1, 2]
    assert "Target:" not in rows[0]["response"]
    assert rows[0]["source"] == "train"
    assert "conversation between" not in rows[0]["response"]
    assert rows[0]["prompt"].startswith("Using the numbers [1, 2, 3]")
    assert rows[0]["words"] == len(rows[0]["response"].split())
    import hashlib
    assert rows[0]["sha256"] == hashlib.sha256(rows[0]["response"].encode()).hexdigest()


def test_join_verdicts_by_cache_key(tmp_path):
    rows = build_rows(LOG)
    cache = tmp_path / "c.jsonl"
    v = {"question_answering": 1, "perspective_shift": 0,
         "conflict_of_perspectives": 0, "reconciliation": 0, "n_personas": 1}
    with cache.open("w") as fh:
        fh.write(json.dumps({"key": cache_key(rows[1]["response"], "claude-sonnet-5"),
                             "model": "claude-sonnet-5", "prompt_version": PROMPT_VERSION,
                             "verdict": v}) + "\n")
        fh.write(json.dumps({"key": "deadbeef", "model": "claude-sonnet-5",
                             "prompt_version": PROMPT_VERSION, "verdict": v}) + "\n")
    out = join_verdicts(rows, cache, models=["claude-sonnet-5", "claude-opus-5"])
    assert len(out) == 1
    assert out[0]["idx"] == 1 and out[0]["step"] == 1
    assert out[0]["judge_model"] == "claude-sonnet-5"
    assert out[0]["prompt_version"] == PROMPT_VERSION
    assert out[0]["question_answering"] == 1


def test_write_corpus_files(tmp_path):
    rows = build_rows(LOG)
    counts = write_corpus(tmp_path, rows, verdicts=[], card="# card")
    assert (tmp_path / "rollouts.jsonl").exists()
    assert (tmp_path / "judgments.jsonl").exists()
    assert "Question_and_Answering" in (tmp_path / "judge_prompt_behaviours.txt").read_text()
    assert "n_perspectives" in (tmp_path / "judge_prompt_persona.txt").read_text()
    assert (tmp_path / "README.md").read_text() == "# card"
    assert counts == {"rollouts": 3, "judgments": 0}
