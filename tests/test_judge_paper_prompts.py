"""The paper's judge prompts, verbatim — tests BEFORE implementation.

We said for three months that the Fig. 4 instrument was unpublished. It is not: the v1
HTML's Supplementary Methods carry the behaviour-counting prompt and the persona
identification prompt in full. Our v1 prompt paraphrased the definitions and folded the
persona count into the same call. The instrument must now be theirs, word for word, and
the persona count must come from their separate persona prompt, whose schema key is
`n_perspectives`.
"""
from __future__ import annotations

import json

import pytest

from rl import judge


def test_default_prompt_version_is_the_papers():
    assert judge.PROMPT_VERSION == "paper-v1"


def test_behaviour_prompt_is_verbatim():
    p = judge.build_prompt("TRACE-X")
    for s in (
        "Your task is to analyze the following text and count how many times behaviors "
        "corresponding to each of the four dimensions appear.",
        "**Text to Analyze:**",
        '"Question_and_Answering": <int>',
        "A question is posed and later answered, as in conversations.",
        '"Let’s try X…? This gives us Y"',
        "A transition to a different idea, viewpoint, assumption, or approach, as in conversations.",
        "Expressions of disagreement, correction, or tension with another perspective.",
        '"Wait, that can’t be right…"',
        "Conflicting views are integrated or resolved into a coherent synthesis.",
        "For each category, count the number of distinct times the behavior occurs in the "
        "chain of thought and return the result as integers. If none are present, use 0.",
    ):
        assert s in p, s
    assert "TRACE-X" in p
    assert "n_personas" not in p            # the persona count is a separate call


def test_persona_prompt_is_verbatim_and_asks_n_perspectives():
    p = judge.build_persona_prompt("TRACE-Y")
    for s in (
        "identify the number of distinct perspectives (agents or voices)",
        'Transitional markers (e.g., "however," "but," "alternatively," "wait," "let me check," "actually," "on the other hand")',
        "BFI-10",
        '"n_perspectives": N',
        "1. Is reserved.",
        "10. Has an active imagination.",
    ):
        assert s in p, s
    assert "TRACE-Y" in p
    assert "segment" not in p.lower()      # segmentation is a third call we do not make


def test_v1_prompt_is_still_available_for_the_old_cache():
    p = judge.build_prompt("t", version="v1")
    assert "n_personas" in p


def test_parse_behaviours_accepts_paper_keys():
    v = judge.parse_behaviours(json.dumps({"Question_and_Answering": 2, "Perspective_Shift": 1,
                                           "Conflict_of_Perspectives": 0, "Reconciliation": 0}))
    assert v == {"question_answering": 2, "perspective_shift": 1,
                 "conflict_of_perspectives": 0, "reconciliation": 0}


def test_parse_behaviours_missing_key_raises():
    with pytest.raises(ValueError, match="Reconciliation"):
        judge.parse_behaviours(json.dumps({"Question_and_Answering": 2, "Perspective_Shift": 1,
                                           "Conflict_of_Perspectives": 0}))


def test_parse_persona_reads_n_perspectives_and_keeps_profiles():
    out = judge.parse_persona(json.dumps({
        "n_perspectives": 2,
        "personality": [["Agree strongly"] * 10, ["Disagree a little"] * 10],
        "domain_expertise": ["setup", "verification"]}))
    assert out["n_personas"] == 2
    assert len(out["personality"]) == 2
    assert out["domain_expertise"] == ["setup", "verification"]


def test_parse_persona_rejects_zero_or_missing():
    with pytest.raises(ValueError):
        judge.parse_persona(json.dumps({"n_perspectives": 0, "personality": [], "domain_expertise": []}))
    with pytest.raises(ValueError, match="n_perspectives"):
        judge.parse_persona(json.dumps({"personality": [], "domain_expertise": []}))


def test_parse_persona_count_must_match_profiles():
    with pytest.raises(ValueError, match="profiles"):
        judge.parse_persona(json.dumps({"n_perspectives": 3, "personality": [["x"] * 10],
                                        "domain_expertise": ["a"]}))


def test_judge_trace_paper_version_makes_two_calls_and_merges(tmp_path):
    prompts = []

    def backend(prompt):
        prompts.append(prompt)
        if "n_perspectives" in prompt:
            return json.dumps({"n_perspectives": 1, "personality": [["Agree a little"] * 10],
                               "domain_expertise": ["arithmetic"]})
        return json.dumps({"Question_and_Answering": 1, "Perspective_Shift": 0,
                           "Conflict_of_Perspectives": 0, "Reconciliation": 0})

    v = judge.judge_trace("trace", backend, cache=tmp_path / "c.jsonl", model="m")
    assert len(prompts) == 2
    assert v["question_answering"] == 1 and v["n_personas"] == 1
    assert v["domain_expertise"] == ["arithmetic"]
    # cached under the paper version, separately from v1
    v2 = judge.judge_trace("trace", backend, cache=tmp_path / "c.jsonl", model="m")
    assert len(prompts) == 2 and v2 == v
    rec = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])
    assert rec["prompt_version"] == "paper-v1"
    assert judge.cache_key("trace", "m") != judge.cache_key("trace", "m", "v1")


def test_judge_trace_v1_is_single_call(tmp_path):
    n = []

    def backend(prompt):
        n.append(1)
        return json.dumps({b: 1 for b in judge.BEHAVIOURS} | {"n_personas": 1})

    judge.judge_trace("t", backend, cache=tmp_path / "c.jsonl", model="m", prompt_version="v1")
    assert len(n) == 1


def test_a_failed_persona_call_fails_the_trace_not_zero(tmp_path):
    def backend(prompt):
        if "n_perspectives" in prompt:
            return "I'd rather not."
        return json.dumps({"Question_and_Answering": 1, "Perspective_Shift": 0,
                           "Conflict_of_Perspectives": 0, "Reconciliation": 0})
    with pytest.raises(ValueError):
        judge.judge_trace("t", backend, cache=tmp_path / "c.jsonl", model="m")
    assert not (tmp_path / "c.jsonl").exists()        # nothing half-written
