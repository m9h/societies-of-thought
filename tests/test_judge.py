"""The paper's LLM-as-judge — tests written BEFORE the implementation.

Fig. 4b is measured by an LLM judge that labels four conversational behaviours and
returns integer counts. Our judge-free marker proxy (analysis/emergence.py) is cheap and
reproducible but it is NOT their instrument, and the whole point of running Fig. 4 is to
speak from experience about their result rather than about a proxy for it.

Two failure modes drive these tests.

The first is a judge that fails open. If a malformed or truncated response is quietly
read as "no behaviours present", every parse failure becomes a zero, and since responses
get longer and more complex as RL proceeds, the failures would concentrate in late
training and manufacture a decline. A parse failure must be a parse failure.

The second is asymmetric cost. Judging every trace twice across three seeds is real
money, so results are cached by content hash -- and a cache that ignores which behaviour
schema or model produced an entry would silently mix instruments across a curve.
"""
from __future__ import annotations

import json

import pytest

from rl.judge import (
    BEHAVIOURS,
    build_prompt,
    cache_key,
    judge_trace,
    parse_verdict,
)


def test_all_four_paper_behaviours_are_present():
    """The paper names four. Our first pass at the proxy measured two, and the two it
    left out are where the interesting disagreement turned out to be."""
    assert set(BEHAVIOURS) == {
        "question_answering", "perspective_shift",
        "conflict_of_perspectives", "reconciliation",
    }


def test_prompt_carries_the_papers_definitions_verbatim():
    p = build_prompt("some trace")
    assert "a question is posed and later answered" in p
    assert "a different idea, viewpoint, assumption, or approach" in p
    assert "disagreement, correction, or tension" in p
    assert "integrated or resolved" in p


def test_prompt_asks_for_counts_and_allows_zero():
    """The paper: 'counts the number of distinct instances of each behaviour, returning
    integer counts (0 if none are present)'. A judge that cannot say zero will not."""
    p = build_prompt("t")
    assert "0" in p and "integer" in p.lower()


def test_prompt_embeds_the_trace():
    assert "UNIQUE-TRACE-MARKER" in build_prompt("UNIQUE-TRACE-MARKER")


def test_parses_a_clean_json_verdict():
    v = parse_verdict(json.dumps({"question_answering": 2, "perspective_shift": 5,
                                  "conflict_of_perspectives": 1, "reconciliation": 0,
                                  "n_personas": 3}))
    assert v["question_answering"] == 2
    assert v["reconciliation"] == 0
    assert v["n_personas"] == 3


def test_parses_json_inside_a_markdown_fence():
    """Every chat model does this eventually, and a judge that dies on it loses whole
    bins of the curve rather than single traces."""
    body = ('Here is my analysis:\n```json\n'
            '{"question_answering": 1, "perspective_shift": 2, '
            '"conflict_of_perspectives": 0, "reconciliation": 1, "n_personas": 2}\n```')
    assert parse_verdict(body)["perspective_shift"] == 2


def test_a_missing_behaviour_is_an_error_not_a_zero():
    """This is the important one. Silently defaulting to 0 turns every parse failure
    into evidence of absence -- and failures rise with trace length, so they would
    concentrate in late training and fabricate a decline."""
    with pytest.raises(ValueError):
        parse_verdict(json.dumps({"question_answering": 1}))


def test_unparseable_output_raises():
    with pytest.raises(ValueError):
        parse_verdict("I'm sorry, I can't analyse that.")


def test_negative_or_non_integer_counts_are_rejected():
    for bad in (-1, 1.5, "two", None):
        payload = {b: 0 for b in BEHAVIOURS}
        payload["n_personas"] = 1
        payload["conflict_of_perspectives"] = bad
        with pytest.raises(ValueError):
            parse_verdict(json.dumps(payload))


def test_cache_key_separates_different_traces():
    assert cache_key("a", "m", "v1") != cache_key("b", "m", "v1")


def test_cache_key_separates_different_judge_models():
    """Mixing two judges inside one curve would make a change of instrument look like
    a change in the model under study."""
    assert cache_key("a", "gemini", "v1") != cache_key("a", "claude", "v1")


def test_cache_key_separates_different_prompt_versions():
    assert cache_key("a", "m", "v1") != cache_key("a", "m", "v2")


def test_judge_trace_uses_the_cache_and_does_not_call_twice(tmp_path):
    calls = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({b: 1 for b in BEHAVIOURS} | {"n_personas": 2})

    cache = tmp_path / "c.jsonl"
    a = judge_trace("trace one", backend=fake, cache=cache, model="fake")
    b = judge_trace("trace one", backend=fake, cache=cache, model="fake")
    assert a == b
    assert len(calls) == 1, "second identical trace must come from cache"


def test_cache_survives_a_new_process(tmp_path):
    def fake(prompt: str) -> str:
        return json.dumps({b: 3 for b in BEHAVIOURS} | {"n_personas": 1})

    cache = tmp_path / "c.jsonl"
    judge_trace("t", backend=fake, cache=cache, model="fake")

    def explode(prompt: str) -> str:
        raise AssertionError("cache miss after restart -- would re-pay for every trace")

    assert judge_trace("t", backend=explode, cache=cache, model="fake")["reconciliation"] == 3


def test_the_token_budget_is_not_tight_enough_to_truncate_the_json():
    """At max_tokens=300 a verbose judge spent its budget on preamble and never emitted
    JSON -- 74% failures. Failures rise with trace length, so a tight budget biases which
    traces get counted, in the same direction as every other length artifact here."""
    import inspect

    from rl.judge import anthropic_backend
    assert inspect.signature(anthropic_backend).parameters["max_tokens"].default >= 1000
