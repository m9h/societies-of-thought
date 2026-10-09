"""Judge backends beyond one model family — tests written BEFORE the implementation.

Norman, Rivera & Hughes (2606.19544) and Yang, Hou & Yang (2607.08535) make the point
this project has been making informally: an LLM-judge score can move when only the judge
changes. Our Fig. 4 null rests on one model family. Closing that needs a backend that can
be pointed at a second family the moment a key exists -- without a network call at
construction time, and without silently mixing judges inside one cache.
"""
from __future__ import annotations

import json

import pytest

from rl import judge


def test_make_backend_bare_name_is_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    call, tag, fam = judge.make_backend("claude-sonnet-5")
    assert callable(call)
    assert tag == "claude-sonnet-5"
    assert fam == "anthropic"


def test_make_backend_provider_prefix_selects_family(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    _, tag, fam = judge.make_backend("gemini/gemini-2.5-pro")
    assert (tag, fam) == ("gemini-2.5-pro", "google")
    _, tag, fam = judge.make_backend("openai/gpt-5.2")
    assert (tag, fam) == ("gpt-5.2", "openai")


def test_make_backend_missing_key_names_the_env_var(monkeypatch):
    """A missing key must fail at construction with the variable name, not 401 mid-run."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        judge.make_backend("gemini/gemini-2.5-pro")


def test_make_backend_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        judge.make_backend("mystery/model-x")


def test_openai_compatible_backend_payload_and_extraction():
    seen = {}

    def fake_post(url, headers, payload):
        seen.update(url=url, headers=headers, payload=payload)
        return {"choices": [{"message": {"content": '{"ok": 1}'}}]}

    call = judge.openai_compatible_backend(
        "gemini-2.5-pro", base_url="https://example/v1", api_key="sekret",
        max_tokens=777, _post=fake_post)
    out = call("PROMPT")
    assert out == '{"ok": 1}'
    assert seen["url"] == "https://example/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sekret"
    p = seen["payload"]
    assert p["model"] == "gemini-2.5-pro"
    assert p["max_tokens"] == 777
    assert p["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert p.get("temperature", 0) == 0          # a judge is not a sampler


def test_openai_compatible_backend_surfaces_api_error():
    def fake_post(url, headers, payload):
        return {"error": {"message": "quota"}}
    call = judge.openai_compatible_backend("m", base_url="u", api_key="k", _post=fake_post)
    with pytest.raises(RuntimeError, match="quota"):
        call("x")


def test_family_inference_from_model_tag():
    assert judge.family_of("claude-opus-5") == "anthropic"
    assert judge.family_of("claude-haiku-4-5-20251001") == "anthropic"
    assert judge.family_of("gemini-2.5-pro") == "google"
    assert judge.family_of("gpt-5.2") == "openai"


def test_cache_key_separates_judges_not_families():
    """Two judges in one family are still two instruments; the cache keys them apart."""
    a = judge.cache_key("t", "claude-sonnet-5")
    b = judge.cache_key("t", "claude-opus-5")
    assert a != b
