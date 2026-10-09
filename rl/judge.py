"""The paper's LLM-as-judge for conversational behaviours.

Fig. 4b is measured by an LLM judge that reads a reasoning trace and counts instances of
four behaviours. The definitions below are the paper's own words; the counting convention
("integer counts, 0 if none are present") is theirs too.

DECLARED DEVIATION. The paper judges with Gemini-2.5-Pro. We judge with whatever backend
is configured -- by default an Anthropic model -- because that is the capable judge this
project has credentials for. The paper's own cross-judge agreement (ICC(3,1) ~ .85
between Gemini-2.5-Pro and GPT-5.2, ~ .76 against human raters) is the reason this is a
declarable deviation rather than a different experiment, but it IS a deviation and the
judge identity is recorded in every cache entry and every result file.

WHY A PARSE FAILURE IS NOT A ZERO. A judge that reads malformed output as "no behaviours
present" fails open. Traces get longer and more complex as RL proceeds, so parse failures
would concentrate in late training, and the curve would show a decline that is entirely
an artifact of the instrument. `parse_verdict` raises instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

BEHAVIOURS = (
    "question_answering",
    "perspective_shift",
    "conflict_of_perspectives",
    "reconciliation",
)

# Bump when the prompt changes. It is part of the cache key, so an edited prompt cannot
# silently mix two instruments inside one curve.
PROMPT_VERSION = "v1"

DEFINITIONS = {
    "question_answering":
        "sequences where a question is posed and later answered",
    "perspective_shift":
        "transition to a different idea, viewpoint, assumption, or approach",
    "conflict_of_perspectives":
        "expressions of disagreement, correction, or tension",
    "reconciliation":
        "conflicting views are integrated or resolved into coherent synthesis",
}

_TEMPLATE = """You are annotating a language model's reasoning trace for conversational \
structure.

Count the number of distinct instances of each of the following behaviours. Return an \
integer count for each, using 0 if none are present.

{definitions}

Also report n_personas: the number of distinct perspectives present in the trace. A trace \
written in a single undifferentiated voice has n_personas = 1.

Judge only what is in the trace. Do not reward or penalise whether the reasoning is \
correct.

Return ONLY a JSON object with exactly these keys and integer values:
{{"question_answering": <int>, "perspective_shift": <int>, \
"conflict_of_perspectives": <int>, "reconciliation": <int>, "n_personas": <int>}}

--- BEGIN TRACE ---
{trace}
--- END TRACE ---
"""


def build_prompt(trace: str) -> str:
    defs = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in DEFINITIONS.items())
    return _TEMPLATE.format(definitions=defs, trace=trace)


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        return brace.group(0)
    raise ValueError(f"no JSON object in judge output: {text[:200]!r}")


def _as_count(value, field: str) -> int:
    # bool is a subclass of int; True would otherwise silently become 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} is not an integer: {value!r}")
    if value < 0:
        raise ValueError(f"{field} is negative: {value}")
    return value


def parse_verdict(text: str) -> dict:
    """Parse a judge response. Raises ValueError rather than defaulting anything to 0."""
    try:
        obj = json.loads(_extract_json(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"judge output is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"judge returned {type(obj).__name__}, not an object")

    out = {}
    for field in (*BEHAVIOURS, "n_personas"):
        if field not in obj:
            raise ValueError(f"judge omitted {field!r} -- refusing to read that as 0")
        out[field] = _as_count(obj[field], field)
    return out


def cache_key(trace: str, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    h = hashlib.sha256()
    h.update(prompt_version.encode())
    h.update(b"\x00")
    h.update(model.encode())
    h.update(b"\x00")
    h.update(trace.encode())
    return h.hexdigest()


def _load_cache(path) -> dict:
    if not path or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            out[rec["key"]] = rec["verdict"]
        except (json.JSONDecodeError, KeyError):
            continue      # a half-written line from a killed run, not a reason to stop
    return out


def judge_trace(trace: str, backend, cache=None, model: str = "unknown") -> dict:
    """Judge one trace, reading through a content-addressed cache.

    The cache is append-only JSONL so a killed run loses at most the line in flight, and
    the key includes both the judge model and the prompt version -- mixing either inside
    one curve would make a change of instrument look like a change in the model.
    """
    key = cache_key(trace, model)
    hits = _load_cache(cache)
    if key in hits:
        return hits[key]

    verdict = parse_verdict(backend(build_prompt(trace)))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "a") as fh:
            fh.write(json.dumps({"key": key, "model": model,
                                 "prompt_version": PROMPT_VERSION,
                                 "verdict": verdict}) + "\n")
    return verdict


def anthropic_backend(model: str = "claude-opus-5", max_tokens: int = 1200):
    """A judge backend backed by the Anthropic API. Requires ANTHROPIC_API_KEY.

    `max_tokens` is generous on purpose. At 300 a more verbose judge ran out of budget
    mid-preamble and never reached its JSON, failing on 74% of traces -- and because
    failures rise with trace length, a truncation budget silently biases which traces
    get counted.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(prompt: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    return call


# --------------------------------------------------------------------------------------
# Other judge families.
#
# Norman, Rivera & Hughes (arXiv 2606.19544) and Yang, Hou & Yang (2607.08535) formalise
# what this project found by hand: a judge score can move with the judge alone. A null
# that rests on one model family is therefore only half a result. The providers below all
# speak the OpenAI chat-completions dialect, so one backend covers Google, OpenAI and the
# aggregators; the key is looked up at construction so a missing key fails before the
# first trace, not as a 401 in the middle of a 256-call run.
# --------------------------------------------------------------------------------------

_PROVIDERS = {
    # provider: (base_url, env var, family, token-limit parameter name)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai",
               "GEMINI_API_KEY", "google", "max_tokens"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "openai",
               "max_completion_tokens"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter",
                   "max_tokens"),
    "nim": ("https://integrate.api.nvidia.com/v1", "NGC_API_KEY", "nvidia", "max_tokens"),
}

_FAMILY_PATTERNS = (
    (r"^claude", "anthropic"), (r"^gemini|^gemma", "google"), (r"^gpt|^o[0-9]", "openai"),
    (r"llama", "meta"), (r"mistral|mixtral", "mistral"), (r"deepseek", "deepseek"),
    (r"qwen", "alibaba"), (r"kimi", "moonshot"), (r"glm", "zhipu"),
)


def family_of(model_tag: str) -> str:
    """Which model family a judge tag belongs to; 'unknown' rather than a guess."""
    t = model_tag.lower().split("/")[-1]
    for pat, fam in _FAMILY_PATTERNS:
        if re.search(pat, t):
            return fam
    return "unknown"


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def openai_compatible_backend(model: str, base_url: str, api_key: str,
                              max_tokens: int = 1200, temperature=0,
                              token_param: str = "max_tokens", _post=None):
    """A judge backend for any OpenAI-chat-compatible endpoint (Gemini, OpenAI, NIM...).

    temperature=0 by default: a judge is a measurement, not a sample. (The Anthropic
    backend above runs at the API default because the published cache was built that
    way; changing it would change the instrument mid-study.) Pass temperature=None to
    omit the field for models that reject it.
    """
    post = _post or _post_json
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    def call(prompt: str) -> str:
        payload = {"model": model, token_param: max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            payload["temperature"] = temperature
        out = post(url, headers, payload)
        if "error" in out:
            raise RuntimeError(f"{model}: {out['error']}")
        try:
            return out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"{model}: malformed completion {str(out)[:200]!r}") from e

    return call


def make_backend(spec: str, max_tokens: int = 1200):
    """`provider/model` -> (callable, cache tag, family).

    A bare model name is Anthropic. The cache tag is the model id alone, so the cache
    key does not change if the same model is reached through a different provider.
    """
    provider, _, model = spec.partition("/")
    if not model:
        provider, model = "anthropic", spec
    if provider == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return anthropic_backend(model, max_tokens=max_tokens), model, "anthropic"
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; known: "
                         f"anthropic, {', '.join(_PROVIDERS)}")
    base_url, env, fam, token_param = _PROVIDERS[provider]
    if env not in os.environ:
        raise RuntimeError(f"{env} is not set (needed for provider {provider!r})")
    temperature = None if provider == "openai" else 0
    call = openai_compatible_backend(model, base_url, os.environ[env], max_tokens=max_tokens,
                                     temperature=temperature, token_param=token_param)
    fam = fam if fam not in ("openrouter", "nvidia") else family_of(model)
    return call, model.split("/")[-1], fam
