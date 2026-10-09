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
import threading
from pathlib import Path

BEHAVIOURS = (
    "question_answering",
    "perspective_shift",
    "conflict_of_perspectives",
    "reconciliation",
)

# Bump when the prompt changes. It is part of the cache key, so an edited prompt cannot
# silently mix two instruments inside one curve.
PROMPT_VERSION = "paper-v1"

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


# --------------------------------------------------------------------------------------
# The paper's prompts, verbatim (arXiv 2601.10825v1, Supplementary Methods: LLM-as-Judge
# prompts). Wording and punctuation are theirs, including curly quotes and ellipses; line
# breaks are reconstructed from the flattened HTML. The v1 prompt above paraphrased these
# definitions and folded the persona count into the same call; for three months this
# project said the instrument was unpublished. It was not. "paper-v1" is now the default.
# --------------------------------------------------------------------------------------

PAPER_BEHAVIOUR_TEMPLATE = """Your task is to analyze the following text and count how many times behaviors \
corresponding to each of the four dimensions appear.

**Text to Analyze:**
{chain_of_thought}

—

You must output a single valid JSON object with the exact schema below and nothing else.
{
  "Question_and_Answering": <int>,
  "Perspective_Shift": <int>,
  "Conflict_of_Perspectives": <int>,
  "Reconciliation": <int>
}

Use the following definitions:
1. **Question and Answering** — A question is posed and later answered, as in conversations. \
(e.g., "Why…? Because…", "What if…? Then…", "How do we know? Well…", "Let’s try X…? This gives us Y")
2. **Perspective Shift** — A transition to a different idea, viewpoint, assumption, or approach, \
as in conversations.
3. **Conflict of Perspectives** — Expressions of disagreement, correction, or tension with another \
perspective. (e.g., "Wait, that can’t be right…", "No, actually…", "This contradicts…")
4. **Reconciliation** — Conflicting views are integrated or resolved into a coherent synthesis. \
(e.g., "So perhaps both are true if…", "Combining these insights…", "This resolves the tension…")

For each category, count the number of distinct times the behavior occurs in the chain of thought \
and return the result as integers. If none are present, use 0.
"""

PAPER_PERSONA_TEMPLATE = """Your task is to analyze the following text to identify the number of distinct \
perspectives (agents or voices). A perspective is defined as a distinct cognitive perspective or \
reasoning role within the text.

Indicators of a perspective may include:
- Transitional markers (e.g., "however," "but," "alternatively," "wait," "let me check," "actually," "on the other hand")
- Shifts between cognitive roles (e.g., problem setup, calculation, verification, error correction, summarization)
- Changes in rhetorical purpose or approach
- Corrections or reconsiderations
- Movement between subproblems
- Domain knowledge
- Personality traits

For each distinct perspective, you will infer its personality by answering the 10 questions of the \
BFI-10 questionnaire as if you were that agent. You will also provide a concise profile of its \
domain expertise.

Your final output must be a single, valid JSON object and nothing else. Do not include any text or \
explanations before or after the JSON object.

**Text to Analyze:**
{chain_of_thought}

—

## **Analysis Instructions**

1. **Identify Perspectives:** Analyze the text to determine the number of distinct voices \
(n_perspectives). Apply the definition above consistently, treating each identifiable shift as a \
boundary between perspectives.
2. **Answer Questionnaire:** For each perspective, answer the 10 BFI-10 questions below from that \
perspective’s point of view. You must use one of these five exact strings for each answer:
   - "Disagree strongly"
   - "Disagree a little"
   - "Neither agree nor disagree"
   - "Agree a little"
   - "Agree strongly"
3. **Profile Expertise:** For each perspective, write a short, open-ended string describing its \
domain expertise and cognitive function.

### **BFI-10 Questionnaire**
Rate the extent to which you, as the identified perspective, agree or disagree with the following \
statements. I see myself as someone who…
1. Is reserved.
2. Is generally trusting.
3. Tends to be lazy.
4. Is relaxed, handles stress well.
5. Has few artistic interests.
6. Is outgoing, sociable.
7. Tends to find fault with others.
8. Does a thorough job.
9. Gets nervous easily.
10. Has an active imagination.

—

## **Required JSON Output Format**
{
  "n_perspectives": N,
  "personality": [
    ["Answer to Q1 for Perspective 1", "Answer to Q2 for Perspective 1", "…", "Answer to Q10 for Perspective 1"],
    ["Answer to Q1 for Perspective 2", "Answer to Q2 for Perspective 2", "…", "Answer to Q10 for Perspective 2"],
    …
  ],
  "domain_expertise": [
    "Open-ended description for Perspective 1.",
    "Open-ended description for Perspective 2.",
    …
  ]
}
"""

_PAPER_KEYS = {
    "Question_and_Answering": "question_answering",
    "Perspective_Shift": "perspective_shift",
    "Conflict_of_Perspectives": "conflict_of_perspectives",
    "Reconciliation": "reconciliation",
}


def build_prompt(trace: str, version: str = None) -> str:
    """The behaviour-counting prompt. Default: the paper's, verbatim."""
    version = version or PROMPT_VERSION
    if version == "v1":
        defs = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in DEFINITIONS.items())
        return _TEMPLATE.format(definitions=defs, trace=trace)
    if version == "paper-v1":
        return PAPER_BEHAVIOUR_TEMPLATE.replace("{chain_of_thought}", trace)
    raise ValueError(f"unknown prompt version {version!r}")


def build_persona_prompt(trace: str) -> str:
    """The paper's persona-identification prompt (without its segmentation follow-up)."""
    return PAPER_PERSONA_TEMPLATE.replace("{chain_of_thought}", trace)


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


def parse_behaviours(text: str) -> dict:
    """Parse the paper-schema behaviour counts (v1 keys accepted too). Raises, never 0."""
    try:
        obj = json.loads(_extract_json(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"judge output is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"judge returned {type(obj).__name__}, not an object")
    out = {}
    for paper_key, ours in _PAPER_KEYS.items():
        if paper_key in obj:
            out[ours] = _as_count(obj[paper_key], paper_key)
        elif ours in obj:
            out[ours] = _as_count(obj[ours], ours)
        else:
            raise ValueError(f"judge omitted {paper_key!r} -- refusing to read that as 0")
    return out


def parse_persona(text: str) -> dict:
    """Parse the paper-schema persona verdict: n_perspectives >= 1 with matching profiles."""
    try:
        obj = json.loads(_extract_json(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"persona output is not valid JSON: {e}") from e
    if not isinstance(obj, dict) or "n_perspectives" not in obj:
        raise ValueError("persona judge omitted n_perspectives")
    n = _as_count(obj["n_perspectives"], "n_perspectives")
    if n < 1:
        raise ValueError(f"n_perspectives must be >= 1, got {n}")
    personality = obj.get("personality") or []
    expertise = obj.get("domain_expertise") or []
    if not isinstance(personality, list) or len(personality) != n:
        raise ValueError(f"n_perspectives={n} but {len(personality)} personality profiles")
    return {"n_personas": n, "personality": personality, "domain_expertise": expertise}


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


_CACHE_LOCK = threading.Lock()


def judge_trace(trace: str, backend, cache=None, model: str = "unknown",
                prompt_version: str = None) -> dict:
    """Judge one trace, reading through a content-addressed cache.

    The cache is append-only JSONL so a killed run loses at most the line in flight, and
    the key includes both the judge model and the prompt version -- mixing either inside
    one curve would make a change of instrument look like a change in the model.

    "paper-v1" is two calls (the paper's behaviour prompt, then its persona prompt) merged
    into one verdict; if either fails the trace fails and nothing is cached.
    """
    prompt_version = prompt_version or PROMPT_VERSION
    key = cache_key(trace, model, prompt_version)
    hits = _load_cache(cache)
    if key in hits:
        return hits[key]

    if prompt_version == "v1":
        verdict = parse_verdict(backend(build_prompt(trace, "v1")))
    else:
        verdict = parse_behaviours(backend(build_prompt(trace, prompt_version)))
        verdict.update(parse_persona(backend(build_persona_prompt(trace))))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_LOCK, open(cache, "a") as fh:
            fh.write(json.dumps({"key": key, "model": model,
                                 "prompt_version": prompt_version,
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
