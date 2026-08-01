"""Reward for the LIAR2 misinformation task (SoT Claim C6).

Mirrors the paper's Countdown reward structure -- `0.9 * accuracy + 0.1 * format`, which
yields 1.0 / 0.1 / 0 -- so the C6 curves are on the same scale as the C5 ones and the two
experiments can be read against each other.

Two hard-won constraints are enforced here:

  * **The "Assistant:" marker is required.** verl's stock scorer locates the response by
    splitting on it and returns None otherwise. Dropping it from the prompt zeroed every
    rollout for 170 steps in Claim B before anyone noticed, because a uniformly-zero reward
    looks like a model that cannot learn rather than a scorer that cannot see.
  * **Answers are parsed the way models actually write them** -- markdown emphasis, a
    restated sentence, hyphen/space variants of "half true". Five separate
    answer-extraction bugs in this project have each turned a working model into an
    apparently broken one.

An answer naming more than one verdict is rejected rather than resolved by precedence: a
trace that says "true or false" has not committed, and guessing for it would inflate
accuracy on exactly the traces where the model is least decided.
"""
from __future__ import annotations

import re

VERDICTS = ("true", "half-true", "false")

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_EMPH = re.compile(r"[*_`]+")
# "half true", "half-true", "halftrue" all mean the same rating.
_HALF = re.compile(r"\bhalf[\s\-_]*true\b", re.I)
_TRUE = re.compile(r"\btrue\b", re.I)
_FALSE = re.compile(r"\bfalse\b", re.I)


def _response_of(solution_str: str) -> str | None:
    """Everything after the assistant marker, or None if there is no marker."""
    for marker in ("Assistant:", "<|im_start|>assistant"):
        if marker in solution_str:
            return solution_str.split(marker, 1)[1]
    return None


def extract_verdict(solution_str: str) -> str | None:
    """Return 'true' | 'half-true' | 'false', or None if not cleanly stated."""
    resp = _response_of(solution_str)
    if resp is None:
        return None
    m = list(_ANSWER.finditer(resp))
    if not m:
        return None
    body = _EMPH.sub("", m[-1].group(1)).strip()
    if not body:
        return None

    # Check half-true first: "half true" also contains "true".
    half = bool(_HALF.search(body))
    stripped = _HALF.sub(" ", body)
    plain_true = bool(_TRUE.search(stripped))
    false = bool(_FALSE.search(stripped))

    named = [v for v, hit in (("half-true", half), ("true", plain_true), ("false", false))
             if hit]
    if len(named) != 1:
        return None          # nothing named, or more than one -- not a commitment
    return named[0]


def compute_score(solution_str: str, ground_truth, method: str = "strict",
                  format_score: float = 0.1, score: float = 1.0) -> float:
    """verl-compatible scorer. `ground_truth` is {"verdict": ...} or the string itself."""
    gold = (ground_truth or {}).get("verdict") if isinstance(ground_truth, dict) \
        else ground_truth
    verdict = extract_verdict(solution_str)
    if verdict is None:
        return 0.0
    return score if verdict == str(gold).strip().lower() else format_score
