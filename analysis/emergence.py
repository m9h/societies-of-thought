"""Fig. 4: do conversational behaviours emerge under accuracy-only RL?

The paper trains Qwen-2.5-3B (base, not instruction-tuned) with PPO on Countdown, rewarding
only `0.9*accuracy + 0.1*format`, and reports that the frequency of two conversational
behaviours -- **Question & Answering** and **Conflict of Perspectives** -- rises across
training. Their instrument is an LLM judge (Fig. 4b, 4e).

This module does two things the judge cannot do for itself:

1. **Step alignment.** verl prints a batch's rollouts to stdout *before* the metric line
   for the step that consumed them. Attributing a trace to the preceding `step:N` shifts
   the entire emergence curve by one batch, and the shifted curve looks just as plausible.
   `parse_rollouts` attributes each trace to the step that *follows* it.

2. **A judge-free proxy.** Counting the surface markers directly gives a cheap curve to
   compare the judge against. Where the two instruments disagree is itself the finding --
   see `rl/judge.py`.

Rates are per 100 words, never per trace: response length grows under RL, so a per-trace
count would rise from length alone and manufacture the effect being tested.
"""
from __future__ import annotations

import re
from collections import defaultdict

from analysis.hse import segment

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RAY = re.compile(r"\((?:main_task|WorkerDict|raylet|pid=)[^)]*\)\s?")
_STEP = re.compile(r"^step:(\d+)\s*-")
# verl's Countdown scorer echoes every graded rollout back to stdout as
#   --------------------------------
#   Target: 18 | Numbers: [50 57 73 52]
#   Extracted equation: None
#   Solution string: A conversation between User and Assistant. ...
# That block lands AFTER the response and BEFORE the next prompt, so a parser that runs
# to the next "User:" swallows it. It did: 78% of traces carried ~45 words of it. That
# diluted every rate, and it handed an LLM judge the phrase "A conversation between User
# and Assistant" -- which biases a persona count in exactly the direction under test.
# Anchored at line start so a response that merely mentions a target is not truncated.
_SCORER_DEBUG = re.compile(
    r"^(?:-{8,}|Target:\s|Extracted equation:|Solution string:)")

_NOISE = re.compile(
    r"^(?:INFO|WARNING|ERROR|DEBUG|\(|Loading checkpoint|Downloading|"
    r"Validation|Adding requests|Processed prompts|\s*$)"
    r"|\d+%\|", re.I)

# "Conflict of Perspectives": a reversal of a position already taken. A bare "but" is a
# conjunction, not a disagreement -- counting it would make the measure track sentence
# length. Every pattern here requires an explicit retraction, objection or alternative.
_CONFLICT = re.compile(
    r"\b(wait|hold on|hmm+|oh no|no,|nope|actually|on second thought|scratch that|"
    r"that'?s (?:wrong|not right|incorrect)|(?:that|this|it) (?:did|does)(?:n'?t| not) work|"
    r"reconsider|rethink|re-?check|let'?s try (?:again|another|a different)|"
    r"alternatively|instead|however|but (?:actually|wait|no|that)|i was wrong|"
    r"my mistake|that fails|doesn'?t equal)\b", re.I)


def strip_log_noise(text: str) -> str:
    """Remove ANSI colour codes and Ray's interleaved worker prefixes."""
    return _RAY.sub("", _ANSI.sub("", text))


def parse_rollouts(text: str) -> list[dict]:
    """Recover `{step, prompt, response}` for every logged rollout.

    A rollout starts at a `User: ` line and runs until the next `User: ` line or the next
    `step:N` metric line. It is attributed to that following step -- the one whose gradient
    it contributed to. Rollouts with no following step line (a run killed mid-batch) were
    never trained on and are dropped.
    """
    clean = strip_log_noise(text)
    out: list[dict] = []
    pending: list[dict] = []
    cur: dict | None = None

    for line in clean.splitlines():
        m = _STEP.match(line.strip())
        if m:
            if cur is not None:
                pending.append(cur)
                cur = None
            step = int(m.group(1))
            for r in pending:
                r["step"] = step
            out.extend(pending)
            pending = []
            continue
        if line.startswith("User: "):
            if cur is not None:
                pending.append(cur)
            cur = {"step": None, "prompt": line[len("User: "):].strip(), "response": ""}
            continue
        if cur is None:
            continue
        if _SCORER_DEBUG.match(line):
            pending.append(cur)      # the response ended; the scorer is talking now
            cur = None
            continue
        if not _NOISE.match(line):
            cur["response"] += line + "\n"

    return out   # `pending` and `cur` are intentionally discarded


def question_answering(text: str) -> int:
    """Count questions the trace poses AND then continues past.

    A trailing question with nothing after it is an unanswered one; the behaviour the
    paper names is question *and answering*.
    """
    n = 0
    for m in re.finditer(r"\?", text):
        if len(text[m.end():].strip()) >= 3:
            n += 1
    return n


def conflict_of_perspectives(text: str) -> int:
    """Count explicit reversals of a position the trace already took."""
    return len(_CONFLICT.findall(text))


# "Reconciliation": conflicting views integrated into a synthesis. A bare conclusion
# ("the answer is 51") is not one -- every trace ends in a conclusion, so counting those
# would make the measure track trace count rather than integration.
_RECONCILE = re.compile(
    r"\b(combin\w+|put(?:ting)? (?:it|them|these) together|taking (?:both|all) "
    r"(?:of these|together)|reconcil\w+|on balance|either way|both (?:approaches|ideas|"
    r"ways) (?:agree|lead|give)|so (?:overall|in the end)|that (?:settles|resolves) it|"
    r"which confirms|consistent with (?:both|the earlier))\b", re.I)


def perspective_shift(text: str) -> int:
    """Count transitions between perspectives.

    N segments means N-1 transitions: a trace in one steady voice has shifted zero times.
    Reusing `analysis.hse.segment` keeps this measure identical to the one the diversity
    work uses, so the two analyses cannot silently drift apart.
    """
    return max(len(segment(text)) - 1, 0)


def reconciliation(text: str) -> int:
    """Count explicit integrations of conflicting views."""
    return len(_RECONCILE.findall(text))


_PATTERNS = {"conflict_of_perspectives": _CONFLICT, "reconciliation": _RECONCILE}


def pattern_dominance(texts, behaviour: str) -> dict:
    """How much of a marker count comes from one repeated string.

    A surface-marker rate is only evidence of a behaviour if the language varies. On the
    claimA log, 99.8% of late-training "conflict of perspectives" matches were the single
    parenthetical "doesn't equal", emitted once per line of a numbered enumeration -- the
    measure read as a 12x rise in dialogic behaviour while every genuinely dialogic
    marker ("however", "nope", "let's try another") fell to near zero. Report this
    beside any marker rate, or the rate can be a template count wearing a costume.
    """
    from collections import Counter

    pat = _PATTERNS[behaviour]
    c = Counter()
    for text in texts:
        for m in pat.finditer(text):
            c[m.group(0).lower().strip()] += 1
    total = sum(c.values())
    if not total:
        return {"total": 0, "top": None, "top_share": 0.0, "distinct": 0}
    top, n = c.most_common(1)[0]
    return {"total": total, "top": top, "top_share": n / total, "distinct": len(c)}


BEHAVIOUR_FNS = {
    "question_answering": question_answering,
    "perspective_shift": perspective_shift,
    "conflict_of_perspectives": conflict_of_perspectives,
    "reconciliation": reconciliation,
}


def behaviours_per_step(rollouts) -> dict[int, dict]:
    """Per-step behaviour rates per 100 words, plus bin size and mean length."""
    by = defaultdict(list)
    for r in rollouts:
        by[r["step"]].append(r)

    out = {}
    for step, rs in sorted(by.items()):
        words = [max(len(r["response"].split()), 1) for r in rs]
        tw = sum(words)
        rec = {
            "n": len(rs),
            "words": tw / len(rs),
            "think_blocks": sum(r["response"].count("<think>") for r in rs) / len(rs),
        }
        for name, fn in BEHAVIOUR_FNS.items():
            counts = [fn(r["response"]) for r in rs]
            rec[f"{name}_rate"] = 100.0 * sum(counts) / tw
            rec[f"{name}_per_trace"] = sum(counts) / len(rs)
        # Short aliases kept for the two behaviours the first pass measured.
        rec["qa_rate"] = rec["question_answering_rate"]
        rec["conflict_rate"] = rec["conflict_of_perspectives_rate"]
        rec["qa_per_trace"] = rec["question_answering_per_trace"]
        rec["conflict_per_trace"] = rec["conflict_of_perspectives_per_trace"]
        out[step] = rec
    return out
