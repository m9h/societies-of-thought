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

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RAY = re.compile(r"\((?:main_task|WorkerDict|raylet|pid=)[^)]*\)\s?")
_STEP = re.compile(r"^step:(\d+)\s*-")
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
        if cur is not None and not _NOISE.match(line):
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


def behaviours_per_step(rollouts) -> dict[int, dict]:
    """Per-step behaviour rates per 100 words, plus bin size and mean length."""
    by = defaultdict(list)
    for r in rollouts:
        by[r["step"]].append(r)

    out = {}
    for step, rs in sorted(by.items()):
        words = [max(len(r["response"].split()), 1) for r in rs]
        qa = [question_answering(r["response"]) for r in rs]
        cf = [conflict_of_perspectives(r["response"]) for r in rs]
        tw = sum(words)
        out[step] = {
            "n": len(rs),
            "words": tw / len(rs),
            "qa_rate": 100.0 * sum(qa) / tw,
            "conflict_rate": 100.0 * sum(cf) / tw,
            "qa_per_trace": sum(qa) / len(rs),
            "conflict_per_trace": sum(cf) / len(rs),
            "think_blocks": sum(r["response"].count("<think>") for r in rs) / len(rs),
        }
    return out
