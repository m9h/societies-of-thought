"""Countdown reward for GRPO.

Mirrors the paper's reward exactly:

    R = 0.9 * accuracy + 0.1 * format

both binary, and *nothing* rewards conversational or cognitive behaviour -- that is the
entire point of the experiment. If dialogue emerges under this reward, it emerged
because it helps, not because we paid for it.

Correctness is delegated to `sot.grade`, the same grader the steering half uses, which
is covered by 42 tests. Sharing it matters: if the RL half and the steering half scored
Countdown differently, their numbers could not be compared.

THE FORMAT CONFOUND (the trap that can silently rig this whole experiment):

Conversation-SFT teaches the model to emit <persona1>/<think1>/<group_consensus>.
The RL format reward wants <think> and <answer>. If we scored format strictly, a
conversation-primed model would be punished *for producing the very scaffolding under
test* -- and we would "discover" that conversational scaffolding hurts, when all we had
done was fine it for existing.

So `format_reward` accepts any think-like block (<think>, <think1>, <thinkN>) and treats
<group_consensus>/<group_solution> as an answer container. The decision is deliberate,
it is stated here, and `--strict-format` exists so the result can be shown not to depend
on it.
"""

from __future__ import annotations

import re

from sot.grade import grade

_THINK = re.compile(r"<think\d*>.*?</think\d*>", re.S | re.I)
_ANSWER = re.compile(r"<answer>.*?</answer>", re.S | re.I)
_GROUP = re.compile(r"<group_(?:consensus|solution)>.*?</group_(?:consensus|solution)>", re.S | re.I)

_STRICT_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def format_reward(completion: str, strict: bool = False) -> float:
    """1.0 iff the completion has a reasoning block and exactly one answer block."""
    think = _STRICT_THINK if strict else _THINK
    has_think = bool(think.search(completion))

    answers = _ANSWER.findall(completion)
    if not strict:
        answers = answers or _GROUP.findall(completion)

    return 1.0 if (has_think and len(answers) == 1) else 0.0


def grade_completion(completion: str, target: int, nums: list[int]):
    """THE single scoring entry point. Both the reward and the eval must use this.

    They diverged once and it inverted a result: the reward path normalised
    <group_consensus> into <answer>, the eval path did not, and the dialogue arm
    scored 0.40 reward while its eval reported 0.0% accuracy and 0.0% parse rate.
    Reported as-is that becomes a confident "monologue beats dialogue" finding
    which is purely an artifact of which code path could parse which format.

    The docstring of accuracy_reward below already warned about exactly this and
    the eval path was written with the bug anyway, so the defence cannot be a
    comment -- it has to be that there is only one function.
    """
    gold = f"{target}|{','.join(map(str, nums))}"
    return grade("countdown", _normalise_answer_container(completion), gold)


def accuracy_reward(completion: str, target: int, nums: list[int]) -> float:
    """1.0 iff the stated equation uses each number once and evaluates to the target.

    A dialogue trace states its final answer in <group_consensus>, not <answer>. Grading
    the raw text would therefore score a CORRECT dialogue answer as wrong -- handing the
    conversation arm ~zero accuracy reward on every problem it actually solved, and
    manufacturing the finding that conversational scaffolding is catastrophic.

    That is the same extraction bug that has now bitten this project three times (the
    Countdown \\boxed{} case, the GPQA \\text{} case, and here). So normalise the answer
    container FIRST, then grade. Note this only changes where we LOOK for the answer;
    the arithmetic is still judged by the shared grader, and a wrong equation stays wrong.
    """
    return 1.0 if grade_completion(completion, target, nums).correct else 0.0


def _normalise_answer_container(completion: str) -> str:
    """Rewrite <group_consensus>/<group_solution> into <answer> so the shared grader sees it."""
    if _ANSWER.search(completion):
        return completion
    m = _GROUP.search(completion)
    if not m:
        return completion
    inner = re.sub(r"</?group_(?:consensus|solution)>", "", m.group(0), flags=re.I).strip()
    return completion + f"\n<answer>{inner}</answer>"


def countdown_reward(completions, target, nums, strict_format: bool = False, **kwargs):
    """TRL GRPO reward signature: returns one float per completion.

    TRL passes dataset columns through as keyword lists, so `target` and `nums` arrive
    as lists aligned with `completions`.
    """
    out = []
    for completion, t, n in zip(completions, target, nums):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        r = 0.9 * accuracy_reward(text, t, list(n)) + 0.1 * format_reward(text, strict_format)
        out.append(r)
    return out
