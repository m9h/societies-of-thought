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


def attempt_reward(completion: str, nums: list[int], strict: bool = False) -> float:
    """1.0 iff the completion is a REAL Countdown attempt: a reasoning block, exactly
    one answer container, AND an expression that uses each given number exactly once
    (target-independent — a valid attempt that misses the target still counts).

    This is the anti-hack replacement for the format slot. `format_reward` pays for
    the <think>/<answer> skeleton regardless of content, so a model can farm the 0.1
    with `<answer>1</answer>` and never do arithmetic — which is exactly what the RL
    probe did (reward climbed via format while accuracy stayed flat). Requiring a valid
    equation using the numbers removes the shortcut: the only way to earn the credit is
    to produce real Countdown expressions, which is a step toward correctness.

    Reuses the shared grader's extraction (via grade_completion's `pred`) so it handles
    the same tag/boxed/LaTeX cases the accuracy path does — no second parser to drift.
    """
    think = _STRICT_THINK if strict else _THINK
    if not think.search(completion):
        return 0.0
    answers = _ANSWER.findall(completion)
    if not strict:
        answers = answers or _GROUP.findall(completion)
    if len(answers) != 1:
        return 0.0

    # Grade with the real target; we only read the extracted expression and whether it
    # parsed as arithmetic. `pred` is the normalised expression the grader found.
    g = grade_completion(completion, 0, nums)  # target irrelevant to the number check
    if not g.parsed or not g.pred:
        return 0.0
    used = [int(t) for t in re.findall(r"\d+", g.pred)]
    return 1.0 if sorted(used) == sorted(nums) else 0.0


def countdown_reward(completions, target, nums, strict_format: bool = False,
                     reward_shape: str = "attempt", **kwargs):
    """TRL GRPO reward signature: returns one float per completion.

    TRL passes dataset columns through as keyword lists, so `target` and `nums` arrive
    as lists aligned with `completions`.

    reward_shape:
      "attempt" (default) — 0.9*accuracy + 0.1*attempt. The 0.1 requires a real
        equation using the numbers, closing the skeleton-farming exploit that made the
        first RL probe format-hack. This is the shape the sweep should use.
      "paper" — 0.9*accuracy + 0.1*format, the paper-faithful reward, kept for the A/B
        that demonstrates the exploit. It pays 0.1 for empty tags.

    Component means for the batch are stashed in LAST_COMPONENTS so the training loop
    can log accuracy and format SEPARATELY — the diagnostic the first probe lacked.
    """
    if reward_shape not in ("attempt", "paper"):
        raise ValueError(f"reward_shape must be 'attempt' or 'paper', got {reward_shape!r}")

    out, accs, fmts = [], [], []
    for completion, t, n in zip(completions, target, nums):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        n = list(n)
        acc = accuracy_reward(text, t, n)
        fmt = (attempt_reward(text, n, strict_format) if reward_shape == "attempt"
               else format_reward(text, strict_format))
        accs.append(acc)
        fmts.append(fmt)
        out.append(0.9 * acc + 0.1 * fmt)

    if accs:
        LAST_COMPONENTS["accuracy"] = sum(accs) / len(accs)
        LAST_COMPONENTS["format"] = sum(fmts) / len(fmts)
        LAST_COMPONENTS["shape"] = reward_shape
    return out


# Batch component means from the most recent countdown_reward call. The training loop
# logs these so "reward went up" can be read as accuracy-vs-format rather than guessed.
LAST_COMPONENTS: dict = {"accuracy": float("nan"), "format": float("nan"), "shape": ""}
