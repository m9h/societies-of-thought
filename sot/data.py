"""Benchmark loading.

Three tasks:

  countdown -- the paper's own task. Present only as a positive control: if our
               harness cannot reproduce 27.1% -> 54.8% on feature 30939, nothing
               downstream is interpretable.
  gpqa      -- GPQA-Diamond, 198 four-way multiple-choice graduate science
               questions. Ungated mirror; the official iDavidRein/gpqa is gated.
  math_hard -- MATH Level-5, the "MATH (Hard)" subset used in the paper's
               benchmark suite and in Open LLM Leaderboard v2.

The two new tasks are the actual test: Countdown accuracy has enormous headroom
for a 8B model (27% baseline), so a large steering effect there may say more
about Countdown than about reasoning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset

GPQA_REPO = "fingertap/GPQA-Diamond"  # ungated 198-item MCQ mirror
MATH_HARD_REPO = "lighteval/MATH-Hard"
COUNTDOWN_REPO = "Jiayi-Pan/Countdown-Tasks-3to4"


@dataclass
class Problem:
    pid: str
    task: str
    prompt: str
    answer: str  # letter for MCQ, boxed expr for MATH, "<target>|<n1,n2,...>" for countdown
    meta: dict


MCQ_TEMPLATE = (
    "{question}\n\n"
    "Think step by step inside <think> </think> tags, then give the letter of the "
    "correct option (A, B, C, or D) inside \\boxed{{}}."
)

MATH_TEMPLATE = (
    "{problem}\n\n"
    "Think step by step inside <think> </think> tags, then put your final answer "
    "inside \\boxed{{}}."
)

# Verbatim from the paper's Methods (Countdown Task Prompt), with the numbers and
# target substituted per problem.
COUNTDOWN_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only "
    "be used once. Show your work in <think> </think> tags. And return the final "
    "answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


def load_problems(task: str, n: int | None = None, seed: int = 0) -> list[Problem]:
    if task == "gpqa":
        ds = load_dataset(GPQA_REPO, split="test")
        problems = [
            Problem(
                pid=f"gpqa-{i}",
                task="gpqa",
                prompt=MCQ_TEMPLATE.format(question=r["question"].strip()),
                answer=r["answer"].strip().upper(),
                meta={},
            )
            for i, r in enumerate(ds)
        ]
    elif task == "math_hard":
        ds = load_dataset(MATH_HARD_REPO, split="test")
        problems = []
        for i, r in enumerate(ds):
            gold = extract_boxed(r["solution"])
            if gold is None:
                continue  # a handful of solutions have no \boxed; ungradeable
            problems.append(
                Problem(
                    pid=f"math-{i}",
                    task="math_hard",
                    prompt=MATH_TEMPLATE.format(problem=r["problem"].strip()),
                    answer=gold,
                    meta={"type": r.get("type"), "level": r.get("level")},
                )
            )
    elif task == "countdown":
        ds = load_dataset(COUNTDOWN_REPO, split="train")
        # The dataset is huge; take a deterministic slice so the control is stable.
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        idx = idx[: (n or 1024)]
        problems = []
        for i in idx:
            r = ds[i]
            nums = list(r["nums"])
            problems.append(
                Problem(
                    pid=f"cd-{i}",
                    task="countdown",
                    prompt=COUNTDOWN_TEMPLATE.format(numbers=nums, target=r["target"]),
                    answer=f"{r['target']}|{','.join(map(str, nums))}",
                    meta={"nums": nums, "target": r["target"]},
                )
            )
        return problems  # already sampled

    else:
        raise ValueError(f"unknown task {task!r}")

    if n is not None and n < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n)
    return problems


def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...}, brace-matched."""
    key = "\\boxed"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    while i < len(text) and text[i] != "{":
        if not text[i].isspace():
            return None
        i += 1
    if i >= len(text):
        return None
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
    return None
