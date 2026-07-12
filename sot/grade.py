"""Answer extraction and grading.

Steering changes how a model talks, and a naive grader can mistake that for a
change in whether it is *right*. Positive steering makes traces chattier and more
self-interrupting, which makes them likelier to run past the token budget or to
trail off without a final answer. If an unparseable trace were dropped rather
than scored wrong, steering would inflate accuracy purely by changing the
denominator. So every attempt is graded, and an unparseable answer is simply
incorrect. `truncated` and `parsed` are recorded per attempt so the analysis can
show the format-failure rate alongside accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data import extract_boxed

_MCQ_PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", re.I),
    re.compile(r"<answer>\s*\(?([A-D])\)?\s*</answer>", re.I),
    re.compile(r"\banswer\s*(?:is|:)\s*\(?([A-D])\)?\b", re.I),
]

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


@dataclass
class Grade:
    correct: bool
    parsed: bool  # did we find a final answer at all?
    pred: str | None


def grade(task: str, completion: str, gold: str) -> Grade:
    if task == "gpqa":
        return _grade_mcq(completion, gold)
    if task == "math_hard":
        return _grade_math(completion, gold)
    if task == "countdown":
        return _grade_countdown(completion, gold)
    raise ValueError(f"unknown task {task!r}")


def _grade_mcq(completion: str, gold: str) -> Grade:
    for pat in _MCQ_PATTERNS:
        m = pat.search(completion)
        if m:
            pred = m.group(1).upper()
            return Grade(correct=pred == gold.upper(), parsed=True, pred=pred)
    return Grade(correct=False, parsed=False, pred=None)


def _grade_math(completion: str, gold: str) -> Grade:
    pred = extract_boxed(completion)
    if pred is None:
        return Grade(correct=False, parsed=False, pred=None)
    return Grade(correct=math_equal(pred, gold), parsed=True, pred=pred)


def math_equal(pred: str, gold: str) -> bool:
    """Symbolic-then-string equivalence.

    math_verify handles the LaTeX cases that string comparison gets wrong
    (\\frac{1}{2} vs 0.5, \\dfrac vs \\frac, trailing units). It is optional so
    the module imports without it, but the runner warns loudly if it is missing,
    because falling back to normalized string match systematically UNDER-counts
    correct MATH answers -- and it would do so equally across conditions, which
    biases the steering contrast toward null rather than toward a false positive.
    """
    if _norm(pred) == _norm(gold):
        return True
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except ImportError:
        return False
    except Exception:
        return False


def _norm(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\!|\\,|\\;|\\ ", "", s)
    s = s.rstrip(".")
    if s.endswith("^\\circ"):
        s = s[: -len("^\\circ")]
    if s.startswith("\\$"):
        s = s[2:]
    return s


def _grade_countdown(completion: str, gold: str) -> Grade:
    target_s, nums_s = gold.split("|")
    target = int(target_s)
    nums = [int(x) for x in nums_s.split(",")]

    m = _ANSWER_TAG.search(completion)
    if not m:
        return Grade(correct=False, parsed=False, pred=None)
    expr = m.group(1).strip().rstrip("=").strip()
    # Models often write "24 = (3+5)*3"; keep the right-hand side.
    if "=" in expr:
        expr = expr.split("=")[-1].strip()

    if not re.fullmatch(r"[0-9+\-*/()\s.]+", expr):
        return Grade(correct=False, parsed=False, pred=expr)

    used = [int(t) for t in re.findall(r"\d+", expr)]
    if sorted(used) != sorted(nums):
        return Grade(correct=False, parsed=True, pred=expr)

    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - charset-restricted above
    except (SyntaxError, ZeroDivisionError, TypeError, NameError):
        return Grade(correct=False, parsed=True, pred=expr)

    return Grade(correct=abs(float(val) - target) < 1e-6, parsed=True, pred=expr)
