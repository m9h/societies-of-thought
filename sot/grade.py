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

try:
    from .extract import extract_boxed
except ImportError:  # standalone use as a reference implementation
    from extract import extract_boxed  # type: ignore

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

# Prose statements of a final letter, e.g. "the answer is C", "**Answer:** D".
# Markdown/LaTeX emphasis around the letter is optional and common.
_MCQ_PROSE = re.compile(
    r"\banswer\b\s*(?:is|:)?\s*[:\-]?\s*"
    r"(?:\*{1,2}|__)?\s*\\?(?:text|textbf|mathrm|mathbf)?\s*\{?\s*"
    r"\(?([A-D])\)?",
    re.I,
)

# A letter, possibly wrapped in LaTeX/markdown, possibly followed by the option's text
# ("C. The planet is denser"). Anchored at the start of the extracted span so it cannot
# match an incidental capital elsewhere.
_MCQ_LETTER = re.compile(
    r"^\s*(?:\*{1,2}|__)?\s*\(?\s*([A-D])\s*[\).:,]?",
)


def _strip_markup(s: str) -> str:
    r"""Unwrap the LaTeX/markdown a reasoning model puts around a bare letter.

    \boxed{\text{(C)}} -> (C);  **D** -> D
    """
    s = s.strip()
    for _ in range(3):  # nested wrappers: \boxed{\textbf{\text{C}}}
        s = re.sub(r"\\(?:text|textbf|textrm|mathrm|mathbf|mathit)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("\\", "")
    s = re.sub(r"\*{1,2}|__", "", s)
    return s.strip()


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
    r"""Extract the final option letter.

    Same failure class as the Countdown \boxed{} bug: a reasoning model states its
    answer in whatever markup it feels like -- \boxed{\text{C}}, \boxed{\textbf{B}},
    \boxed{C. The planet is denser}, "**Answer:** D" -- and a grader that accepts only
    \boxed{C} scores every one of those as unparseable, i.e. WRONG. On GPQA that would
    silently manufacture a null result.

    Sources are tried most-explicit first. Within prose we take the LAST statement, so
    "Option B looks plausible, but the answer is D" resolves to D, not B.
    """
    for extract in (extract_boxed, _answer_tag_content):
        raw = extract(completion)
        if raw is not None:
            m = _MCQ_LETTER.match(_strip_markup(raw))
            if m:
                pred = m.group(1).upper()
                return Grade(correct=pred == gold.upper(), parsed=True, pred=pred)

    matches = _MCQ_PROSE.findall(completion)
    if matches:
        pred = matches[-1].upper()  # the final answer the model settles on
        return Grade(correct=pred == gold.upper(), parsed=True, pred=pred)

    return Grade(correct=False, parsed=False, pred=None)


def _answer_tag_content(completion: str) -> str | None:
    m = _ANSWER_TAG.search(completion)
    return m.group(1) if m else None


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


def _delatex(s: str) -> str:
    r"""Turn the LaTeX the model actually emits into evaluable arithmetic.

    \frac{98}{7} -> (98)/(7);  \times -> *;  \div -> /;  strip \left \right $ \! etc.
    """
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[!,;: ]", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("{", "(").replace("}", ")")
    return s


def _grade_countdown(completion: str, gold: str) -> Grade:
    target_s, nums_s = gold.split("|")
    target = int(target_s)
    nums = [int(x) for x in nums_s.split(",")]

    # The prompt asks for <answer></answer>, and the paper's RL reward scores format
    # on exactly that. But R1-distill frequently ignores the tag and answers in LaTeX:
    #     \boxed{\frac{98}{7} + (34 - 27) + 7 = 30}
    # Accepting only the tag marked 74% of baseline traces "unparseable" -- correct
    # answers included -- which dragged baseline accuracy to 5.5% against the paper's
    # 27.1%. That measures the model's willingness to use a tag, not its arithmetic.
    # So fall back to \boxed{}. An answer we cannot find at all is still WRONG, never
    # dropped; this widens what counts as *found*, it does not forgive being wrong.
    m = _ANSWER_TAG.search(completion)
    if m:
        expr = m.group(1)
    else:
        boxed = extract_boxed(completion)
        if boxed is None:
            return Grade(correct=False, parsed=False, pred=None)
        expr = boxed

    expr = _delatex(expr).strip().rstrip("=").strip()
    # Models write both "24 = (3+5)*3" and "(3+5)*3 = 24". Keep the side that is an
    # expression rather than the bare target.
    if "=" in expr:
        parts = [p.strip() for p in expr.split("=") if p.strip()]
        non_trivial = [p for p in parts if not re.fullmatch(r"-?\d+(?:\.\d+)?", p)]
        expr = (non_trivial or parts)[0]

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
