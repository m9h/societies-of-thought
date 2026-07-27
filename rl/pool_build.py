"""Reconstruct the paper's 8,262-problem priming pool (Supplementary Table 9).

The paper's SFT priming data is drawn from general reasoning benchmarks and is
**out of domain** relative to the Countdown RL task. That relationship is the whole
content of Claim C5: does conversational structure learned on unrelated problems
transfer and accelerate RL elsewhere? Priming on Countdown itself -- which is what we
did in the first run -- tests something else.

What is recoverable and what is not
-----------------------------------
Recoverable: the pool COMPOSITION. Supplementary Table 9 gives per-subtask counts that
sum to exactly 8,262 (asserted in `rl.paper_spec`). We match every count.

Not recoverable: WHICH problems the authors sampled where a benchmark is larger than
its listed count (MMLU-Pro 432 of ~12k; GPQA main 380 of 448). We take the first N by a
fixed seed. Declared in `paper_spec.DEVIATIONS['pool_problem_identity']`.

IFEval (524 problems) is in the pool for composition fidelity but cannot contribute
priming examples: the priming set is filtered to instances that "reach correct
answers", and the paper itself notes IFEval "tasks are excluded due to difficulties in
accuracy evaluation". Such records are emitted with `gradable=False`.

Run on a machine with network + HF_TOKEN:
    python -m rl.pool_build --out rl/data/pool.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rl import paper_spec as S

LETTERS = "ABCDEFGHIJKLMNOP"


def _mc(question: str, options: list[str]) -> str:
    """Render a multiple-choice problem the way the benchmarks present it."""
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return f"{question}\n{opts}"


def _boxed(solution: str) -> str | None:
    """Pull the final \\boxed{...} payload out of a MATH solution, brace-balanced."""
    i = solution.rfind("\\boxed")
    if i < 0:
        return None
    j = solution.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(solution)):
        if solution[k] == "{":
            depth += 1
        elif solution[k] == "}":
            depth -= 1
            if depth == 0:
                return solution[j + 1:k].strip()
    return None


def _take(rows, n, seed=0):
    """Deterministic subsample to exactly n (or everything, if fewer exist)."""
    import random

    rows = list(rows)
    if len(rows) <= n:
        return rows
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def _rec(pid, source, subtask, task, answer, gradable=True, options=None) -> dict:
    """`options` is the ordered choice list for multiple-choice items.

    It is stored so the grader can accept an answer given as option TEXT, not just as
    a letter. Without it, a model replying "10^-4 eV" to a letter-gold item scores
    wrong, which silently removed GPQA, MMLU-Pro and MUSR from the priming corpus.
    """
    return {"pid": pid, "source": source, "subtask": subtask, "task": task,
            "answer": answer, "gradable": gradable, "options": options or []}


# --- per-benchmark adapters ---------------------------------------------------

def load_bbh(counts: dict[str, int]) -> list[dict]:
    from datasets import load_dataset

    out = []
    for sub, n in counts.items():
        ds = load_dataset("lukaemon/bbh", sub, split="test")
        for i, r in enumerate(_take(ds, n)):
            out.append(_rec(f"bbh/{sub}/{i}", "bbh", sub, r["input"], str(r["target"])))
    return out


# The canonical GPQA repo is gated. `Wanfq/gpqa` is an open mirror carrying identical
# per-config counts (diamond 198 / main 448 / extended 546) and the same column schema.
# Preference order: canonical first, mirror only if the canonical repo is unreachable.
GPQA_REPOS = ("Idavidrein/gpqa", "Wanfq/gpqa")


def _load_gpqa_config(cfg: str):
    from datasets import load_dataset

    last = None
    for repo in GPQA_REPOS:
        try:
            return load_dataset(repo, cfg, split="train"), repo
        except Exception as e:  # gated / unauthenticated / offline
            last = e
    raise RuntimeError(f"no GPQA source reachable for {cfg}: {last}")


def load_gpqa(counts: dict[str, int]) -> list[dict]:
    cfg = {"gpqa_diamond": "gpqa_diamond", "gpqa_extended": "gpqa_extended",
           "gpqa_main": "gpqa_main"}
    out = []
    for sub, n in counts.items():
        ds, repo = _load_gpqa_config(cfg[sub])
        if repo != GPQA_REPOS[0]:
            print(f"  note: {sub} via open mirror {repo} (canonical repo gated)")
        for i, r in enumerate(_take(ds, n)):
            correct = r["Correct Answer"].strip()
            opts = [correct, r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                    r["Incorrect Answer 3"]]
            opts = [str(o).strip() for o in opts]
            # deterministic shuffle so the answer is not always (A)
            order = sorted(range(4), key=lambda k: hash((sub, i, k)) & 0xFFFF)
            shown = [opts[k] for k in order]
            letter = LETTERS[shown.index(correct)]
            out.append(_rec(f"gpqa/{sub}/{i}", "gpqa", sub,
                            _mc(r["Question"].strip(), shown), letter,
                            options=shown))
    return out


def load_math_hard(counts: dict[str, int]) -> list[dict]:
    from datasets import load_dataset

    name = {"algebra": "algebra", "counting_and_probability": "counting_and_probability",
            "geometry": "geometry", "intermediate_algebra": "intermediate_algebra",
            "number_theory": "number_theory", "prealgebra": "prealgebra",
            "precalculus": "precalculus"}
    out = []
    for sub, n in counts.items():
        ds = load_dataset("lighteval/MATH-Hard", name[sub], split="test")
        rows = [r for r in ds if _boxed(r["solution"])]
        for i, r in enumerate(_take(rows, n)):
            out.append(_rec(f"math/{sub}/{i}", "math_hard", sub, r["problem"],
                            _boxed(r["solution"])))
    return out


def load_mmlu_pro(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for i, r in enumerate(_take(ds, n)):
        opts = list(r["options"])
        out.append(_rec(f"mmlu_pro/{i}", "mmlu_pro", r.get("category", "mixed"),
                        _mc(r["question"], opts), str(r["answer"]).strip(),
                        options=opts))
    return out


def load_musr(counts: dict[str, int]) -> list[dict]:
    from datasets import load_dataset

    # The Hub split is pluralised for one subtask; Table 9 names it in the singular.
    split_name = {"object_placement": "object_placements"}
    out = []
    for sub, n in counts.items():
        ds = load_dataset("TAUR-Lab/MuSR", split=split_name.get(sub, sub))
        for i, r in enumerate(_take(ds, n)):
            choices = r["choices"]
            if isinstance(choices, str):
                import ast
                choices = ast.literal_eval(choices)
            gold = LETTERS[int(r["answer_index"])]
            task = f"{r['narrative']}\n\n{r['question']}\n" + "\n".join(
                f"({LETTERS[j]}) {c}" for j, c in enumerate(choices))
            out.append(_rec(f"musr/{sub}/{i}", "musr", sub, task, gold,
                            options=[str(c) for c in choices]))
    return out


def load_ifeval(n: int) -> list[dict]:
    """In the pool for composition fidelity; not gradable, so never primed on."""
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    return [_rec(f"ifeval/{i}", "ifeval", "instruction_following", r["prompt"], "",
                 gradable=False)
            for i, r in enumerate(_take(ds, n))]


# --- grading ------------------------------------------------------------------

_NORM = re.compile(r"[\s$\\,]+")
_PROSE = re.compile(
    r"^(so\s+)?(the\s+)?(final\s+|correct\s+)?(answer|solution|option|choice)"
    r"\s*(is|:|=)?\s*", re.I)


def _unbox(a: str) -> str:
    """Unwrap a trailing \\boxed{...}, brace-balanced."""
    i = a.rfind("\\boxed")
    if i < 0:
        return a
    j = a.find("{", i)
    if j < 0:
        return a
    depth = 0
    for k in range(j, len(a)):
        if a[k] == "{":
            depth += 1
        elif a[k] == "}":
            depth -= 1
            if depth == 0:
                return a[j + 1:k].strip()
    return a


def normalise_answer(a: str) -> str:
    a = (a or "").strip()
    a = _PROSE.sub("", a).strip()
    a = _unbox(a).strip()
    a = re.sub(r"^\(([A-P])\)$", r"\1", a, flags=re.I)
    a = re.sub(r"^\\text\{(.*)\}$", r"\1", a)
    return _NORM.sub("", a).lower().rstrip(".")


def _letter(a: str) -> str | None:
    """Leading choice letter, however the model dressed it up: 'C', '(C)', 'C.',
    'C) 10^-4 eV', 'C: foo'."""
    a = _unbox(_PROSE.sub("", (a or "").strip()).strip()).strip()
    m = re.match(r"^\(?([A-Pa-p])\)?\s*(?:[.):\-]|$)", a)
    return m.group(1).upper() if m else None


def is_correct(pred: str, gold: str, options: list[str] | None = None) -> bool:
    """Grade an answer, tolerating how models actually write them.

    Accepts a bare letter, a parenthesised letter, letter-plus-text, a \\boxed{...}
    payload, a prose prefix ("the answer is ..."), or -- when `options` is supplied --
    the option TEXT for a letter gold. The narrow version of this function silently
    excluded every letter-gold benchmark from the priming corpus.
    """
    p, g = normalise_answer(pred), normalise_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True

    lp, lg = _letter(pred), _letter(gold)
    if lp and lg and lp == lg:
        return True

    # Gold is a letter and the model answered with the option's text (or vice versa).
    if options:
        idx = {LETTERS[i]: normalise_answer(o) for i, o in enumerate(options)}
        if lg and lg in idx and idx[lg] and idx[lg] == p:
            return True
        if lp and lp in idx and idx[lp] and idx[lp] == g:
            return True
        if lg and lg in idx and idx[lg] and idx[lg] in p:
            return True
    return False


def build() -> list[dict]:
    pool: list[dict] = []
    pool += load_bbh(S.POOL_BBH)
    pool += load_gpqa(S.POOL_GPQA)
    pool += load_math_hard(S.POOL_MATH_HARD)
    pool += load_mmlu_pro(S.POOL_OTHER["mmlu_pro"])
    pool += load_musr(S.POOL_MUSR)
    pool += load_ifeval(S.POOL_OTHER["ifeval"])
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the paper's 8,262-problem pool.")
    ap.add_argument("--out", type=Path, default=Path("rl/data/pool.json"))
    ap.add_argument("--allow-short", action="store_true",
                    help="write the pool even if a source is unavailable (records a gap)")
    args = ap.parse_args()

    pool = build()
    by_source: dict[str, int] = {}
    for r in pool:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    gradable = sum(1 for r in pool if r["gradable"])

    print(f"pool: {len(pool)} problems (paper: {S.POOL_TOTAL}); gradable {gradable}")
    for k, v in sorted(by_source.items()):
        print(f"  {k:12} {v}")

    if len(pool) != S.POOL_TOTAL and not args.allow_short:
        raise SystemExit(
            f"pool is {len(pool)}, paper specifies {S.POOL_TOTAL}. Fix the sources or "
            "rerun with --allow-short and declare the gap in paper_spec.DEVIATIONS."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pool))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
