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


def _rec(pid, source, subtask, task, answer, gradable=True) -> dict:
    return {"pid": pid, "source": source, "subtask": subtask, "task": task,
            "answer": answer, "gradable": gradable}


# --- per-benchmark adapters ---------------------------------------------------

def load_bbh(counts: dict[str, int]) -> list[dict]:
    from datasets import load_dataset

    out = []
    for sub, n in counts.items():
        ds = load_dataset("lukaemon/bbh", sub, split="test")
        for i, r in enumerate(_take(ds, n)):
            out.append(_rec(f"bbh/{sub}/{i}", "bbh", sub, r["input"], str(r["target"])))
    return out


def load_gpqa(counts: dict[str, int]) -> list[dict]:
    from datasets import load_dataset

    cfg = {"gpqa_diamond": "gpqa_diamond", "gpqa_extended": "gpqa_extended",
           "gpqa_main": "gpqa_main"}
    out = []
    for sub, n in counts.items():
        ds = load_dataset("Idavidrein/gpqa", cfg[sub], split="train")
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
                            _mc(r["Question"].strip(), shown), letter))
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
        out.append(_rec(f"mmlu_pro/{i}", "mmlu_pro", r.get("category", "mixed"),
                        _mc(r["question"], list(r["options"])), str(r["answer"]).strip()))
    return out


def load_musr(counts: dict[str, int]) -> list[dict]:
    from datasets import load_dataset

    out = []
    for sub, n in counts.items():
        ds = load_dataset("TAUR-Lab/MuSR", split=sub)
        for i, r in enumerate(_take(ds, n)):
            choices = r["choices"]
            if isinstance(choices, str):
                import ast
                choices = ast.literal_eval(choices)
            gold = LETTERS[int(r["answer_index"])]
            task = f"{r['narrative']}\n\n{r['question']}\n" + "\n".join(
                f"({LETTERS[j]}) {c}" for j, c in enumerate(choices))
            out.append(_rec(f"musr/{sub}/{i}", "musr", sub, task, gold))
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


def normalise_answer(a: str) -> str:
    a = (a or "").strip()
    a = re.sub(r"^\(([A-P])\)$", r"\1", a)
    a = re.sub(r"^\\text\{(.*)\}$", r"\1", a)
    return _NORM.sub("", a).lower().rstrip(".")


def is_correct(pred: str, gold: str) -> bool:
    """Strict-ish equality after normalisation, plus a bare-letter match for MC."""
    p, g = normalise_answer(pred), normalise_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    mp = re.fullmatch(r"\(?([a-p])\)?", p)
    mg = re.fullmatch(r"\(?([a-p])\)?", g)
    return bool(mp and mg and mp.group(1) == mg.group(1))


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
