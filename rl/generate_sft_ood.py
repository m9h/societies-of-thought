"""Generate the paper's SFT priming data with the paper's teacher and prompts.

Replaces `rl/generate_sft.py`, which primed on Countdown (the RL task itself) via a 72B
teacher. Both were deviations. This module:

  * uses `paper_spec.TEACHER_MODEL` -- Qwen/Qwen2.5-32B-Instruct, self-hosted with vLLM,
    which is what the paper used. (An earlier run substituted the 72B because one API
    provider did not list the 32B. Self-hosting removes that constraint entirely.)
  * draws problems from the out-of-domain 8,262 pool (`rl.pool_build`), not Countdown;
  * uses the verbatim generation prompts from Supplementary Methods;
  * keeps only instances that reach correct answers, and keeps the SAME problems for
    both arms -- the paper's invariant: "both conditions are trained on identical
    problems and correct answers".

Run on a GPU host:
    python -m rl.generate_sft_ood --pool rl/data/pool.json --out rl/data/ood \
        --attempt 2500 --tp 2
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from rl import paper_spec as S
from rl.pool_build import is_correct

PERSONA_SLOT = (
    "<persona{i}>\n[Brief persona for thinker {i} – personality traits, domain "
    "expertises, and reasoning styles]\n</persona{i}>\n"
)
THINK_SLOT = "<think{i}>\n…\n</think{i}>\n"

_GROUP = re.compile(r"<group_solution>(.*?)</group_solution>", re.S | re.I)
_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)


def dialogue_prompt(task: str, n_thinkers: int) -> str:
    return S.DIALOGUE_PROMPT.format(
        task=task,
        n_thinkers={2: "2", 3: "3", 4: "4"}[n_thinkers],
        persona_slots="".join(PERSONA_SLOT.format(i=i) for i in range(1, n_thinkers + 1)),
        think_slots="".join(THINK_SLOT.format(i=i) for i in range(1, n_thinkers + 1)),
    )


def monologue_prompt(task: str) -> str:
    return S.MONOLOGUE_PROMPT.format(task=task)


def extract_dialogue_answer(text: str) -> str | None:
    m = _GROUP.search(text)
    return m.group(1).strip() if m else None


def extract_monologue_answer(text: str) -> str | None:
    """Monologue traces reason in <think>…</think> then answer after it."""
    m = _THINK.search(text)
    if not m:
        return None
    tail = text[m.end():].strip()
    if not tail:
        return None
    tail = re.sub(r"^(the\s+)?(final\s+)?answer\s*(is)?\s*[:\-]?\s*", "", tail,
                  flags=re.I)
    return tail.split("\n")[0].strip() or None


def well_formed_dialogue(text: str) -> bool:
    """The paper's tag structure must actually be present, or it is not a dialogue."""
    return ("<cast_of_characters>" in text and "</cast_of_characters>" in text
            and re.search(r"<think1>", text, re.I) is not None
            and re.search(r"<think2>", text, re.I) is not None
            and _GROUP.search(text) is not None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-faithful SFT priming generation.")
    ap.add_argument("--pool", type=Path, default=Path("rl/data/pool.json"))
    ap.add_argument("--out", type=Path, default=Path("rl/data/ood"))
    ap.add_argument("--attempt", type=int, default=2500,
                    help="problems to attempt; need enough BOTH-correct to reach 600")
    ap.add_argument("--tp", type=int, default=2, help="vLLM tensor parallel size")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = [r for r in json.loads(args.pool.read_text()) if r["gradable"]]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    problems = pool[: args.attempt]
    print(f"pool {len(pool)} gradable; attempting {len(problems)}")

    from vllm import LLM, SamplingParams

    llm = LLM(model=S.TEACHER_MODEL, tensor_parallel_size=args.tp,
              dtype="bfloat16", trust_remote_code=True, seed=args.seed,
              gpu_memory_utilization=0.90, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_tokens,
                        seed=args.seed)

    def chat(prompts: list[str]) -> list[str]:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        return [o.outputs[0].text for o in llm.generate(texts, sp)]

    n_thinkers = [rng.choice(S.DIALOGUE_N_THINKERS) for _ in problems]
    print("generating dialogue traces ...")
    dia = chat([dialogue_prompt(p["task"], n) for p, n in zip(problems, n_thinkers)])
    print("generating monologue traces ...")
    mon = chat([monologue_prompt(p["task"]) for p in problems])

    matched = []
    for p, d, m in zip(problems, dia, mon):
        if not well_formed_dialogue(d):
            continue
        da, ma = extract_dialogue_answer(d), extract_monologue_answer(m)
        if da is None or ma is None:
            continue
        if not (is_correct(da, p["answer"]) and is_correct(ma, p["answer"])):
            continue
        matched.append({"pid": p["pid"], "source": p["source"], "subtask": p["subtask"],
                        "task": p["task"], "answer": p["answer"],
                        "dialogue": d.strip(), "monologue": m.strip()})

    need = S.SFT_N_TRAIN + S.SFT_N_VAL
    print(f"both-correct and well-formed: {len(matched)} / {len(problems)} "
          f"({100*len(matched)/max(len(problems),1):.1f}%); need {need}")
    if len(matched) < need:
        raise SystemExit(
            f"only {len(matched)} matched instances; need {need}. Re-run with a larger "
            f"--attempt (try {int(args.attempt * need / max(len(matched),1) * 1.3)})."
        )

    rng.shuffle(matched)
    keep = matched[:need]
    split = {"train": keep[: S.SFT_N_TRAIN], "val": keep[S.SFT_N_TRAIN:]}

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        for arm in ("dialogue", "monologue"):
            recs = [{"pid": r["pid"], "source": r["source"], "subtask": r["subtask"],
                     "task": r["task"], "answer": r["answer"], arm: r[arm]}
                    for r in rows]
            (args.out / f"{arm}_{name}.json").write_text(json.dumps(recs, indent=1))
            print(f"  wrote {arm}_{name}.json  ({len(recs)})")

    srcs: dict[str, int] = {}
    for r in keep:
        srcs[r["source"]] = srcs.get(r["source"], 0) + 1
    print("priming set composition:", dict(sorted(srcs.items())))
    dw = sorted(len(r["dialogue"].split()) for r in keep)[len(keep) // 2]
    mw = sorted(len(r["monologue"].split()) for r in keep)[len(keep) // 2]
    print(f"median words -- dialogue {dw}, monologue {mw}  (ratio {dw/max(mw,1):.2f}x)")


if __name__ == "__main__":
    main()
