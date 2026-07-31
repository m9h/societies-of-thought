"""Run QwQ-32B over the NON-MATH domains of our pool, keeping correct AND incorrect traces.

WHY
---
`results/qwq/FINDINGS.md` reports a null: within QwQ, at matched trace length, perspective
diversity does not predict correctness. Its stated limitation is that the corpus
(NuminaMath) is **math only**, while SoT's pool spans BBH/GPQA/MATH-Hard/MMLU-Pro/MUSR.
Math has a specific failure mode -- QwQ flails and doubles its trace length when stuck --
and the null might not generalise.

This generates the missing corpus: the *same model* (QwQ-32B, one of the two SoT builds its
diversity claims on) over chemistry, physics, biology, law, deductive reasoning and the
rest of our reconstructed pool.

WHAT IS DIFFERENT FROM `modal_teacher.py`
-----------------------------------------
That app keeps only problems where BOTH arms answer correctly -- it builds *priming* data.
Here the incorrect traces are the entire point, because the measurement is
correct-vs-incorrect. Everything is kept, labelled, and nothing is filtered on outcome.

    modal run modal_qwq_domains.py --attempt 4000 --shards 4
"""
from __future__ import annotations

import modal

APP = "sot-qwq-domains"
app = modal.App(APP)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.6", "vllm>=0.8", "transformers>=4.51", "accelerate",
                 "safetensors", "datasets", "numpy", "huggingface_hub", "hf_transfer")
    .add_local_dir("rl", remote_path="/root/rl")
)

cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("sot-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false",
       "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "PYTHONPATH": "/root"}

MODEL = "Qwen/QwQ-32B"          # the paper's own subject model
GPU = "A100-80GB:2"

# Reasoning traces are long; QwQ in particular thinks at length before answering.
#
# TRUNCATION IS NOT A NEUTRAL LOSS. A trace cut off at the token cap never reaches an
# answer, so it is graded incorrect -- which manufactures a class of traces that are
# simultaneously MAXIMUM length and ALWAYS wrong. In a correct-vs-incorrect diversity
# comparison that is the confound in its purest form. At 4096 tokens, QwQ hit the cap on
# most GPQA problems and measured accuracy fell to 16%, below the 25% chance floor.
#
# So: a generous cap, AND every record carries vLLM's finish_reason so truncated traces
# can be excluded downstream rather than silently scored as failures.
MAX_TOKENS = 16384


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=120 * 60, env=ENV, retries=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def generate_shard(shard: int, n_shards: int, attempt: int, seed: int = 0,
                   source: str = "", samples: int = 1, tag: str = "") -> dict:
    import json
    import random
    import re
    from pathlib import Path

    from rl.pool_build import is_correct

    pool = json.loads(Path("/out/pool.json").read_text())
    # Non-math only: NuminaMath already covers math, and the open question is whether the
    # null generalises beyond it.
    pool = [r for r in pool if r["gradable"] and r["source"] != "math_hard"]
    if source:
        # Restrict to one domain. Used to push a single domain's estimate: GPQA is the
        # only one where the length-matched effect survived, and it needs more data.
        pool = [r for r in pool if r["source"] == source]
    rng = random.Random(seed)
    rng.shuffle(pool)
    problems = pool[:attempt][shard::n_shards]
    print(f"shard {shard}: {len(problems)} non-math problems", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, tensor_parallel_size=2, dtype="bfloat16", seed=seed,
              gpu_memory_utilization=0.90, max_model_len=20480)
    tok = llm.get_tokenizer()
    # `n` samples per problem: the within-problem comparison needs the SAME problem to
    # yield both correct and incorrect traces, which only happens with repeated sampling.
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=MAX_TOKENS, seed=seed,
                        n=samples)

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": p["task"]}], tokenize=False,
        add_generation_prompt=True) for p in problems]
    gen = llm.generate(prompts, sp)
    # Flatten (problem, sample) -> one record each, keeping the problem id so the
    # within-problem analysis can group on it.
    outs = [[(c.text, c.finish_reason) for c in o.outputs] for o in gen]

    _BOX = re.compile(r"\\boxed\{([^{}]*)\}")

    def final_answer(text: str) -> str:
        """QwQ ends with the answer; take a boxed payload if present, else the last line."""
        m = list(_BOX.finditer(text))
        if m:
            return m[-1].group(1).strip()
        tail = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return tail[-1] if tail else ""

    recs, n_ok, n_trunc = [], 0, 0
    for p, cands in zip(problems, outs):
        for k, (o, finish) in enumerate(cands):
            truncated = (finish == "length")
            n_trunc += truncated
            ans = final_answer(o)
            ok = bool(is_correct(ans, p["answer"], p.get("options") or None))
            n_ok += ok
            recs.append({"pid": p["pid"], "sample": k, "source": p["source"],
                         "subtask": p["subtask"], "answer": p["answer"],
                         "extracted": ans[:200], "correct": ok, "truncated": truncated,
                         "finish_reason": finish, "response": o.strip()})

    Path(f"/out/qwqdom_{tag}{shard:03d}.json").write_text(json.dumps(recs))
    out.commit()
    print(f"shard {shard}: {len(recs)} traces, {n_ok} correct "
          f"({100*n_ok/max(len(recs),1):.1f}%), {n_trunc} truncated "
          f"({100*n_trunc/max(len(recs),1):.1f}%)", flush=True)
    return {"shard": shard, "n": len(recs), "n_correct": n_ok, "n_truncated": n_trunc}


@app.function(image=image, volumes={"/out": out}, timeout=30 * 60, env=ENV)
def assemble() -> dict:
    import json
    from collections import Counter
    from pathlib import Path

    rows = []
    for f in sorted(Path("/out").glob("qwqdom_*.json")):
        rows += json.loads(f.read_text())
    # With repeated sampling a pid appears once per sample, so dedupe on (pid, sample).
    seen, uniq = set(), []
    for r in rows:
        key = (r["pid"], r.get("sample", 0))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    Path("/out/qwq_domains.json").write_text(json.dumps(uniq))
    out.commit()

    by_src = Counter(r["source"] for r in uniq)
    acc = {s: round(sum(r["correct"] for r in uniq if r["source"] == s)
                    / max(sum(1 for r in uniq if r["source"] == s), 1), 3)
           for s in by_src}
    trunc = Counter(r["source"] for r in uniq if r.get("truncated"))
    # Accuracy among traces that actually FINISHED -- the only honest number, since a
    # truncated trace was never given the chance to be right.
    fin = [r for r in uniq if not r.get("truncated")]
    acc_fin = {s: round(sum(r["correct"] for r in fin if r["source"] == s)
                        / max(sum(1 for r in fin if r["source"] == s), 1), 3)
               for s in by_src}
    return {"n": len(uniq), "n_correct": sum(r["correct"] for r in uniq),
            "n_truncated": sum(1 for r in uniq if r.get("truncated")),
            "by_source": dict(by_src), "accuracy_by_source": acc,
            "truncated_by_source": dict(trunc),
            "accuracy_finished_only": acc_fin, "n_finished": len(fin)}


@app.local_entrypoint()
def main(attempt: int = 4000, shards: int = 4, seed: int = 0,
         source: str = "", samples: int = 1, tag: str = ""):
    results = list(generate_shard.starmap(
        [(i, shards, attempt, seed, source, samples, tag) for i in range(shards)]))
    tot = sum(r["n"] for r in results)
    ok = sum(r["n_correct"] for r in results)
    tr = sum(r.get("n_truncated", 0) for r in results)
    print(f"generated {tot} traces, {ok} correct ({100*ok/max(tot,1):.1f}%), "
          f"{tr} truncated ({100*tr/max(tot,1):.1f}%)")
    print(assemble.remote())
