"""Generate the paper's SFT priming traces on Modal, with the paper's teacher.

WHY MODAL AND NOT A POD
-----------------------
This job was first attempted on hand-rolled RunPod pods and burned ~$7 across three
launches without ever running: a guessed image tag that did not exist, then a missing
PUBLIC_KEY so sshd never started. Meanwhile the J-space work next door
(jacobian-lens/modal_olmo_ladder.py) has been running multi-GPU jobs on Modal without
any of that. This app copies that pattern: no SSH, no image tags, no provisioning
polling, and per-second billing so an idle GPU costs nothing.

WHAT IT PRODUCES
----------------
The priming corpus for Claim B (C5), built to `rl.paper_spec`:

  * teacher  = Qwen/Qwen2.5-32B-Instruct  (the paper's; NOT the 72B we substituted)
  * problems = the 8,262-problem out-of-domain pool (BBH/GPQA/MATH-Hard/MMLU-Pro/
               MUSR/IFEval) -- NOT Countdown, which is the RL task
  * prompts  = verbatim from Supplementary Methods
  * kept     = only problems where BOTH arms reach the correct answer, so the two
               conditions cover identical problems (the paper's stated invariant)

SHARDING
--------
Generation is embarrassingly parallel over problems, so it fans out over N workers and
the shards are concatenated. Each shard writes its own file before returning, so a
worker dying costs one shard rather than the run -- the failure that cost this project
a full PPO run when a pod filled its disk mid-save.

    modal run modal_teacher.py --attempt 2500 --shards 4
"""
from __future__ import annotations

import modal

APP = "sot-teacher"
app = modal.App(APP)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.6", "vllm>=0.8", "transformers>=4.51", "accelerate",
                 "safetensors", "datasets", "numpy", "huggingface_hub", "hf_transfer")
    # Ship `rl/` directly rather than `pip install git+...`: this repo's pyproject
    # declares the jacobian-lens package (name="jlens", packages=["jlens"]) and there
    # is no jlens/ here, so a git install would not make `rl` importable. Caught in
    # preflight rather than in a paid container.
    .add_local_dir("rl", remote_path="/root/rl")
)

cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("sot-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false",
       "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONPATH": "/root"}

# 32B bf16 is ~64GB of weights: two 80GB cards give room for weights + KV cache.
GPU = "A100-80GB:2"


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=90 * 60, env=ENV, retries=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def generate_shard(shard: int, n_shards: int, attempt: int, seed: int = 0,
                   offset: int = 0, tag: str = "a") -> dict:
    """Generate dialogue+monologue for this shard's slice and keep matched-correct."""
    import json
    import random
    from pathlib import Path

    from rl import paper_spec as S
    from rl.pool_build import is_correct
    from rl.generate_sft_ood import (dialogue_prompt, monologue_prompt,
                                     extract_dialogue_answer, extract_monologue_answer,
                                     well_formed_dialogue)

    pool = json.loads(Path("/out/pool.json").read_text())
    pool = [r for r in pool if r["gradable"]]
    rng = random.Random(seed)
    rng.shuffle(pool)
    # `offset` lets a later run cover only problems the first run never attempted, so a
    # top-up costs the remainder rather than a full regeneration. The shuffle is seeded,
    # so pool[offset:offset+attempt] is a stable, disjoint slice across runs.
    problems = pool[offset:offset + attempt][shard::n_shards]
    print(f"shard {tag}{shard}/{n_shards}: {len(problems)} problems "
          f"(offset {offset})", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=S.TEACHER_MODEL, tensor_parallel_size=2, dtype="bfloat16",
              seed=seed, gpu_memory_utilization=0.90, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1536, seed=seed)

    def chat(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        return [o.outputs[0].text for o in llm.generate(texts, sp)]

    ns = [rng.choice(S.DIALOGUE_N_THINKERS) for _ in problems]
    dia = chat([dialogue_prompt(p["task"], n) for p, n in zip(problems, ns)])
    mon = chat([monologue_prompt(p["task"]) for p in problems])

    # Count WHY instances are dropped. A silently over-strict grader would bias which
    # problems reach the priming set, so the rejection profile is reported, not assumed.
    matched, why = [], {"bad_dialogue_tags": 0, "no_dialogue_answer": 0,
                        "no_monologue_answer": 0, "dialogue_wrong": 0,
                        "monologue_wrong": 0, "both_wrong": 0, "kept": 0}
    for p, d, m in zip(problems, dia, mon):
        if not well_formed_dialogue(d):
            why["bad_dialogue_tags"] += 1
            continue
        da, ma = extract_dialogue_answer(d), extract_monologue_answer(m)
        if da is None:
            why["no_dialogue_answer"] += 1
            continue
        if ma is None:
            why["no_monologue_answer"] += 1
            continue
        opts = p.get("options") or None
        dok = is_correct(da, p["answer"], opts)
        mok = is_correct(ma, p["answer"], opts)
        if not (dok and mok):
            why["both_wrong" if not (dok or mok) else
                ("monologue_wrong" if dok else "dialogue_wrong")] += 1
            continue
        why["kept"] += 1
        matched.append({**{k: p[k] for k in ("pid", "source", "subtask", "task",
                                             "answer")},
                        "dialogue": d.strip(), "monologue": m.strip()})
    print(f"shard {tag}{shard} rejection profile: {why}", flush=True)

    # Write before returning: a dying worker must cost one shard, not the run.
    Path(f"/out/shard_{tag}{shard:03d}.json").write_text(json.dumps(matched))
    out.commit()
    print(f"shard {tag}{shard}: {len(matched)}/{len(problems)} matched-correct", flush=True)
    return {"shard": f"{tag}{shard}", "attempted": len(problems),
            "matched": len(matched), "why": why}


@app.function(image=image, volumes={"/out": out}, timeout=20 * 60, env=ENV)
def assemble(seed: int = 0) -> dict:
    """Concatenate shards into the paper's 500/100 split, matched across arms."""
    import json
    import random
    from pathlib import Path

    from rl import paper_spec as S

    rows = []
    for f in sorted(Path("/out").glob("shard_*.json")):
        rows += json.loads(f.read_text())
    # A pid must appear once: shards are disjoint, but be explicit about the invariant.
    seen, uniq = set(), []
    for r in rows:
        if r["pid"] not in seen:
            seen.add(r["pid"])
            uniq.append(r)

    need = S.SFT_N_TRAIN + S.SFT_N_VAL
    if len(uniq) < need:
        return {"ok": False, "matched": len(uniq), "need": need,
                "hint": "increase --attempt and rerun"}

    random.Random(seed).shuffle(uniq)
    keep = uniq[:need]
    d = Path("/out/ood")
    d.mkdir(parents=True, exist_ok=True)
    for name, rs in (("train", keep[:S.SFT_N_TRAIN]), ("val", keep[S.SFT_N_TRAIN:])):
        for arm in ("dialogue", "monologue"):
            recs = [{"pid": r["pid"], "source": r["source"], "subtask": r["subtask"],
                     "task": r["task"], "answer": r["answer"], arm: r[arm]} for r in rs]
            (d / f"{arm}_{name}.json").write_text(json.dumps(recs, indent=1))
    out.commit()

    comp: dict[str, int] = {}
    for r in keep:
        comp[r["source"]] = comp.get(r["source"], 0) + 1
    dw = sorted(len(r["dialogue"].split()) for r in keep)[len(keep) // 2]
    mw = sorted(len(r["monologue"].split()) for r in keep)[len(keep) // 2]
    return {"ok": True, "matched": len(uniq), "kept": len(keep), "composition": comp,
            "median_words": {"dialogue": dw, "monologue": mw,
                             "ratio": round(dw / max(mw, 1), 2)}}


@app.function(image=image, volumes={"/out": out}, timeout=30 * 60, env=ENV)
def upload_pool(pool_bytes: bytes) -> int:
    """Ship the locally-built pool (CPU-only work) into the volume."""
    from pathlib import Path

    Path("/out/pool.json").write_bytes(pool_bytes)
    out.commit()
    import json
    return len(json.loads(pool_bytes))


@app.local_entrypoint()
def main(attempt: int = 2500, shards: int = 4, seed: int = 0,
         pool: str = "rl/data/pool.json", offset: int = 0, tag: str = "a"):
    from pathlib import Path

    n = upload_pool.remote(Path(pool).read_bytes())
    print(f"pool uploaded: {n} problems")

    results = list(generate_shard.starmap(
        [(i, shards, attempt, seed, offset, tag) for i in range(shards)]))
    tot_a = sum(r["attempted"] for r in results)
    tot_m = sum(r["matched"] for r in results)
    print(f"generated: {tot_m}/{tot_a} matched-correct "
          f"({100 * tot_m / max(tot_a, 1):.1f}%)")

    print(assemble.remote(seed))
