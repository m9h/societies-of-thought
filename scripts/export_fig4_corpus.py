"""Export the Fig. 4 corpus: every training rollout of the faithful run, plus verdicts.

The authors ship no code and no traces. A complete 250-step PPO run on their exact
configuration, with every rollout parsed and a judged subsample, is the artifact another
replicator needs. Verdicts join to rollouts by the same content hash the judge cache uses
(`rl.judge.cache_key`), so the join is verifiable from the published files alone.

    python scripts/export_fig4_corpus.py --repo mhough/sot-fig4-countdown-ppo-rollouts [--public]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from analysis.emergence import parse_rollouts
from rl.judge import BEHAVIOURS, PROMPT_VERSION, build_prompt, cache_key


def build_rows(log_text: str) -> list[dict]:
    rows = []
    for i, r in enumerate(parse_rollouts(log_text)):
        rows.append({"idx": i, "step": r["step"], "prompt": r["prompt"],
                     "response": r["response"], "words": len(r["response"].split()),
                     "sha256": hashlib.sha256(r["response"].encode()).hexdigest()})
    return rows


def _load_cache_full(path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        try:
            rec = json.loads(line)
            out[rec["key"]] = rec
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def join_verdicts(rows, cache_path, models) -> list[dict]:
    cache = _load_cache_full(cache_path)
    out = []
    for r in rows:
        for m in models:
            rec = cache.get(cache_key(r["response"], m))
            if rec is None:
                continue
            out.append({"idx": r["idx"], "step": r["step"], "sha256": r["sha256"],
                        "judge_model": m, "prompt_version": rec.get("prompt_version", PROMPT_VERSION),
                        **{k: rec["verdict"][k] for k in (*BEHAVIOURS, "n_personas")}})
    return out


def write_corpus(out_dir, rows, verdicts, prompt_text: str, card: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rollouts.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with (out / "judgments.jsonl").open("w") as fh:
        for v in verdicts:
            fh.write(json.dumps(v) + "\n")
    (out / f"judge_prompt_{PROMPT_VERSION}.txt").write_text(prompt_text)
    (out / "README.md").write_text(card)
    return {"rollouts": len(rows), "judgments": len(verdicts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("results/emergence/fig4_s0_final.log"))
    ap.add_argument("--cache", type=Path, default=Path("results/emergence/judge_cache.jsonl"))
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"])
    ap.add_argument("--card", type=Path, default=Path("docs/dataset_card_fig4_corpus.md"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true", help="publish publicly (default private)")
    ap.add_argument("--local-only", type=Path, default=None, help="write here, do not upload")
    a = ap.parse_args()

    rows = build_rows(a.log.read_text(errors="ignore"))
    verdicts = join_verdicts(rows, a.cache, a.models)
    prompt_text = build_prompt("<TRACE>")
    per_model = {}
    for v in verdicts:
        per_model[v["judge_model"]] = per_model.get(v["judge_model"], 0) + 1
    print("rollouts:", len(rows), "judgments:", per_model)

    if a.local_only:
        print(write_corpus(a.local_only, rows, verdicts, prompt_text, a.card.read_text()))
        return
    with tempfile.TemporaryDirectory() as td:
        counts = write_corpus(td, rows, verdicts, prompt_text, a.card.read_text())
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))      # None -> local login
        api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)
        api.upload_folder(folder_path=td, repo_id=a.repo, repo_type="dataset",
                          commit_message=f"Fig. 4 corpus: {counts}")
    print(f"https://huggingface.co/datasets/{a.repo} ({'public' if a.public else 'private'})")


if __name__ == "__main__":
    main()
