"""Export the priming corpus as a paired HF dataset (one row = one problem, both arms).

The paper ships no code or data, so the reconstructed corpus is the artifact other
replicators actually need. Pairing the arms in a single row is the point: the design
invariant is that both conditions cover identical problems with identical answers.

    HF_TOKEN=... python scripts/export_hf_dataset.py --repo mhough/sot-priming-traces-dialogue-monologue

Creates the repo PRIVATE by default. Publishing is a deliberate, separate act.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def build(src: Path, out: Path) -> dict[str, int]:
    counts = {}
    for split, name in (("train", "train"), ("val", "val")):
        dia = {r["pid"]: r for r in json.loads((src / f"dialogue_{split}.json").read_text())}
        mon = {r["pid"]: r for r in json.loads((src / f"monologue_{split}.json").read_text())}
        if set(dia) != set(mon):
            raise SystemExit(f"{split}: arms cover different problems -- invariant broken")
        rows = [{"pid": p, "source": dia[p]["source"], "subtask": dia[p]["subtask"],
                 "task": dia[p]["task"], "answer": dia[p]["answer"],
                 "dialogue": dia[p]["dialogue"], "monologue": mon[p]["monologue"]}
                for p in sorted(dia)]
        with (out / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        counts[split] = len(rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("rl/data/ood"))
    ap.add_argument("--card", type=Path, default=Path("docs/dataset_card_priming_traces.md"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true", help="publish publicly (default private)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        counts = build(args.src, out)
        (out / "README.md").write_text(args.card.read_text())
        print("paired rows:", counts)

        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.repo, repo_type="dataset",
                        private=not args.public, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="dataset",
                          commit_message="paired dialogue/monologue priming traces")
    print(f"https://huggingface.co/datasets/{args.repo} "
          f"({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
