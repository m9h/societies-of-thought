"""Judge-human agreement: a blind annotation sheet and its scorer.

Norman et al. (2606.19544) make agreement with human raters the validity anchor for an
LLM judge; the paper reports ~.76 against humans for its judge and we have never
measured ours. The sheet shows a person the traces in shuffled order with no training
step and no verdict attached, against the paper's own four definitions, and the
ratings are scored with the same chance-corrected statistics used judge-to-judge.

    python -m analysis.human_annotation make --judged results/emergence/cross_judge.json \
        --log results/emergence/fig4_s0_final.log --out results/emergence/human
    python -m analysis.human_annotation score --ratings results/emergence/human/ratings.csv \
        --key results/emergence/human/answer_key.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from analysis.judge_agreement import FIELDS, format_agreement, pairwise_agreement
from rl.judge import BEHAVIOURS, PAPER_BEHAVIOUR_TEMPLATE, PAPER_PERSONA_TEMPLATE


def select_traces(judged, n: int = 50, seed: int = 1) -> list[dict]:
    """Equal quota per bin (±1), sampled within bin, then shuffled so the order carries
    no information about training progress."""
    rng = random.Random(seed)
    by = {}
    for r in judged:
        by.setdefault(r["bin"], []).append(r)
    bins = sorted(by)
    base, extra = divmod(n, len(bins))
    extra_bins = set(rng.sample(bins, extra))
    out = []
    for b in bins:
        q = base + (1 if b in extra_bins else 0)
        pool = by[b]
        out.extend(pool if len(pool) <= q else rng.sample(pool, q))
    rng.shuffle(out)
    return out


_INSTRUCTIONS = """# Annotation sheet

For each trace, count how many times behaviours corresponding to each of the four
dimensions appear, using the definitions below (the paper's, verbatim). For each category,
count the number of distinct times the behaviour occurs; if none are present, use 0. Then
report **n personas** using the paper's persona definition, also below. Rate only what is
in the trace. Enter your counts in `ratings_template.csv`.

{defs}

The traces are in random order. Do not look at `answer_key.json` until you have finished.
"""


def write_sheet(selected, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(selected))))
    ids = [f"T{i + 1:0{width}d}" for i in range(len(selected))]

    beh = PAPER_BEHAVIOUR_TEMPLATE.split("Use the following definitions:")[1]
    beh = beh.split("For each category")[0].strip()
    per = PAPER_PERSONA_TEMPLATE.split("For each distinct perspective")[0].strip()
    defs = ("## Behaviour definitions\n\n" + beh
            + "\n\n## Persona definition (n personas = number of distinct perspectives)\n\n" + per)
    parts = [_INSTRUCTIONS.format(defs=defs)]
    key = {}
    for rid, r in zip(ids, selected):
        parts.append(f"\n---\n\n## {rid}\n\n```text\n{r['response'].rstrip()}\n```\n")
        key[rid] = {"step": r["step"], "bin": r["bin"],
                    "sha256": hashlib.sha256(r["response"].encode()).hexdigest(),
                    **{f: r[f] for f in FIELDS}}
    sheet = out / "sheet.md"
    sheet.write_text("".join(parts))
    key_path = out / "answer_key.json"
    key_path.write_text(json.dumps(key, indent=1))
    tmpl = out / "ratings_template.csv"
    with tmpl.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        w.writeheader()
        for rid in ids:
            w.writerow({"id": rid, **{f: "" for f in FIELDS}})
    return {"sheet": sheet, "key": key_path, "template": tmpl}


def score_ratings(ratings_csv, key_json) -> dict:
    key = json.loads(Path(key_json).read_text())
    human, judge, blank = [], [], 0
    with Path(ratings_csv).open() as fh:
        for row in csv.DictReader(fh):
            if row["id"] not in key:
                raise ValueError(f"unknown id {row['id']!r}")
            vals = [row[f] for f in FIELDS]
            if any(v is None or v.strip() == "" for v in vals):
                blank += 1
                continue
            human.append({f: int(row[f]) for f in FIELDS})      # non-int raises, not 0
            judge.append({f: key[row["id"]][f] for f in FIELDS})
    rows = pairwise_agreement({"human": human, "judge": judge}) if human else []
    return {"n_rated": len(human), "n_blank": blank, "agreement": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--judged", type=Path, required=True, help="cross_judge.json")
    m.add_argument("--judge", default=None, help="which judge's verdicts form the key")
    m.add_argument("--log", type=Path, required=True)
    m.add_argument("--n", type=int, default=50)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", type=Path, default=Path("results/emergence/human"))
    s = sub.add_parser("score")
    s.add_argument("--ratings", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True)
    a = ap.parse_args()

    if a.cmd == "make":
        from analysis.emergence import parse_rollouts
        from analysis.judge_emergence import stratified_sample
        d = json.loads(a.judged.read_text())
        tag = a.judge or d["primary"]
        rollouts = parse_rollouts(a.log.read_text(errors="ignore"))
        if d.get("source", "all") != "all":          # the judged file's series, not the whole log
            rollouts = [r for r in rollouts if r.get("source", "train") == d["source"]]
        sample = stratified_sample(rollouts, d["bins"], d["per_bin"], seed=d["seed"])
        for v in d["verdicts"][tag]:                 # the resample must be the judged sample
            assert sample[v["idx"]]["step"] == v["step"], "sample mismatch: wrong log or series"
        judged = [{**sample[v["idx"]], **{f: v[f] for f in FIELDS}, "bin": v["bin"]}
                  for v in d["verdicts"][tag]]
        sel = select_traces(judged, n=a.n, seed=a.seed)
        paths = write_sheet(sel, a.out)
        print(f"{len(sel)} traces from judge {tag}; wrote", *paths.values())
    else:
        rep = score_ratings(a.ratings, a.key)
        print(f"rated {rep['n_rated']}, blank {rep['n_blank']}")
        print(format_agreement(rep["agreement"]))


if __name__ == "__main__":
    main()
