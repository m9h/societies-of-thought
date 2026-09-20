"""Score an RL run's traces with the paper's LLM judge, binned by training step.

The judge-free marker proxy in `analysis.emergence` is cheap and reproducible, but on the
claimA log it turned out to be measuring one repeated template rather than a behaviour
(see `pattern_dominance`). Fig. 4b is an LLM-judge result, so answering it needs the
judge.

Sampling is stratified by training step so every bin gets the same number of traces: the
number of rollouts verl happens to print per step varies, and judging "whatever is in the
log" would weight bins by log verbosity rather than by training progress.

    python -m analysis.judge_emergence --log results/rl_ab/tz_train_claimA.log \
        --bins 10 --per-bin 20 --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.emergence import BEHAVIOUR_FNS, parse_rollouts, pattern_dominance
from rl.judge import BEHAVIOURS, anthropic_backend, judge_trace


def stratified_sample(rollouts, n_bins: int, per_bin: int, seed: int = 0):
    """Equal traces per training-progress bin, so bins are comparable."""
    steps = sorted({r["step"] for r in rollouts})
    edges = np.array_split(np.array(steps), n_bins)
    rng = random.Random(seed)
    out = []
    for i, edge in enumerate(edges):
        members = set(edge.tolist())
        pool = [r for r in rollouts if r["step"] in members]
        take = pool if len(pool) <= per_bin else rng.sample(pool, per_bin)
        for r in take:
            out.append({**r, "bin": i, "bin_lo": int(edge[0]), "bin_hi": int(edge[-1])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--per-bin", type=int, default=20)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--cache", type=Path, default=Path("results/emergence/judge_cache.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/emergence/judged.json"))
    a = ap.parse_args()

    rollouts = parse_rollouts(a.log.read_text(errors="ignore"))
    sample = stratified_sample(rollouts, a.bins, a.per_bin)
    print(f"{len(rollouts)} rollouts -> {len(sample)} judged "
          f"({a.bins} bins x {a.per_bin}), judge={a.model}")

    backend = anthropic_backend(a.model)
    judged, failures = [], 0
    for i, r in enumerate(sample):
        try:
            v = judge_trace(r["response"], backend=backend, cache=a.cache, model=a.model)
        except (ValueError, Exception) as e:            # a refusal or a transport error
            failures += 1
            print(f"  [{i}] judge failed: {type(e).__name__}: {e}", flush=True)
            continue
        judged.append({**{k: r[k] for k in ("step", "bin", "bin_lo", "bin_hi")},
                       "words": len(r["response"].split()), **v})
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    # A parse failure is never a zero, so a bin that lost traces must say so.
    by = defaultdict(list)
    for j in judged:
        by[j["bin"]].append(j)

    print(f"\njudged {len(judged)}, failed {failures}")
    hdr = f"{'steps':>12}{'n':>5}{'words':>8}" + "".join(f"{b[:9]:>11}" for b in BEHAVIOURS) + f"{'personas':>10}"
    print("\n" + hdr)
    print(" " * 25 + "  ---------- instances per 100 words ----------")
    rows = []
    for b in sorted(by):
        js = by[b]
        tw = sum(j["words"] for j in js)
        row = {"bin": b, "lo": js[0]["bin_lo"], "hi": js[0]["bin_hi"], "n": len(js),
               "words": tw / len(js),
               "n_personas": sum(j["n_personas"] for j in js) / len(js)}
        for beh in BEHAVIOURS:
            row[f"{beh}_rate"] = 100.0 * sum(j[beh] for j in js) / tw
        rows.append(row)
        print(f"{f'{row['lo']}-{row['hi']}':>12}{row['n']:>5}{row['words']:>8.0f}"
              + "".join(f"{row[f'{b}_rate']:>11.3f}" for b in BEHAVIOURS)
              + f"{row['n_personas']:>10.2f}")

    # The proxy on exactly the same traces, so judge and proxy are directly comparable.
    print("\nJUDGE-FREE PROXY on the same sampled traces:")
    print(hdr)
    proxy_rows = []
    for b in sorted(by):
        texts = [r["response"] for r in sample if r["bin"] == b]
        tw = sum(max(len(t.split()), 1) for t in texts)
        row = {"bin": b, "n": len(texts), "words": tw / len(texts)}
        for beh, fn in BEHAVIOUR_FNS.items():
            row[f"{beh}_rate"] = 100.0 * sum(fn(t) for t in texts) / tw
        dom = pattern_dominance(texts, "conflict_of_perspectives")
        row["conflict_dominance"] = dom
        proxy_rows.append(row)
        lo, hi = by[b][0]["bin_lo"], by[b][0]["bin_hi"]
        print(f"{f'{lo}-{hi}':>12}{row['n']:>5}{row['words']:>8.0f}"
              + "".join(f"{row[f'{x}_rate']:>11.3f}" for x in BEHAVIOURS)
              + f"{'':>10}   top={dom['top']!r} {dom['top_share']:.0%}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"log": a.log.name, "judge_model": a.model, "bins": a.bins,
         "per_bin": a.per_bin, "n_judged": len(judged), "n_failed": failures,
         "judge": rows, "proxy": proxy_rows, "traces": judged}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
