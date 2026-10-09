"""Re-score the published Fig. 4 sample with other judges and report agreement.

The published numbers are one stratified sample (8 bins x 32, seed 0) of the FINAL log,
scored by one judge. A second judge replicates that result only if it scores the SAME
traces, so this runner regenerates the sample and then insists that every primary-judge
verdict come from the cache: a single cache miss means the sample is not the published
one, and the run stops rather than quietly comparing two different samples.

    python -m analysis.cross_judge --log results/emergence/fig4_s0_final.log \
        --primary claude-sonnet-5 --judges claude-opus-5 claude-haiku-4-5-20251001 \
        gemini/gemini-2.5-pro --out results/emergence/cross_judge.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.emergence import parse_rollouts
from analysis.judge_agreement import bin_curves, format_agreement, pairwise_agreement
from analysis.judge_emergence import stratified_sample
from rl.judge import _load_cache, cache_key, family_of, judge_trace, make_backend


def run_cross_judge(sample, backends: dict, cache, primary: str,
                    progress=None) -> dict:
    hits = _load_cache(cache)
    verdicts, failures = {}, {}
    for tag, backend in backends.items():
        vs, fails = [], 0
        for i, r in enumerate(sample):
            if tag == primary:
                key = cache_key(r["response"], tag)
                if key not in hits:
                    raise RuntimeError(
                        f"trace {i} (step {r['step']}) is not in cache for primary judge "
                        f"{tag!r}: this sample is not the published one")
                v = hits[key]
            else:
                try:
                    v = judge_trace(r["response"], backend, cache=cache, model=tag)
                except Exception as e:                     # parse failure or transport
                    fails += 1
                    if progress:
                        progress(f"  [{tag} {i}] failed: {type(e).__name__}: {str(e)[:120]}")
                    continue
            vs.append({"idx": i, **{k: r[k] for k in ("step", "bin", "bin_lo", "bin_hi")},
                       "words": len(r["response"].split()), **v})
            if progress and i % 25 == 0 and tag != primary:
                progress(f"  {tag}: {i}/{len(sample)}")
        verdicts[tag], failures[tag] = vs, fails

    agreement = []
    tags = list(backends)
    for a_i in range(len(tags)):
        for b_i in range(a_i + 1, len(tags)):
            ta, tb = tags[a_i], tags[b_i]
            ia = {v["idx"]: v for v in verdicts[ta]}
            ib = {v["idx"]: v for v in verdicts[tb]}
            common = sorted(set(ia) & set(ib))
            agreement.extend(pairwise_agreement(
                {ta: [ia[i] for i in common], tb: [ib[i] for i in common]}))

    return {"n": len(sample), "judges": tags, "primary": primary,
            "families": {t: family_of(t) for t in tags},
            "verdicts": verdicts, "failures": failures,
            "agreement": agreement, "curves": bin_curves(verdicts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--per-bin", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primary", default="claude-sonnet-5")
    ap.add_argument("--judges", nargs="+", required=True,
                    help="provider/model specs; bare names are Anthropic")
    ap.add_argument("--cache", type=Path, default=Path("results/emergence/judge_cache.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/emergence/cross_judge.json"))
    a = ap.parse_args()

    rollouts = parse_rollouts(a.log.read_text(errors="ignore"))
    sample = stratified_sample(rollouts, a.bins, a.per_bin, seed=a.seed)
    print(f"{len(rollouts)} rollouts -> {len(sample)} ({a.bins} x {a.per_bin}, seed {a.seed})")

    backends = {a.primary: None}
    for spec in a.judges:
        call, tag, fam = make_backend(spec)
        backends[tag] = call
        print(f"judge {tag} ({fam})")

    out = run_cross_judge(sample, backends, cache=a.cache, primary=a.primary,
                          progress=lambda s: print(s, flush=True))
    out.update(log=a.log.name, bins=a.bins, per_bin=a.per_bin, seed=a.seed)

    print("\nfailures:", out["failures"])
    print("\n" + format_agreement(out["agreement"]))
    print("\nper-bin means (n_personas | conflict | question_answering | shift | reconciliation)")
    for tag, rows in out["curves"].items():
        print(f"  {tag}")
        for r in rows:
            print(f"    {r.get('bin_lo', '?'):>4}-{r.get('bin_hi', '?'):<4} n={r['n']:>3} "
                  f"{r['n_personas']:.2f} | {r['conflict_of_perspectives']:.3f} | "
                  f"{r['question_answering']:.3f} | {r['perspective_shift']:.3f} | "
                  f"{r['reconciliation']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
