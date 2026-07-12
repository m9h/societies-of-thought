"""Analysis: per-condition accuracy, and the contrast that actually matters.

The headline number is NOT "steering improved accuracy". A single feature moving
accuracy is unsurprising -- SAE steering perturbs the residual stream, and on a
task with headroom, perturbation alone can help. The paper's claim is specifically
that CONVERSATIONAL features help and comparable non-conversational ones do not.

So the estimand is a difference-in-differences:

    (steered - baseline | conversational candidate)
  - (steered - baseline | control matched on sparsity and magnitude)

If that is ~0, the "society of thought" mechanism is not doing the work, whatever
the raw steering effect looks like.

Uncertainty is bootstrapped by resampling PROBLEMS, not attempts: attempts on the
same problem are not independent, and treating them as such would shrink the
intervals dramatically and manufacture significance.

We also report parse rate and truncation, because a steering condition that makes
the model ramble can lose accuracy purely by running out of tokens -- a formatting
failure, not a reasoning failure, and the two would otherwise be indistinguishable.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/sweep.jsonl"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {args.results}")
    rng = np.random.default_rng(args.seed)

    for task in sorted({r["task"] for r in rows}):
        trs = [r for r in rows if r["task"] == task]
        print(f"\n{'='*78}\n{task}  (n={len(trs)} attempts)\n{'='*78}")

        base = [r for r in trs if r["feature"] == -1]
        if not base:
            print("  no baseline rows; cannot form contrasts")
            continue
        base_by_pid = _by_pid(base)
        b_acc, b_lo, b_hi = _boot(base_by_pid, rng, args.boot)
        print(f"\nbaseline (unsteered): acc={b_acc:.1%} [{b_lo:.1%}, {b_hi:.1%}]   "
              f"parsed={_rate(base,'parsed'):.1%}  truncated={_rate(base,'truncated'):.1%}")

        print(f"\n{'feature':>8} {'role':<17} {'alpha':>6} {'acc':>7} {'95% CI':>16} "
              f"{'d vs base':>10} {'parsed':>7} {'trunc':>6}")
        print("-" * 90)

        cells = defaultdict(list)
        for r in trs:
            if r["feature"] == -1:
                continue
            cells[(r["feature"], r["role"], r["alpha"])].append(r)

        deltas_by_role = defaultdict(list)
        for (fid, role, alpha), rs in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
            by_pid = _by_pid(rs)
            acc, lo, hi = _boot(by_pid, rng, args.boot)
            d, d_lo, d_hi = _boot_paired(by_pid, base_by_pid, rng, args.boot)
            star = "*" if (d_lo > 0 or d_hi < 0) else " "
            print(f"{fid:>8} {role:<17} {alpha:>6} {acc:>6.1%} "
                  f"[{lo:>5.1%},{hi:>6.1%}] {d:>+9.1%}{star} "
                  f"{_rate(rs,'parsed'):>6.1%} {_rate(rs,'truncated'):>5.1%}")
            if alpha is not None and alpha > 0:
                deltas_by_role[role].append((fid, d))

        # The estimand.
        cand = [d for role, v in deltas_by_role.items()
                if role in ("anchor", "candidate") for _, d in v]
        ctrl = [d for role, v in deltas_by_role.items()
                if role.startswith("control") for _, d in v]
        if cand and ctrl:
            did = float(np.mean(cand) - np.mean(ctrl))
            lo, hi = _boot_diff(cand, ctrl, rng, args.boot)
            print(f"\n  conversational candidates: mean d = {np.mean(cand):+.1%}  (n={len(cand)} features)")
            print(f"  matched controls:          mean d = {np.mean(ctrl):+.1%}  (n={len(ctrl)} features)")
            print(f"  DIFFERENCE-IN-DIFFERENCES: {did:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
            if lo <= 0 <= hi:
                print("  => CI spans zero: no evidence that CONVERSATIONAL features are special.")
                print("     Any raw steering effect here is not specific to the paper's mechanism.")
            else:
                print("  => conversational features differ from matched controls.")

        _markers(trs)


def _by_pid(rows) -> dict[str, list[int]]:
    d = defaultdict(list)
    for r in rows:
        d[r["pid"]].append(int(r["correct"]))
    return d


def _rate(rows, key) -> float:
    return float(np.mean([bool(r[key]) for r in rows])) if rows else float("nan")


def _boot(by_pid, rng, n_boot):
    pids = list(by_pid)
    means = np.array([np.mean(by_pid[p]) for p in pids])
    point = float(means.mean())
    idx = rng.integers(0, len(pids), size=(n_boot, len(pids)))
    dist = means[idx].mean(axis=1)
    return point, float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def _boot_paired(by_pid, base_by_pid, rng, n_boot):
    """Paired on problem: only problems present in both conditions."""
    pids = [p for p in by_pid if p in base_by_pid]
    if not pids:
        return float("nan"), float("nan"), float("nan")
    d = np.array([np.mean(by_pid[p]) - np.mean(base_by_pid[p]) for p in pids])
    idx = rng.integers(0, len(pids), size=(n_boot, len(pids)))
    dist = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def _boot_diff(a, b, rng, n_boot):
    a, b = np.array(a), np.array(b)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    dist = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def _markers(rows) -> None:
    """Did steering actually make the traces more dialogic?

    This is the mechanism check. If accuracy moves but these do not, the paper's
    story does not explain the movement -- and if these move but accuracy does not,
    the intervention worked and the mechanism simply does not pay off here.
    """
    pos = [r for r in rows if (r["alpha"] or 0) > 0 and r["role"] in ("anchor", "candidate")]
    base = [r for r in rows if r["feature"] == -1]
    if not pos or not base:
        return
    keys = list(base[0]["markers"])
    print("\n  trace markers (per trace, baseline -> steered+):")
    for k in keys:
        b = np.mean([r["markers"][k] for r in base])
        p = np.mean([r["markers"][k] for r in pos])
        arrow = "up" if p > b else ("down" if p < b else "flat")
        print(f"    {k:22s} {b:6.2f} -> {p:6.2f}  ({arrow})")
    bt = np.mean([r["n_tokens"] for r in base])
    pt = np.mean([r["n_tokens"] for r in pos])
    print(f"    {'trace length (tokens)':22s} {bt:6.0f} -> {pt:6.0f}")


if __name__ == "__main__":
    main()
