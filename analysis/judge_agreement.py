"""Chance-corrected agreement between judges (and between a judge and a human).

Exact match is the number everyone quotes and the one that says least: on a label that
is nearly constant (our n_personas is 1 on every trace) two judges that both say 1
agree 100% while measuring nothing. Norman, Rivera & Hughes (2606.19544) report 33-41
point deflation from exact match to Cohen's kappa across 541k judgments. So every
comparison here reports exact match, kappa (unweighted and quadratic), ICC(3,1) and
Spearman's rho side by side, and returns NaN -- not 1.0 -- when there is no variance
to correct for.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np

from rl.judge import BEHAVIOURS

FIELDS = (*BEHAVIOURS, "n_personas")


def cohen_kappa(a, b, weights=None) -> float:
    """Cohen's kappa; `weights` in {None, 'linear', 'quadratic'} uses the category
    VALUES (these are counts) for distance. NaN when chance disagreement is zero."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("ratings must be aligned")
    cats = np.array(sorted(set(a.tolist()) | set(b.tolist())))
    k = len(cats)
    if k < 2:
        return math.nan
    pos = {c: i for i, c in enumerate(cats.tolist())}
    obs = np.zeros((k, k))
    for x, y in zip(a.tolist(), b.tolist()):
        obs[pos[x], pos[y]] += 1
    n = obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    d = np.abs(cats[:, None] - cats[None, :]).astype(float)
    span = float(cats.max() - cats.min())
    if weights is None:
        w = (d > 0).astype(float)
    elif weights == "linear":
        w = d / span
    elif weights == "quadratic":
        w = (d / span) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denom = (w * exp).sum()
    if denom == 0:
        return math.nan
    return float(1.0 - (w * obs).sum() / denom)


def icc31(ratings) -> float:
    """ICC(3,1), two-way mixed, single rater, consistency (Shrout & Fleiss 1979).
    `ratings` is n targets x k raters. NaN when there is no between-target variance."""
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return math.nan
    grand = x.mean()
    rm = x.mean(1, keepdims=True)
    cm = x.mean(0, keepdims=True)
    msr = k * ((rm - grand) ** 2).sum() / (n - 1)
    mse = ((x - rm - cm + grand) ** 2).sum() / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse
    if denom == 0:
        return math.nan
    return float((msr - mse) / denom)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = spearmanr(a, b).statistic
    return float(r) if r is not None else math.nan


def pairwise_agreement(verdicts: dict, fields=FIELDS) -> list[dict]:
    """Every pair of judges, every field. `verdicts` maps judge tag -> list of verdict
    dicts, aligned by index (the same trace at the same position for every judge)."""
    tags = list(verdicts)
    lengths = {t: len(verdicts[t]) for t in tags}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"verdict lists are not aligned: {lengths}")
    rows = []
    for ta, tb in itertools.combinations(tags, 2):
        for f in fields:
            a = [v[f] for v in verdicts[ta]]
            b = [v[f] for v in verdicts[tb]]
            rows.append({
                "pair": (ta, tb), "field": f, "n": len(a),
                "exact": float(np.mean(np.array(a) == np.array(b))) if a else math.nan,
                "kappa": cohen_kappa(a, b),
                "kappa_quadratic": cohen_kappa(a, b, "quadratic"),
                "icc31": icc31(np.column_stack([a, b])) if a else math.nan,
                "spearman": _spearman(a, b),
                "mean_a": float(np.mean(a)) if a else math.nan,
                "mean_b": float(np.mean(b)) if b else math.nan,
            })
    return rows


def bin_curves(verdicts: dict, fields=FIELDS) -> dict:
    """Per-judge, per-bin mean of every field: does the second judge see the same CURVE,
    not just the same traces?"""
    out = {}
    for tag, vs in verdicts.items():
        by = {}
        for v in vs:
            by.setdefault(v["bin"], []).append(v)
        rows = []
        for b in sorted(by):
            row = {"bin": b, "n": len(by[b])}
            for k in ("bin_lo", "bin_hi"):
                if k in by[b][0]:
                    row[k] = by[b][0][k]
            for f in fields:
                row[f] = float(np.mean([v[f] for v in by[b]]))
            rows.append(row)
        out[tag] = rows
    return out


def format_agreement(rows) -> str:
    def f(x):
        return "   nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:6.2f}"
    lines = [f"{'pair':<44}{'field':<26}{'n':>5}{'exact':>7}{'kappa':>7}{'kappa_q':>8}"
             f"{'ICC31':>7}{'rho':>7}{'mean_a':>8}{'mean_b':>8}"]
    for r in rows:
        lines.append(f"{' vs '.join(r['pair']):<44}{r['field']:<26}{r['n']:>5}"
                     f"{f(r['exact'])}{f(r['kappa'])} {f(r['kappa_quadratic'])}"
                     f"{f(r['icc31'])}{f(r['spearman'])}  {f(r['mean_a'])}  {f(r['mean_b'])}")
    return "\n".join(lines)
