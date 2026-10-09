"""Render a cross_judge JSON as the markdown tables FINDINGS quotes.

    python -m analysis.cross_judge_report results/emergence/cross_judge_train.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

from rl.judge import BEHAVIOURS

SHORT = {"question_answering": "Q&A", "perspective_shift": "shift",
         "conflict_of_perspectives": "conflict", "reconciliation": "reconc.",
         "n_personas": "personas"}


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def render(out: dict) -> str:
    lines = [f"**{out.get('source', '?')} series, prompt `{out.get('prompt_version', '?')}`, "
             f"n = {out['n']} traces per judge.** Failures: "
             + ", ".join(f"{k}: {v}" for k, v in out["failures"].items()) + "."]
    for tag in out["judges"]:
        vs = out["verdicts"].get(tag, [])
        if not vs:
            continue
        gt1 = defaultdict(list)
        for v in vs:
            gt1[v["bin"]].append(v["n_personas"] > 1)
        lines.append(f"\n*{tag}* — mean count per trace, by training-step bin:\n")
        lines.append("| steps | n | Q&A | shift | conflict | reconc. | personas | >1 persona | words |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        words = defaultdict(list)
        for v in vs:
            words[v["bin"]].append(v["words"])
        for r in out["curves"][tag]:
            b = r["bin"]
            lines.append(f"| {r.get('bin_lo', '?')}–{r.get('bin_hi', '?')} | {r['n']} | "
                         + " | ".join(_f(r[k]) for k in BEHAVIOURS)
                         + f" | {_f(r['n_personas'])} | {_f(sum(gt1[b]) / max(len(gt1[b]), 1))} "
                         f"| {_f(sum(words[b]) / max(len(words[b]), 1), 0)} |")
    if out.get("agreement"):
        lines.append("\nAgreement on the same traces (chance-corrected beside exact match):\n")
        lines.append("| pair | field | n | exact | κ | κ_quad | ICC(3,1) | ρ | mean A | mean B |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in out["agreement"]:
            a, b = r["pair"]
            lines.append(f"| {a} vs {b} | {SHORT.get(r['field'], r['field'])} | {r['n']} | "
                         f"{_f(r['exact'])} | {_f(r['kappa'])} | {_f(r['kappa_quadratic'])} | "
                         f"{_f(r['icc31'])} | {_f(r['spearman'])} | {_f(r['mean_a'])} | {_f(r['mean_b'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(render(json.load(open(path))))
