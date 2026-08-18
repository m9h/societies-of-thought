"""Diversity-on-drift: the SoT mediation test in DDM coordinates (GPQA, QwQ-32B).

Data: per_trace from results/qwq/hse_domains_v2.json, GPQA subset — 500 problems
x 6 samples (truncated traces were already excluded upstream; that exclusion is
a real caveat, recorded in RESULTS.md).

Model: DDM with choice = correct/error, RT = kilowords. Trial-level diversity
(z-scored hse_norm) regresses on BOTH drift and boundary, with a problem random
intercept on drift soaking difficulty:

    v ~ 1 + hse_z + (1|pid)      a ~ 1 + hse_z   (log link)

Reading: if perspective diversity improves evidence quality (the paper's C2/C3),
the hse_z coefficient on v is positive after length is jointly modelled. If
diversity only tracks deliberation, it lands on a instead. The repo's HSE
within-problem estimate was a null (+0.0023 [-0.0032, +0.0078]); this is the
same question asked with length inside the likelihood instead of matched away.

Both hse_norm and rt are products of the same generation process — this is
descriptive mediation, not causal identification. Same status as the paper's SEM.
"""
import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import hssm

p = argparse.ArgumentParser()
p.add_argument("--domain", default="gpqa")
p.add_argument("--center-within", action="store_true",
               help="center hse_norm within problem before z-scoring")
p.add_argument("--mixed-only", action="store_true",
               help="keep only problems with both correct and incorrect traces. "
               "Without this, the 79%% of GPQA problems that are always-right or "
               "always-wrong (see GPQA_within_problem.md) give their random "
               "intercepts no finite optimum (complete separation) and the "
               "chains do not converge (observed r_hat 3-4).")
p.add_argument("--draws", type=int, default=1000)
p.add_argument("--tune", type=int, default=1000)
p.add_argument("--chains", type=int, default=4)
args = p.parse_args()

HERE = Path(__file__).resolve().parent
src = json.loads((HERE.parent.parent / "results/qwq/hse_domains_v2.json").read_text())
rows = [r for r in src["per_trace"] if r["source"] == args.domain]
df = pd.DataFrame(rows)
df["rt"] = df["words"] / 1000.0
df["response"] = np.where(df["correct"], 1, -1)
df["pid"] = df["pid"].astype(str)
if args.center_within:
    # within-problem centering: the coefficient is then identified purely from
    # variation among samples of the SAME problem, the faithful DDM analog of
    # GPQA_within_problem.md (between-problem structure cannot leak in)
    dev = df["hse_norm"] - df.groupby("pid")["hse_norm"].transform("mean")
    df["hse_z"] = dev / dev.std()
else:
    df["hse_z"] = (df["hse_norm"] - df["hse_norm"].mean()) / df["hse_norm"].std()
if args.mixed_only:
    outcome_kinds = df.groupby("pid")["correct"].nunique()
    mixed = outcome_kinds[outcome_kinds == 2].index
    df = df[df.pid.isin(mixed)]
    print(f"mixed-outcome problems only: {df.pid.nunique()} problems, {len(df)} traces")
data = df[["rt", "response", "hse_z", "pid"]].copy()
print(f"{args.domain}: {len(data)} traces, {data.pid.nunique()} problems, "
      f"acc={(data.response == 1).mean():.3f}, rt mean={data.rt.mean():.2f} kw, "
      f"min={data.rt.min():.3f}")

t_upper = max(0.001, float(data.rt.min()) * 0.95)
model = hssm.HSSM(
    data=data,
    model="ddm",
    loglik_kind="analytical",
    include=[
        {"name": "v", "formula": "v ~ 1 + hse_z + (1|pid)"},
        {"name": "a", "formula": "a ~ 1 + hse_z", "link": "log"},
    ],
    t=hssm.Prior("Uniform", lower=0.001, upper=t_upper),
)
idata = model.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                     cores=args.chains, target_accept=0.9)

out = HERE / f"idata_{args.domain}_diversity.nc"
idata.to_netcdf(str(out))

summ = az.summary(idata, filter_vars="like", var_names=["v_", "a_", "t", "z"])
summ = summ[~summ.index.str.contains(r"\|pid\[", regex=True)]
pd.set_option("display.width", 200)
print(summ.to_string())
summ.to_csv(str(out).replace(".nc", "_summary.csv"))

post = idata.posterior
for name, transform in [("v_hse_z", None), ("a_hse_z", "log-scale")]:
    if name in post:
        d = post[name].values.ravel()
        hdi = az.hdi(d)
        note = f" ({transform})" if transform else ""
        print(f"{name}{note}: {d.mean():+.4f} [{hdi[0]:+.4f}, {hdi[1]:+.4f}]  "
              f"P(>0)={np.mean(d > 0):.3f}")
print("Saved:", out.name)
