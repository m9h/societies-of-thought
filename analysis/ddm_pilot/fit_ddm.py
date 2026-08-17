"""DDM pilot on the societies-of-thought Countdown steering dose ladder.

Choice = correct/error (accuracy coding: upper bound = correct).
RT proxy = thinking length in kilowords (words/1000). Stopping is endogenous
(the model decides when to commit an answer), which is what licenses the mapping.

Model A: v ~ 0 + C(alpha), a ~ 0 + C(alpha), t ~ Uniform(0, 0.1), z = 0.5.
Model B (--hier): adds (1|pid) random intercept on v (200 problems).

The contrast of interest: does steering dose move the boundary a (deliberation
quantity) or the drift v (evidence quality per kiloword)?
"""
import argparse
import sys

import arviz as az
import numpy as np
import pandas as pd

import hssm

p = argparse.ArgumentParser()
p.add_argument("--hier", action="store_true", help="add (1|pid) on v")
p.add_argument("--drop-ceiling", action="store_true",
               help="sensitivity: drop near-ceiling (likely truncated) traces")
p.add_argument("--draws", type=int, default=500)
p.add_argument("--tune", type=int, default=500)
p.add_argument("--chains", type=int, default=2)
args = p.parse_args()

df = pd.read_csv("countdown_ladder.csv")
if args.drop_ceiling:
    df = df[df.near_ceiling == 0]
df["alpha"] = df["alpha"].astype(str)  # categorical
df["pid"] = df["pid"].astype(str)
data = df[["rt", "response", "alpha", "pid"]].copy()
print(f"{len(data)} trials, rt range [{data.rt.min():.3f}, {data.rt.max():.3f}] kilowords")
print(data.groupby("alpha").agg(n=("rt", "size"), acc=("response", lambda s: (s == 1).mean()),
                                rt_mean=("rt", "mean")))

v_formula = "v ~ 0 + C(alpha)" + (" + (1|pid)" if args.hier else "")

model = hssm.HSSM(
    data=data,
    model="ddm",
    loglik_kind="analytical",
    include=[
        {"name": "v", "formula": v_formula},
        {"name": "a", "formula": "a ~ 0 + C(alpha)", "link": "log"},
    ],
    t=hssm.Prior("Uniform", lower=0.001, upper=0.105),
)
print(model)

idata = model.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                     cores=args.chains, target_accept=0.9)

out = ("idata_hier.nc" if args.hier else "idata_flat.nc")
if args.drop_ceiling:
    out = out.replace(".nc", "_noceil.nc")
idata.to_netcdf(out)

summ = az.summary(idata, filter_vars="like", var_names=["v_", "a_", "t", "z"])
pd.set_option("display.width", 200)
print(summ.to_string())
summ.to_csv(out.replace(".nc", "_summary.csv"))

post = idata.posterior
avar = "a_C(alpha)"
vvar = "v_C(alpha)"
levels = [str(x) for x in post[avar].coords[f"{avar}_dim"].values]
a_draws = np.exp(post[avar].values.reshape(-1, len(levels)))  # log link
v_draws = post[vvar].values.reshape(-1, len(levels))

print("\nPer-condition posterior (mean [94% HDI]):")
rows = []
for i, lev in enumerate(levels):
    a_hdi = az.hdi(a_draws[:, i])
    v_hdi = az.hdi(v_draws[:, i])
    rows.append({"alpha": lev,
                 "a_mean": a_draws[:, i].mean(), "a_lo": a_hdi[0], "a_hi": a_hdi[1],
                 "v_mean": v_draws[:, i].mean(), "v_lo": v_hdi[0], "v_hi": v_hdi[1]})
    print(f"  alpha={lev:>6}: a = {a_draws[:, i].mean():.3f} [{a_hdi[0]:.3f}, {a_hdi[1]:.3f}]"
          f"   v = {v_draws[:, i].mean():+.3f} [{v_hdi[0]:+.3f}, {v_hdi[1]:+.3f}]")

base = levels.index("0.0")
print("\nContrasts vs baseline (P = posterior prob of increase):")
for i, lev in enumerate(levels):
    if i == base:
        continue
    da = a_draws[:, i] - a_draws[:, base]
    dv = v_draws[:, i] - v_draws[:, base]
    print(f"  alpha={lev:>6}: dA = {da.mean():+.3f} (P(a up)={np.mean(da > 0):.3f})"
          f"   dV = {dv.mean():+.3f} (P(v up)={np.mean(dv > 0):.3f})")

json_out = {"levels": levels, "rows": rows}
pd.DataFrame(rows).to_csv(out.replace(".nc", "_params.csv"), index=False)
print("\nDone. Saved:", out)
