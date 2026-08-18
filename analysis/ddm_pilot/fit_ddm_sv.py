"""Robustness: does the boundary/drift decomposition survive drift variability?

The PPC on the plain DDM reproduced marginal accuracy and mean RT but underfit
the fast-correct/slow-error asymmetry, whose textbook remedy is inter-trial
drift variability (sv). This refits the dose ladder with model="ddm_sdv".

Restricted to alpha <= 1.0: the degenerate alpha=1.693 condition (a ~ 5.3) is
qualitatively settled and only strains the parameterization.
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import hssm

p = argparse.ArgumentParser()
p.add_argument("--draws", type=int, default=1000)
p.add_argument("--tune", type=int, default=1000)
p.add_argument("--chains", type=int, default=4)
args = p.parse_args()

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "countdown_ladder.csv")
df = df[df.alpha <= 1.0]
df["alpha"] = df["alpha"].astype(str)
data = df[["rt", "response", "alpha"]].copy()
print(f"{len(data)} trials (alpha <= 1.0)")

model = hssm.HSSM(
    data=data,
    model="ddm_sdv",
    loglik_kind="analytical",
    include=[
        {"name": "v", "formula": "v ~ 0 + C(alpha)"},
        {"name": "a", "formula": "a ~ 0 + C(alpha)", "link": "log"},
    ],
    t=hssm.Prior("Uniform", lower=0.001, upper=0.105),
)
idata = model.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                     cores=args.chains, target_accept=0.9)
idata.to_netcdf(str(HERE / "idata_sv.nc"))

summ = az.summary(idata, filter_vars="like", var_names=["v_", "a_", "t", "z", "sv"])
pd.set_option("display.width", 200)
print(summ.to_string())
summ.to_csv(HERE / "idata_sv_summary.csv")

post = idata.posterior
levels = [str(x) for x in post["a_C(alpha)"].coords["a_C(alpha)_dim"].values]
a_draws = np.exp(post["a_C(alpha)"].values.reshape(-1, len(levels)))
v_draws = post["v_C(alpha)"].values.reshape(-1, len(levels))
base = levels.index("0.0")
print("\nWith sv — contrasts vs baseline:")
for i, lev in enumerate(levels):
    if i == base:
        continue
    da = a_draws[:, i] - a_draws[:, base]
    dv = v_draws[:, i] - v_draws[:, base]
    print(f"  alpha={lev:>6}: dA = {da.mean():+.3f} (P(a up)={np.mean(da > 0):.3f})"
          f"   dV = {dv.mean():+.3f} (P(v up)={np.mean(dv > 0):.3f})")
