"""LaRT companion fit on the same Countdown dose ladder.

Mapping: each steering dose = one 'model' row (N=6), each Countdown problem =
one item (J=200). response[i,j] = correct, latency[i,j] = words (+1 guard).

Caveat stated up front: LaRT's regime is N ~ 100 models; with N=6 the
ability-speed correlation rho is estimated from six latent pairs and should be
treated as descriptive only. theta (ability) and tau (speed) per dose are the
usable output.
"""
import json

import numpy as np
import pandas as pd
from lart import fit_lart

df = pd.read_csv("countdown_ladder.csv")
alphas = [0.0, 0.25, 0.5, 0.678, 1.0, 1.693]
R = np.zeros((6, 200), dtype=int)
L = np.zeros((6, 200), dtype=float)
for i, a in enumerate(alphas):
    sub = df[df.alpha == a].sort_values("pid")
    assert len(sub) == 200
    R[i] = sub["correct"].to_numpy()
    L[i] = sub["words"].to_numpy() + 1.0

fit = fit_lart(R, L, seed=42)
out = {
    "alphas": alphas,
    "theta_ability": [float(x) for x in np.ravel(fit.theta)],
    "tau_speed": [float(x) for x in np.ravel(fit.tau)],
    "rho": float(np.ravel(fit.rho)[0]) if np.size(fit.rho) else None,
}
print(json.dumps(out, indent=2))
with open("lart_fit.json", "w") as f:
    json.dump(out, f, indent=2)
