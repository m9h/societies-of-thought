"""Give a verl PPO run a seed it does not otherwise have.

The TinyZero/verl vintage pinned here exposes no seed: the resolved config printed into
our own PPO logs contains no `seed` key anywhere. "Three seeds" therefore cannot mean
passing `seed=0,1,2` on the command line -- it has to be a knob we own and can show.

The knob is the order of the training set. A different problem order sends PPO down a
different trajectory from step 1, it is reproducible from an integer, and unlike a
framework default it is visible in the artifact itself: two runs are demonstrably
differently seeded because their parquets differ.

Rows move whole. Permuting a single column would decouple each problem from its ground
truth and train on noise while still drawing a plausible reward curve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def permute_train(src, dst, seed: int) -> int:
    """Write `src` to `dst` with rows permuted by `seed`. Returns the row count.

    Seed 0 permutes like any other seed; leaving it as the identity would make one arm
    of a sweep the unshuffled original and not comparable to the rest.
    """
    df = pd.read_parquet(src)
    order = np.random.default_rng(seed).permutation(len(df))
    out = df.iloc[order].reset_index(drop=True)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    return len(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    print(f"{permute_train(a.src, a.dst, a.seed)} rows -> {a.dst} (seed {a.seed})")
