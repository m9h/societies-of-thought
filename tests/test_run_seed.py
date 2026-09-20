"""Seeding a verl PPO run — tests written BEFORE the implementation.

The TinyZero/verl vintage this project pins exposes no seed anywhere: a grep of the
resolved config printed into our own PPO logs finds no `seed` key at all. So "three
seeds" cannot mean "pass seed=0,1,2"; it has to be a knob we own.

The knob is the training-set permutation. Different problem order gives a different
optimisation trajectory, it is reproducible from an integer, and it is visible in the
artifact itself (the parquet) rather than buried in a framework default.

The failure this guards against is the quiet one: a "seeded" sweep where every arm
actually trains on the identical order and the three curves differ only by GPU
nondeterminism, which would understate seed variance and tell the authors something
false about their own experiment.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rl.run_seed import permute_train


@pytest.fixture
def parquet(tmp_path):
    p = tmp_path / "train.parquet"
    pd.DataFrame({"prompt": [f"p{i}" for i in range(50)],
                  "data_source": ["countdown"] * 50,
                  "reward_model": [{"ground_truth": i} for i in range(50)]}).to_parquet(p)
    return p


def test_permutation_keeps_every_row(parquet, tmp_path):
    out = tmp_path / "s0.parquet"
    n = permute_train(parquet, out, seed=0)
    a, b = pd.read_parquet(parquet), pd.read_parquet(out)
    assert n == len(a) == len(b)
    assert sorted(a["prompt"]) == sorted(b["prompt"]), "rows must be permuted, not resampled"


def test_different_seeds_give_different_orders(parquet, tmp_path):
    permute_train(parquet, tmp_path / "s0.parquet", seed=0)
    permute_train(parquet, tmp_path / "s1.parquet", seed=1)
    a = pd.read_parquet(tmp_path / "s0.parquet")["prompt"].tolist()
    b = pd.read_parquet(tmp_path / "s1.parquet")["prompt"].tolist()
    assert a != b, "two seeds produced an identical order -- the sweep would be unseeded"


def test_same_seed_is_reproducible(parquet, tmp_path):
    permute_train(parquet, tmp_path / "a.parquet", seed=7)
    permute_train(parquet, tmp_path / "b.parquet", seed=7)
    assert (pd.read_parquet(tmp_path / "a.parquet")["prompt"].tolist()
            == pd.read_parquet(tmp_path / "b.parquet")["prompt"].tolist())


def test_seed_zero_still_permutes(parquet, tmp_path):
    """Seed 0 must not be a silent no-op -- otherwise one arm of the sweep is the
    unshuffled original and is not comparable to the others."""
    permute_train(parquet, tmp_path / "s0.parquet", seed=0)
    assert (pd.read_parquet(parquet)["prompt"].tolist()
            != pd.read_parquet(tmp_path / "s0.parquet")["prompt"].tolist())


def test_columns_are_preserved_exactly(parquet, tmp_path):
    """verl's scorer dispatches on data_source and reads reward_model.ground_truth.
    Losing or reordering columns zeroes every reward, silently."""
    permute_train(parquet, tmp_path / "s.parquet", seed=3)
    a, b = pd.read_parquet(parquet), pd.read_parquet(tmp_path / "s.parquet")
    assert list(a.columns) == list(b.columns)


def test_ground_truth_travels_with_its_prompt(parquet, tmp_path):
    """The permutation must move whole rows. Shuffling one column independently would
    decouple each problem from its answer and train on pure noise -- while still
    producing a plausible-looking loss curve."""
    permute_train(parquet, tmp_path / "s.parquet", seed=5)
    b = pd.read_parquet(tmp_path / "s.parquet")
    for _, row in b.iterrows():
        assert row["reward_model"]["ground_truth"] == int(row["prompt"][1:])
