"""Fidelity gate for the Fig. 4 (C4) emergence run.

Fig. 4 is the paper's headline: Qwen-2.5-3B, *base*, no priming of any kind, rewarded
only for accuracy, developing conversational behaviour anyway. The claim dies the moment
anything conversational enters through the starting weights, so this file's real job is
to make "un-primed" un-fakeable -- and to hold the same PPO hyperparameters the C5 gate
already holds, so the two experiments stay comparable.

The last invariant is about durability rather than fidelity: for C5 the artifact was the
val curve and the log was scaffolding; here the generated traces ARE the measurement and
they exist only in the log. A run whose log cannot leave the pod is a run that spends
fourteen hours producing nothing, which has already happened once on this project.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from rl import paper_spec as S

FIG4 = Path(__file__).resolve().parents[1] / "scripts" / "fig4_pod.sh"


def _text() -> str:
    return FIG4.read_text()


def _flag(text: str, key: str) -> str | None:
    m = re.search(rf"{re.escape(key)}=([^\s\\]+)", text)
    return m.group(1) if m else None


# --- the arm must be the paper's un-primed base model ------------------------


def test_base_model_is_the_papers_base_not_an_instruct_variant():
    """Qwen2.5-3B-Instruct is already conversational. Emergence measured from it would
    be measuring instruction tuning, which is the one thing Fig. 4 claims it is not."""
    m = re.search(r"MODEL:-([^\}]+)\}", _text())
    assert m, "fig4 script does not set a default MODEL"
    assert m.group(1) == "Qwen/Qwen2.5-3B"
    assert "Instruct" not in m.group(1)


def test_there_is_no_priming_step():
    """Any SFT before PPO turns Fig. 4 into Fig. 8. The word should not appear as a
    command anywhere in the script."""
    t = _text()
    assert "rl.sft_prime" not in t, "fig4 must not prime -- that is the C5 experiment"
    assert "sft_dialogue" not in t and "sft_monologue" not in t


def test_ppo_starts_from_the_base_model_not_a_checkpoint():
    t = _text()
    for flag in ("actor_rollout_ref.model.path", "critic.model.path"):
        assert _flag(t, flag) == '"$MODEL"', f"{flag} must be the base model"


# --- the same PPO as C5, so the two experiments are comparable ---------------


@pytest.mark.parametrize("flag,spec_key", [
    ("data.train_batch_size", "train_batch_size"),
    ("data.val_batch_size", "val_batch_size"),
    ("data.max_prompt_length", "max_prompt_length"),
    ("data.max_response_length", "max_response_length"),
    ("actor_rollout_ref.actor.ppo_mini_batch_size", "ppo_mini_batch_size"),
    ("actor_rollout_ref.rollout.n", "rollout_n"),
])
def test_ppo_hparams_match_the_paper(flag, spec_key):
    got = _flag(_text(), flag)
    assert got is not None, f"fig4 script does not set {flag}"
    assert int(got) == S.PPO_HPARAMS[spec_key]


def test_lrs_and_kl_match_the_paper():
    t = _text()
    assert float(_flag(t, "actor_rollout_ref.actor.optim.lr")) == S.PPO_HPARAMS["actor_lr"]
    assert float(_flag(t, "critic.optim.lr")) == S.PPO_HPARAMS["critic_lr"]
    assert float(_flag(t, "algorithm.kl_ctrl.kl_coef")) == S.PPO_HPARAMS["kl_coef"]


def test_step_count_and_eval_cadence_match_the_paper():
    t = _text()
    assert int(_flag(t, "trainer.test_freq")) == S.RL_EVAL_EVERY
    m = re.search(r"STEPS:-(\d+)", t)
    assert m and int(m.group(1)) == S.PPO_HPARAMS["total_steps"]


def test_sampling_temperature_is_one():
    """Emergence is a claim about what the policy generates. Lowering temperature to
    make the curve cleaner would suppress exactly the variability under study."""
    assert float(_flag(_text(), "actor_rollout_ref.rollout.temperature")) == 1.0


# --- the seed has to be real -------------------------------------------------


def test_the_seed_reaches_the_training_data():
    t = _text()
    assert "rl.run_seed" in t, "SEED is declared but never used to permute anything"
    assert 'train_s${SEED}.parquet' in t


def test_the_eval_set_is_not_permuted():
    """Seeds must vary the training trajectory, not the yardstick. A permuted val set
    would make the three accuracy curves incomparable."""
    t = _text()
    assert 'data.val_files="$DATA/test.parquet"' in t
    assert "test_s${SEED}" not in t


def test_the_smoke_gate_proves_the_permutation_happened():
    """A seed that silently no-ops gives three identical runs reported as three seeds --
    which would understate seed variance in exactly the direction that flatters us."""
    assert "would be unseeded" in _text()


# --- durability: the traces are the result -----------------------------------


def test_the_run_refuses_to_start_without_a_way_to_save_the_traces():
    t = _text()
    assert "REFUSING TO START" in t
    assert "ALLOW_NO_MIRROR" in t, "there must be an explicit opt-out, not a silent one"


def test_ppo_checkpoints_are_still_disabled_by_default():
    """~16GB each; three arms once filled a 200GB disk mid-save."""
    assert 'trainer.save_freq="${SAVE_FREQ:--1}"' in _text()


# --- the dependency chain has to fail where it breaks ------------------------


def test_a_failed_verl_install_stops_the_run():
    """verl's install failed once and the script marched on into a data build that
    could not work. The first visible symptom was a missing parquet, which reads like a
    data bug rather than a failed dependency install."""
    t = _text()
    assert "verl did NOT install" in t
    assert t.index("verl did NOT install") < t.index("building data"), \
        "the verl check must come before the data build it gates"


def test_the_dependency_chain_is_pinned_rather_than_resolved():
    """Resolving `vllm<=0.6.3` from scratch sends pip backtracking to uvicorn 0.5.2,
    whose setup.py opens a README.md absent from the sdist. Two pod boots died there,
    both reporting a missing parquet rather than a failed install."""
    t = _text()
    assert '"vllm==0.6.3"' in t, "vllm must be pinned, not left to the resolver"
    assert '"uvicorn[standard]==0.30.6"' in t, "uvicorn is the package it backtracks on"
    assert "--no-deps" in t, "verl must not re-resolve what is already installed"


def test_every_verl_requirement_is_named_explicitly():
    """If a requirement is only implied, a partial install looks like success until
    something three steps later cannot import it."""
    t = _text()
    for pkg in ("accelerate", "codetiming", "datasets", "dill", "hydra-core",
                "pybind11", "ray", '"tensordict<0.6"', '"transformers<4.48"'):
        assert pkg in t, f"{pkg} is a verl requirement and is not installed explicitly"


def test_the_data_build_verifies_its_own_output():
    t = _text()
    assert "data build produced no" in t
