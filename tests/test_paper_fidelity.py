"""The fidelity gate.

These tests exist because we twice shipped a run that deviated from the paper and then
described it as a replication. They fail if our runnable configs drift from
`rl.paper_spec`, so the question "is this actually a replication?" is answered by CI
rather than by assertion.

If a test here fails, either fix the config or add an entry to paper_spec.DEVIATIONS
with a reason -- and then the run is NOT a replication and must not be described as one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from rl import paper_spec as S

POD = Path(__file__).resolve().parents[1] / "scripts" / "claimB_pod.sh"


def _pod_text() -> str:
    return POD.read_text()


def _flag(text: str, key: str) -> str | None:
    """Pull `key=value` out of the pod script's PPO command line."""
    m = re.search(rf"{re.escape(key)}=([^\s\\]+)", text)
    return m.group(1) if m else None


# --- the pool ----------------------------------------------------------------


def test_pool_reconstructs_the_papers_8262():
    assert S.POOL_TOTAL == 8262


def test_pool_excludes_the_rl_task():
    """The whole point of C5: priming is OUT OF DOMAIN relative to Countdown."""
    names = (set(S.POOL_BBH) | set(S.POOL_GPQA) | set(S.POOL_MATH_HARD)
             | set(S.POOL_MUSR) | set(S.POOL_OTHER))
    assert not any("countdown" in n.lower() for n in names)


def test_priming_data_is_not_built_from_countdown():
    """Guards the exact mistake we made: priming on the RL task itself.

    Countdown records carry `numbers`/`target`. Out-of-domain records must not.
    """
    import json

    d = Path(__file__).resolve().parents[1] / "rl" / "data" / "ood"
    if not d.exists():
        pytest.skip("out-of-domain priming set not built yet")
    for split in ("dialogue_train.json", "monologue_train.json"):
        rows = json.loads((d / split).read_text())
        assert rows, f"{split} is empty"
        bad = [r for r in rows if "numbers" in r and "target" in r]
        assert not bad, (
            f"{split}: {len(bad)} records look like Countdown problems. Priming must be "
            "out-of-domain relative to the RL task."
        )
        assert {"source", "task", "answer"} <= set(rows[0]), (
            f"{split}: records must record their benchmark provenance"
        )


def test_priming_split_sizes_match_the_paper():
    import json

    d = Path(__file__).resolve().parents[1] / "rl" / "data" / "ood"
    if not d.exists():
        pytest.skip("out-of-domain priming set not built yet")
    for arm in ("dialogue", "monologue"):
        assert len(json.loads((d / f"{arm}_train.json").read_text())) == S.SFT_N_TRAIN
        assert len(json.loads((d / f"{arm}_val.json").read_text())) == S.SFT_N_VAL


def test_arms_are_matched_on_the_same_problems():
    """Dialogue and monologue must cover identical problems -- the paper's invariant:
    'both conditions are trained on identical problems and correct answers'."""
    import json

    d = Path(__file__).resolve().parents[1] / "rl" / "data" / "ood"
    if not d.exists():
        pytest.skip("out-of-domain priming set not built yet")
    for split in ("train", "val"):
        dia = {r["pid"] for r in json.loads((d / f"dialogue_{split}.json").read_text())}
        mon = {r["pid"] for r in json.loads((d / f"monologue_{split}.json").read_text())}
        assert dia == mon, f"{split}: arms cover different problems"


# --- the teacher -------------------------------------------------------------


def test_teacher_is_the_papers_teacher():
    assert S.TEACHER_MODEL == "Qwen/Qwen2.5-32B-Instruct"


def test_generator_uses_the_spec_teacher():
    """No silent substitution of a different-sized model."""
    gen = Path(__file__).resolve().parents[1] / "rl" / "generate_sft_ood.py"
    if not gen.exists():
        pytest.skip("out-of-domain generator not written yet")
    src = gen.read_text()
    assert "paper_spec" in src and "TEACHER_MODEL" in src, (
        "generator must take its teacher from paper_spec, not hardcode one"
    )
    # Prose may discuss the old 72B substitution; what must not reappear is a 72B
    # model *identifier*. Walk real string constants, ignoring docstrings/comments.
    import ast

    tree = ast.parse(src)
    docstrings = {ast.get_docstring(n) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
    ids = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and n.value not in docstrings
           and "/" in n.value and not n.value.strip().count(" ")
           and "72b" in n.value.lower()]
    assert not ids, f"a 72B model identifier is still referenced: {ids}"


# --- the prompts -------------------------------------------------------------


def test_dialogue_prompt_matches_the_paper():
    p = S.DIALOGUE_PROMPT
    for required in ("<cast_of_characters>", "<group_solution>", "<conversation>",
                     "disagreements can be helpful"):
        assert required in p, f"dialogue prompt missing {required!r}"


def test_monologue_prompt_matches_the_paper():
    assert "<think>" in S.MONOLOGUE_PROMPT and "</think>" in S.MONOLOGUE_PROMPT
    assert "Do not answer directly without reasoning" in S.MONOLOGUE_PROMPT


# --- SFT hyperparameters -----------------------------------------------------


@pytest.mark.parametrize("key,want", [
    ("total_epochs", 5), ("train_batch_size", 64), ("context_window", 2048),
    ("peak_lr", 1e-5), ("warmup_ratio", 0.10),
])
def test_sft_spec_values(key, want):
    assert S.SFT_HPARAMS[key] == want


def test_pod_script_uses_paper_sft_hparams():
    t = _pod_text()
    m = re.search(r"--epochs\s+(\S+)", t)
    assert m, "pod script does not set --epochs"
    assert float(m.group(1)) == float(S.SFT_HPARAMS["total_epochs"]), (
        f"SFT epochs {m.group(1)} != paper's {S.SFT_HPARAMS['total_epochs']}"
    )
    m = re.search(r"--max-len\s+(\d+)", t)
    assert m and int(m.group(1)) >= S.SFT_HPARAMS["context_window"], (
        "SFT context window is below the paper's 2048"
    )


# --- PPO hyperparameters -----------------------------------------------------


@pytest.mark.parametrize("flag,spec_key", [
    ("data.train_batch_size", "train_batch_size"),
    ("data.val_batch_size", "val_batch_size"),
    ("data.max_prompt_length", "max_prompt_length"),
    ("data.max_response_length", "max_response_length"),
    ("actor_rollout_ref.actor.ppo_mini_batch_size", "ppo_mini_batch_size"),
    ("actor_rollout_ref.rollout.n", "rollout_n"),
])
def test_pod_script_ppo_matches_paper(flag, spec_key):
    got = _flag(_pod_text(), flag)
    assert got is not None, f"pod script does not set {flag}"
    assert int(got) == S.PPO_HPARAMS[spec_key], (
        f"{flag}={got} but the paper specifies {S.PPO_HPARAMS[spec_key]}"
    )


def test_pod_script_ppo_lrs_and_kl():
    t = _pod_text()
    assert float(_flag(t, "actor_rollout_ref.actor.optim.lr")) == S.PPO_HPARAMS["actor_lr"]
    assert float(_flag(t, "critic.optim.lr")) == S.PPO_HPARAMS["critic_lr"]
    assert float(_flag(t, "algorithm.kl_ctrl.kl_coef")) == S.PPO_HPARAMS["kl_coef"]


def test_pod_script_eval_cadence_matches_paper():
    got = _flag(_pod_text(), "trainer.test_freq")
    assert got and int(got) == S.RL_EVAL_EVERY, (
        f"test_freq={got} but the paper evaluates every {S.RL_EVAL_EVERY} steps"
    )


def test_pod_script_runs_the_papers_step_count():
    t = _pod_text()
    m = re.search(r"STEPS:-(\d+)", t)
    assert m and int(m.group(1)) == S.PPO_HPARAMS["total_steps"]


# --- the Llama-specific control we previously missed --------------------------


def test_llama_persona_concatenation_is_implemented():
    """The paper concatenates personas into one <think> block for Llama ONLY, to
    length-match the conditions. We shipped a Llama run without it."""
    from rl import claimB_data

    assert hasattr(claimB_data, "concatenate_personas"), (
        "claimB_data must expose concatenate_personas() for the Llama condition"
    )


def test_llama_concatenation_produces_a_single_think_block():
    from rl.claimB_data import concatenate_personas

    trace = ("<cast_of_characters><persona1>a</persona1><persona2>b</persona2>"
             "</cast_of_characters><conversation><think1> first </think1>"
             "<think2> second </think2></conversation>"
             "<group_solution> 42 </group_solution>")
    out = concatenate_personas(trace)
    assert out.count("<think>") == 1 and out.count("</think>") == 1
    assert "<think1>" not in out and "<persona1>" not in out
    assert "first" in out and "second" in out
    assert "42" in out


# --- deviations must be declared ---------------------------------------------


def test_every_deviation_is_declared_with_a_reason():
    for name, reason in S.DEVIATIONS.items():
        assert len(reason) > 60, f"deviation {name!r} needs a real reason, not a stub"
