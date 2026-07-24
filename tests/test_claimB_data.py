"""Claim B data conversion: the two arms must differ ONLY in the form of reasoning.

Claim B (the paper's main event) primes a base model on multi-agent DIALOGUE vs
single-voice MONOLOGUE over IDENTICAL problems with IDENTICAL answers, then runs
the SAME PPO from each. Everything downstream rests on two properties this module
must guarantee, and these tests pin both:

  1. The PROMPT is identical across arms for a given problem, and does NOT
     pre-open a <think> tag. Tier-0/Claim A used TinyZero's stock prompt ending
     in "<think>", which would force the dialogue arm's <persona1> opening
     out-of-distribution the instant PPO starts. So Claim B shares one prompt,
     ending at "Assistant:", for all arms -- a documented deviation from stock
     TinyZero, re-run for the baseline arm too.

  2. Every SFT response is GRADABLE by the stock scorer. The dialogue traces
     state their answer in <group_consensus>; TinyZero's PPO scorer only reads
     <answer>. Ungradable dialogue responses would hand the conversation arm ~0
     reward on problems it actually solved -- the exact bug rl/reward.py's
     _normalise_answer_container exists to prevent. So both arms' responses must
     extract a <answer> that scores 1.0 under the same grader PPO will use.

The length confound (dialogue traces are longer, hence more priming tokens) is
real and inherited from the paper; test_length_confound_is_measured records its
direction rather than pretending it away.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.claimB_data import (
    COUNTDOWN_PROMPT,
    length_stats,
    make_prompt,
    ppo_records,
    sft_records,
)
from rl.reward import accuracy_reward

DATA = Path(__file__).resolve().parent.parent / "rl" / "data"
ARMS = ("dialogue", "monologue")


# --- the prompt -----------------------------------------------------------------

def test_prompt_states_the_numbers_and_target():
    p = make_prompt([43, 25, 33, 26], 77)
    assert "43" in p and "25" in p and "33" in p and "26" in p
    assert "77" in p


def test_prompt_does_not_preopen_a_think_tag():
    """The whole reason Claim B needs its own prompt: a trailing '<think>' would
    force the dialogue arm's <persona1> opening out-of-distribution."""
    p = make_prompt([1, 2, 3], 6)
    assert not p.rstrip().endswith("<think>"), (
        "prompt pre-opens <think>; dialogue priming (<persona1> ...) would be "
        "out-of-distribution under PPO"
    )
    assert p.rstrip().endswith("Assistant:")


def test_prompt_is_deterministic():
    assert make_prompt([1, 2, 3], 6) == make_prompt([1, 2, 3], 6)


# --- the two arms share the prompt, differ only in response ----------------------

def test_arms_produce_identical_prompts_per_problem():
    """The controlled comparison: for each shared pid, the prompt must be byte-for-byte
    identical between arms. Only the response may differ."""
    d = {r["pid"]: r for r in sft_records(DATA / "dialogue_train.json", "dialogue")}
    m = {r["pid"]: r for r in sft_records(DATA / "monologue_train.json", "monologue")}
    assert set(d) == set(m), "arms cover different problems -- experiment is confounded"
    for pid in d:
        assert d[pid]["prompt"] == m[pid]["prompt"], f"pid {pid}: prompts differ across arms"


def test_responses_differ_between_arms():
    """Control for the test above -- if responses were also identical the swap does
    nothing. Dialogue carries persona structure; monologue does not."""
    d = {r["pid"]: r for r in sft_records(DATA / "dialogue_train.json", "dialogue")}
    m = {r["pid"]: r for r in sft_records(DATA / "monologue_train.json", "monologue")}
    pid = next(iter(d))
    assert d[pid]["response"] != m[pid]["response"]
    assert "<persona1>" in d[pid]["response"]
    assert "<persona" not in m[pid]["response"]


# --- gradability: the property that keeps the dialogue arm from being sabotaged ---

@pytest.mark.parametrize("arm", ARMS)
def test_every_sft_response_is_gradable_and_correct(arm):
    """THE keystone. Every response, graded by the SAME grader PPO uses, must
    extract a <answer> that scores 1.0. A dialogue response whose answer lived
    only in <group_consensus> would score 0 here -- and would have handed the
    conversation arm near-zero reward on problems it actually solved."""
    recs = sft_records(DATA / f"{arm}_train.json", arm)
    assert recs, "no records"
    for r in recs:
        prob = _problem_for(arm, r["pid"])
        assert "<answer>" in r["response"], f"pid {r['pid']}: response has no <answer> to grade"
        score = accuracy_reward(r["response"], prob["target"], list(prob["numbers"]))
        assert score == 1.0, f"pid {r['pid']} ({arm}): response does not score 1.0"


@pytest.mark.parametrize("arm", ARMS)
def test_response_ends_at_the_answer(arm):
    """Nothing should trail the final <answer>, or the model learns to keep
    generating after committing -- and the PPO rollout would too."""
    recs = sft_records(DATA / f"{arm}_train.json", arm)
    for r in recs[:20]:
        assert r["response"].rstrip().endswith("</answer>")


# --- verl PPO schema ------------------------------------------------------------

def test_ppo_records_have_the_verl_rl_schema():
    rows = [{"numbers": [43, 25, 33, 26], "target": 77, "pid": 1},
            {"numbers": [1, 2, 3], "target": 6, "pid": 2}]
    recs = ppo_records(rows, split="train")
    assert len(recs) == 2
    r = recs[0]
    assert r["data_source"] == "countdown"
    assert isinstance(r["prompt"], list) and r["prompt"][0]["role"] == "user"
    assert r["prompt"][0]["content"] == make_prompt([43, 25, 33, 26], 77)
    gt = r["reward_model"]["ground_truth"]
    assert gt["target"] == 77 and list(gt["numbers"]) == [43, 25, 33, 26]
    assert r["extra_info"]["split"] == "train"


# --- the confound, recorded not hidden ------------------------------------------

def test_length_confound_is_measured():
    """Dialogue traces are longer -- more priming tokens. This is inherited from
    the paper and is a real confound (form vs sheer compute). Record its
    direction; if it ever flips, this test says so."""
    d = sft_records(DATA / "dialogue_train.json", "dialogue")
    m = sft_records(DATA / "monologue_train.json", "monologue")
    ds = length_stats([r["response"] for r in d])
    ms = length_stats([r["response"] for r in m])
    assert ds["median_chars"] > 0 and ms["median_chars"] > 0
    # documented expectation: dialogue is the longer arm
    assert ds["median_chars"] >= ms["median_chars"], (
        f"length confound flipped: dialogue {ds['median_chars']} < monologue "
        f"{ms['median_chars']} median chars -- update the write-up"
    )


# --- helpers --------------------------------------------------------------------

_CACHE: dict = {}


def _problem_for(arm: str, pid: int) -> dict:
    if arm not in _CACHE:
        _CACHE[arm] = {r["pid"]: r for r in json.loads((DATA / f"{arm}_train.json").read_text())}
    return _CACHE[arm][pid]
