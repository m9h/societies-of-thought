"""The eval path and the reward path must agree on what counts as an answer.

They did not, and it silently inverted a result.

`rl/reward.py` normalises dialogue-style answer containers before grading:

    _normalise_answer_container("... <group_consensus> 1+2 </group_consensus>")
        -> appends "<answer>1+2</answer>"

`rl/train_grpo.py::EvalAndLog` called `grade()` directly, with no normalisation.
So on the dialogue arm:

    reward during training      0.40 -> 0.43   (max 0.57)
    eval accuracy reported      0.0%, parse 0.0%

The model was scoring roughly SEVEN TIMES the baseline arm's reward while the
eval reported total failure, because the dialogue SFT traces end in
<group_consensus> and 0 of 500 contain an <answer> tag. Reported as-is, this
would have produced a confident "monologue beats dialogue" conclusion that was
entirely an artifact of which formats each code path could parse.

That is worse than a crash. A crash stops you.

tests/test_sft_pairing.py already checks that the two arms cover identical
problems with identical answers -- the invariant the paper's design rests on.
It does not check that both arms' OUTPUT FORMAT is gradeable, which is a
separate invariant and the one that broke.

These tests pin the agreement itself: anything the reward path can score, the
eval path must be able to parse.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from rl.reward import _normalise_answer_container, accuracy_reward
from sot.grade import grade

REPO = pathlib.Path(__file__).resolve().parents[1]

DIALOGUE_TAIL = (
    "<think1> 43 + 33 is 76. </think1>\n"
    "<think2> And 26 - 25 is 1. </think2>\n"
    "<group_consensus> (43 + 33) + (26 - 25) </group_consensus>"
)
MONOLOGUE_TAIL = "<think> 43 + 33 - 25 + 26 </think>\n<answer> 43 + 33 - 25 + 26 </answer>"
GOLD = "77|43,33,25,26"


def eval_grade(completion: str):
    """What EvalAndLog does. Must see everything the reward path sees."""
    return grade("countdown", _normalise_answer_container(completion), GOLD)


# --- the bug, stated directly --------------------------------------------------

def test_dialogue_completion_is_parseable_by_the_eval_path():
    """The regression. Before the fix this parsed as nothing, giving 0% accuracy
    on an arm that was scoring 0.4 reward."""
    g = eval_grade(DIALOGUE_TAIL)
    assert g.parsed, "eval path cannot parse a <group_consensus> answer"
    assert g.correct, "(43+33)+(26-25) = 77 -- this is a correct answer"


def test_monologue_completion_is_parseable_by_the_eval_path():
    g = eval_grade(MONOLOGUE_TAIL)
    assert g.parsed and g.correct


def test_reward_and_eval_agree_on_both_formats():
    """THE invariant. Whatever the reward path scores as correct, the eval path
    must also score as correct -- otherwise the two halves of the experiment are
    measuring different things and the comparison between arms is meaningless."""
    for label, text in [("dialogue", DIALOGUE_TAIL), ("monologue", MONOLOGUE_TAIL)]:
        r = accuracy_reward(text, 77, [43, 33, 25, 26])
        g = eval_grade(text)
        assert (r == 1.0) == bool(g.correct), (
            f"{label}: reward says {r}, eval says correct={g.correct}. "
            "The reward path and the eval path disagree."
        )


def test_unnormalised_dialogue_would_not_parse():
    """Pins WHY the fix is needed, so nobody 'simplifies' the normalisation away.
    Grading the raw dialogue completion finds nothing."""
    raw = grade("countdown", DIALOGUE_TAIL, GOLD)
    assert not raw.parsed, (
        "raw <group_consensus> now parses without normalisation -- if the grader "
        "was widened to handle it directly, EvalAndLog no longer needs the "
        "normalisation call, but check that deliberately rather than by accident"
    )


# --- the shipped data, not just synthetic strings ------------------------------

@pytest.mark.parametrize("arm", ["dialogue", "monologue"])
def test_shipped_sft_traces_are_gradeable_by_the_eval_path(arm):
    """The synthetic cases above could pass while the real traces still fail.
    Every trace was verified correct at generation time by generate_sft.py, so
    the eval path should score essentially all of them correct."""
    path = REPO / f"rl/data/{arm}_train.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    rows = json.loads(path.read_text())[:50]
    ok = 0
    for r in rows:
        gold = f"{r['target']}|{','.join(map(str, r['numbers']))}"
        g = grade("countdown", _normalise_answer_container(r[arm]), gold)
        ok += bool(g.parsed)
    assert ok >= int(0.95 * len(rows)), (
        f"{arm}: only {ok}/{len(rows)} shipped traces parse through the eval path. "
        "generate_sft.py verified every one of them at generation time, so a low "
        "rate here means the eval path disagrees with the generator."
    )


# --- guard the real code path, not a local re-implementation -------------------
# The first version of this file tested a local `eval_grade` helper that applied
# the normalisation. It passed while EvalAndLog still had the bug -- a test of
# the fix rather than of the code. These check the actual call sites.

def test_train_grpo_does_not_call_the_raw_grader():
    """EvalAndLog must go through grade_completion. A bare grade() call there is
    the bug: it skips <group_consensus> normalisation and reports a working
    dialogue arm as 0% accuracy."""
    src = (REPO / "rl/train_grpo.py").read_text()
    body = src.split("class EvalAndLog")[1].split("\ndef main")[0]
    assert "grade_completion(" in body, "EvalAndLog does not use grade_completion"
    bare = [l.strip() for l in body.splitlines()
            if "grade(" in l and "grade_completion(" not in l and not l.strip().startswith("#")]
    assert not bare, f"EvalAndLog still calls the raw grader: {bare}"


def test_accuracy_reward_delegates_to_the_same_function():
    """One scoring path, not two that happen to agree today."""
    src = (REPO / "rl/reward.py").read_text()
    body = src.split("def accuracy_reward")[1].split("\ndef ")[0]
    assert "grade_completion(" in body, "accuracy_reward no longer shares the entry point"


def test_shared_entry_point_normalises():
    from rl.reward import grade_completion
    g = grade_completion(DIALOGUE_TAIL, 77, [43, 33, 25, 26])
    assert g.parsed and g.correct
