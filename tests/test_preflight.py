"""A launch gate that must pass, on CPU, before any GPU is rented.

WHY THIS EXISTS. On 2026-07-19 three RunPod pods ran the A/B on code with a
result-inverting bug: the eval path and reward path graded dialogue answers
differently, so the dialogue arm reported 0% accuracy while scoring 0.4 reward.
It cost ~$40 and was catchable for free -- the dialogue SFT traces contain zero
<answer> tags in 500 rows, which a local check reads in milliseconds.

The bug was not the deepest problem. The deepest problem was that NOTHING had to
pass before money flowed. There were tests, but no gate that the launch script
was required to clear. So this module is that gate: `python -m rl.preflight`
returns nonzero if the experiment is not ready to run, and the pod runner calls
it before the first `train_grpo` invocation.

Everything here runs on CPU in milliseconds. It checks the invariants that, if
violated, waste GPU hours producing meaningless curves:

  1. both SFT arms cover identical problems with identical targets
     (the paper's design; a difference otherwise is content, not format)
  2. every shipped SFT trace parses through the SAME grader the eval uses
     (the bug that inverted the result)
  3. both arms' format earns format reward
     (a strict-format regression would punish dialogue for its scaffolding)
  4. the eval path and reward path are the one shared function, structurally
     (so they cannot silently diverge again)
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from rl.preflight import PreflightError, preflight


def test_preflight_passes_on_the_shipped_data():
    """The committed rl/data must be launch-ready, or the harness cannot run."""
    report = preflight()
    assert report["ready"] is True, report
    assert report["dialogue_parse_rate"] >= 0.95
    assert report["monologue_parse_rate"] >= 0.95
    assert report["problems_identical"] is True


def test_preflight_rejects_mismatched_problem_sets(tmp_path):
    """If the two arms solved different problems, format-vs-content is confounded."""
    import json

    (tmp_path / "dialogue_train.json").write_text(json.dumps([
        {"numbers": [1, 2, 3], "target": 6, "dialogue":
         "<think1> 1+2+3 </think1>\n<group_consensus> 1 + 2 + 3 </group_consensus>"}]))
    (tmp_path / "monologue_train.json").write_text(json.dumps([
        {"numbers": [4, 5, 6], "target": 15, "monologue":
         "<think> 4+5+6 </think>\n<answer> 4 + 5 + 6 </answer>"}]))
    with pytest.raises(PreflightError, match="(?i)problem"):
        preflight(data_dir=tmp_path)


def test_preflight_rejects_unparseable_arm(tmp_path):
    """THE regression. An arm whose traces the eval path cannot parse would train
    and report 0% -- exactly the dialogue failure. This is what must fail here
    rather than on a rented GPU."""
    import json

    prob = {"numbers": [43, 33, 25, 26], "target": 77}
    # dialogue trace with NO answer container of any kind -> ungradeable
    (tmp_path / "dialogue_train.json").write_text(json.dumps([
        {**prob, "dialogue": "<think1> hmm let me think about this </think1>"}]))
    (tmp_path / "monologue_train.json").write_text(json.dumps([
        {**prob, "monologue": "<think> 43+33-25+26 </think>\n<answer> 43 + 33 - 25 + 26 </answer>"}]))
    with pytest.raises(PreflightError, match="(?i)parse|gradeable|answer"):
        preflight(data_dir=tmp_path)


def test_preflight_returns_actionable_report_not_just_bool():
    """A gate is only useful if a failure says what to fix. The report must name
    which arm and which check failed."""
    report = preflight()
    assert "dialogue_parse_rate" in report
    assert "monologue_parse_rate" in report
    assert "problems_identical" in report
    assert "checks" in report and isinstance(report["checks"], dict)


def test_preflight_verifies_the_shared_grader_structurally():
    """Data can be fine today while the code paths silently diverge tomorrow.
    Preflight also asserts EvalAndLog and accuracy_reward share grade_completion,
    so a future edit that reintroduces the split fails the gate."""
    report = preflight()
    assert report["checks"]["shared_grader"] is True


# --- the gate must be wired into the runner, not just exist ---------------------

def test_runner_calls_preflight_before_training():
    """A gate nobody runs is not a gate. The pod runner must invoke rl.preflight
    and must do it BEFORE the first train_grpo call, or a broken experiment
    reaches the GPU again."""
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    script = (repo / "scripts/run_rl_arm.sh").read_text()
    assert "rl.preflight" in script, "runner does not call rl.preflight"
    pre = script.index("rl.preflight")
    train = script.index("rl.train_grpo")
    assert pre < train, "preflight must run BEFORE training, not after"
    # and it must be able to stop the run
    gate = script[pre - 60:train]
    assert "exit 1" in gate or "exit 1" in script[:train], (
        "preflight failure must abort the run (exit 1), not just print"
    )
