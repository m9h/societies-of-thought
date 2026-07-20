"""Launch gate: refuse to spend GPU on an experiment that cannot produce a valid result.

`python -m rl.preflight` exits nonzero if the A/B is not ready. The pod runner
calls it before the first training step, so a data or wiring bug fails on CPU in
milliseconds rather than after hours on a rented box.

The checks are exactly the invariants whose violation wastes money silently:

  problems_identical   dialogue and monologue must cover the same problems with
                       the same targets. Otherwise a difference between arms is
                       content, not reasoning format -- the paper's whole design.

  <arm>_parse_rate     every shipped trace must grade through grade_completion,
                       the SAME entry point EvalAndLog uses. The dialogue traces
                       end in <group_consensus> with zero <answer> tags; if the
                       eval could not parse them the arm reports 0% while scoring
                       real reward, which is the bug that cost ~$40 on 2026-07-19.

  <arm>_format_rate    both formats must earn format reward. A strict-format
                       regression would punish the dialogue arm for producing the
                       very scaffolding under test.

  shared_grader        EvalAndLog and accuracy_reward must both route through
                       grade_completion, structurally, so the two paths cannot
                       diverge again on a future edit.

None of this needs a GPU, a model download, or the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from rl.reward import format_reward, grade_completion

DATA_DIR = Path(__file__).resolve().parent / "data"
MIN_PARSE_RATE = 0.95
MIN_FORMAT_RATE = 0.95


class PreflightError(RuntimeError):
    """The experiment is not ready to run. The message names the failed check."""


def _problem_key(row: dict) -> tuple:
    return (tuple(sorted(row["numbers"])), row["target"])


def _load(arm: str, data_dir: Path) -> list[dict]:
    path = data_dir / f"{arm}_train.json"
    if not path.exists():
        raise PreflightError(f"{arm}: {path} does not exist -- nothing to train on")
    return json.loads(path.read_text())


def _check_shared_grader() -> bool:
    """EvalAndLog and accuracy_reward must both call grade_completion.

    Source-level, not behavioural: two paths can agree on today's inputs and
    diverge on tomorrow's. The structural guarantee is that there is one function.
    """
    repo = Path(__file__).resolve().parent.parent
    train = (repo / "rl/train_grpo.py").read_text()
    eval_body = train.split("class EvalAndLog")[1].split("\ndef main")[0]
    if "grade_completion(" not in eval_body:
        return False
    if any("grade(" in l and "grade_completion(" not in l and not l.strip().startswith("#")
           for l in eval_body.splitlines()):
        return False
    reward = (repo / "rl/reward.py").read_text()
    reward_body = reward.split("def accuracy_reward")[1].split("\ndef ")[0]
    return "grade_completion(" in reward_body


def preflight(data_dir: Path = DATA_DIR, *, arms=("dialogue", "monologue")) -> dict:
    """Return a readiness report. Raise PreflightError on any hard failure.

    The report is actionable: every rate and every boolean check is named, so a
    caller (or a human reading the pod log) can see which arm and which invariant
    is the problem without rerunning.
    """
    report: dict = {"checks": {}}

    loaded = {arm: _load(arm, data_dir) for arm in arms}

    # 1. identical problem sets and targets
    keysets = {arm: {_problem_key(r) for r in rows} for arm, rows in loaded.items()}
    a, b = arms
    identical = keysets[a] == keysets[b]
    report["problems_identical"] = identical
    report["checks"]["problems_identical"] = identical
    if not identical:
        only_a = len(keysets[a] - keysets[b])
        only_b = len(keysets[b] - keysets[a])
        raise PreflightError(
            f"problem sets differ: {only_a} only in {a}, {only_b} only in {b}. "
            "The arms must solve identical problems, or a between-arm difference "
            "is content rather than reasoning format."
        )

    # 2 & 3. every trace parses through the eval grader and earns format reward
    for arm, rows in loaded.items():
        parsed = fmt = 0
        for r in rows:
            g = grade_completion(r[arm], r["target"], r["numbers"])
            parsed += bool(g.parsed)
            fmt += 1 if format_reward(r[arm]) == 1.0 else 0
        n = len(rows)
        parse_rate = parsed / n if n else 0.0
        format_rate = fmt / n if n else 0.0
        report[f"{arm}_parse_rate"] = parse_rate
        report[f"{arm}_format_rate"] = format_rate
        report["checks"][f"{arm}_parse"] = parse_rate >= MIN_PARSE_RATE
        report["checks"][f"{arm}_format"] = format_rate >= MIN_FORMAT_RATE
        if parse_rate < MIN_PARSE_RATE:
            raise PreflightError(
                f"{arm}: only {parse_rate:.1%} of {n} traces grade through the eval "
                f"path (need {MIN_PARSE_RATE:.0%}). generate_sft.py verified every "
                "trace at generation time, so a low rate means the eval grader "
                "cannot see this arm's answer format -- the bug that reports a "
                "working arm as 0% accuracy."
            )
        if format_rate < MIN_FORMAT_RATE:
            raise PreflightError(
                f"{arm}: only {format_rate:.1%} of traces earn format reward "
                f"(need {MIN_FORMAT_RATE:.0%}). Check format_reward accepts this "
                "arm's scaffolding."
            )

    # 4. the two grading paths are structurally one function
    shared = _check_shared_grader()
    report["checks"]["shared_grader"] = shared
    if not shared:
        raise PreflightError(
            "EvalAndLog and accuracy_reward do not both route through "
            "grade_completion. The eval path and reward path can diverge, which "
            "is what inverted the dialogue result. Restore the shared entry point."
        )

    report["ready"] = True
    return report


def main() -> int:
    try:
        report = preflight()
    except PreflightError as e:
        print(f"PREFLIGHT FAILED: {e}")
        return 1
    print("PREFLIGHT OK -- safe to launch")
    for arm in ("dialogue", "monologue"):
        print(f"  {arm:10s} parse={report[f'{arm}_parse_rate']:.1%} "
              f"format={report[f'{arm}_format_rate']:.1%}")
    print(f"  problems identical: {report['problems_identical']}")
    print(f"  shared grader:      {report['checks']['shared_grader']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
