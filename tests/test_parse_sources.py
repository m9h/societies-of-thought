"""Parser corrections found while cross-judging — tests BEFORE implementation.

Three things the first parse got wrong, each one visible only once the judge verdicts were
read against the paper's Methods text:

1. verl prints its validation generations (greedy, n=1) into the same log, after a
   `validation generation end` marker and before the `step:N` line that carries the val
   score. The parser attributed them to that step as if they were temperature-1 training
   rollouts: ~20% of "training" traces were greedy validation samples, concentrated at
   steps 0, 10, 20, ... The paper judges validation generations, so these must be kept --
   as their own series.
2. TinyZero's scorer prints its verdict (`Invalid equation`, `No equation found`,
   `Correct equation: ...`, `Wrong result: ...`) AFTER the solution string; 2,239 of 2,555
   responses ended with it. Everything after `<|endoftext|>` is not model output.
3. `Assistant:` is the prompt's last token, not the generation's first.
"""
from __future__ import annotations

from analysis.emergence import parse_rollouts

RAY = "\x1b[36m(main_task pid=5487)\x1b[0m "


def _echo(nums, target, body, verdict="Invalid equation"):
    return (f"{RAY}--------------------------------\n"
            f"{RAY}Target: {target} | Numbers: {nums}\n"
            f"{RAY}Extracted equation: None\n"
            f"{RAY}Solution string: A conversation between User and Assistant.\n"
            f"{RAY}User: Using the numbers {nums}, create an equation that equals {target}.\n"
            f"{RAY}Assistant: {body}<|endoftext|>\n"
            f"{RAY}{verdict}\n")


LOG = (
    f"{RAY}validation generation end\n"
    + _echo("[1, 2]", 3, "<think> 1+2 </think>\n<answer>1+2</answer>", "Correct equation: 1+2 = 3")
    + f"{RAY}\"Initial validation metrics: {{'val/test_score/countdown': 0.1}}\"\n"
    + f"{RAY}step:0 - val/test_score/countdown:0.104\n"
    + _echo("[4, 5]", 9, "<think> 4+5 </think>\n<answer>4+5</answer>", "Wrong result: equation = 9, target = 9")
    + _echo("[7, 8]", 15, "<think> 7+8 </think>\n<answer>7+8</answer>")
    + f"{RAY}step:1 - global_seqlen/min:1\n"
    + _echo("[2, 3]", 5, "<think> 2+3 </think>\n<answer>2+3</answer>", "No equation found")
    + f"{RAY}validation generation end\n"
    + _echo("[6, 6]", 12, "<think> 6+6 </think>\n<answer>6+6</answer>")
    + f"{RAY}step:10 - global_seqlen/min:1 - val/test_score/countdown:0.3\n"
)


def test_validation_echoes_are_tagged_not_mixed_in():
    r = parse_rollouts(LOG)
    assert [(x["step"], x["source"]) for x in r] == [
        (0, "validation"), (1, "train"), (1, "train"), (10, "train"), (10, "validation")]


def test_response_ends_at_endoftext_and_drops_scorer_verdict():
    r = parse_rollouts(LOG)
    for x in r:
        assert "<|endoftext|>" not in x["response"]
        for bad in ("Invalid equation", "No equation found", "Correct equation", "Wrong result"):
            assert bad not in x["response"], (bad, x["response"])


def test_assistant_prefix_is_prompt_not_generation():
    r = parse_rollouts(LOG)
    assert r[0]["response"].startswith("<think>")
    assert not any(x["response"].lstrip().startswith("Assistant") for x in r)


def test_scorer_verdict_line_without_endoftext_is_still_stripped():
    log = (f"{RAY}step:4 - x\n"
           + f"{RAY}--------------------------------\n{RAY}Target: 3 | Numbers: [1, 2]\n"
           + f"{RAY}Extracted equation: None\n{RAY}Solution string: A conversation\n"
           + f"{RAY}User: Using the numbers [1, 2], create an equation that equals 3.\n"
           + f"{RAY}Assistant: <think> truncated at max_tokens 1+\n"
           + f"{RAY}No equation found\n"
           + f"{RAY}step:5 - x\n")
    r = parse_rollouts(log)
    assert len(r) == 1
    assert r[0]["response"].strip() == "<think> truncated at max_tokens 1+"
