"""Re-grading saved traces, and the truncation flag it repairs.

`model.generate` returns every row in a batch padded to the batch's LONGEST sequence.
The sweep computed `truncated = len(row) >= max_new_tokens`, which is a property of the
BATCH, not of the sequence: one runaway generation marked all 24 of its batch-mates
truncated. Observed: 96% "truncation" where the truth was 12%.

That matters because truncation is the metric that distinguishes a *formatting* failure
from a *reasoning* failure -- the whole point of tracking it. `n_tokens` (a real
per-sequence count) was always right, so the flag is recoverable exactly.
"""

from __future__ import annotations

import json
import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from sot.regrade import main as regrade_main


def _row(**kw):
    base = dict(
        task="countdown", pid="cd-1", sample=0, layer=15, hook_layer=15,
        feature=-1, role="baseline", alpha=0.0, strength=0.0, scope="all",
        correct=False, parsed=False, pred=None, gold="32|25,30,3,4",
        n_tokens=500, truncated=True, markers={}, trace="",
    )
    base.update(kw)
    return base


def _run(tmp_path, rows, max_new_tokens=4096):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "out.jsonl"
    sys.argv = ["regrade", "--results", str(src), "--out", str(out),
                "--max-new-tokens", str(max_new_tokens)]
    regrade_main()
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_truncation_is_per_sequence_not_per_batch(tmp_path):
    """A short trace in a batch with a long one is NOT truncated."""
    rows = [
        _row(pid="short", n_tokens=500, truncated=True),   # mislabelled by the old bug
        _row(pid="long", n_tokens=4096, truncated=True),   # genuinely truncated
    ]
    out = _run(tmp_path, rows, max_new_tokens=4096)
    by = {r["pid"]: r for r in out}
    assert by["short"]["truncated"] is False, "short trace must not inherit its batch's truncation"
    assert by["long"]["truncated"] is True


def test_regrade_rescues_latex_answers_without_forgiving_wrong_ones(tmp_path):
    r"""The core asymmetry: widen what counts as FOUND, never what counts as RIGHT."""
    rows = [
        # correct, stated in LaTeX -> was scored wrong+unparsed, must now be right
        _row(pid="latex-ok", trace=r"\boxed{(30 - 25 + 3) * 4 = 32}",
             correct=False, parsed=False),
        # stated in LaTeX but the arithmetic is wrong -> must STAY wrong
        _row(pid="latex-bad", trace=r"\boxed{(30 - 25) * 4 = 32}",
             correct=False, parsed=False),
        # degenerate babble -> no answer at all -> wrong and unparsed
        _row(pid="babble", trace="oh wait no wait oh wait", correct=False, parsed=False),
    ]
    out = _run(tmp_path, rows)
    by = {r["pid"]: r for r in out}

    assert by["latex-ok"]["correct"] is True and by["latex-ok"]["parsed"] is True
    assert by["latex-bad"]["correct"] is False, "re-grading must not forgive bad arithmetic"
    assert by["babble"]["correct"] is False and by["babble"]["parsed"] is False


def test_regrade_refuses_without_traces(tmp_path):
    """Fail loudly rather than silently re-grading nothing."""
    src = tmp_path / "in.jsonl"
    r = _row()
    r.pop("trace")
    src.write_text(json.dumps(r))
    sys.argv = ["regrade", "--results", str(src)]
    with pytest.raises(SystemExit, match="--save-traces"):
        regrade_main()
