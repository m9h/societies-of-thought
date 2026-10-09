"""Judge-human agreement sheet — tests BEFORE implementation.

Norman et al. make judge-human agreement the validity anchor. We have never measured it.
The sheet must be blind (no judge verdicts, no training step visible), stratified across
training so the human sees early and late traces, and scorable with the same statistics
used for judge-judge agreement.
"""
from __future__ import annotations

import csv
import json
import random

import pytest

from analysis.human_annotation import score_ratings, select_traces, write_sheet
from rl.judge import BEHAVIOURS

FIELDS = (*BEHAVIOURS, "n_personas")


def _judged(n=256, bins=8):
    rng = random.Random(0)
    out = []
    for i in range(n):
        v = {f: rng.randint(0, 3) for f in BEHAVIOURS}
        v["n_personas"] = 1
        out.append({"step": i, "bin": i * bins // n, "response": f"trace number {i}", **v})
    return out


def test_select_is_stratified_and_sized():
    sel = select_traces(_judged(), n=50, seed=1)
    assert len(sel) == 50
    per_bin = {}
    for r in sel:
        per_bin[r["bin"]] = per_bin.get(r["bin"], 0) + 1
    assert set(per_bin) == set(range(8))
    assert max(per_bin.values()) - min(per_bin.values()) <= 1


def test_sheet_is_blind(tmp_path):
    sel = select_traces(_judged(), n=50, seed=1)
    paths = write_sheet(sel, tmp_path)
    sheet = (tmp_path / "sheet.md").read_text()
    for phrase in ("A question is posed and later answered, as in conversations.",
                   "A transition to a different idea, viewpoint, assumption, or approach",
                   "Expressions of disagreement, correction, or tension with another perspective.",
                   "Conflicting views are integrated or resolved into a coherent synthesis.",
                   "identify the number of distinct perspectives (agents or voices)",
                   "Transitional markers"):
        assert phrase in sheet, phrase                 # the paper's definitions, verbatim
    assert "step" not in sheet.lower().replace("step-by-step", "")
    assert "judge" not in sheet.lower()
    # verdict numbers must not leak: the sheet carries ids and traces only
    key = json.loads((tmp_path / "answer_key.json").read_text())
    assert len(key) == 50
    for rid, rec in key.items():
        assert rid in sheet
        assert all(f in rec for f in FIELDS)
        assert "step" in rec and "bin" in rec
    assert set(paths) == {"sheet", "key", "template"}


def test_sheet_order_is_shuffled_relative_to_training():
    sel = select_traces(_judged(), n=50, seed=1)
    steps = [r["step"] for r in sel]
    assert steps != sorted(steps)


def test_template_columns_and_scoring_roundtrip(tmp_path):
    sel = select_traces(_judged(), n=50, seed=1)
    write_sheet(sel, tmp_path)
    key = json.loads((tmp_path / "answer_key.json").read_text())
    tmpl = tmp_path / "ratings_template.csv"
    with tmpl.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0].keys() == {"id", *FIELDS}
    assert len(rows) == 50
    # fill the template with the judge's own answers -> perfect agreement
    filled = tmp_path / "human.csv"
    with filled.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        w.writeheader()
        for r in rows:
            w.writerow({"id": r["id"], **{f: key[r["id"]][f] for f in FIELDS}})
    rep = score_ratings(filled, tmp_path / "answer_key.json")
    qa = next(r for r in rep["agreement"] if r["field"] == "question_answering")
    assert qa["exact"] == pytest.approx(1.0)
    assert qa["kappa"] == pytest.approx(1.0)
    assert rep["n_rated"] == 50


def test_blank_rows_are_skipped_and_counted(tmp_path):
    sel = select_traces(_judged(), n=50, seed=1)
    write_sheet(sel, tmp_path)
    key = json.loads((tmp_path / "answer_key.json").read_text())
    filled = tmp_path / "human.csv"
    ids = list(key)
    with filled.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        w.writeheader()
        for i, rid in enumerate(ids):
            if i < 10:
                w.writerow({"id": rid, **{f: key[rid][f] for f in FIELDS}})
            else:
                w.writerow({"id": rid, **{f: "" for f in FIELDS}})
    rep = score_ratings(filled, tmp_path / "answer_key.json")
    assert rep["n_rated"] == 10
    assert rep["n_blank"] == 40
