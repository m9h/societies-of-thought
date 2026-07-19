"""The layer sweep grid must live in exactly one place.

It did not. As of 2026-07-18 there were THREE grids in play and no two agreed:

    scripts/run_stages.sh:117   --layers 9 12 15 18 21 24
    calibration on disk         feature_stats_L{5,10,15,20,25,30}_mixed.json
    completed rows              layer 5 only (572 rows, math_hard)

So the committed script asks for a grid whose calibration does not exist, the
calibration on disk was computed for a grid nothing references, and the only
sweep that actually ran used a third thing. A sweep launched from the script
would have recalibrated five layers from scratch and thrown away the 572 rows
already paid for -- while the calibration for the layers it skipped sat unused.

The chosen grid is 5/10/15/20/25/30 for two reasons:

  cost     calibration exists for all six, and the completed rows are at layer
           5, so both are reusable. The script's grid has calibration for 15
           only.

  design   it spans 16%-94% of a 32-layer model roughly uniformly. The script's
           grid spans 28%-75% clustered around layer 15 -- which bakes in the
           paper's prior about where the mechanism lives. Our own weight
           analysis puts layer 15 at rank 21/32, and the 2026 steering
           literature puts the usual sweet spot nearer 75% depth, so a grid
           centred on 15 is the one shape we should NOT commit to.

These tests pin the grid and, more importantly, pin that the shell script does
not carry its own copy of it.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from sot.registry import layer_sweep_grid, resolve_model

REPO = Path(__file__).resolve().parents[1]
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def test_grid_is_the_calibrated_one():
    assert layer_sweep_grid(DEEPSEEK) == (5, 10, 15, 20, 25, 30)


def test_every_grid_layer_resolves_under_the_mixture_the_sweep_uses():
    """slimpj exists at layer 15 only, so a multi-layer sweep must use `mixed`.
    A grid entry that cannot resolve would fail after the model is on the GPU."""
    for layer in layer_sweep_grid(DEEPSEEK):
        spec = resolve_model(DEEPSEEK, layer, mixture="mixed")
        assert spec.layer == layer
        assert spec.neuronpedia is not None, "the sweep selects features by label"


def test_grid_spans_the_full_depth_rather_than_clustering_on_layer_15():
    """Regression guard on the DESIGN, not just the numbers.

    The rejected grid (9..24) sat entirely within the middle half of the model.
    If someone re-centres the grid on the paper's layer, this fails and makes
    them say why."""
    grid = layer_sweep_grid(DEEPSEEK)
    n_layers = 32
    assert min(grid) / n_layers < 0.25, "must probe the early layers"
    assert max(grid) / n_layers > 0.85, "must probe the late layers"


def test_calibration_exists_for_every_grid_layer():
    """The grid is only cheap because these files already exist. If the grid
    changes without recalibration, say so here rather than discovering it as a
    missing-file crash partway through a paid sweep."""
    missing = [
        layer for layer in layer_sweep_grid(DEEPSEEK)
        if not (REPO / f"results/steering/feature_stats_L{layer}_mixed.json").exists()
    ]
    if missing:
        pytest.skip(f"calibration absent for {missing} (results/ may be untracked here)")


def test_run_stages_does_not_hardcode_its_own_grid():
    """THE ACTUAL BUG. Three sources of truth is what let them drift apart.

    The sweep stage must interpolate the grid from Python, not carry a literal
    list that can silently diverge from what was calibrated."""
    script = (REPO / "scripts/run_stages.sh").read_text()

    # Find every multi-layer --layers argument in the file.
    multi = [
        m.group(1).strip()
        for m in re.finditer(r"--layers\s+((?:\d+\s+){1,}\d+)", script)
    ]
    assert not multi, (
        f"scripts/run_stages.sh hardcodes layer grid(s) {multi}. The sweep grid "
        "lives in sot.registry.layer_sweep_grid; the script must interpolate it "
        "(e.g. --layers $(python -m sot.sweep_grid)) so the two cannot drift."
    )
