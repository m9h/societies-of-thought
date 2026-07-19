"""Gemma Scope 2 publishes THREE different layer sets, and we conflated two of them.

Verified against the live HF tree and the Neuronpedia S3 listing on 2026-07-18:

  Neuronpedia LABELS        16, 31, 40, 41, 53
    (v1/gemma-3-27b-it/<L>-gemmascope-2-res-<width>/)  widths 16k/65k/262k,
    except layer 41 which has 16k/262k only.

  HF WEIGHTS, flagship      16, 31, 40, 53          resid_post/
    widths 16k/65k/262k/1m, l0 small/medium/big.

  HF WEIGHTS, everything    0..61  (all 62)         resid_post_all/
    widths 16k/262k ONLY, l0 small/big ONLY -- there is no `medium`.

These are not the same set and they are not nested. Layer 41 has labels and has
weights in resid_post_all, but is absent from the flagship directory. Layer 20 has
weights in resid_post_all but no labels at all.

The bug this pins: registry.py used the LABEL set as the WEIGHT-availability set.
That greenlights layer 41 for steering and then download_gemma_sae -- whose own list
is the flagship [16,31,40,53] -- rejects it. The contradiction only fires after the
model is resident on a GPU.

The second bug: our default is l0="medium", which exists ONLY in the flagship
directory. Any of the other 58 layers requested at the default l0 is a 404 on a path
we asserted was valid -- exactly the "dead prefix looks like an unlabelled SAE"
failure sources.py warns about in its own docstring.

Steering needs WEIGHTS. Feature selection by description needs LABELS. Availability
is therefore a function of what you are about to do, which is why they are separate
names here rather than one frozenset.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from sot.sources import (
    GEMMA_SCOPE_ALL_LAYERS,
    GEMMA_SCOPE_FLAGSHIP_LAYERS,
    NEURONPEDIA_GEMMA3_27B_LAYERS,
    gemma_sae_location,
    neuronpedia_source,
)


# --- the three sets are distinct ------------------------------------------------

def test_the_three_layer_sets_are_not_the_same():
    assert NEURONPEDIA_GEMMA3_27B_LAYERS == frozenset({16, 31, 40, 41, 53})
    assert GEMMA_SCOPE_FLAGSHIP_LAYERS == frozenset({16, 31, 40, 53})
    assert GEMMA_SCOPE_ALL_LAYERS == frozenset(range(62))


def test_layer_41_has_labels_but_no_flagship_weights():
    """The exact discrepancy that made the registry inconsistent with the loader."""
    assert 41 in NEURONPEDIA_GEMMA3_27B_LAYERS
    assert 41 not in GEMMA_SCOPE_FLAGSHIP_LAYERS
    assert 41 in GEMMA_SCOPE_ALL_LAYERS


# --- weights: which directory, and which width/l0 are legal there ---------------

@pytest.mark.parametrize("layer", [16, 31, 40, 53])
def test_flagship_layers_resolve_to_resid_post(layer):
    loc = gemma_sae_location(layer, width="16k", l0="medium")
    assert loc.startswith("resid_post/"), loc
    assert loc == f"resid_post/layer_{layer}_width_16k_l0_medium"


@pytest.mark.parametrize("layer", [0, 5, 20, 41, 61])
def test_non_flagship_layers_resolve_to_resid_post_all(layer):
    loc = gemma_sae_location(layer, width="16k", l0="big")
    assert loc == f"resid_post_all/layer_{layer}_width_16k_l0_big"


def test_medium_l0_outside_the_flagship_is_rejected_not_404d():
    """resid_post_all ships small/big only. Our DEFAULT is medium, so this is the
    live failure path: layer 20 at defaults must fail loudly here, not on the GPU."""
    with pytest.raises(ValueError, match="(?i)l0.*medium|medium.*not"):
        gemma_sae_location(20, width="16k", l0="medium")


def test_65k_width_outside_the_flagship_is_rejected():
    """65k and 1m exist only for the four flagship layers."""
    with pytest.raises(ValueError, match="(?i)width"):
        gemma_sae_location(20, width="65k", l0="big")


def test_flagship_keeps_its_rich_width_and_l0_options():
    assert gemma_sae_location(16, width="1m", l0="medium").startswith("resid_post/")
    assert gemma_sae_location(31, width="65k", l0="small").startswith("resid_post/")


def test_layer_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="(?i)layer"):
        gemma_sae_location(62, width="16k", l0="big")


# --- labels: unchanged, and still guarded ---------------------------------------

def test_neuronpedia_still_guards_on_the_label_set():
    """Layer 20 has WEIGHTS but no published labels -- feature selection by
    description genuinely cannot work there, and must say so."""
    assert neuronpedia_source("google/gemma-3-27b-it", 41) == (
        "gemma-3-27b-it", "41-gemmascope-2-res-16k")
    with pytest.raises(ValueError, match="(?i)not available|label"):
        neuronpedia_source("google/gemma-3-27b-it", 20)


def test_layer_41_has_no_65k_labels():
    """Layer 41's S3 export has 16k and 262k only -- the one asymmetry in the set."""
    with pytest.raises(ValueError, match="(?i)65k|width"):
        neuronpedia_source("google/gemma-3-27b-it", 41, width="65k")
