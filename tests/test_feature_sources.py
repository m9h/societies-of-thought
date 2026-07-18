"""Neuronpedia source resolution -- making feature selection model-agnostic.

Feature selection reads Neuronpedia's S3 export to get per-feature labels, max
activations and firing rates. Those live under

    v1/<neuronpedia_model_id>/<source_id>/{explanations,features}/batch-N.jsonl.gz

Both parts differ per model family, and getting either wrong yields an empty download
that looks exactly like "this model has no features" rather than "you asked for the
wrong path":

    model                       np model id           source id
    DeepSeek-R1-Distill-8B      deepseek-r1-distill-  15-llamascope-slimpj-res-32k
                                llama-8b
    Gemma 3 27B IT              gemma-3-27b-it        16-gemmascope-2-res-16k

Verified against the live S3 listing: gemma-3-27b-it has res SAEs only at layers
16/31/40/41/53 (widths 16k/65k/262k), plus transcoders at every layer 0-61. Asking for
a res source at, say, layer 20 is a silent empty result -- so it must raise instead.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from sot.sources import GEMMA3_27B_RES_LAYERS, neuronpedia_source


def test_llama_scope_source_unchanged():
    """The existing Llama path must keep resolving exactly as before."""
    np_model, src = neuronpedia_source("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                                       layer=15, mixture="slimpj")
    assert np_model == "deepseek-r1-distill-llama-8b"
    assert src == "15-llamascope-slimpj-res-32k"


def test_gemma_scope_source():
    np_model, src = neuronpedia_source("google/gemma-3-27b-it", layer=16, width="16k")
    assert np_model == "gemma-3-27b-it"
    assert src == "16-gemmascope-2-res-16k"


@pytest.mark.parametrize("layer", sorted(GEMMA3_27B_RES_LAYERS))
def test_every_advertised_gemma_layer_resolves(layer):
    """The layers we claim to support must all produce a source id."""
    _, src = neuronpedia_source("google/gemma-3-27b-it", layer=layer, width="16k")
    assert src == f"{layer}-gemmascope-2-res-16k"


def test_unavailable_gemma_layer_raises_instead_of_returning_a_dead_path():
    """THE TRAP. Gemma 3 27B has res SAEs at 16/31/40/41/53 only. A request for layer 20
    must fail loudly -- a dead S3 prefix downloads zero batches, which is indistinguishable
    from 'this SAE has no labelled features'."""
    with pytest.raises(ValueError, match="(?i)layer 20.*(not available|no res)"):
        neuronpedia_source("google/gemma-3-27b-it", layer=20, width="16k")


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="(?i)unknown model"):
        neuronpedia_source("meta-llama/Llama-3.1-70B", layer=15)
