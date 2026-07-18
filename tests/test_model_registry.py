"""Model registry -- one place that knows how to steer each supported model.

run_sweep hardcoded DeepSeek-R1-Distill-Llama-8B and the Llama Scope loader. To run the
same experiment on Gemma 3 27B (a model the paper never tested) the sweep has to be
model-agnostic, and the per-model differences have to live in exactly one place instead
of being re-derived at each call site.

Three things differ per model, and each has burned us once:

  loader        Llama Scope vs Gemma Scope store weights transposed, with log vs raw
                thresholds, with and without dataset-wise rescaling.
  hook layer    Llama Scope's Neuronpedia metadata says resid_pre; the SAE config says
                resid_post and reconstruction proved the config right. Gemma states
                `model.layers.N.output` unambiguously. The offset from "SAE layer" to
                "layer whose output we hook" is therefore per-family.
  layers        Gemma publishes residual SAEs at 16/31/40/41/53 only. Llama Scope
                (slimpj) publishes layer 15 only.

The registry is data, so a wrong entry fails as a test here rather than as a plausible
number on a rented GPU.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest

from sot.registry import MODELS, resolve_model


def test_known_models_are_registered():
    assert "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" in MODELS
    assert "google/gemma-3-27b-it" in MODELS


def test_deepseek_resolves_to_llama_scope_at_layer_15():
    spec = resolve_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", layer=15)
    assert spec.sae_kind == "llama_scope"
    assert spec.layer == 15
    # resid_post(L) is the output of layer L -> hook layer L itself (offset 0),
    # settled empirically by reconstruction, NOT by Neuronpedia's metadata.
    assert spec.hook_layer == 15


def test_gemma_resolves_to_gemma_scope():
    spec = resolve_model("google/gemma-3-27b-it", layer=16)
    assert spec.sae_kind == "gemma_scope"
    assert spec.layer == 16
    assert spec.hook_layer == 16          # config says model.layers.16.output
    assert spec.neuronpedia == ("gemma-3-27b-it", "16-gemmascope-2-res-16k")


@pytest.mark.parametrize("layer", [16, 31, 40, 41, 53])
def test_all_published_gemma_layers_resolve(layer):
    spec = resolve_model("google/gemma-3-27b-it", layer=layer)
    assert spec.layer == layer


def test_gemma_layer_without_a_published_sae_raises():
    """A dead layer must fail here, not silently download zero features on a GPU box."""
    with pytest.raises(ValueError, match="(?i)not available"):
        resolve_model("google/gemma-3-27b-it", layer=20)


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="(?i)unknown model"):
        resolve_model("mistralai/Mistral-7B-v0.3", layer=15)


def test_default_layer_is_the_papers_layer_for_deepseek():
    """Layer 15 is the paper's claimed mechanism site; keep it the default so the
    existing results stay reproducible without passing --layer."""
    spec = resolve_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    assert spec.layer == 15


def test_anchor_feature_is_model_specific_not_layer_specific():
    """Feature 30939 is the paper's conversational-surprise feature IN THE LLAMA SCOPE
    SAE. Index 30939 in Gemma Scope is an unrelated feature -- different SAE, different
    width, different training. Anchoring Gemma on it would silently seed the whole
    candidate search from a meaningless direction.

    The old guard was `if layer == 15`, which worked only because Gemma happens to have
    no layer 15. Make it explicit so it survives a new model or a new layer."""
    llama = resolve_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", layer=15)
    gemma = resolve_model("google/gemma-3-27b-it", layer=16)
    assert llama.anchor_feature == 30939, "the paper's feature, for the paper's model"
    assert gemma.anchor_feature is None, (
        "Gemma has no known conversational anchor; selection must fall back to the "
        "lexicon rather than reuse an index from a different SAE"
    )
