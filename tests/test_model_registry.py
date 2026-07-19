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
  layers        Gemma publishes resid_post SAEs at all 62 layers but Neuronpedia
                LABELS at only 16/31/40/41/53 -- weights and labels are different,
                non-nested sets. Llama Scope (slimpj) publishes layer 15 only.

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


def test_gemma_layer_out_of_range_raises():
    """62 decoder blocks; layer 62 does not exist at all."""
    with pytest.raises(ValueError, match="(?i)not available"):
        resolve_model("google/gemma-3-27b-it", layer=62)


def test_unlabelled_gemma_layer_resolves_for_steering_with_neuronpedia_none():
    """Weights and labels are different sets. Gemma publishes resid_post SAEs at all
    62 layers but explanations at only 5, so layer 20 IS steerable -- it just cannot
    have its features chosen by description.

    This used to raise, because resolve_model built a Neuronpedia source id
    unconditionally. That blocked 57 steerable layers, which is the whole reason the
    Gemma arm looked like it had 5 layers to sweep instead of 62."""
    spec = resolve_model("google/gemma-3-27b-it", layer=20)
    assert spec.layer == 20
    assert spec.hook_layer == 20
    assert spec.neuronpedia is None, (
        "no published explanations at layer 20 -- record the absence rather than "
        "raising, so steering still works"
    )


def test_labelled_gemma_layer_still_gets_its_source():
    assert resolve_model("google/gemma-3-27b-it", layer=41).neuronpedia == (
        "gemma-3-27b-it", "41-gemmascope-2-res-16k")


def test_caller_errors_are_not_swallowed_by_the_no_labels_path():
    """resolve_model catches NoLabelsPublished so unlabelled layers still resolve.
    It must NOT catch bare ValueError -- an unknown model or a bad mixture is a
    caller error, and swallowing it turns a typo into a silent no-op. This is the
    same failure mode that hid a jlens bug behind `except Exception: continue`."""
    with pytest.raises(ValueError, match="(?i)unknown model"):
        resolve_model("mistralai/Mistral-7B-v0.3", layer=15)
    with pytest.raises(ValueError, match="(?i)unknown mixture"):
        resolve_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 15, mixture="nope")


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


# --- layer availability is MIXTURE-dependent for Llama Scope -------------------
# Regression guard. The registry first hardcoded layers={15} for DeepSeek, which is true
# only of the pure-slimpj SAE (the paper's). The slimpj+openr1 "mixed" suite is published
# for ALL 32 layers -- and that is what the layer sweep uses. Flattening the two blocked
# 5 of the 6 layers in a sweep we had already started running.

@pytest.mark.parametrize("layer", [5, 10, 15, 20, 25, 30])
def test_mixed_mixture_exposes_all_layers(layer):
    spec = resolve_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", layer, mixture="mixed")
    assert spec.layer == layer
    assert spec.neuronpedia[1] == f"{layer}-llamascope-slimpj-openr1-res-32k"


def test_slimpj_mixture_is_layer_15_only():
    """The paper's own SAE genuinely exists at one layer -- that constraint is real."""
    assert resolve_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 15,
                         mixture="slimpj").layer == 15
    with pytest.raises(ValueError, match="(?i)not available"):
        resolve_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 20, mixture="slimpj")
