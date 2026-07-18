"""One place that knows how to steer each supported model.

The sweep used to hardcode DeepSeek-R1-Distill-Llama-8B and the Llama Scope loader.
Running the same experiment on Gemma 3 27B -- a model the paper never tested -- means
the per-model differences have to live somewhere single rather than being re-derived at
each call site. Every one of these has cost us a wrong run at least once:

  sae_kind    Llama Scope and Gemma Scope store weights transposed relative to each
              other, with log vs raw JumpReLU thresholds, and with vs without
              dataset-wise rescaling. Separate loaders, chosen here.

  hook_layer  Which layer's OUTPUT carries the residual stream the SAE reads. For Llama
              Scope this was settled empirically by reconstruction (52.5% vs 27.5%
              explained variance) AGAINST Neuronpedia's published metadata, which says
              resid_pre and is wrong. Gemma states `model.layers.N.output` in its own
              config, so there is nothing to guess.

  layers      Which layers actually have a published residual SAE. Llama Scope slimpj:
              layer 15 only. Gemma 3 27B: 16/31/40/41/53 only.

Registry entries are data, so a wrong one fails in tests rather than as a plausible
number on a rented GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from sot.sources import GEMMA3_27B_RES_LAYERS, neuronpedia_source


@dataclass(frozen=True)
class ModelSpec:
    model: str
    layer: int
    sae_kind: str            # "llama_scope" | "gemma_scope"
    hook_layer: int          # layer whose OUTPUT we hook (resid_post)
    neuronpedia: tuple[str, str]   # (neuronpedia_model_id, source_id)
    d_model: int
    # Index of a KNOWN conversational feature in THIS model's SAE, or None.
    # 30939 is the paper's feature in Llama Scope; index 30939 in Gemma Scope is an
    # unrelated feature (different SAE, width, training run). None => fall back to the
    # lexicon selector rather than seeding the search from a meaningless direction.
    anchor_feature: int | None


@dataclass(frozen=True)
class _Entry:
    sae_kind: str
    layers: frozenset[int]
    default_layer: int
    d_model: int
    anchor_feature: int | None = None
    # offset from the SAE's layer index to the layer whose output we hook.
    # 0 for both families: resid_post(L) IS the output of layer L.
    hook_offset: int = 0


MODELS: dict[str, _Entry] = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": _Entry(
        sae_kind="llama_scope",
        layers=frozenset({15}),      # the pure-slimpj SAE the paper used exists only here
        default_layer=15,            # the paper's claimed mechanism site
        d_model=4096,
        anchor_feature=30939,        # the paper's conversational-surprise feature
    ),
    "google/gemma-3-27b-it": _Entry(
        sae_kind="gemma_scope",
        layers=GEMMA3_27B_RES_LAYERS,
        default_layer=16,
        d_model=5376,
        anchor_feature=None,         # no known anchor; lexicon selector picks one
    ),
}


def resolve_model(model: str, layer: int | None = None, *, width: str = "16k",
                  mixture: str = "slimpj") -> ModelSpec:
    """Resolve a model (+ optional layer) into everything the sweep needs."""
    entry = MODELS.get(model)
    if entry is None:
        raise ValueError(
            f"unknown model {model!r}. Register it in sot/registry.py -- the sweep must "
            "not guess a loader, hook point or layer set."
        )

    layer = entry.default_layer if layer is None else layer
    if layer not in entry.layers:
        raise ValueError(
            f"layer {layer} is not available for {model}: published residual SAEs are at "
            f"{sorted(entry.layers)}. Steering a layer with no SAE is not a thing you can "
            "do quietly -- it would download zero features and look like an unlabelled SAE."
        )

    kw = {"width": width} if entry.sae_kind == "gemma_scope" else {"mixture": mixture}
    return ModelSpec(
        model=model,
        layer=layer,
        sae_kind=entry.sae_kind,
        hook_layer=layer + entry.hook_offset,
        neuronpedia=neuronpedia_source(model, layer, **kw),
        d_model=entry.d_model,
        anchor_feature=entry.anchor_feature,
    )


def load_sae_for(spec: ModelSpec, device: str = "cuda", **kw):
    """Dispatch to the right SAE loader for this model family."""
    if spec.sae_kind == "llama_scope":
        from sot.sae import load_sae

        return load_sae(spec.layer, kw.get("mixture", "slimpj"), device=device)
    if spec.sae_kind == "gemma_scope":
        from sot.gemma_sae import download_gemma_sae

        return download_gemma_sae(spec.model, spec.layer,
                                  width=kw.get("width", "16k"),
                                  l0=kw.get("l0", "medium"), device=device)
    raise ValueError(f"no loader for sae_kind {spec.sae_kind!r}")
