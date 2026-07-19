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

  layers      Which layers actually have a published residual SAE -- and for Llama Scope
              this depends on the MIXTURE, not just the model. The pure-slimpj SAE (the
              paper's) exists at layer 15 only; the slimpj+openr1 "mixed" suite covers all
              32. Collapsing the two blocked 5 of the 6 layers in a sweep already running,
              so availability is a function of (model, mixture).
              Gemma 3 27B: resid_post SAEs at ALL 62 layers (four in the flagship
              resid_post/ dir, the rest in resid_post_all/). Neuronpedia publishes
              LABELS at only 16/31/40/41/53 -- a different, non-nested set. Using the
              label set as weight-availability blocked 57 steerable layers and
              greenlit layer 41, which the loader then rejects.

Registry entries are data, so a wrong one fails in tests rather than as a plausible
number on a rented GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from sot.sources import (
    GEMMA_SCOPE_ALL_LAYERS,
    NoLabelsPublished,
    neuronpedia_source,
)


@dataclass(frozen=True)
class ModelSpec:
    model: str
    layer: int
    sae_kind: str            # "llama_scope" | "gemma_scope"
    hook_layer: int          # layer whose OUTPUT we hook (resid_post)
    # (neuronpedia_model_id, source_id), or None when this layer has SAE weights
    # but no published explanations. Steering works there; selecting features by
    # description does not. Callers that need labels must check and say so --
    # see sot/features.py, which raises rather than downloading a dead prefix.
    neuronpedia: tuple[str, str] | None
    d_model: int
    # Index of a KNOWN conversational feature in THIS model's SAE, or None.
    # 30939 is the paper's feature in Llama Scope; index 30939 in Gemma Scope is an
    # unrelated feature (different SAE, width, training run). None => fall back to the
    # lexicon selector rather than seeding the search from a meaningless direction.
    anchor_feature: int | None


ALL_LLAMA_LAYERS = frozenset(range(32))     # DeepSeek-R1-Distill-Llama-8B has 32 layers


@dataclass(frozen=True)
class _Entry:
    sae_kind: str
    # mixture -> available layers. Gemma has no mixtures, so it uses the "" key.
    layers_by_mixture: dict[str, frozenset[int]]
    default_layer: int
    d_model: int
    anchor_feature: int | None = None
    # offset from the SAE's layer index to the layer whose output we hook.
    # 0 for both families: resid_post(L) IS the output of layer L.
    hook_offset: int = 0
    # Which layers the sweep probes. ONE source of truth -- scripts/run_stages.sh
    # interpolates this rather than carrying its own list. See sweep_grid below.
    sweep_grid: tuple[int, ...] = ()


MODELS: dict[str, _Entry] = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": _Entry(
        sae_kind="llama_scope",
        layers_by_mixture={
            "slimpj": frozenset({15}),   # the paper's SAE genuinely exists at one layer
            "mixed": ALL_LLAMA_LAYERS,   # slimpj+openr1: all 32 -- what the layer sweep uses
            "openr1": ALL_LLAMA_LAYERS,
        },
        default_layer=15,            # the paper's claimed mechanism site
        d_model=4096,
        anchor_feature=30939,        # the paper's conversational-surprise feature
        # Spans 16%-94% of the 32 layers roughly uniformly. Deliberately NOT
        # centred on layer 15: our weight analysis puts 15 at rank 21/32, and
        # calibration already exists for exactly these six, so this grid is both
        # the cheaper and the less question-begging choice. See
        # tests/test_sweep_grid.py and docs/why_layers.md.
        sweep_grid=(5, 10, 15, 20, 25, 30),
    ),
    "google/gemma-3-27b-it": _Entry(
        sae_kind="gemma_scope",
        # WEIGHTS, not labels. All 62 layers have a resid_post SAE (four in the
        # flagship resid_post/ dir, the rest in resid_post_all/). This used to be
        # the Neuronpedia LABEL set {16,31,40,41,53}, which both blocked 57 layers
        # that are steerable and greenlit layer 41, which download_gemma_sae then
        # rejected -- a contradiction that only fires once the model is on a GPU.
        layers_by_mixture={"": GEMMA_SCOPE_ALL_LAYERS},
        default_layer=16,
        d_model=5376,
        anchor_feature=None,         # no known anchor; lexicon selector picks one
        # Constrained by LABELS, not weights. Steering works at all 62 layers,
        # but our feature selection reads Neuronpedia explanations, which exist
        # only at these five. Sweeping an unlabelled layer would mean picking
        # features some other way -- a different experiment, not a wider one.
        sweep_grid=(16, 31, 40, 41, 53),
    ),
}


def layer_sweep_grid(model: str) -> tuple[int, ...]:
    """The layers the sweep probes for this model. One source of truth.

    scripts/run_stages.sh interpolates this via `python -m sot.sweep_grid`
    instead of carrying its own list -- three copies of the grid is exactly how
    the script, the calibration on disk and the completed rows came to disagree.
    """
    entry = MODELS.get(model)
    if entry is None:
        raise ValueError(f"unknown model {model!r}; register it in sot/registry.py")
    if not entry.sweep_grid:
        raise ValueError(f"no sweep grid defined for {model!r}")
    return entry.sweep_grid


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
    key = mixture if entry.sae_kind == "llama_scope" else ""
    available = entry.layers_by_mixture.get(key)
    if available is None:
        raise ValueError(f"unknown mixture {mixture!r} for {model}; "
                         f"try {sorted(entry.layers_by_mixture)}")
    if layer not in available:
        shown = sorted(available)
        shown = shown if len(shown) <= 8 else f"{shown[:4]}...{shown[-2:]}"
        raise ValueError(
            f"layer {layer} is not available for {model} (mixture={mixture!r}): published "
            f"residual SAEs are at {shown}. Steering a layer with no SAE is not something "
            "you can do quietly -- it would download zero features and look like an "
            "unlabelled SAE."
        )

    kw = {"width": width} if entry.sae_kind == "gemma_scope" else {"mixture": mixture}
    # A layer can have weights but no labels (Gemma publishes SAEs at all 62 layers
    # and explanations at 5). That must not block steering, so record the absence
    # instead of raising here -- feature selection is where it actually matters.
    try:
        np_source = neuronpedia_source(model, layer, **kw)
    except NoLabelsPublished:
        np_source = None

    return ModelSpec(
        model=model,
        layer=layer,
        sae_kind=entry.sae_kind,
        hook_layer=layer + entry.hook_offset,
        neuronpedia=np_source,
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
