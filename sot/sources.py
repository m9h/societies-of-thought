"""Where a model's features live on Neuronpedia.

Feature selection pulls labels, max activations and firing rates from Neuronpedia's S3
export at

    v1/<neuronpedia_model_id>/<source_id>/{explanations,features}/batch-N.jsonl.gz

Both path components are family-specific. Getting either wrong downloads zero batches,
which is indistinguishable from "this SAE has no labelled features" -- so this module
owns the mapping and raises rather than handing back a path that will quietly yield
nothing.

Layer availability is NOT uniform. Verified against the live S3 listing (2026-07):
gemma-3-27b-it publishes residual SAEs at layers 16/31/40/41/53 only (widths 16k/65k/
262k), though transcoders exist at every layer 0-61. Requesting a res SAE at any other
layer is a dead prefix.
"""

from __future__ import annotations

# Residual-SAE layers actually published for Gemma 3 27B (see module docstring).
GEMMA3_27B_RES_LAYERS = frozenset({16, 31, 40, 41, 53})
GEMMA_WIDTHS = frozenset({"16k", "65k", "262k"})

_LLAMA_MIXTURES = {
    "slimpj": "llamascope-slimpj-res-32k",
    "mixed": "llamascope-slimpj-openr1-res-32k",
    "openr1": "llamascope-openr1-res-32k",
}


def neuronpedia_source(
    model: str,
    layer: int,
    *,
    mixture: str = "slimpj",
    width: str = "16k",
) -> tuple[str, str]:
    """Return (neuronpedia_model_id, source_id) for a model + layer.

    >>> neuronpedia_source("google/gemma-3-27b-it", layer=16)
    ('gemma-3-27b-it', '16-gemmascope-2-res-16k')
    """
    m = model.lower()

    if "deepseek-r1-distill-llama-8b" in m:
        if mixture not in _LLAMA_MIXTURES:
            raise ValueError(f"unknown mixture {mixture!r}; try {sorted(_LLAMA_MIXTURES)}")
        return "deepseek-r1-distill-llama-8b", f"{layer}-{_LLAMA_MIXTURES[mixture]}"

    if "gemma-3-27b" in m:
        np_model = "gemma-3-27b-it" if m.endswith("-it") else "gemma-3-27b"
        if width not in GEMMA_WIDTHS:
            raise ValueError(f"unknown width {width!r}; try {sorted(GEMMA_WIDTHS)}")
        if layer not in GEMMA3_27B_RES_LAYERS:
            raise ValueError(
                f"layer {layer} is not available: Gemma 3 27B publishes residual SAEs "
                f"only at {sorted(GEMMA3_27B_RES_LAYERS)}. Asking for another layer "
                "yields a dead S3 prefix that looks like an unlabelled SAE."
            )
        return np_model, f"{layer}-gemmascope-2-res-{width}"

    raise ValueError(
        f"unknown model {model!r} -- add its Neuronpedia mapping here rather than "
        "guessing a source id at the call site."
    )
