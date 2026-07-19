"""Where a model's SAE weights and feature labels live.

Two different questions, two different answers, and conflating them cost us a
registry that greenlit a layer its own loader rejects.

WEIGHTS come from the Gemma Scope 2 HF repos (google/gemma-scope-2-27b-it, ...).
Steering needs these and nothing else.

LABELS come from Neuronpedia's S3 export at

    v1/<neuronpedia_model_id>/<source_id>/{explanations,features}/batch-N.jsonl.gz

Feature *selection by description* needs these. Getting the path wrong downloads
zero batches, which is indistinguishable from "this SAE has no labelled features".

For Gemma 3 27B these are THREE non-nested sets, verified against the live HF tree
and S3 listing on 2026-07-18:

    Neuronpedia labels      16, 31, 40, 41, 53
                            widths 16k/65k/262k, except layer 41 = 16k/262k only

    HF weights, flagship    16, 31, 40, 53              resid_post/
                            widths 16k/65k/262k/1m, l0 small/medium/big

    HF weights, all layers  0..61                       resid_post_all/
                            widths 16k/262k ONLY, l0 small/big ONLY -- no medium

Layer 41 has labels and has weights in resid_post_all, but is absent from the
flagship directory. Layer 20 has weights but no labels. So "is layer L available"
has no answer until you say what for.

The l0 asymmetry is the sharp edge: our default is l0="medium", which exists only
in the flagship directory. Requesting any of the other 58 layers at the default
would 404 on a path we had asserted was valid. gemma_sae_location() raises instead.

tests/test_gemma_scope_layout.py pins all of it.
"""

from __future__ import annotations


class NoLabelsPublished(ValueError):
    """This layer has SAE weights but no published Neuronpedia explanations.

    Distinct from a bad model/mixture/width, which is a caller error. Steering
    works on an unlabelled layer; selecting features by description does not.
    Callers that can proceed without labels catch THIS, never bare ValueError --
    swallowing the caller errors too is how a typo becomes a silent no-op.
    """


# --- Gemma 3 27B: labels (Neuronpedia) ------------------------------------------
NEURONPEDIA_GEMMA3_27B_LAYERS = frozenset({16, 31, 40, 41, 53})
# Layer 41's S3 export has 16k and 262k only -- the one asymmetry in the set.
_NEURONPEDIA_GEMMA_WIDTHS = frozenset({"16k", "65k", "262k"})
_NEURONPEDIA_GEMMA_WIDTHS_BY_LAYER = {41: frozenset({"16k", "262k"})}

# --- Gemma 3 27B: weights (HF gemma-scope-2) ------------------------------------
GEMMA_SCOPE_FLAGSHIP_LAYERS = frozenset({16, 31, 40, 53})
GEMMA_SCOPE_ALL_LAYERS = frozenset(range(62))   # gemma-3-27b has 62 decoder blocks

_FLAGSHIP_WIDTHS = frozenset({"16k", "65k", "262k", "1m"})
_FLAGSHIP_L0 = frozenset({"small", "medium", "big"})
_ALL_WIDTHS = frozenset({"16k", "262k"})
_ALL_L0 = frozenset({"small", "big"})

# Backwards-compatible alias. Callers that meant LABELS were right to use this;
# callers that meant WEIGHTS were the bug. Kept so the name resolves, but new code
# should say which one it means.
GEMMA3_27B_RES_LAYERS = NEURONPEDIA_GEMMA3_27B_LAYERS
GEMMA_WIDTHS = _NEURONPEDIA_GEMMA_WIDTHS

_LLAMA_MIXTURES = {
    "slimpj": "llamascope-slimpj-res-32k",
    "mixed": "llamascope-slimpj-openr1-res-32k",
    "openr1": "llamascope-openr1-res-32k",
}


def gemma_sae_location(layer: int, *, width: str = "16k", l0: str = "medium") -> str:
    """Return the repo-relative path to a Gemma Scope 2 resid_post SAE.

    Picks the flagship `resid_post/` directory when the layer is one of the four
    that has it (richer width and l0 options), and `resid_post_all/` otherwise.

    >>> gemma_sae_location(16, width="16k", l0="medium")
    'resid_post/layer_16_width_16k_l0_medium'
    >>> gemma_sae_location(20, width="16k", l0="big")
    'resid_post_all/layer_20_width_16k_l0_big'
    """
    if layer not in GEMMA_SCOPE_ALL_LAYERS:
        raise ValueError(
            f"layer {layer} is out of range: gemma-3-27b has "
            f"{len(GEMMA_SCOPE_ALL_LAYERS)} decoder blocks (0-61)."
        )

    if layer in GEMMA_SCOPE_FLAGSHIP_LAYERS:
        subdir, widths, l0s = "resid_post", _FLAGSHIP_WIDTHS, _FLAGSHIP_L0
    else:
        subdir, widths, l0s = "resid_post_all", _ALL_WIDTHS, _ALL_L0

    if width not in widths:
        raise ValueError(
            f"width {width!r} is not published for layer {layer} in {subdir}/: "
            f"available {sorted(widths)}. Widths 65k and 1m exist only for the "
            f"flagship layers {sorted(GEMMA_SCOPE_FLAGSHIP_LAYERS)}."
        )
    if l0 not in l0s:
        raise ValueError(
            f"l0 {l0!r} is not published for layer {layer} in {subdir}/: "
            f"available {sorted(l0s)}. Note l0='medium' -- our default -- exists "
            f"ONLY for the flagship layers {sorted(GEMMA_SCOPE_FLAGSHIP_LAYERS)}; "
            "every other layer must pass l0='small' or 'big' explicitly."
        )

    return f"{subdir}/layer_{layer}_width_{width}_l0_{l0}"


def neuronpedia_source(
    model: str,
    layer: int,
    *,
    mixture: str = "slimpj",
    width: str = "16k",
) -> tuple[str, str]:
    """Return (neuronpedia_model_id, source_id) for a model + layer.

    This is about LABELS. A layer can have weights (and be steerable) while having
    no labels here -- see gemma_sae_location().

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
        if layer not in NEURONPEDIA_GEMMA3_27B_LAYERS:
            raise NoLabelsPublished(
                f"layer {layer} has no published Neuronpedia labels: Gemma 3 27B "
                f"exports explanations only at {sorted(NEURONPEDIA_GEMMA3_27B_LAYERS)}. "
                "Asking for another layer yields a dead S3 prefix that looks like an "
                "unlabelled SAE. (Weights DO exist at every layer 0-61 -- see "
                "gemma_sae_location -- so this layer is steerable but its features "
                "cannot be selected by description.)"
            )
        allowed = _NEURONPEDIA_GEMMA_WIDTHS_BY_LAYER.get(layer, _NEURONPEDIA_GEMMA_WIDTHS)
        if width not in allowed:
            raise NoLabelsPublished(
                f"width {width!r} has no labels for layer {layer}; available "
                f"{sorted(allowed)}."
            )
        return np_model, f"{layer}-gemmascope-2-res-{width}"

    raise ValueError(
        f"unknown model {model!r} -- add its Neuronpedia mapping here rather than "
        "guessing a source id at the call site."
    )
