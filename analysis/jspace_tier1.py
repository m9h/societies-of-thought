# Copyright 2026 The Antigravity Authors
# SPDX-License-Identifier: Apache-2.0
"""Tier 1: Capture J-space activations during steered reasoning runs and measure workspace diversity."""

from typing import Any, Tuple
import numpy as np
import torch
import jax

import jlens
from jlens.hooks import ActivationRecorder
from analysis.hse import segment, hierarchic_social_entropy, MIN_SEGMENTS, MIN_SEG_CHARS


def get_token_range(encoded, char_start: int, char_end: int) -> Tuple[int, int]:
    """Resolves character start/end boundaries into token start/end indices in the encoded text."""
    token_start = None
    for char_idx in range(char_start, char_end):
        token_idx = encoded.char_to_token(char_idx)
        if token_idx is not None:
            token_start = token_idx
            break

    token_end = None
    for char_idx in range(char_end - 1, char_start - 1, -1):
        token_idx = encoded.char_to_token(char_idx)
        if token_idx is not None:
            token_end = token_idx
            break

    return token_start, token_end


def analyze_trace_workspace_diversity(
    model: torch.nn.Module,
    tokenizer: Any,
    trace: str,
    lens: jlens.JacobianLens,
    layer: int,
) -> Tuple[float, float, float]:
    """Segments the trace, extracts layer-L hidden states, projects them to the
    J-space, and computes Hierarchic Social Entropy (HSE) on the projected vectors.

    Returns:
        (hse, hse_normalised, mean_pairwise_distance)
    """
    # 1. Segment trace at perspective shift markers
    segs = segment(trace)
    if len(segs) < MIN_SEGMENTS:
        return float("nan"), float("nan"), float("nan")

    # 2. Tokenize trace to get input IDs and character-to-token mappings
    encoded = tokenizer(trace, return_tensors="pt")
    input_ids = encoded["input_ids"]

    # Wrap model for jlens hooks
    lm = jlens.from_hf(model, tokenizer)

    # 3. Forward pass to record hidden states at target layer
    with ActivationRecorder(lm.layers, at=[layer]) as recorder:
        with torch.no_grad():
            model(input_ids.to(model.device))
        h = recorder.activations[layer][0]  # Shape: [seq_len, d_model]

    # 4. Project residual stream vectors into the final basis via J-lens transport
    y = lens.transport(h, layer)  # Shape: [seq_len, d_model]

    # 5. Extract character start/end bounds for segment matching
    # Find boundary indices by matching cues
    from analysis.hse import SHIFT
    cuts = [m.start() for m in SHIFT.finditer(trace)]
    bounds = [0] + cuts + [len(trace)]

    # Collect workspace vectors for each valid segment
    v_segs = []
    for a, b in zip(bounds, bounds[1:]):
        seg_str = trace[a:b].strip()
        if len(seg_str) < MIN_SEG_CHARS:
            continue

        token_start, token_end = get_token_range(encoded, a, b)
        if token_start is None or token_end is None:
            continue

        # Average the J-space vectors across token span of this segment
        segment_vectors = y[token_start : token_end + 1]
        v_seg = segment_vectors.mean(dim=0)
        v_segs.append(v_seg)

    if len(v_segs) < MIN_SEGMENTS:
        return float("nan"), float("nan"), float("nan")

    # Stack into [n_segments, d_model] matrix
    E = torch.stack(v_segs)

    # 6. Normalize vectors and compute pairwise cosine distance matrix
    E = E.float()
    E_norm = E / (E.norm(dim=-1, keepdim=True) + 1e-9)
    # Cosine distance: 1 - cosine similarity
    D = 1.0 - (E_norm @ E_norm.T)
    D = D.detach().cpu().numpy()
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)

    # 7. Compute Hierarchic Social Entropy (HSE)
    return hierarchic_social_entropy(D)
