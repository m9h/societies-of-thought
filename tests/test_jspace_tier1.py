# Copyright 2026 The Antigravity Authors
# SPDX-License-Identifier: Apache-2.0
"""Test suite for J-space Tier 1 workspace diversity analysis.

Exercises the Red stage of Red-Green TDD.
"""

import pytest
import torch
import numpy as np

# Mock datasets module at load time
import sys
import types
sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

from tests.test_jspace import tiny, _long_prompt


def test_analyze_trace_workspace_diversity_red(tiny):
    """The RED stage: Verify that the Tier 1 analysis function can be imported
    and runs on a mock trace using the tiny model.
    Initially, this should fail because the analysis/jspace_tier1.py module
    does not exist.
    """
    model, tok = tiny
    
    # We import the function we want to implement
    from analysis.jspace_tier1 import analyze_trace_workspace_diversity
    import jlens
    from jlens_lab import fit_converged
    
    # 1. Fit a tiny lens on the tiny model
    lm = jlens.from_hf(model, tok)
    prompts = [_long_prompt()] * 10
    lens, _ = fit_converged(lm, prompts, source_layers=[1], min_prompts=5, verbose=False)
    
    # 2. Define a mock trace containing surprise markers to trigger segmentation
    # We want at least 3 segments to calculate meaningful social entropy.
    mock_trace = "The initial calculation is extremely straightforward and simple. Wait, actually, let me reconsider the first step because of a small mistake. Oh, hold on, it works out after all if we check the final addition step."
    
    # 3. Call the analysis function
    # It should return: (hse, hse_norm, mean_dist)
    hse, hse_n, md = analyze_trace_workspace_diversity(
        model=model,
        tokenizer=tok,
        trace=mock_trace,
        lens=lens,
        layer=1,
    )
    
    # 4. Assertions on the output types and values
    assert isinstance(hse, float)
    assert isinstance(hse_n, float)
    assert isinstance(md, float)
    assert np.isfinite(hse)
    assert np.isfinite(hse_n)
    assert np.isfinite(md)
    assert md >= 0.0
