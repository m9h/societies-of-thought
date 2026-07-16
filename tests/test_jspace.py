"""J-space pipeline tests -- the ones whose absence cost half a day.

The Tier-0 experiment failed on a GPU box with `no prompt produced a Jacobian`, three
times, because I called an unfamiliar API without a red test first. The root cause was
invisible: fit_converged wraps each prompt in `except Exception: continue`, so the REAL
error is swallowed and only the downstream symptom (jac_sum is None) survives.

These run on a 4-layer in-memory Llama in seconds on CPU. Two jobs:
  1. exercise from_hf -> jacobian_for_prompt -> fit -> jacobians on a real (tiny) model,
     with the exception NOT swallowed, so any API misuse shows its true error here and
     never on a paid GPU again.
  2. pin the geometry (jspace_subspace, align) against planted directions where the
     answer is known -- this is the part that turns into the actual finding.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# analysis.jspace_tier0 imports sot.sae (needs datasets stub) at module load
import sys
import types
sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

from analysis.jspace_tier0 import align, jspace_subspace

jlens = pytest.importorskip("jlens")
pytest.importorskip("jlens_lab")


# --------------------------------------------------------------------------
# a real, tiny Llama -- small enough to fit a lens on CPU in a test
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4,
        vocab_size=tok.vocab_size, max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    return model, tok


def _long_prompt() -> str:
    # must exceed skip_first (16) + 1 tokens after tokenisation
    return ("The Jacobian lens transports a residual stream vector into the final basis "
            "and decodes it with the model's own unembedding into a ranked list of "
            "vocabulary tokens, one step at a time, across many positions and prompts.")


def test_jacobian_for_prompt_returns_a_tensor_not_a_swallowed_error(tiny):
    """The red test. Calls the INNER function directly -- no `except: continue` -- so a
    real API error (wrong arg, layer out of range, encode failure) surfaces HERE, on CPU,
    instead of vanishing into 'no prompt produced a Jacobian' on a GPU."""
    from jlens.fitting import jacobian_for_prompt

    model, tok = tiny
    lm = jlens.from_hf(model, tok)
    assert lm.n_layers == 4

    per_prompt, seq_len, n_valid = jacobian_for_prompt(
        lm, _long_prompt(), source_layers=[1],
        dim_batch=8, max_seq_len=128, skip_first=16,
    )
    assert 1 in per_prompt
    J = per_prompt[1]
    assert J.shape == (lm.d_model, lm.d_model), f"J_1 should be d x d, got {tuple(J.shape)}"
    assert torch.isfinite(J).all()
    assert n_valid > 0


def test_bad_layer_index_raises_loudly_not_silently(tiny):
    """The half-day bug in one assertion. fit_converged swallows per-prompt errors, so a
    misconfigured call fails as 'no prompt produced a Jacobian' with no cause. The direct
    estimator must raise the REAL error -- which is why the tier-0 script probes with it
    before the fit."""
    from jlens.fitting import jacobian_for_prompt

    model, tok = tiny
    lm = jlens.from_hf(model, tok)  # 4 layers; layer 15 does not exist
    with pytest.raises(Exception):
        jacobian_for_prompt(lm, _long_prompt(), source_layers=[15],
                            max_seq_len=128, skip_first=16)


def test_fit_converged_populates_jacobians(tiny):
    """End to end: the wrapper must yield a lens whose .jacobians has the source layer."""
    from jlens_lab import fit_converged

    model, tok = tiny
    lm = jlens.from_hf(model, tok)
    prompts = [_long_prompt()] * 40
    lens, report = fit_converged(lm, prompts, source_layers=[1], min_prompts=5, verbose=False)
    assert 1 in lens.jacobians
    assert lens.jacobians[1].shape == (lm.d_model, lm.d_model)


def test_jspace_reps_are_directions_in_residual_space(tiny):
    """jspace_reps[l] must be [n_tokens, d_model] -- vocab pullbacks at layer l."""
    from jlens_lab import fit_converged
    from jlens_lab.geometry import jspace_reps

    model, tok = tiny
    lm = jlens.from_hf(model, tok)
    lens, _ = fit_converged(lm, [_long_prompt()] * 20, source_layers=[1],
                            min_prompts=5, verbose=False)
    W_U = model.get_output_embeddings().weight.detach()
    reps = jspace_reps(lens, W_U, n_tokens=128, device="cpu")
    assert reps[1].shape[1] == lm.d_model


# --------------------------------------------------------------------------
# the geometry that becomes the finding -- planted directions, known answers
# --------------------------------------------------------------------------

def test_align_is_one_for_an_in_subspace_direction():
    """A direction built entirely from J-space rows must have alignment ~1."""
    torch.manual_seed(0)
    reps = torch.randn(200, 64)               # 200 vocab pullbacks in 64-d residual space
    Q = jspace_subspace(reps, var=0.99)       # its top subspace
    d = Q[:, 0] * 3.0 + Q[:, 1] * -2.0        # a vector inside span(Q)
    assert align(d, Q) == pytest.approx(1.0, abs=1e-4)


def test_align_is_near_zero_for_an_orthogonal_direction():
    """A direction in the complement of J-space must have alignment ~0."""
    torch.manual_seed(1)
    # low-rank J-space so an orthogonal complement provably exists
    basis = torch.linalg.qr(torch.randn(64, 64)).Q
    reps = (torch.randn(200, 10) @ basis[:, :10].T)   # J-space spans only dims 0..9
    Q = jspace_subspace(reps, var=0.999)
    ortho = basis[:, 40]                               # a dimension J-space cannot reach
    assert align(ortho, Q) < 0.15


def test_align_of_a_random_direction_scales_with_subspace_fraction():
    """Sanity: a random direction's expected alignment ~ sqrt(k/d)."""
    torch.manual_seed(2)
    reps = torch.randn(300, 64)
    Q = jspace_subspace(reps, var=0.90)
    k, d = Q.shape[1], Q.shape[0]
    vals = [align(torch.randn(d), Q) for _ in range(200)]
    assert np.mean(vals) == pytest.approx((k / d) ** 0.5, abs=0.1)
