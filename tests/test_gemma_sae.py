"""Gemma Scope 2 SAE loading -- the format traps, pinned before any GPU time.

Gemma Scope stores its SAE differently from Llama Scope in three ways that all fail
SILENTLY. Each one produced a plausible-looking wrong answer for us on the Llama side,
so each gets a test here rather than a discovery on a rented H100:

  1. w_enc is [d_model, d_sae] and w_dec is [d_sae, d_model] -- TRANSPOSED relative to
     Llama Scope's encoder.weight/decoder.weight. Load them the Llama way and you either
     get a shape error (lucky) or, at square-ish shapes, silent garbage.
  2. `threshold` is the RAW JumpReLU threshold. Llama Scope stores
     `log_jumprelu_threshold` and we exp() it. exp()-ing a raw threshold gives e^theta
     instead of theta -- every feature's gate moves, nothing errors.
  3. There is NO dataset_average_activation_norm, i.e. no dataset-wise rescaling. Our
     Llama loader asserts dataset-wise; the Gemma path must use scale 1.0. Reusing the
     Llama rescale would mis-size every steering vector (the exact 2.5x bug we hit).

The reconstruction test is the ground truth: an SAE only reconstructs its own activations
if encoder/decoder orientation AND the activation function are both right.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest
import torch
from safetensors.torch import save_file

from sot.gemma_sae import load_gemma_sae

D_MODEL, D_SAE = 32, 128


@pytest.fixture
def gemma_scope_file(tmp_path):
    """A tiny file in the REAL Gemma Scope 2 layout (w_enc/w_dec/b_enc/b_dec/threshold).

    Built so the SAE actually reconstructs: an orthonormal dictionary, encoder as the
    transpose of the decoder, thresholds low enough to pass signal through.
    """
    # d_sae != d_model deliberately: a square dictionary would hide a transpose bug,
    # which is the main thing this fixture exists to catch. The first d_model features
    # are a real orthonormal dictionary; the rest are zero vectors that gate off, so
    # reconstruction is exact instead of counting each direction several times.
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL))
    dec = torch.zeros(D_SAE, D_MODEL)          # Gemma layout: [d_sae, d_model]
    dec[:D_MODEL] = q.T                        # one distinct direction per live feature
    enc = dec.T.contiguous()                    # Gemma layout: [d_model, d_sae]

    path = tmp_path / "params.safetensors"
    save_file(
        {
            "w_enc": enc,
            "w_dec": dec,
            "b_enc": torch.zeros(D_SAE),
            "b_dec": torch.zeros(D_MODEL),
            "threshold": torch.full((D_SAE,), 1e-6),   # RAW, not log
        },
        str(path),
    )
    (tmp_path / "config.json").write_text(
        '{"hf_hook_point_in":"model.layers.16.output","width":128,'
        '"architecture":"jump_relu","l0":53,"model_name":"google/gemma-3-27b-it"}'
    )
    return tmp_path


def test_loads_gemma_layout_with_correct_orientation(gemma_scope_file):
    """decoder must end up [d_model, d_sae] in OUR convention, whatever the file says."""
    sae = load_gemma_sae(gemma_scope_file, device="cpu")
    assert sae.d_model == D_MODEL and sae.d_sae == D_SAE
    assert sae.decoder.shape == (D_MODEL, D_SAE), (
        f"decoder must be [d_model, d_sae]; got {tuple(sae.decoder.shape)} -- "
        "w_dec is stored TRANSPOSED in Gemma Scope"
    )
    assert sae.encoder.shape == (D_SAE, D_MODEL)


def test_threshold_is_used_raw_not_exponentiated(gemma_scope_file):
    """Gemma stores the threshold directly. exp()-ing it silently moves every gate."""
    sae = load_gemma_sae(gemma_scope_file, device="cpu")
    assert torch.allclose(sae.threshold, torch.full((D_SAE,), 1e-6), atol=1e-9), (
        "threshold must be raw; exp(1e-6) would be ~1.0 and gate everything off"
    )


def test_no_dataset_wise_rescaling(gemma_scope_file):
    """Gemma Scope has no dataset_average_activation_norm -> scale is exactly 1.0."""
    sae = load_gemma_sae(gemma_scope_file, device="cpu")
    assert sae.sae_to_real == 1.0


def test_reconstructs_its_own_activations(gemma_scope_file):
    """Ground truth. Wrong orientation or wrong activation fn -> reconstruction collapses.

    This is the check that finally settled the Llama Scope hook-point question, so it is
    the check the Gemma loader has to pass before it is trusted.
    """
    sae = load_gemma_sae(gemma_scope_file, device="cpu")
    torch.manual_seed(1)
    # NOT isotropic Gaussian: JumpReLU gates off negative coefficients, so half of a
    # Gaussian's projections are killed and EV caps around 0.35 no matter how correct
    # the loader is. A JumpReLU SAE can only represent NON-NEGATIVE combinations of its
    # dictionary, so the test data has to live in that cone -- as real activations do.
    coeffs = torch.rand(64, D_MODEL)                 # strictly positive
    x = coeffs @ sae.decoder[:, :D_MODEL].T          # a cone combination of live features

    acts = sae.encode(x)
    x_hat = sae.decode(acts)

    num = ((x - x_hat) ** 2).sum()
    den = ((x - x.mean(0, keepdim=True)) ** 2).sum()
    ev = float(1 - num / den)
    assert ev > 0.9, f"explained variance {ev:.3f} -- orientation or activation fn is wrong"


def test_steering_vector_raises_that_features_activation(gemma_scope_file):
    """The invariant the whole experiment rests on: add s*d_f, feature f rises by ~s."""
    sae = load_gemma_sae(gemma_scope_file, device="cpu")
    f, s = 7, 3.0
    x = sae.decoder[:, f] * 2.0            # start with the feature already active
    before = sae.encode(x)[f]
    after = sae.encode(x + sae.steering_vector(f, s))[f]
    assert float(after - before) == pytest.approx(s, abs=1e-4)
