"""The steering math -- the part that decides whether we ran the intended experiment.

Four published facts about this SAE disagree with each other or with intuition, and
every one of them fails SILENTLY:

  * the config says resid_post, Neuronpedia's metadata says resid_pre (one layer apart)
  * activations are rescaled dataset-wise, so decoder columns are NOT in residual space
  * the activation is JumpReLU, not ReLU
  * Neuronpedia's published max-activations are on a different scale again (~2.5x)

A wrong choice anywhere here still produces plausible accuracy numbers -- just for a
different intervention than the one you meant. These tests pin the invariants with a
synthetic SAE, so they run on CPU in milliseconds and cannot be fooled by a
plausible-looking result.

The load-bearing property:

    Steering feature f at SAE-space strength s must raise feature f's own measured
    activation by ~s. That is the definition of "steer this feature by s", and it is
    the thing every scaling bug breaks.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

import pytest
import torch

from sot.sae import LlamaScopeSAE
from sot.steering import steer

D_MODEL, D_SAE = 16, 64
DATASET_AVG_NORM = 11.575  # the real value shipped with the layer-15 SAE


def make_sae(seed: int = 0) -> LlamaScopeSAE:
    """A small SAE with an (approximately) dual encoder/decoder pair.

    Real SAEs are trained so that encoding a decoder column recovers that feature;
    we construct that property directly with an orthonormal basis so the round-trip
    identity below is exact rather than approximate.
    """
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL, generator=g))
    decoder = torch.zeros(D_MODEL, D_SAE)
    encoder = torch.zeros(D_SAE, D_MODEL)
    for i in range(D_SAE):
        v = q[:, i % D_MODEL]
        decoder[:, i] = v
        encoder[i] = v  # dual basis: e_i . d_i = 1
    return LlamaScopeSAE(
        layer=15, mixture="slimpj", d_model=D_MODEL, d_sae=D_SAE,
        decoder=decoder, encoder=encoder,
        encoder_bias=torch.zeros(D_SAE), decoder_bias=torch.zeros(D_MODEL),
        log_jumprelu_threshold=torch.full((D_SAE,), -20.0),  # effectively no threshold
        dataset_avg_norm=DATASET_AVG_NORM,
    )


def test_sae_to_real_matches_dataset_wise_definition():
    """x_sae = x_real * sqrt(d_model)/avg_norm, so real = sae * avg_norm/sqrt(d_model)."""
    sae = make_sae()
    assert sae.sae_to_real == pytest.approx(DATASET_AVG_NORM / D_MODEL**0.5)


def test_steering_raises_that_features_activation_by_the_requested_amount():
    """The invariant. Add s*d_f to the REAL stream -> feature f's activation rises by s.

    This catches the rescaling bug in both directions: forget to convert and the
    activation moves by s * (avg_norm/sqrt(d)) instead of s; convert twice and it moves
    by s * (sqrt(d)/avg_norm).

    The invariant holds only for a feature that is ALREADY ACTIVE. JumpReLU hard-zeroes
    anything below threshold, so for a silent feature the pre-activation is clipped and
    the *measured* rise is smaller than s. (Written first with an unseeded random x,
    this test was flaky for exactly that reason -- it passed or failed depending on
    whether the random vector happened to activate the feature. That is a real property
    of steering, not a nuisance: steering a silent feature does not move its reported
    activation by the full requested amount.)
    """
    sae = make_sae()
    f, s = 7, 3.0
    # Start from a vector on which feature f is already firing, so no clipping.
    x_real = sae.decoder[:, f] * 2.0 * sae.sae_to_real

    before = sae.encode(x_real)[f]
    assert before.item() > 0, "precondition: the feature must be active to begin with"

    after = sae.encode(x_real + sae.steering_vector(f, s))[f]
    assert (after - before).item() == pytest.approx(s, abs=1e-4)


def test_steering_a_silent_feature_still_activates_it():
    """Complement to the above: a below-threshold feature is switched ON by steering.

    This is the case that actually matters in the experiment -- feature 30939 fires on
    0.017% of tokens, so on almost every token it is silent and steering is what turns
    it on. The measured activation lands at s minus whatever the (negative)
    pre-activation was, so it must be positive and bounded by s.
    """
    sae = make_sae()
    f, s = 11, 4.0
    x_real = sae.decoder[:, (f + 1) % D_SAE] * 3.0 * sae.sae_to_real  # f itself silent

    assert sae.encode(x_real)[f].item() == pytest.approx(0.0, abs=1e-6)
    after = sae.encode(x_real + sae.steering_vector(f, s))[f].item()
    assert 0 < after <= s + 1e-4


def test_steering_vector_lives_in_real_space_not_sae_space():
    """A guard against the specific bug of adding the SAE-space vector directly.

    The two differ by avg_norm/sqrt(d_model) ~ 2.9x here (and ~5.5x in the real SAE).
    Steering with the unconverted vector 'works' -- it just applies a different dose.
    """
    sae = make_sae()
    v_real = sae.steering_vector(3, 1.0)
    v_sae_naive = sae.decoder[:, 3] * 1.0
    ratio = (v_real.norm() / v_sae_naive.norm()).item()
    assert ratio == pytest.approx(sae.sae_to_real, rel=1e-5)
    assert ratio != pytest.approx(1.0, abs=0.05), "real and SAE space must not be conflated"


def test_jumprelu_zeroes_subthreshold_features():
    """JumpReLU, not ReLU: below-threshold activations are hard-zeroed.

    Using a plain ReLU leaves thousands of small activations alive; each is negligible
    but they all get multiplied by decoder columns and summed, and the reconstruction
    acquires a large amount of spurious mass (observed: -2600% explained variance).
    """
    sae = make_sae()
    sae.log_jumprelu_threshold = torch.full((D_SAE,), torch.tensor(2.0).log().item())
    x_sae = torch.zeros(D_MODEL)
    x_sae += sae.decoder[:, 5] * 5.0   # well above threshold
    x_sae += sae.decoder[:, 6] * 0.5   # below threshold -> must be zeroed

    acts = sae.encode_sae_space(x_sae)
    assert acts[5].item() == pytest.approx(5.0, abs=1e-4)
    assert acts[6].item() == 0.0, "sub-threshold feature must be hard-zeroed, not kept"


# ---------------------------------------------------------------------------
# The hook itself
# ---------------------------------------------------------------------------

class ToyBlock(torch.nn.Module):
    def forward(self, x):  # noqa: D102
        return (x,)


class ToyModel(torch.nn.Module):
    """Mimics the HF Llama attribute path the hook relies on: model.model.layers[i]."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ToyBlock() for _ in range(3)])


def test_hook_adds_delta_to_the_named_layer_and_removes_itself():
    model = ToyModel()
    delta = torch.ones(4) * 2.0
    x = torch.zeros(1, 1, 4)

    with steer(model, 1, delta):
        out = model.model.layers[1](x)[0]
    assert torch.allclose(out, torch.full((1, 1, 4), 2.0))

    # and the hook must be gone afterwards, or every later condition is contaminated
    after = model.model.layers[1](x)[0]
    assert torch.allclose(after, torch.zeros(1, 1, 4)), "hook leaked past its context"


def test_hook_is_a_true_noop_when_delta_is_none():
    """The baseline runs through this same path; it must be bit-identical to unsteered."""
    model = ToyModel()
    x = torch.randn(1, 3, 4)
    with steer(model, 0, None):
        out = model.model.layers[0](x)[0]
    assert torch.equal(out, x)


def test_scope_generated_leaves_the_prompt_untouched():
    """scope='generated' must skip the prefill pass and steer only decode steps.

    The paper says it adds the vector 'at each token generation step'. Steering the
    prefill as well perturbs the model's representation of the QUESTION, which is a
    different intervention entirely.
    """
    model = ToyModel()
    delta = torch.ones(4)
    prefill = torch.zeros(1, 5, 4)  # seq_len > 1 -> the prompt
    decode = torch.zeros(1, 1, 4)   # seq_len == 1 -> a generated token

    with steer(model, 2, delta, scope="generated"):
        p = model.model.layers[2](prefill)[0]
        d = model.model.layers[2](decode)[0]

    assert torch.allclose(p, torch.zeros(1, 5, 4)), "prompt must NOT be steered"
    assert torch.allclose(d, torch.ones(1, 1, 4)), "generated token must be steered"
