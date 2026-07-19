"""llama3 RoPE frequency rescaling for penzai.

WHY. penzai's Llama converter refuses Llama-3.1 models -- correctly -- because
`rope_scaling={"rope_type":"llama3",...}` is not in its handled set
(variants/llama.py:96). That refusal is what makes penzai safer than the HF Flax
path, which silently hardcodes the rotary base (see docs/flax_rope_bug.md). But
it also means penzai cannot load DeepSeek-R1-Distill-Llama-8B until llama3
scaling exists.

WHAT IS ACTUALLY MISSING. Only the frequency rescale. penzai's ApplyRoPE already
takes `max_wavelength`, and llamalike_common passes `rope_wavelength=
hf_config.rope_theta` -- so the BASE is handled. Llama-3.1 additionally rescales
the inverse frequencies piecewise by wavelength, and that is the gap.

ORACLE. transformers' own `ROPE_INIT_FUNCTIONS["llama3"]`
(`_compute_llama3_parameters`). Testing against a reimplementation of the same
formula would only prove I can copy; testing against the shipped function proves
the numbers agree with what the model was trained under.

These tests need jax + penzai and skip cleanly without them, since the main
suite is the PyTorch steering side.
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("penzai")

from penzai_backend.llama3_rope import llama3_inv_freq

# The real values from deepseek-ai/DeepSeek-R1-Distill-Llama-8B's config.json.
LLAMA31 = dict(
    factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
    original_max_position_embeddings=8192,
)
HEAD_DIM = 128
ROPE_THETA = 500000.0


def _hf_reference(head_dim: int, rope_theta: float, **scaling):
    """Ground truth straight out of transformers."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    cfg = transformers.LlamaConfig(
        hidden_size=head_dim * 4, num_attention_heads=4, head_dim=head_dim,
        rope_theta=rope_theta,
        rope_scaling={"rope_type": "llama3", **scaling},
        max_position_embeddings=131072,
    )
    inv_freq, attn_factor = ROPE_INIT_FUNCTIONS["llama3"](cfg, device="cpu")
    return inv_freq.numpy(), attn_factor


def test_matches_transformers_reference_exactly():
    """The whole point: our numbers must be the ones the model was trained under."""
    ref, _ = _hf_reference(HEAD_DIM, ROPE_THETA, **LLAMA31)
    ours = llama3_inv_freq(HEAD_DIM, ROPE_THETA, **LLAMA31)
    assert ours.shape == ref.shape
    rel = abs(ours - ref) / abs(ref)
    assert rel.max() < 1e-6, f"max relative error {rel.max():.2e}"


def test_attention_scaling_is_unity_for_llama3():
    """llama3 returns attention_factor 1.0 -- if that ever changes, the layer
    would need to apply it, so pin the assumption rather than assume it."""
    _, attn_factor = _hf_reference(HEAD_DIM, ROPE_THETA, **LLAMA31)
    assert attn_factor == 1.0


def test_high_frequency_components_are_untouched():
    """Short wavelengths (wavelen < old_ctx/high_freq_factor) must pass through
    unscaled. These carry local position, which llama3 deliberately preserves."""
    inv = llama3_inv_freq(HEAD_DIM, ROPE_THETA, **LLAMA31)
    plain = jnp.asarray(1.0 / (ROPE_THETA ** (jnp.arange(0, HEAD_DIM, 2) / HEAD_DIM)))
    wavelen = 2 * math.pi / plain
    high_cut = LLAMA31["original_max_position_embeddings"] / LLAMA31["high_freq_factor"]
    mask = wavelen < high_cut
    assert bool(mask.any()), "test is vacuous if no component is high-frequency"
    assert jnp.allclose(inv[mask], plain[mask], rtol=1e-6)


def test_low_frequency_components_are_divided_by_factor():
    """Long wavelengths (wavelen > old_ctx/low_freq_factor) get inv_freq/factor.
    That division is what extends the context."""
    inv = llama3_inv_freq(HEAD_DIM, ROPE_THETA, **LLAMA31)
    plain = jnp.asarray(1.0 / (ROPE_THETA ** (jnp.arange(0, HEAD_DIM, 2) / HEAD_DIM)))
    wavelen = 2 * math.pi / plain
    low_cut = LLAMA31["original_max_position_embeddings"] / LLAMA31["low_freq_factor"]
    mask = wavelen > low_cut
    assert bool(mask.any()), "test is vacuous if no component is low-frequency"
    assert jnp.allclose(inv[mask], plain[mask] / LLAMA31["factor"], rtol=1e-6)


def test_medium_band_is_strictly_between_the_two_regimes():
    """The smoothing band must interpolate, not overshoot -- a sign error here
    would still pass the two boundary tests above."""
    inv = llama3_inv_freq(HEAD_DIM, ROPE_THETA, **LLAMA31)
    plain = jnp.asarray(1.0 / (ROPE_THETA ** (jnp.arange(0, HEAD_DIM, 2) / HEAD_DIM)))
    wavelen = 2 * math.pi / plain
    lo = LLAMA31["original_max_position_embeddings"] / LLAMA31["low_freq_factor"]
    hi = LLAMA31["original_max_position_embeddings"] / LLAMA31["high_freq_factor"]
    mid = (wavelen >= hi) & (wavelen <= lo)
    assert bool(mid.any()), "test is vacuous if the smoothing band is empty"
    assert bool(jnp.all(inv[mid] <= plain[mid] * (1 + 1e-6)))
    assert bool(jnp.all(inv[mid] >= plain[mid] / LLAMA31["factor"] * (1 - 1e-6)))


def test_factor_one_is_a_no_op():
    """factor=1 with matched band factors must reduce to plain RoPE. A scaling
    implementation that cannot be turned off is hiding a bug."""
    inv = llama3_inv_freq(HEAD_DIM, ROPE_THETA, factor=1.0, low_freq_factor=1.0,
                          high_freq_factor=4.0, original_max_position_embeddings=8192)
    plain = jnp.asarray(1.0 / (ROPE_THETA ** (jnp.arange(0, HEAD_DIM, 2) / HEAD_DIM)))
    assert jnp.allclose(inv, plain, rtol=1e-6)


@pytest.mark.parametrize("theta", [10000.0, 500000.0])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_matches_reference_across_configs(theta, head_dim):
    ref, _ = _hf_reference(head_dim, theta, **LLAMA31)
    ours = llama3_inv_freq(head_dim, theta, **LLAMA31)
    assert abs(ours - ref).max() / abs(ref).max() < 1e-6


# --- layer-level: the rotation itself, not just the frequencies ----------------
# The tests above only exercise llama3_inv_freq. Llama3RoPE._apply_1d
# REIMPLEMENTS penzai's rotation (split-in-half, cos/sin mix), and a sign or
# ordering error there passes every test above. These catch that.

def test_layer_with_factor_one_is_identical_to_penzai_ApplyRoPE():
    """THE key wiring test. With scaling disabled the layer must reproduce
    penzai's own ApplyRoPE bit for bit -- any divergence is a bug in my
    rotation, not in the frequencies."""
    from penzai import pz
    from penzai.nn.embeddings import ApplyRoPE
    from penzai_backend.llama3_rope import Llama3RoPE

    base = ApplyRoPE(embedding_axis="embedding", max_wavelength=ROPE_THETA,
                     positions_input_name="token_positions")
    mine = Llama3RoPE.from_apply_rope(
        base, factor=1.0, low_freq_factor=1.0, high_freq_factor=4.0,
        original_max_position_embeddings=8192)

    key = jax.random.key(0)
    x = pz.nx.wrap(jax.random.normal(key, (5, HEAD_DIM))).tag("seq", "embedding")
    pos = pz.nx.wrap(jnp.arange(5)).tag("seq")

    a = base(x, token_positions=pos).unwrap("seq", "embedding")
    b = mine(x, token_positions=pos).unwrap("seq", "embedding")
    assert jnp.allclose(a, b, atol=1e-5), f"max diff {abs(a-b).max():.2e}"


def test_layer_with_real_scaling_differs_from_plain_rope():
    """Control for the test above: with the real factor=8 it must NOT match,
    or the swap is a no-op and we have changed nothing."""
    from penzai import pz
    from penzai.nn.embeddings import ApplyRoPE
    from penzai_backend.llama3_rope import Llama3RoPE

    base = ApplyRoPE(embedding_axis="embedding", max_wavelength=ROPE_THETA,
                     positions_input_name="token_positions")
    mine = Llama3RoPE.from_apply_rope(base, **LLAMA31)

    x = pz.nx.wrap(jax.random.normal(jax.random.key(0), (5, HEAD_DIM))).tag("seq", "embedding")
    pos = pz.nx.wrap(jnp.arange(5)).tag("seq")
    a = base(x, token_positions=pos).unwrap("seq", "embedding")
    b = mine(x, token_positions=pos).unwrap("seq", "embedding")
    assert not jnp.allclose(a, b, atol=1e-5), "llama3 scaling had no effect"


def test_swaps_into_a_real_penzai_model_by_type():
    """The injection path: no string paths, no fork of penzai."""
    from penzai import pz
    from penzai.models.transformer.variants import llamalike_common as L
    from penzai.nn.embeddings import ApplyRoPE
    from penzai_backend.llama3_rope import Llama3RoPE

    cfg = L.LlamalikeTransformerConfig(
        num_kv_heads=2, query_head_multiplier=1, embedding_dim=32, projection_dim=8,
        mlp_hidden_dim=64, num_decoder_blocks=2, vocab_size=64,
        mlp_variant="geglu_approx", tie_embedder_and_logits=False)
    model = L.build_llamalike_transformer(cfg, init_base_rng=jax.random.key(0))

    patched = pz.select(model).at_instances_of(ApplyRoPE).apply(
        lambda r: Llama3RoPE.from_apply_rope(r, **LLAMA31))

    assert len(pz.select(patched).at_instances_of(ApplyRoPE).selected_by_path) == 0
    assert len(pz.select(patched).at_instances_of(Llama3RoPE).selected_by_path) == 4

    seq = pz.nx.wrap(jnp.arange(6)).tag("seq")
    out = patched(seq, token_positions=pz.nx.wrap(jnp.arange(6)).tag("seq"))
    assert out.named_shape == {"seq": 6, "vocabulary": 64}


def test_wrong_rope_type_is_rejected():
    """Applying llama3 scaling to a yarn/linear config would silently compute
    the wrong frequencies -- the exact failure mode this module exists to avoid."""
    from penzai.nn.embeddings import ApplyRoPE
    from penzai_backend.llama3_rope import Llama3RoPE

    base = ApplyRoPE(embedding_axis="embedding", max_wavelength=ROPE_THETA,
                     positions_input_name="token_positions")
    with pytest.raises(ValueError, match="(?i)not llama3"):
        Llama3RoPE.from_apply_rope(base, rope_type="yarn", **LLAMA31)
