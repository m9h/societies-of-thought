"""The Flax RoPE patch must actually recover correctness, not just change numbers.

Structure mirrors tests/test_penzai_loader.py, and for the same reason: the
error being fixed is invisible at short sequence lengths. llama3 rescales LOW
frequencies, so at seq=8 the whole bug is worth ~8e-06 and any tolerance loose
enough to pass a correct implementation also passes a broken one.

Measured ladder on this exact tiny model at 512 tokens (rel. max logit error vs
HF PyTorch):

    Flax unpatched                      5.7e-03
    Flax + correct rope_theta only      8.6e-04
    Flax + rope_theta + llama3 scaling  3.7e-07

The three-rung ladder is the point. A patch that only fixed the base would land
at 8.6e-04 and still be wrong, so the suite asserts each rung separately rather
than just "patched is better than unpatched".

Needs transformers < 5 (v5 deleted the Flax classes), so these skip elsewhere.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

if not hasattr(transformers, "FlaxLlamaForCausalLM"):
    pytest.skip("transformers >= 5 removed the Flax classes", allow_module_level=True)

from flax_compat.rope_fix import corrected_inv_freq, patched_llama_rope, rope_settings

SEQ = 512
THETA = 500000.0
SCALING = {
    "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
TMP = "/tmp/claude-1000/ropetest/flaxfix"


@pytest.fixture(scope="module")
def reference():
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        # NOT 131072: Flax builds a dense causal mask of
        # max_position_embeddings^2 booleans at setup, which at 131072 is ~17GB
        # and hangs the test. 16384 keeps the mask ~268MB while still exceeding
        # original_max_position_embeddings=8192, which the config validator
        # requires and which keeps the llama3 bands in their real positions.
        max_position_embeddings=16384,
        rope_theta=THETA, rope_scaling={"rope_type": "llama3", **SCALING},
        torch_dtype="float32")
    torch.manual_seed(0)
    pt = transformers.LlamaForCausalLM(cfg).eval()
    pt.save_pretrained(TMP, safe_serialization=True)
    ids = np.array([[i % 64 for i in range(SEQ)]], dtype=np.int32)
    with torch.no_grad():
        logits = pt(torch.tensor(ids, dtype=torch.long)).logits[0].numpy()
    return cfg, ids, logits


def _flax_rel(ids, ref):
    fx = transformers.FlaxLlamaForCausalLM.from_pretrained(
        TMP, from_pt=True, dtype=np.float32)
    got = np.asarray(fx(ids).logits)[0]
    return abs(ref - got).max() / abs(ref).max()


def test_unpatched_flax_is_wrong(reference):
    """Baseline. If this ever passes, the upstream bug was fixed and this whole
    module can be deleted -- so assert the bug still exists."""
    _, ids, ref = reference
    assert _flax_rel(ids, ref) > 1e-3, (
        "unpatched Flax now agrees with PyTorch; upstream may have fixed the "
        "hardcoded rotary base. Re-check before keeping this patch."
    )


def test_fixing_only_the_base_is_not_enough(reference):
    """The interesting rung. Correcting rope_theta alone improves things by an
    order of magnitude and is STILL wrong, because llama3 scaling is separate.
    A patch that stopped here would look like a fix and not be one."""
    _, ids, ref = reference
    with patched_llama_rope(THETA, None):
        rel = _flax_rel(ids, ref)
    assert 1e-5 < rel < 1e-3, f"expected ~8.6e-04, got {rel:.3e}"


def test_full_patch_matches_pytorch(reference):
    """The claim: base + llama3 scaling recovers correctness outright."""
    _, ids, ref = reference
    with patched_llama_rope(THETA, SCALING):
        rel = _flax_rel(ids, ref)
    assert rel < 1e-5, f"max relative logit difference {rel:.3e}"


def test_patch_is_reverted_on_exit(reference):
    """It monkeypatches a module global, so a leak would silently affect every
    later model built in this process."""
    from transformers.models.llama import modeling_flax_llama as fl
    before = fl.create_sinusoidal_positions
    with patched_llama_rope(THETA, SCALING):
        assert fl.create_sinusoidal_positions is not before
    assert fl.create_sinusoidal_positions is before


def test_patch_reverts_even_if_the_body_raises():
    from transformers.models.llama import modeling_flax_llama as fl
    before = fl.create_sinusoidal_positions
    with pytest.raises(RuntimeError):
        with patched_llama_rope(THETA, SCALING):
            raise RuntimeError("boom")
    assert fl.create_sinusoidal_positions is before


def test_from_config_reads_the_real_config(reference):
    cfg, ids, ref = reference
    with patched_llama_rope.from_config(cfg):
        rel = _flax_rel(ids, ref)
    assert rel < 1e-5, f"max relative logit difference {rel:.3e}"


def test_unsupported_rope_type_raises_rather_than_no_op():
    cfg = transformers.LlamaConfig(
        rope_theta=THETA, rope_scaling={"rope_type": "yarn", "factor": 8.0},
        max_position_embeddings=16384)
    with pytest.raises(ValueError, match="(?i)yarn|not implemented"):
        rope_settings(cfg)


def test_matches_transformers_own_llama3_reference():
    """Same oracle the penzai implementation is held to, so the two backends are
    verified against one another's standard rather than each other."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    head_dim = 128
    cfg = transformers.LlamaConfig(
        hidden_size=head_dim * 4, num_attention_heads=4, head_dim=head_dim,
        rope_theta=THETA, rope_scaling={"rope_type": "llama3", **SCALING},
        max_position_embeddings=16384)
    ref, _ = ROPE_INIT_FUNCTIONS["llama3"](cfg, device="cpu")
    ours = corrected_inv_freq(head_dim, THETA, SCALING)
    assert abs(ours - ref.numpy()).max() / abs(ref.numpy()).max() < 1e-6
