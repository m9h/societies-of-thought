"""End-to-end: does a converted Llama-3.1 model match HF PyTorch numerically?

This is the gate that matters. tests/test_llama3_rope.py proves the frequency
math matches transformers' oracle and that the layer reproduces penzai's
rotation -- but neither says anything about whether WEIGHT CONVERSION is right.
A transposed projection or a mis-mapped norm would pass every test there and
still produce a model that is quietly not the one on disk.

So: build a model with Llama-3.1's exact config (rope_theta=500000,
rope_scaling=llama3), run it through HF PyTorch and through the penzai loader,
and require the logits to agree.

Deliberately TINY and randomly initialised. That is not a weakness here the way
it was for the Flax magnitude question -- weight conversion either maps tensors
correctly or it does not, and a 2-layer model exercises every distinct parameter
kind (embed, q/k/v/o, gate/up/down, two norms, unembed) that an 8B model has.
Scale would add confidence about numerics under bf16, not about correctness of
the mapping.

SEQUENCE LENGTH IS LOAD-BEARING, and the control below is what revealed it.
llama3 rescales LOW frequencies, i.e. long-range position, which barely rotate
at small positions. Measured on this exact tiny model, HF-with-scaling vs
HF-without:

    seq     8      7.9e-06        <-- BELOW our 1e-4 agreement tolerance
    seq    64      9.8e-05        <-- about equal to it
    seq   256      4.1e-04
    seq   512      8.6e-04        <-- 8.6x headroom
    seq  8192      3.2e-03        (saturates near original_max_position_embeddings)

So at seq=8 a converter that ignored rope_scaling ENTIRELY would still pass an
agreement test at 1e-4. The first draft of this file tested at seq=8 and was
therefore nearly vacuous. Real validation happens at 512, where the tolerance is
comfortably inside the effect being tested for.

test_scaling_effect_exceeds_agreement_tolerance pins that separation, so if
anyone tightens the sequence length or loosens the tolerance the suite says why
that is not allowed rather than going quietly green.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("penzai")
transformers = pytest.importorskip("transformers")

from penzai import pz

from penzai_backend.loader import llama3_to_penzai

SEQ = 512          # see module docstring: below ~256 the test is vacuous
SMOKE_SEQ = 8      # short path, for shape/wiring only -- proves nothing about scaling
TOL = 1e-4         # relative logit agreement
LLAMA31_SCALING = {
    "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
    "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
}


def _tiny_hf(rope_scaling, seed=0):
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=131072, rope_theta=500000.0,
        rope_scaling=rope_scaling, torch_dtype="float32",
    )
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(cfg).eval()


def _pt_logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor(ids, dtype=torch.long)[None, :]).logits[0].numpy()


def _pz_logits(pz_model, ids):
    toks = pz.nx.wrap(jnp.asarray(ids)).tag("seq")
    out = pz_model(toks, token_positions=pz.nx.wrap(jnp.arange(len(ids))).tag("seq"))
    return jax.device_get(out.unwrap("seq", "vocabulary"))


def test_converted_model_matches_hf_pytorch():
    """THE gate. If this fails, nothing built on the penzai backend is trustworthy."""
    hf = _tiny_hf(LLAMA31_SCALING)
    pzm = llama3_to_penzai(hf)
    ids = [i % 64 for i in range(SEQ)]

    a, b = _pt_logits(hf, ids), _pz_logits(pzm, ids)
    assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
    rel = abs(a - b).max() / (abs(a).max() or 1.0)
    assert rel < TOL, f"max relative logit difference {rel:.3e} at seq={SEQ}"


def test_scaling_effect_exceeds_agreement_tolerance():
    """Makes the agreement test non-vacuous BY CONSTRUCTION.

    If llama3 scaling moves the HF logits by less than the tolerance we accept
    when comparing backends, then a converter that ignored scaling completely
    would pass. This asserts real separation -- and fails loudly if someone
    shortens SEQ or loosens TOL to the point where it disappears."""
    ids = [i % 64 for i in range(SEQ)]
    with_s = _pt_logits(_tiny_hf(LLAMA31_SCALING), ids)
    without = _pt_logits(_tiny_hf(None), ids)
    effect = abs(with_s - without).max() / (abs(with_s).max() or 1.0)
    assert effect > 5 * TOL, (
        f"llama3 scaling moves logits by {effect:.2e} at seq={SEQ}, which is not "
        f"comfortably above the {TOL:.0e} agreement tolerance. The agreement test "
        "cannot distinguish applying the scaling from ignoring it. Raise SEQ."
    )


def test_short_sequence_agreement_is_a_smoke_test_only():
    """Documents the trap rather than pretending it is validation. At seq=8 the
    scaling effect (~8e-06) is below tolerance, so this checks shapes and wiring
    and nothing more."""
    hf = _tiny_hf(LLAMA31_SCALING)
    pzm = llama3_to_penzai(hf)
    ids = list(range(SMOKE_SEQ))
    a, b = _pt_logits(hf, ids), _pz_logits(pzm, ids)
    assert a.shape == b.shape
    assert abs(a - b).max() / (abs(a).max() or 1.0) < TOL


def test_converter_refuses_a_rope_type_it_does_not_implement():
    """yarn/longrope configs must fail loudly rather than silently getting
    llama3 math or no scaling at all."""
    hf = _tiny_hf({"rope_type": "yarn", "factor": 8.0})
    with pytest.raises(ValueError, match="(?i)yarn|not implemented|unsupported"):
        llama3_to_penzai(hf)


def test_plain_llama_without_scaling_still_converts():
    """A Llama-2-style config (no rope_scaling) must keep working -- the loader
    should add llama3 handling, not require it."""
    hf = _tiny_hf(None)
    pzm = llama3_to_penzai(hf)
    ids = [i % 64 for i in range(SEQ)]
    rel = abs(_pt_logits(hf, ids) - _pz_logits(pzm, ids)).max() / (
        abs(_pt_logits(hf, ids)).max() or 1.0)
    assert rel < TOL, f"max relative logit difference {rel:.3e}"


def test_agreement_holds_at_longer_context():
    """Scaling's effect keeps growing with position (2.5e-03 at 2048, 3.2e-03 at
    8192), so agreement at 4096 is a stronger claim than at 512."""
    hf = _tiny_hf(LLAMA31_SCALING)
    pzm = llama3_to_penzai(hf)
    ids = [i % 64 for i in range(4096)]
    a, b = _pt_logits(hf, ids), _pz_logits(pzm, ids)
    rel = abs(a - b).max() / (abs(a).max() or 1.0)
    assert rel < TOL, f"max relative logit difference at 4096 tokens {rel:.3e}"


# --- transformers 5.x moved the RoPE config -------------------------------------
# v4:  config.rope_theta = 500000.0
#      config.rope_scaling = {"rope_type": "llama3", "factor": 8.0, ...}
# v5:  config.rope_parameters = {"rope_type": "llama3", "rope_theta": 500000.0,
#                                "factor": 8.0, ...}   and NO rope_scaling attr
#
# Reading only rope_scaling means that under v5 rope_type resolves to None, the
# loader concludes "no scaling needed", and silently produces an unscaled model.
# That is the exact silent-wrong-math failure this module exists to prevent,
# reintroduced by a dependency bump. Verified against the real config on
# 2026-07-19 with transformers 5.13.1.

class _V5Config:
    """Minimal stand-in for a transformers 5.x LlamaConfig.

    Deliberately raises AttributeError for rope_scaling / rope_theta, as v5 does
    -- a Mock returning None would let a buggy loader pass."""

    rope_parameters = {
        "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }

    def __getattr__(self, name):
        if name in ("rope_scaling", "rope_theta"):
            raise AttributeError(name)
        raise AttributeError(name)


class _V5Model:
    config = _V5Config()


def test_reads_transformers_v5_rope_parameters():
    """The loader must find llama3 scaling under the v5 layout, not silently
    conclude there is none."""
    from penzai_backend.loader import rope_settings

    rope_type, scaling = rope_settings(_V5Model().config)
    assert rope_type == "llama3"
    assert scaling["factor"] == 8.0
    assert scaling["original_max_position_embeddings"] == 8192


def test_reads_transformers_v4_rope_scaling():
    """The v4 layout must keep working -- this is what the numerical tests above
    actually exercise."""
    from penzai_backend.loader import rope_settings

    hf = _tiny_hf(LLAMA31_SCALING)
    rope_type, scaling = rope_settings(hf.config)
    assert rope_type == "llama3"
    assert scaling["factor"] == 8.0


def test_unscaled_config_reports_no_rope_type_under_both_layouts():
    from penzai_backend.loader import rope_settings

    assert rope_settings(_tiny_hf(None).config)[0] in (None, "default")

    class _V5Plain:
        rope_parameters = {"rope_type": "default", "rope_theta": 500000.0}
    assert rope_settings(_V5Plain())[0] in (None, "default")


def test_v5_yarn_is_still_refused():
    """The refusal must survive the layout change too, or v5 users get silent
    wrong math for unsupported rope types."""
    from penzai_backend.loader import llama3_to_penzai

    class _V5Yarn:
        rope_parameters = {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 8.0}

    class _M:
        config = _V5Yarn()

    with pytest.raises(ValueError, match="(?i)yarn|not implemented"):
        llama3_to_penzai(_M())
