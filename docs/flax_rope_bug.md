# HF Flax Llama computes the wrong RoPE for Llama-3 models

**Status:** code-level finding, verified 2026-07-18. Empirical confirmation
(logits diff vs PyTorch) is the outstanding step — see "How to confirm" below.

**Who this affects:** anything using `jlens-jax` (or any HF-Flax path) on a
Llama-3-family model. That includes `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`,
the model the Societies-of-Thought paper uses and the one our J-space work
targets.

## The finding

`transformers.models.llama.modeling_flax_llama` builds its rotary frequencies as:

```python
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
```

The base **10000 is hardcoded**. Grepping the whole module:

    mentions rope_scaling : False
    mentions llama3       : False
    mentions rope_theta   : False

The implementation predates Llama 3 and was never updated — it reads neither
config field, though both are present on the config object.

Our model's `config.json` says:

    rope_theta   = 500000.0
    rope_scaling = {"rope_type": "llama3", "factor": 8.0,
                    "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192}

So the Flax path uses a rotary base **50x smaller** than the one the model was
trained with, and skips llama3 frequency rescaling entirely. This is not a
long-context edge case: it changes the positional encoding at every position,
so every hidden state and therefore every Jacobian derived from it is computed
on a model that does not match the trained weights.

It fails **silently**. There is no warning; the model loads and produces
plausible-looking logits.

## Why this was easy to miss

`FlaxLlamaForCausalLM` exists and accepts the checkpoint, so the failure mode is
"works, but wrong" rather than "won't load". Nothing in the stack objects:

  - `transformers` 4.57.6 still ships the Flax Llama class
  - the weights load without shape errors (RoPE has no parameters)
  - `jlens_jax/models.py` finds its FlaxLayout and returns hidden states

Contrast penzai, which **refuses** the same model: its converter validates
config against `LlamaConfig()` defaults and raises on the unhandled
`rope_scaling` key (`variants/llama.py:96`). Penzai's dormancy looked like the
liability; it turns out its strictness is the asset. A library that refuses is
strictly safer than one that quietly substitutes different math.

## Blast radius

Any `jlens-jax` result on a Llama-3-family model computed through the Flax
backend should be treated as unvalidated until re-run or confirmed. J-space
fits, transported activations, and anything downstream (Tier 0 alignment
numbers, Tier 1 workspace diversity) inherit the error if they came through
this path.

Note the PyTorch `jlens` / `jlens-lab` path is NOT affected — it uses HF PyTorch
modelling code, which reads `rope_theta` and `rope_scaling` correctly. Only the
JAX/Flax port has this problem.

## How to confirm empirically

Load the same checkpoint through `FlaxLlamaForCausalLM` and
`LlamaForCausalLM`, run one identical short prompt, and compare final logits.
If this finding is right they will diverge substantially even at ~10 tokens
(the base differs at every position, not just long ones). A max-abs-diff near
zero would refute it.

This is worth doing before acting, because the claim rests on reading the
source rather than observing an output.

## Options

1. **Patch the Flax RoPE** — reimplement `create_sinusoidal_positions` to read
   `rope_theta` and apply llama3 rescaling. Smallest change, but it means
   monkeypatching a deprecated code path in a frozen dependency branch
   (`transformers<5`, since v5 removed Flax entirely).
2. **Move the backend to penzai** — see `spikes/penzai_backend_spike.py`, which
   shows penzai can satisfy `jlens_jax/protocol.py`'s
   `forward_with_intermediates` contract, with RoPE injectable by type via
   `pz.select(...).at_instances_of(ApplyRoPE)`. Penzai needs the same ~40 lines
   of llama3 RoPE, but it is not on a removed code path and it fails loudly
   rather than silently.
3. **Stay on PyTorch `jlens`** for Llama-3-family work and use the JAX port only
   where it has been validated.

Option 2 is the one that also unblocks Gemma 3: HF Flax has `FlaxGemmaForCausalLM`
(Gemma 1) but **no Gemma 2 and no Gemma 3**, so the Flax path cannot reach the
model we want for the layer sweep regardless of the RoPE issue.

## Environment facts behind this (Spark, 2026-07-18)

    ~/Workspace/jlens-jax/.venv : jax 0.11.0, transformers 4.57.6, flax 0.12.7
    societies-of-thought/.venv  : transformers 5.14.1 -- Flax classes ALL GONE
    ~/dev/wwj/.venv             : jax 0.10.1  (42 passed / 4 failed; the 4 are a
                                  missing `weightwatcher` oracle dep, not JAX)

jax 0.11.0 itself is fine — `jlens-jax` already runs on it, and penzai runs on
it too despite a 15-month-old release. JAX version was never the problem.
