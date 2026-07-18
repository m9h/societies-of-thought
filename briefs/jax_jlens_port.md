# Brief: port the Jacobian lens to JAX

Port Anthropic's `jlens` (the Jacobian lens from *"Verbalizable Representations Form a
Global Workspace in Language Models"*) from PyTorch to JAX/Flax, targeting
`google-deepmind/gemma`, and validate it numerically against the existing PyTorch
lenses.

This brief is for an autonomous agent. **Read all of it before writing code.** The gates
are not suggestions: a stage that fails its gate means *stop and report*, not "proceed
and hope."

---

## Why this is worth doing

**The maths fits JAX better than PyTorch.** The lens is

```
lens_l(h) = unembed( J_l @ h ),    J_l = E[ ∂h_final / ∂h_l ]
```

That expectation over prompts and positions is a `vjp` inside a `vmap`. In PyTorch the
reference implementation hand-rolls it: replicate the prompt `dim_batch` times, retain the
graph, and loop one-hot cotangents over output dimensions in chunks. In JAX it is
`jax.vmap(jax.vjp(...))` and the compiler does the batching. Expect materially less code
and better hardware utilisation.

**JAX is empty here.** There is no JAX SAE library, no JAX TransformerLens, no JAX
NNsight. Penzai is dormant; HF `transformers` v5 removed Flax entirely. Meanwhile
`google-deepmind/gemma` (Flax NNX) is actively maintained and Gemma is a first-class
`jlens` target. This is an unclaimed, useful artifact.

**It unlocks free compute.** Google's **TPU Research Cloud**
(`sites.research.google/trc`) grants free TPUs and — unusually — accepts *unaffiliated
individuals*. TPUs need JAX. Today every interpretability run in this project costs rented
NVIDIA hours; a JAX lens makes them free.

**Licensing is clear.** Upstream `jlens` is **Apache-2.0** and explicitly *"not maintained
and not accepting contributions"* — so a port cannot be rejected upstream, and the licence
permits it. Preserve the copyright notice and state clearly that this is a derived port.

---

## The oracle — this is what makes the task gateable

**You are not guessing whether the port is correct. Pre-fitted PyTorch lenses exist.**

`huggingface.co/neuronpedia/jacobian-lens` (MIT) contains fitted lenses for ~38 models,
each as `<model>/jlens/<dataset>/<model>_jacobian_lens.pt`, alongside the `config.yaml`
recording the exact fit command and a `*_convergence.csv`. Relevant entries include
`gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-2-9b-it`, `gemma-2-2b`, `gemma-3-270m`,
`llama3.1-8b`.

**Start with `gemma-3-270m`** — smallest, fits anywhere, fastest iteration.

Their fits used, per `config.yaml`: `Salesforce/wikitext` (`wikitext-103-raw-v1`, train),
`text_field=text`, `max_chars=2000`, `n_prompts=1000`, `dim_batch=64`, `max_seq_len=128`,
`dtype=bfloat16`, `min_prompts=100`, `stop_window=10`, `stop_at_delta=0.002`. **Match
these exactly** when comparing, or you are comparing two different estimators.

---

## Gates

**Gate 0 — the reference runs.** Before writing any JAX, fit a lens for `gemma-3-270m`
with upstream PyTorch `jlens` using the config above, and confirm it converges. If you
cannot reproduce a PyTorch lens, you have no oracle and nothing downstream is meaningful.
**Stop and report.**

**Gate 1 — single-prompt Jacobian agreement.** For one fixed prompt and one layer,
`J_l` from your JAX implementation must match PyTorch's `jacobian_for_prompt` elementwise.
Report `max_abs_err`, `rel_fro_err = ||J_jax - J_torch||_F / ||J_torch||_F`, and cosine
similarity of the flattened matrices. **Target `rel_fro_err < 1e-3` in float32.** If you
cannot hit that on ONE prompt, the averaging machinery is irrelevant — fix this first.

**Gate 2 — fitted-lens agreement.** Fit the full JAX lens on `gemma-3-270m` with the
config above and compare against the downloaded `.pt`: per-layer `rel_fro_err`, plus
agreement of the *decoded readout* (top-k tokens from `unembed(J_l @ h)` for a held-out
set of `h`). Top-10 token overlap should be high; report it, don't hand-wave it. A lens
that is numerically close but decodes differently is not a working port.

**Gate 3 — scale.** Only after Gates 1–2 pass. Fit `gemma-2-2b` or `gemma-3-12b-it` and
compare to its published lens. Report wall-clock and peak memory vs the PyTorch reference
on the same hardware — the performance claim above is a hypothesis, so **measure it**. If
JAX is slower, say so.

---

## Known traps

These cost real time on the PyTorch side. Assume the JAX port inherits them.

- **`fit_converged` swallows per-prompt errors.** Upstream wraps each prompt in
  `except Exception: continue`, so a genuine failure (bad layer index, dtype, OOM,
  device-placement) silently yields `jac_sum is None` and surfaces only as
  `"no prompt produced a Jacobian"`. **Always probe the inner per-prompt function
  directly, un-swallowed, before running a fit.** Half a day was lost to this.
- **A probe only guards what it exercises.** A probe at `dim_batch=8` passed while the
  fit at `dim_batch=128` OOM'd. Probe with the *same* parameters the real run uses.
- **Skip the BOS token.** Its residual norm is ~40x an ordinary token (an attention sink
  the lens was never meant to model). Upstream's `skip_first=16` handles this; don't
  quietly change it while "simplifying."
- **`stop_at_delta=0.002` needs ~400–500 prompts.** `mean_rel_change` decays ~`0.85/n`.
  Supply too few prompts and it exits un-converged — check `report.converged`, do not
  assume. An under-fit lens does not announce itself.
- **Precision.** Their fits are `bfloat16` with the Jacobian accumulated in float32.
  Reproduce that split; comparing a float32 JAX fit to a bfloat16 PyTorch fit will show
  disagreement that is precision, not a bug. Report the dtype of every number.
- **Layer indexing.** In HF, `hidden_states[i]` is the *input* to layer `i`, so
  `resid_post(L) == hidden_states[L+1]`. Flax NNX Gemma will index differently. Verify
  against the oracle rather than reasoning it out.

---

## Deliverables

1. A JAX/Flax implementation with a `fit` and a `transport`/`lens_logits` equivalent.
2. `results/oracle_comparison.json` — per-model, per-layer `rel_fro_err`, cosine
   similarity, top-k decode overlap, dtypes, and the exact fit config used.
3. A benchmark table: JAX vs PyTorch wall-clock and peak memory, same hardware, same
   config.
4. `REPORT.md` answering, in order: did Gate 1 pass and at what error? Gate 2? Gate 3?
   Is JAX actually faster, and by how much? What does NOT match, and why?
5. Every number in `REPORT.md` traceable to a file in `results/`.

## Do not

- **Never simulate, extrapolate, or hand-write a number you did not measure.** No
  `np.random` anywhere near a reported result. (A previous agent on this project
  hard-coded a paper's published figures, added Gaussian noise to fake three seeds, and
  reported "successfully replicates." Every deliverable looked perfect. Do not do this.)
- Do not report agreement without stating the tolerance and the dtype.
- Do not skip a gate because the next stage looks more interesting.
- **If you cannot make it work, say so and stop.** A blocked port honestly reported is
  the outcome I want; a fabricated success is worthless and detectable.

## Environment notes

- Upstream: `github.com/anthropics/jacobian-lens` (Apache-2.0, `pip install -e .`).
  Needs `torch>=2.6` (it imports `DTensor`) and `transformers>=5.5`; after upgrading
  torch, also upgrade `torchvision` and remove `torchaudio` or you get an undefined-symbol
  crash on import.
- `jlens.from_hf` already supports the Gemma layout natively — no layout registration
  needed on the PyTorch side.
- JAX target: `google-deepmind/gemma` (Flax NNX, actively maintained). Penzai is dormant
  (last commit 2025-06) — usable for reference, do not depend on it.
- Free TPUs: TPU Research Cloud accepts unaffiliated individuals. NSF NAIRR does not
  (institutional email required).
