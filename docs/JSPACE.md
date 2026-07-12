# The same claim, twice

*Why this repo's design is the missing control in Anthropic's global-workspace paper.*
*Written 2026-07-12. Companion work lives in `~/Workspace/jacobian-lens`.*

## Two papers, one shape

**"Reasoning Models Generate Societies of Thought"** (Kim, Lai, Scherrer, Agüera y Arcas
& Evans, [arXiv:2601.10825](https://arxiv.org/abs/2601.10825)) intervenes on one SAE
feature in one layer of one 8B model, reports a large raw effect on one task, and reads
it as evidence that reasoning models simulate *internal voices that question, disagree,
and reconcile*.

**"Verbalizable Representations Form a Global Workspace in Language Models"**
([transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html),
code at [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens),
Apache-2.0) identifies a subspace via an expected input–output Jacobian, ablates it,
reports that multi-hop reasoning and translation collapse while sentiment, MMLU,
acceptability judgements and one-step recall survive, and reads it as evidence for a
*global workspace* in the sense of Baars and Dehaene.

Strip the vocabulary and both are the same move:

> Perturb a chosen subspace of a transformer. Observe a large behavioural effect.
> Name the subspace after a theory from cognitive science.

That move can be right. It is not right *by default*, and neither paper does the thing
that would make it right.

## The confound, stated precisely

This repo already articulates it for the Societies-of-Thought claim, in the README:

> *"Perturb the residual stream and the model tries more things" would produce the same
> result with no society of thought anywhere in the story.*

The J-space version is the same sentence with two nouns swapped, and it is *sharper*,
because of how the J-space is defined. `J_l = E[∂h_final/∂h_l]` — the J-space is,
by construction, **the subspace with the largest influence on the output.** So:

- Ablating it removes the directions the output is *most sensitive to*.
- Tasks differ enormously in how much output perturbation they tolerate. Multi-hop
  reasoning is a long chain in which an early error compounds. Sentiment classification
  is a one-shot, highly redundant, margin-heavy decision.
- Therefore **"ablate the highest-influence subspace → the fragile tasks break and the
  robust tasks don't"** is a prediction of *task fragility*. It requires no workspace,
  no broadcast, and no global anything.

The observed dissociation is real. What is unestablished is that it is *about the
workspace* rather than *about perturbation sensitivity*. Anthropic's paper reports the
raw effect. As far as I can find, it does not report an ablation of a **matched control
subspace** — same rank, same effect on activation variance or output KL, but *not*
selected for verbalizability. Without that arm, the effect size is uninterpretable in
exactly the way this repo says the Countdown effect is uninterpretable.

## The control they're missing is the control we built

The estimand here is already the right one. From the README:

```
  (steered − baseline | conversational feature)
− (steered − baseline | matched control feature)
```

Ported to the workspace claim:

```
  (ablated − baseline | J-space, rank k)
− (ablated − baseline | control subspace, rank k, matched on ‖Δh‖ and output KL)
```

If that difference-in-differences is ≈ 0, then the J-space is not doing anything a
magnitude-matched random subspace wouldn't do, *however large the raw ablation effect
looks* — and "global workspace" collapses to "the directions that matter most," which is
a tautology with a press release.

Two design points carry over unchanged, and both are load-bearing:

1. **Match on magnitude, not just on rank.** The paper's controls (and Kim et al.'s)
   leave "bigger perturbation" as an unexcluded explanation. Matching on effect size is
   what converts a raw effect into an identified one.
2. **Unparseable output scores as wrong, never dropped.** Ablation makes generations
   degenerate. A trace that never emits an answer must count against the condition, or
   ablation "spares" a task by shrinking its denominator. `sot/grade.py` already does
   this; any J-space port must too.

## What is already running

`~/Workspace/jacobian-lens` (fork of `anthropics/jacobian-lens`, `upstream` wired):

- `experiments/randomization_control.py` — the **model-randomization sanity check**
  (Adebayo et al. 2018, *Sanity Checks for Saliency Maps*). Fit a J-lens on a model whose
  transformer blocks are re-initialized but whose embedding, final norm and unembedding
  are the trained ones. Score two ways per layer: does the readout predict the **true next
  token** (learned structure), or does it merely **echo the current token** (the embedding
  riding the additive residual stream up to a trained unembedding)?

  If the random-blocks lens is empty on both, the J-lens is sensitive to learned structure
  and passes. If it predicts nothing but echoes strongly, then coherent, human-legible
  lens readouts survive with **zero learned structure** — and every J-lens figure has to be
  scored against that floor rather than against chance. Nobody has published this. Anthropic
  did not run it; Neel Nanda replicated only the *positive* finding, and only on Qwen.

The J-space ablation DiD described above is the natural second experiment, and it is
this repo's method applied to their object.

## The methodological rhyme

Both projects have now been saved by the same discipline: **verify the instrument
empirically; never trust the metadata.**

| Here | There |
|---|---|
| SAE config says `resid_post`, Neuronpedia says `resid_pre`. Reconstruction settles it — 52.5% vs 27.5% explained variance. The config was right. | `model._init_weights()` in transformers v5 is a **silent no-op** on an already-loaded model. The "randomized" control was fully trained. Caught only by asserting the weights actually changed. |
| Neuronpedia's max-activations are **~3.1× off** our scale. Sizing steering off them makes every intervention 3× weaker than intended, silently — and it still "works." | A lens fit on a model you *believe* is random, but isn't, produces a clean, confident, entirely wrong "it passes." |

In both cases the broken version runs to completion and prints plausible numbers. That is
the whole danger, and it is why these harnesses are built as gates.

## Status

- Hook point: **resolved** (`resid_post`, by reconstruction).
- Calibration: launched on the DGX Spark; **DGX went offline mid-run** (site power loss,
  2026-07-12). Status unverified — rerun `./scripts/run_stages.sh calibrate`.
- Countdown control gate, main sweep: **not yet run.**
- J-lens randomization control: running on Qwen3-0.6B.

## Why this matters beyond either paper

The field is accumulating claims of the form *"a cognitive-science construct has been
found inside a language model."* Their evidentiary standard is currently set by whoever
publishes first, and it is a raw effect with an unmatched control. Both of these repos
exist to insist on a difference-in-differences instead. A null result from either is
informative — which is the point, and why it is worth running them properly.

See also: `~/.claude/.../memory/project_gwt_jspace.md` for the Global Workspace Theory
landscape (Butlin & Long's indicator properties; what Anthropic concedes; COGITATE), and
`project_jspace_replication.md` for the access/infrastructure map.
