# Does conversational-feature steering survive off Countdown?

A probe of the causal claim in **"Reasoning Models Generate Societies of Thought"**
(Kim, Lai, Scherrer, Agüera y Arcas & Evans, [arXiv:2601.10825](https://arxiv.org/abs/2601.10825)).

## The claim under test

The paper's mechanistic centerpiece: steering a single sparse-autoencoder feature
in DeepSeek-R1-Distill-Llama-8B — feature **30939**, layer 15, "a discourse marker
for surprise, realization, or acknowledgment" — **doubles reasoning accuracy on
Countdown, 27.1% → 54.8%**, and suppressing it drops accuracy to 23.8%. The paper
reads this as evidence that reasoning models work by simulating a *society of
thought*: internal voices that question, disagree, and reconcile.

That is a large claim resting on a narrow base: **one hand-picked feature, at one
layer, in one 8B distilled model, on one task.** And Countdown is a peculiar task
to hang it on. It is a search puzzle with a weak baseline and enormous headroom,
where the winning strategy is to enumerate more candidate expressions before
committing. "Perturb the residual stream and the model tries more things" would
produce the same result with no society of thought anywhere in the story.

So this repo runs the experiment the authors didn't:

1. **Off Countdown.** GPQA-Diamond and MATH Level-5 — both from the paper's own
   benchmark suite, neither with Countdown's headroom, neither rewarding blind
   enumeration.
2. **Beyond one feature.** Conversational candidates selected two independent
   ways, against controls **matched on sparsity and activation magnitude** — so
   the controls differ from the candidates in *meaning*, not in how big or how
   rare the perturbation is. The paper's controls were not matched this way, which
   leaves "bigger perturbation" as an unexcluded explanation.
3. **Beyond one layer.** All 32 layers have published SAEs.

The estimand is a difference-in-differences, not a raw effect:

```
  (steered − baseline | conversational feature)
− (steered − baseline | matched control feature)
```

If that is ≈ 0, the *society-of-thought* mechanism is not what's doing the work —
however large the raw steering effect looks.

## Four traps, and how they were resolved

Every one of these produces plausible-looking numbers and a *different experiment*
than the one you meant to run. None throws an error. They were found by demanding
that the SAE **reconstruct** the residual stream — its own training objective, and
the only ground truth here that doesn't depend on trusting published metadata.

**1. The published metadata names the wrong hook point.** The SAE's `config.json`
says `blocks.15.hook_resid_post`; Neuronpedia's metadata for the *same SAE* says
`blocks.15.hook_resid_pre`. Those are different tensors, one layer apart.
Reconstruction settles it: `resid_post` gives **52.5%** explained variance,
`resid_pre` **27.5%**. The config is right; **Neuronpedia is mislabeled.**

**2. BOS is an attention sink.** Its residual norm here is **466** against ~11 for
every ordinary token. The SAE was never trained to model it, and leaving it in
makes reconstruction error ~25× the variance *at every hook point and every
scaling* — which looks exactly like "the whole setup is broken." Excluded
everywhere.

**3. The activation function is JumpReLU, not ReLU.** Features fire only above a
per-feature threshold (`log_jumprelu_threshold`). A plain ReLU keeps thousands of
sub-threshold activations alive; each is individually negligible, but they all get
multiplied by decoder columns and summed, and the reconstruction acquires a large
amount of spurious mass.

**4. The SAE lives in a rescaled space.** `norm_activation: dataset-wise` — the SAE
saw activations rescaled so the dataset-average norm maps to `sqrt(d_model)`.
Confirmed empirically: the stored `dataset_avg_norm` is 11.575 and the measured
mean residual norm is 11.0–12.1. Steering vectors must be converted back to real
space exactly once (`sae.steering_vector`).

## Strength units, and why Neuronpedia's numbers can't set them

Raw `s` is meaningless across features whose activation scales differ by orders of
magnitude, so strength is parameterized as `alpha` = multiples of a feature's own
max activation.

But **max activation must be measured in our units.** At the verified hook point,
feature 30939 peaks at **18.4** on the very contexts the paper's Fig. 2a prints as
5.78 / 5.75 / 4.75 — a consistent **~3.1×** offset. Reconstruction says our scaling
is the correct one, so Neuronpedia's displayed activations simply live on a
different scale. Sizing `alpha` off their number would make every intervention ~3×
weaker than intended, silently. `sot/calibrate.py` therefore measures max
activations directly over SlimPajama (the SAE's own training corpus), and the sweep
refuses to run without it.

Feature *selection* is unaffected by this — it only compares features to each
other, and the offset is common to all of them.

For the Countdown positive control, `--raw-strengths` reproduces the paper's exact
`s = ±10` units so the numbers are directly comparable.

## Running it

```bash
./scripts/setup.sh                  # uv venv + torch (cu130) + deps

./scripts/run_stages.sh hook        # REQUIRED. resolve the hook point by reconstruction
./scripts/run_stages.sh calibrate   # REQUIRED. measure max-acts in our units
./scripts/run_stages.sh smoke       # 8 problems, end-to-end wiring check
./scripts/run_stages.sh control     # GATE: reproduce 27.1% -> 54.8% on Countdown
./scripts/run_stages.sh main        # THE EXPERIMENT: GPQA + MATH-Hard
./scripts/run_stages.sh layers      # layer sweep (only if `main` shows an effect)
```

Stages are gates. **If `control` does not roughly reproduce the paper's Countdown
numbers, the harness is wrong and nothing downstream is interpretable** — fix that
before reading anything into `main`. Runs are resumable; completed cells are
skipped on restart.

Needs one GPU with ≥20GB free (8B bf16 ≈ 16GB + SAE). Validated on a GB10
(DGX Spark) with torch 2.13/cu130.

## What the results mean

| Countdown control | GPQA / MATH-Hard | Reading |
|---|---|---|
| reproduces | candidates beat matched controls | Society-of-thought mechanism generalizes. The paper's claim is stronger than it proved. |
| reproduces | DiD ≈ 0 | The steering effect is real but **not conversational** — it's perturbation, and Countdown's headroom flattered it. |
| reproduces | nothing moves anywhere | Effect is Countdown-specific. The mechanistic claim doesn't survive contact with real reasoning tasks. |
| fails | — | Our harness is wrong. Stop; debug the hook point and the SAE-space rescaling. |

A null result here is *informative*, not a failed experiment — which is why the
analysis reports parse and truncation rates too. Positive steering makes traces
chattier and likelier to run past the token budget, and a trace that never emits
an answer must be scored **wrong**, not dropped — otherwise steering "improves"
accuracy by shrinking the denominator.

## Layout

```
sot/sae.py            SAE loading; the SAE-space <-> real-space rescaling
sot/steering.py       activation-addition hook on a decoder layer
sot/validate_hook.py  preflight: resolves the hook-point contradiction
sot/features.py       feature selection + sparsity/magnitude-matched controls
sot/data.py           GPQA-Diamond, MATH-Hard, Countdown
sot/grade.py          answer extraction; unparseable == wrong
sot/run_sweep.py      the sweep (resumable)
sot/analyze.py        accuracy, problem-clustered bootstrap CIs, the DiD
```

## Provenance of the artifacts

The paper ships **no code and no data**. Everything here is rebuilt from public
sources:

- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (MIT)
- SAEs: `OpenMOSS-Team/Llama-Scope-R1-Distill` (Apache-2.0). The paper's
  `15-llamascope-slimpj-res-32k` is the `800M-Slimpajama-0-OpenR1-Math-220k/L15R`
  subdirectory. Only layer 15 exists for the pure-SlimPajama mixture; all 32
  layers exist for the SlimPajama+OpenR1 mixture, which is what the layer sweep
  uses.
- Feature explanations, max-activations, firing rates: Neuronpedia's S3 export
  (GPT-4o-mini autointerp — the same model the paper cites; note the *website* now
  serves regenerated Claude explanations for some features, so pin the source).
- GPQA-Diamond: `fingertap/GPQA-Diamond` (the official `iDavidRein/gpqa` is gated).
- MATH-Hard: `lighteval/MATH-Hard`. Countdown: `Jiayi-Pan/Countdown-Tasks-3to4`.

What is *not* reproducible: the authors' 8,262 generated traces and their
LLM-as-judge annotations. The judge prompts and rubrics are in the paper's
~82-page supplement, as prose.
