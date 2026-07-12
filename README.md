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

## Two traps this code is built around

**1. The SAE lives in a rescaled space.** These SAEs are trained with
`norm_activation: dataset-wise`: they never saw the raw residual stream, only
activations rescaled so the dataset-average norm maps to `sqrt(d_model)`. Decoder
columns, feature activations, and Neuronpedia's reported max-activations are all
in *that* space. Steering strengths must be converted back exactly once
(`sae.steering_vector`). Getting it backwards rescales every intervention by
~an order of magnitude and still "works" — it just isn't the experiment you meant
to run.

**2. The published metadata contradicts itself about where the SAE attaches.**
The SAE's own `config.json` says `blocks.15.hook_resid_post`. Neuronpedia's
feature metadata for the *same SAE* says `blocks.15.hook_resid_pre`. Those are
different tensors, one layer apart. `sot/validate_hook.py` settles it empirically —
it encodes both candidate sites and keeps whichever reproduces feature 30939's
published behaviour (fires on "Oh!"-type surprise markers, max activation ≈ 5.906).
**The sweep refuses to run until this is resolved.**

## Strength units

The paper steers at `s = ±10`. Feature 30939's max activation is 5.906, so the
paper's `+10` is ≈ **1.7× max-act**. Raw `s` is meaningless across features whose
activation scales differ by orders of magnitude, so the sweep parameterizes
strength as `alpha` = multiples of each feature's own max activation
(`--alphas -2 -1 1 2`). `--raw-strengths` reproduces the paper's exact units for
the Countdown control.

## Running it

```bash
./scripts/setup.sh              # uv venv + torch (cu130) + deps

./scripts/run_stages.sh hook     # REQUIRED. resolve resid_pre vs resid_post
./scripts/run_stages.sh smoke    # 8 problems, end-to-end wiring check
./scripts/run_stages.sh control  # GATE: reproduce 27.1% -> 54.8% on Countdown
./scripts/run_stages.sh main     # THE EXPERIMENT: GPQA + MATH-Hard
./scripts/run_stages.sh layers   # layer sweep (only if `main` shows an effect)
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
