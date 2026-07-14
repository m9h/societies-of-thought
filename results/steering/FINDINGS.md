# The society-of-thought steering effect is a Countdown artifact

A test of the mechanistic claim in **"Reasoning Models Generate Societies of Thought"**
(Kim, Lai, Scherrer, Agüera y Arcas & Evans, [arXiv:2601.10825](https://arxiv.org/abs/2601.10825)).

The paper reports that steering a single "conversational surprise" SAE feature (30939,
layer 15) in DeepSeek-R1-Distill-Llama-8B **doubles** Countdown accuracy, 27.1% → 54.8%,
and reads this as evidence that reasoning works by simulating a *society of thought* —
diverse internal voices that question, disagree and reconcile.

We reproduce the Countdown effect at a third of its reported size, and find that on the
paper's **own** benchmarks it **reverses**.

---

## 1. The harness reproduces the paper

| | ours | paper |
|---|---|---|
| Countdown baseline (unsteered) | **24.0%** | 27.1% |
| feature 30939 sparsity | 0.00017 | 0.00016 (Neuronpedia) |
| SAE reconstruction (layer 15, resid_post) | 52.5% EV | — |

The feature is the right one: it fires **only** on surprise markers (` Oh`), and reads
exactly 0.000 on non-conversational text.

## 2. On Countdown the effect is real — but ~1/3 the reported size, and non-monotonic

Steering strength `α` in multiples of the feature's calibrated max activation.
(The paper's `s = ±10` is unit-ambiguous; both readings — α=0.678 and α=1.693 — are on
this ladder.) n=200 problems/condition, paired bootstrap over problems.

| α | accuracy | Δ vs baseline | 95% CI | P(Δ>0) |
|---|---|---|---|---|
| 0 (baseline) | 24.0% | — | — | — |
| 0.25 | 23.0% | −1.0 | [−7.5, +5.5] | 35% |
| 0.50 | 31.0% | +7.0 | [−0.5, +14.5] | 96% |
| 0.678 | 29.0% | +5.0 | [−3.0, +13.0] | 88% |
| **1.00** | **34.0%** | **+10.0** | **[+2.5, +18.0]** | **99.4%** |
| 1.693 | 3.5% | −20.5 | [−26.5, −14.5] | 0% |

**An inverted U.** Steering helps up to a point, then the model collapses into
degenerate babble — literally `"5-isopropyl-3,4-dimethylcyclochoh! Wait, no, wait, no,
wait, it's a cyclo-oh, no, wait..."` — scoring 3.5% with 96% of traces unparseable.

**+10 points, not +28.** The paper's doubling does not reproduce at any dose.

## 3. On the paper's own benchmarks, the effect REVERSES

The paper never steers on GPQA or MATH — it only *observes* traces there. Its entire
causal claim rests on Countdown. We ran the missing cell, at the dose that was optimal on
Countdown (α=1.0), with **controls matched on sparsity and max-activation** (the paper's
controls were not).

### MATH-Hard (n=100 problems × 8 conditions; baseline 62.0%, 38% truncation)

| feature | role | accuracy | Δ vs baseline |
|---|---|---|---|
| **30939** | **anchor (the paper's feature)** | **40.0%** | **−22.0** ✱ |
| 3114 | conversational candidate | 59.0% | −3.0 |
| 10126 | conversational candidate | 56.0% | −6.0 |
| 20402 | conversational candidate | 54.0% | −8.0 |
| 5993 | matched control | 61.0% | −1.0 |
| 22600 | matched control | 62.0% | +0.0 |
| 26919 | matched control | 53.0% | −9.0 |

- conversational candidates: mean Δ = **−9.8%**
- matched controls: mean Δ = **−3.3%**
- **difference-in-differences: −6.4%** [−14.8, +1.5]

✱ CI excludes zero.

**The same feature, at the same dose, that gives +10 on Countdown costs −22 on MATH-Hard.**

### GPQA-Diamond — REPORTED FOR COMPLETENESS ONLY, DO NOT RELY ON IT

At a 4096-token budget the DiD was −2.9% [−6.2, −0.8] (conversational features
significantly *worse* than matched controls). **But that run truncated 79% of traces.** A
16k-token re-run of the baseline gives 34.4% accuracy at 100% parse rate, against 14.0%
at 31% parse under the 4096 budget — i.e. the short-budget GPQA numbers measured *"did it
finish"*, not *"did it reason"*. We stopped the full 16k sweep (≈$19, 6h) because the
MATH arm already carries the finding and the GPQA effect points the same way.

*(Truncation biases toward the null: every arm faces the same ceiling, and an unparseable
trace scores WRONG rather than being dropped. So it can shrink a difference, never
manufacture one.)*

## 4. The mediation story fails on its own terms

The paper's SEM claims conversational behaviour *mediates* the accuracy gain. Steering
lets us test that directly, and the two come apart:

| trace marker (MATH-Hard) | baseline | steered α=1.0 |
|---|---|---|
| self-interruption (`wait`, `hmm`, `oh`) | 15.6 | **25.1** ↑ |
| contradiction (`but`, `however`, `actually`) | 15.4 | **21.5** ↑ |
| questions | 4.3 | **6.3** ↑ |
| first-person plural (`we`, `let's`) | 14.5 | **16.3** ↑ |
| **accuracy** | **62.0%** | **40.0%** ↓ |

**The intervention works.** The traces become measurably more dialogic, exactly as the
paper predicts. And reasoning gets *worse*. You can induce the society of thought and get
a dumber model.

---

## 5. Interpretation

Countdown is a **search** task: combine 3–4 numbers to hit a target. The winning move is
to enumerate more candidate expressions before committing, and the baseline sits at 24%
with enormous headroom. On such a task, *"perturb the model into trying more things"* and
*"make the model reason better"* are **empirically indistinguishable**.

GPQA and MATH-Hard remove that confusion. You cannot brute-force your way to knowing that
a Diels-Alder reaction forms a six-membered ring. There, the same intervention just makes
the model babble the *form* of self-correction (`"wait — no — actually"`) without doing
any of the epistemic work those words normally accompany.

**A parsimonious account of the paper's own descriptive findings.** RL on verifiable
correctness selects for **verification and backtracking**, because checking your work and
abandoning dead ends raise P(correct). That is a *search* property, not a social one. But
English expresses self-audit in dialogic form — a mind checking itself *sounds like* two
people arguing. The conversational structure is therefore plausibly a **stylistic
signature of self-correction, not its cause**. The paper measured the shadow and steered
it; steering a shadow harder does not move the object.

## 6. Independent support: accuracy and "society" come apart

Huot, Kaisers & Lapata, *[When is Routing Meaningful? Diversity and Robustness in Language
Model Societies](https://arxiv.org/abs/2607.09197)* (cs.MA, Jul 2026), reach the same
dissociation from the opposite direction — at the multi-**model** level rather than inside
one model's activations.

Their argument is that routing over a society of models is evaluated almost entirely on
accuracy and cost, and that this is not enough. Two further properties decide whether a
society is *meaningful*: **behavioural differentiation** among the actors, and **routing
stability**. Their central line:

> "High task accuracy is compatible with violating both properties, since a router can
> operate over a **redundant society** or assign queries inconsistently."

They conclude that **"accuracy and meaningfulness can sharply diverge."**

That is our finding, mirrored:

| | they show | we show |
|---|---|---|
| dissociation | high accuracy with a **redundant** (fake) society | a **real** society with degraded accuracy |
| level | between models (routing) | inside one model (steering) |

Both cases break the inference Kim et al. rely on. Their design measures conversational
behaviour and accuracy *together*, on a task where both happen to rise, and concludes that
the first mediates the second. Two independent lines of evidence now say that inference is
unsafe: you can have the accuracy without the society, and — as we show — the society
without the accuracy.

**Two instruments from that paper are directly applicable here, and we have not used them:**

- **Hierarchic Social Entropy.** Kim et al. quantify "perspective diversity" with an
  LLM-as-judge that first *infers* the personas and then *measures* their spread — the
  judge is scoring constructs it invented. HSE is a principled, judge-free diversity
  metric. Applying it to R1's traces would test their descriptive claim (which we do not
  dispute) on much firmer ground, and would answer whether R1's internal voices are
  genuinely differentiated or a **redundant society wearing dialogic clothes**.
- **Perturbation robustness.** Their robustness metric asks whether a society survives
  being poked. Our dose-response *is* that experiment, and the society fails it: the
  inverted-U collapses into degenerate babble (3.5%, 96% unparseable) at the paper's own
  upper dose. In their terms, this society is not robust.

## 7. Hierarchic Social Entropy: diversity is a correlate, not a lever

We applied Huot et al.'s judge-free diversity measure to the Countdown dose ladder (1,200
saved traces). Each trace is segmented at the paper's *own* perspective-shift cues
(wait / but / however / actually / ...), the segments are embedded locally, and Balch's HSE
is integrated from the single-linkage dendrogram. No LLM-judge anywhere — unlike the paper's
"perspective diversity", which is an LLM that infers the personas it then scores.

Steering strength `α` in multiples of calibrated max-activation. `analysis/hse.py`, 8 tests.

| α | segments/trace | HSE/log₂N | mean dist | accuracy |
|---|---|---|---|---|
| 0 (baseline) | 21.4 | **0.236** | 0.320 | 15.2% |
| 0.5 | 29.5 | 0.212 | 0.332 | 28.5% |
| 1.0 | 44.6 | 0.190 | 0.340 | 31.5% |
| 1.693 | 54.7 | **0.190** | 0.360 | 3.6% |

**Two findings, and they pull apart the paper's claim precisely.**

**(a) Steering makes the society bigger and proportionally MORE REDUNDANT.** Segment count
more than doubles (+155% at α=1.693), but **normalised diversity FALLS 20%** (0.236 → 0.190).
Each added "voice" contributes less distinct information. This is Huot et al.'s *redundant
society* — many actors, little differentiation — induced on demand. (Raw mean-distance does
rise, but that is a trap: it peaks at α=1.693, the degenerate 3.6%-accuracy dose, where the
model emits `"wait, no, wait, no, wait"`. Incoherent text embeds far apart because it is
noise, not because it holds distinct viewpoints. Diversity of noise is not diversity of
thought.)

**(b) The mediation the paper asserts is real — but steering does not deliver it.** The
paper's actual claim is that diversity *mediates* accuracy. That predicts a *within-condition*
relationship: holding the intervention fixed, the more-diverse traces should be the correct
ones. They are:

| α | more-diverse → correct | 95% CI |
|---|---|---|
| 0 (baseline) | **+0.072** | [+0.029, +0.113] ✱ |
| 0.678 | **+0.045** | [+0.019, +0.072] ✱ |
| 1.0 | **+0.063** | [+0.041, +0.085] ✱ |
| 1.693 | +0.029 | [−0.029, +0.082] |

So genuine reasoning diversity *is* associated with getting the answer right — even in the
unsteered model. The paper found a real correlation. **What it got wrong is that steering the
conversational feature buys that diversity.** It doesn't: it buys redundancy, then noise.
And at the degenerate dose the mediation link vanishes entirely (CI spans zero) — the
clearest sign that what steering maximises is incoherence, not viewpoint.

**The synthesis.** Diversity of *authentic* reasoning correlates with success. The paper
mistook that correlation for a lever, and pulling the lever produces the *form* of diversity
without the substance — which is exactly why accuracy falls (§3–4) while the dialogic markers
rise. This is the within-model instance of Huot et al.'s between-model result: a society can
look richer while being emptier.

## 8. Limitations (ours, and the field's)

- **The model is not actually a reasoning model.** `DeepSeek-R1-Distill-Llama-8B` was
  never RL'd; it is Llama-3.1-8B *supervised-fine-tuned to imitate* R1's outputs. The
  paper's causal claims rest on it too — the field's only public reasoning-model SAE is
  for this one distill. So *the mechanistic understanding of reasoning models currently
  rests on a model that is an impersonation of one.* Training an SAE on a genuinely RL'd
  reasoner (QwQ-32B) is the obvious next step and nobody has done it.
- One feature family, one layer (15), one dose ladder.
- GPQA arm truncation-limited (see above).
- Our claim is scoped exactly as the paper's is: within this model, on these benchmarks.

## 9. A correction to the public record

Neuronpedia lists this SAE's hook point as **`blocks.15.hook_resid_pre`**. The SAE's own
config says **`resid_post`**, and reconstruction settles it decisively (52.5% vs 27.5%
explained variance). Anyone replicating from the published metadata would **silently steer
the wrong layer**. Neuronpedia's reported max-activations are also on a different scale
from the SAE's own (~2.5×), so steering strengths sized from them are ~2.5× weaker than
intended.

---

## Reproducing

```bash
./scripts/run_stages.sh hook       # resolve the hook point by reconstruction
./scripts/run_stages.sh calibrate  # measure max-acts in OUR units
./scripts/run_stages.sh control    # the Countdown dose-response
./scripts/run_stages.sh main       # GPQA + MATH with matched controls
```
53 tests: `pytest tests/`. Raw traces: `results/steering/*.jsonl` (5,664 attempts).
