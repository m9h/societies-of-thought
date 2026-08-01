# Within QwQ, perspective diversity carries no independent signal about correctness
> **UPDATE 2026-07-31 — the non-math extension is complete; see `GPQA_within_problem.md`.**
> Across four non-math domains (6,689 QwQ traces) the pooled length-matched effect is small
> (+0.0066) and only GPQA separated (+0.0110, 1,003 pairs). Holding the *problem* fixed via
> 6 samples per problem, GPQA gives **+0.0023 [−0.0032, +0.0078] — no difference**, and the
> between-problem estimate sits 3.1 SE outside that interval, so it is excluded rather than
> unresolved. The two significant results in this whole line of work (math negative, GPQA
> positive) point in opposite directions and each dissolves under its own control. The
> math-only limitation below is now closed.

*Judge-free test of SoT's mediation claim on 10,000 QwQ traces. Code: `analysis/hse_qwq.py`
(measurement), `analysis/hse_qwq_length.py` (length control), `analysis/hse_qwq_compare.py`
(cross-run stability). Canonical data: `hse_shuffled_minilm.json`,
`hse_length_matched.json`.*

## The claim

SoT argues that reasoning models outperform instruction-tuned models because they simulate
a *diverse* internal society, and that this diversity **accounts for** the accuracy
advantage. Diversity is measured by an LLM judge that infers personas and scores their
spread, over 8,262 traces.

The cross-model comparison cannot isolate diversity: DeepSeek-R1 differs from DeepSeek-V3
in many ways at once. The **within-model** version is decidable. Hold the model, the
decoding, and the task distribution fixed; vary only the outcome. If diversity mediates
accuracy, correct traces must be more diverse than incorrect ones.

`PrimeIntellect/NuminaMath-QwQ-CoT-5M` (MIT) makes this cheap: 5,138,102 **QwQ** traces
with a per-trace `correct` label. QwQ-32B is one of the two models SoT builds its
descriptive and diversity claims on.

## Result

**At matched trace length, diversity does not differ between correct and incorrect traces.**

| estimate of hse_norm difference (correct − incorrect) | value | shrinkage |
|---|---|---|
| unadjusted, whole sample (n = 10,000) | −0.0186 [−0.0237, −0.0135] ✱ | — |
| stratified on length, 8 quantile bins | −0.0041 [−0.0085, +0.0002] | 78% |
| **1:1 length-matched, ±2% caliper (3,010 pairs)** | **−0.0003 [−0.0066, +0.0060]** | **99%** |

✱ = 95% CI excludes zero. Matched pairs averaged 1,195 vs 1,205 words — balanced to 1%.

The unadjusted numbers are large and consistent, and they are **length**:

| | correct | incorrect | ratio |
|---|---|---|---|
| words per trace | 924 | 2,100 | **2.3×** |
| segments per trace | 10.9 | 30.2 | **2.8×** |
| hse (raw) | 0.8127 | 1.1985 | — |
| mean pairwise distance | 0.3826 | 0.4819 | — |
| hse_norm | 0.2746 | 0.2932 | — |

When QwQ gets a problem wrong it flails: 2.3× the words, 2.8× the perspective shifts. Every
diversity measure follows length, and once length is held fixed **nothing is left**.

## What this does and does not say about the paper

**It is a null against the mediation claim.** SoT's C2/C3 require diversity to explain
accuracy. Within a single reasoning model, on a single domain, at matched trace length,
diversity has no association with correctness in either direction. Whatever separates a
correct QwQ trace from an incorrect one, it is not how many differentiated voices it
contains.

**It also retracts a result we briefly believed.** An earlier version of this file reported
that correct traces are *less* diverse (−0.0186, CI excluding zero, stable across sampling
and embedders). That is superseded: it was length, not diversity. We do **not** claim the
society is actively harmful.

**It leaves the descriptive observation intact.** QwQ traces are dialogic, and perspective
shifts track **struggle** — they are dense where the model is failing. That is consistent
with the account in `results/steering/FINDINGS.md` (dialogic markers as the exhaust of a
search process rather than its engine), but that interpretation now rests on the *steering*
result, where intervention raised dialogic behaviour and accuracy fell. This comparison is
a null and cannot carry it.

## Three conclusions from one dataset — read this before trusting any of them

This analysis reached three different answers. The sequence is the most useful thing in this
file.

| # | method | hse_norm difference | reads as |
|---|---|---|---|
| 1 | single-voice traces **dropped** | **+0.0134** [+0.0075, +0.0193] ✱ | correct MORE diverse |
| 2 | single-voice traces **scored 0** | **−0.0186** [−0.0237, −0.0135] ✱ | correct LESS diverse |
| 3 | **length-matched** | **−0.0003** [−0.0066, +0.0060] | **no difference** |

Both (1) and (2) had confidence intervals excluding zero. Both were artifacts.

**(1) was a filtering error.** Traces with fewer than 3 shifts cannot support a dendrogram,
so we dropped them — removing 707 correct vs 365 incorrect, i.e. deleting the low-diversity
tail of the correct group specifically. The paper's own convention settles it:

> If a reasoning trace contained only a single implicit voice, **E = 0**

A single voice is zero diversity: a measurement, not a missing value.

**(2) was a length confound.** `hse_norm` divides out segment *count* via log2(N), which is
not the same as controlling for length. Matching on words removed 99% of the effect.

**What the robustness checks did and did not catch.** We varied the sampling (streamed
prefix → seeded 100k shuffle), the sample size (2,000 → 5,000 per class), and the embedder
(MiniLM-L6 → mpnet-base). The sign was stable across all of them — `hse_qwq_compare.py`
reported *"SIGN STABLE across 3 paper-convention runs"*. **That stability told us nothing
about the confound**, because none of those choices touched length. Reproducibility is not
validity, and three consistent runs of a confounded estimator are still confounded.

## Limits

- **Math only.** NuminaMath, whereas SoT's pool spans BBH/GPQA/MATH/MMLU-Pro/MUSR.
- **Segmentation is a heuristic** — a regex over the paper's cue words, not a semantic
  parse. Same heuristic as our steering analysis, so the two are comparable, but it is not
  the paper's LLM judge.
- **Not the paper's instrument.** HSE has never been compared against the authors' judge on
  shared inputs. Our null is a null *for HSE*; if the two measures disagree, that is a
  finding about the measures and it is the most informative thing left to do.
- **Matching cannot reach exactly zero bias.** Residual imbalance inside the ±2% caliper is
  real; read the 99% shrinkage rather than the nominal p-value. A purely length-driven
  effect shrinks by most of its size, which is what happened here.
- **A null is not proof of absence.** With 3,010 matched pairs the CI is [−0.0066, +0.0060],
  so effects below ~0.007 in hse_norm (≈2% relative) remain possible.
- **Correlational throughout.**

## Next

1. **Apply this length control to our steering results.** Steering changed dialogic markers
   *and* trace length. If that result is also length-mediated, the interpretation in
   `results/steering/FINDINGS.md` needs the same correction this file just took — and it is
   currently load-bearing for the whole project. This is now the highest priority in the
   repository, ahead of any new experiment.
2. **Run the paper's judge-based measure on these same traces**, with the same length
   control, to test whether the null is about diversity or about HSE.
3. **DeepSeek-R1 traces**, the paper's other subject model, if an openly-licensed corpus
   with correctness labels exists.
