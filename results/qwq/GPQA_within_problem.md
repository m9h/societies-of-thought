# The GPQA diversity effect is between-problem structure, not diversity

*Completed 2026-07-31. Code: `analysis/hse_domains.py`, `analysis/within_problem.py` (both
written test-first). Data: `hse_domains_v2.json`, `within_problem_gpqa.json`. Corpus:
`modal_qwq_domains.py`, 6,689 QwQ-32B traces including 500 GPQA problems × 6 samples.*

## The claim under test

Across four non-math domains, GPQA was the one place where perspective diversity predicted
correctness after length matching: **+0.0133, p = 0.0003**, surviving Bonferroni. That was
the first result in this project supporting SoT's mediation claim rather than nulling it,
so it got the harshest test available.

The weakness: length-matched pairs come from **different problems**. Problem difficulty is
uncontrolled, and difficulty plausibly drives both the error rate and the trace structure.

## Result

| estimate | difference | n |
|---|---|---|
| between-problem, length-matched | **+0.0110** [+0.0078, +0.0143] ✱ | 1,003 pairs |
| **within-problem (problem held fixed)** | **+0.0023** [−0.0032, +0.0078] | 168 problems |

Holding the problem fixed, diversity does **not** differ between a model's correct and
incorrect traces. Within-pair length balance is excellent: 5,349 vs 5,399 mean words.

**This is not an underpowered null.** The within-problem standard error is 0.0028, so the
between-problem estimate of +0.0110 sits **3.1 SE above** the within-problem one and lies
outside its confidence interval. An effect of the size measured between problems is
positively excluded, not merely unresolved.

The between-problem effect itself replicated cleanly at scale — +0.0133 on 183 pairs became
+0.0110 on 1,003 pairs. It is real, reproducible, and **not about diversity**. It reflects
systematic differences between the problems QwQ tends to get right and the ones it tends to
get wrong.

## An unexpected finding: QwQ is problem-determined, not stochastic

Only **168 of 767** GPQA problems yielded both a correct and an incorrect trace:

| | n | share |
|---|---|---|
| always correct (6/6) | 388 | 51% |
| always incorrect (0/6) | 211 | 28% |
| mixed | **168** | **22%** |

If the six samples were independent draws at the measured 61.2% accuracy, ~94% of problems
would be mixed. Observing 22% means QwQ's success on a GPQA problem is close to
deterministic at temperature 0.6 — it either knows the problem or it doesn't, and resampling
rarely changes the outcome.

We predicted 93% informative problems on the independence assumption, and that prediction
was wrong by a factor of four. It cost nothing here (168 problems still gave a decisive
answer) but it is a real result about QwQ worth recording: **per-problem accuracy is
bimodal, not binomial.**

It also explains the between-problem effect. If problems partition into "known" and
"unknown", then comparing correct traces to incorrect traces across problems is largely
comparing *known problems to unknown ones* — two different populations that differ in many
ways, of which trace diversity is one symptom.

## Where this leaves the diversity claim

Every level of control now points the same way:

| test | result |
|---|---|
| QwQ, math, unadjusted | correct traces less diverse ✱ |
| QwQ, math, length-matched | **no difference** |
| QwQ, non-math pooled, length-matched | +0.0066 ✱ (small) |
| QwQ, GPQA, length-matched | +0.0110 ✱ |
| **QwQ, GPQA, within-problem** | **no difference** |

The two significant results (math negative, GPQA positive) point in *opposite* directions
and both dissolve under the appropriate control — length in one case, problem identity in
the other. That pattern is what a confounded measurement looks like, not a mechanism.

SoT's C2/C3 require diversity to account for the accuracy advantage. Within a single
reasoning model, holding the problem fixed, it accounts for nothing.

## Limits

- **Conditioned on middling difficulty.** Only mixed-outcome problems are informative, which
  by construction excludes the 79% QwQ answers consistently. If diversity matters only where
  the model is reliably right or reliably wrong, this design cannot see it — though it is
  unclear what mechanism would work that way.
- **168 problems.** Enough to exclude the between-problem effect size, not enough to exclude
  a much smaller one. The CI admits effects up to +0.0078.
- **One model, one instrument.** HSE, not the paper's LLM judge. Comparing both on shared
  inputs remains the most informative outstanding check.
- **GPQA only.** The within-problem control has not been run on BBH, MuSR or MMLU-Pro, whose
  between-problem estimates were null anyway.
- **Truncation excluded** (347 traces, 10.4% of GPQA). Truncated traces are max-length and
  always-wrong; including them would manufacture the confound this design exists to remove.

## Method note

Both analysis modules were written **before** the data existed — `within_problem.py` failed
with `ModuleNotFoundError` across 9 tests before implementation. That ordering mattered:
this project has three retracted or corrected results this week (the drop-filter inversion,
the length confound, the steering redundancy claim), every one produced by building the
analysis after seeing the numbers. Here the test that would have caught a difficulty
confound was written before there was a confound to find.
