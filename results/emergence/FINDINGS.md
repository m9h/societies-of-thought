# Fig. 4 (C4): the instrument decides the answer

> **Read §4.10–4.11 first (2026-10-08).** Under the paper's own prompts, run verbatim,
> Fig. 4b/4e **reproduce** on our run: conflict and persona count rise over training under
> two judges. Under our earlier paraphrased prompt (§4–4.9) nothing rose. The title stands,
> but the direction of the dependence is the reverse of what §4–4.9 implied — and the
> judge-free series in §4.10 shows what the paper's instrument is counting.

*Status 2026-09-19. The faithful multi-seed run is in flight; everything below is from
`results/rl_ab/tz_train_claimA.log` — un-primed Qwen-2.5-3B, PPO on Countdown,
accuracy-only reward, 232 steps, 1,099 logged rollouts — which is the right experiment
under a **non-faithful config** (`rollout.n=1`, train batch 256) and a **single run**.
Read §5 before quoting any of it.*

## 0. This question was answered once before, and lost

Commit `ed8d42e` (2026-07-23) ran this analysis on this log and concluded *"no dialogic
society emerges — systematic SEARCH does,"* naming the same `(Doesn't equal N)` enumeration
template reported in §3 below. That finding was written **into the commit message and
nowhere else**: it produced no results file, so it never reached `HANDOFF.md`, and after
the next context compaction the project's own roadmap recorded C4 as "never run."

Two lessons, both cheap to act on:

- **A result that lives only in a commit message is a result you will lose.** The standing
  discipline here is "write the analysis before the data exists." Its missing counterpart
  is "write the result to a file, or it did not happen."
- **The July conclusion was right but under-built.** It rested on the 3-segment threshold
  (*"only 9/300 early and 0/300 late traces clear it"*) — the same filter whose asymmetry
  produced the retracted steering result. This version replaces it with the paper's own
  instrument plus a positive control, which is what makes the null defensible rather than
  merely correct.

## 1. The claim

> Controlled reinforcement learning experiments reveal that base models increase
> conversational behaviors when rewarded solely for reasoning accuracy. — abstract

Fig. 4b reports the frequency of conversational behaviours rising across PPO steps on
Qwen-2.5-3B (base, no instruction tuning), measured by an **LLM judge** over four
behaviours: question–answering, perspective shift, conflict of perspectives,
reconciliation. Fig. 4e reports judge-inferred persona counts.

## 2. What a surface-marker instrument says

Counting the markers directly, per 100 words (rates, not per-trace counts — response
length more than doubles, 166 → 357 words):

| steps | Q&A | shift | conflict | reconciliation |
|---|---|---|---|---|
| 1–23 | 0.041 | 0.207 | 0.254 | 0.332 |
| 50–72 | 0.042 | 0.329 | 0.709 | 0.699 |
| 98–119 | 0.007 | 0.015 | 2.302 | 0.423 |
| 210–232 | 0.000 | 0.003 | **3.069** | 0.307 |

Read naively: conflict of perspectives rises **12×**, and Fig. 4 replicates.

## 3. ⚠ That 12× is one template, and the reading is RETRACTED

Decomposing the matches by which string produced them:

| steps | total | top patterns |
|---|---|---|
| 1–48 | 185 | however=77, let's try another=22, instead=18 |
| 96–140 | 1,916 | **doesn't equal=1,720** |
| 188–232 | 2,095 | **doesn't equal=2,090** |

**2,090 of 2,095 late matches — 99.8% — are the single string "doesn't equal."** It is a
stock parenthetical the model emits once per line of a numbered enumeration:

> `1. 89 - 64 - 6 = 21 (Doesn't equal 19)`
> `2. 89 - 64 + 6 = 31 (Doesn't equal 19)`
> `3. 89 - 6 + 64 = 93 (Doesn't equal 19)` …

Every genuinely dialogic marker moves the **other** way: *however* 77 → 1, *let's try
another* 22 → 2, *nope* 26 → 0. The model is not learning to disagree with itself. It is
converging on a brute-force enumeration template that a marker-counting instrument scores
as disagreement.

`analysis.emergence.pattern_dominance` now reports, for any marker rate, what share comes
from a single repeated string. This project has already retracted one result to
asymmetric filtering; a marker rate quoted without a dominance figure beside it is the
same class of claim.

## 4. What the paper's instrument says

`rl/judge.py` implements the LLM-as-judge with the paper's four behaviours and its
counting convention (*"integer counts, 0 if none are present"*). Stratified sample, 25
traces per bin, **250 judged, 0 failures**, on traces with the scorer's debug echo removed
(see §4.3):

| steps | Q&A | shift | conflict | reconciliation | **n_personas** |
|---|---|---|---|---|---|
| 1–23 | 0.810 | 0.463 | 0.000 | 0.000 | **1.00** |
| 25–49 | 0.512 | 0.845 | 0.178 | 0.133 | **1.00** |
| 50–72 | 0.239 | 0.455 | 0.193 | 0.046 | **1.00** |
| 73–97 | 0.320 | 0.439 | 0.093 | 0.053 | **1.00** |
| 98–119 | 0.254 | 0.069 | 0.000 | 0.000 | **1.00** |
| 120–141 | 0.351 | 0.198 | 0.000 | 0.015 | **1.00** |
| 142–165 | 0.288 | 0.163 | 0.013 | 0.038 | **1.00** |
| 166–187 | 0.293 | 0.056 | 0.000 | 0.000 | **1.00** |
| 188–209 | 0.394 | 0.181 | 0.000 | 0.000 | **1.00** |
| 210–232 | 0.251 | 0.172 | 0.000 | 0.013 | **1.00** |

**Every one of the 250 traces scored `n_personas = 1`.** Not a mean pulled down by
outliers — the distribution is `{1: 250}`. Conflict of perspectives is exactly zero in six
of ten bins and zero in the last. Question–answering *falls*, 0.810 → 0.251. Perspective
shift *falls*, 0.463 → 0.172. **Nothing rises.**

A second judge (a different model, same prompt, 100 traces) returns `n_personas = 1.00` in
every bin as well.

### §4.3 A contaminant that was working against this null

verl's Countdown scorer echoes each graded rollout back to stdout as
`Target: … / Extracted equation: … / Solution string: <the whole prompt>`, and that block
lands *after* the response and *before* the next prompt. The first parser ran to the next
`User:` and swallowed it — **78% of traces carried ~45 words of it**, including the literal
phrase *"A conversation between User and Assistant."*

That inflated every word denominator, and it handed the judge a sentence that biases a
persona count toward two. The judge answered 1.00 anyway. So the null survived a
contaminant pushing against it, and the table above is recomputed from clean text.

Separately, the judge's token budget was raised 300 → 1200. At 300 a more verbose judge
spent its budget on preamble and never reached its JSON, failing on 74% of traces — and
since failures rise with trace length, the budget was itself a length artifact.

### The two instruments agree on three behaviours out of four

This is the part that makes the disagreement worth taking seriously. If judge and proxy
simply measured different things, neither would be informative about the other. They do
not: over the ten bins, per behaviour,

| behaviour | judge, first → last | proxy, first → last | same direction? | r |
|---|---|---|---|---|
| question_answering | 0.810 → 0.251 | 0.039 → 0.000 | **yes** | **+0.75** |
| perspective_shift | 0.463 → 0.172 | 0.116 → 0.000 | **yes** | **+0.83** |
| **conflict_of_perspectives** | 0.000 → 0.000 | 0.116 → **3.373** | **no** | **−0.53** |
| reconciliation | ~0 | 0.540 → 0.397 | levels differ | +0.88 |

Perspective shift tracks at r = +0.83 and question–answering at +0.75, both falling
under both instruments. Reconciliation is flat under both, at different levels. The instruments part
company on **exactly one** behaviour — the one whose marker count is 100% a single
repeated string.

That is a surgical result rather than a nihilistic one. Marker counting is not useless
here; it fails on one pattern, for a reason that is visible in the decomposition, and the
failure happens to land on the behaviour Fig. 4's headline rests on.

### The control that makes this credible

A judge that always answers 1 would produce this table on any input. It does not:

| trace set | n_personas | traces with >1 | shift | reconciliation |
|---|---|---|---|---|
| teacher **dialogue** corpus (known multi-persona) | **2.87** | **93%** | 2.07 | 0.87 |
| teacher **monologue** corpus (known single voice) | 1.00 | 0% | 0.20 | 0.07 |
| **RL traces, step > 200** (decontaminated) | **1.00** | **0%** | 0.40 | 0.07 |

n = 15 each, 0 judge failures. Per-trace counts, not rates.

Same judge, same prompt, same session. It recovers ~3 personas when personas are present,
in 93% of traces. On late-RL traces it recovers one — and the whole row is
indistinguishable from the known-monologue corpus.

**So the two instruments disagree about Fig. 4, and the disagreement is not noise.** Where
marker-counting sees a 12× rise in dialogic behaviour, an LLM judge sees a single voice
that never becomes two.

### A design note on the judge

> ⚠ Correction (2026-10-08): the paragraphs in this section and in §6 that treat the
> paper's prompt as unavailable are wrong; both prompts are in the v1 Supplementary Methods
> and are now the default instrument (`paper-v1`). See §4.10.

A parse failure is never read as zero. Traces lengthen under RL, so judge failures
concentrate in late training, and a fail-open judge would manufacture precisely the
decline being looked for. `parse_verdict` raises. The cache key carries both judge model
and prompt version, so two instruments cannot be silently mixed inside one curve.

## 4.5 The faithful config, mid-run (seed 0, step 166 of 250)

The run in flight uses the paper's configuration — `rollout.n=4`, train batch 128,
un-primed Qwen-2.5-3B, accuracy-only reward — and it is learning: validation on Countdown
climbs 0.602 → 0.668 by step 166, landing where the C5 faithful run's baseline arm
finished (0.661 at step 250).

**Judge, 250 traces, 25 per bin, 0 failures:**

| steps | Q&A | shift | conflict | reconciliation | **n_personas** |
|---|---|---|---|---|---|
| 0–16 | 0.618 | 0.545 | 0.254 | 0.036 | **1.00** |
| 17–33 | 0.957 | 0.478 | 0.217 | 0.130 | **1.00** |
| 34–50 | 0.520 | 0.565 | 0.045 | 0.090 | **1.00** |
| 51–67 | 0.452 | 1.157 | 0.108 | 0.072 | **1.00** |
| 68–84 | 0.303 | 0.389 | **0.000** | 0.058 | **1.00** |
| 102–118 | 0.467 | 0.409 | **0.000** | 0.019 | **1.00** |
| 135–150 | 0.545 | 0.981 | **0.000** | 0.044 | **1.00** |
| 151–166 | 0.469 | 0.664 | **0.000** | 0.117 | **1.00** |

Conflict of perspectives declines from 0.254 to **exactly zero from step 68 onward**.
Perspective shift is noisy with no trend (0.31–1.16). Question–answering drifts down.
Nothing emerges.

**Across both runs: 500 traces judged, `n_personas` distribution `{1: 500}`.** Two
configurations, two judge models, zero exceptions.

### ⚠ Two cautions this run adds

**A "rise then fall" was visible in the marker proxy and is NOT confirmed.** The proxy
showed conflict nearly doubling to step ~40 then decaying — which is the dynamic a
co-author predicted, citing Gandhi et al. The judge shows no rise at all over the same
traces (0.254 → 0.217 → 0.045). It was not written up as a finding, and it should not be.

**The proxy's agreement with the judge does not generalise.** On the `n=1` log the two
instruments tracked on three behaviours of four (perspective shift r = +0.83). On the
faithful run the correlations are ~0 across the board:

| behaviour | r (n=1 log) | r (faithful run) |
|---|---|---|
| question_answering | +0.75 | **−0.13** |
| perspective_shift | +0.83 | **+0.08** |
| conflict_of_perspectives | −0.53 | +0.42 |
| reconciliation | +0.88 | **−0.09** |

So the earlier "they agree on three of four" was a property of that one log — plausibly
because both instruments were watching the same collapse — not a general validation of
marker counting. **Surface-marker behaviour rates should not be trusted on this question
at all.** What survives across every cut is the judge's persona count, and it is 1.

## 4.9 FINAL: the complete 250-step run on the paper's configuration

*Seed 0 finished 2026-09-21. Un-primed Qwen-2.5-3B, PPO on Countdown, reward = accuracy
only, `rollout.n=4`, train batch 128, 250 steps. 2,555 logged rollouts. Full log at
`results/emergence/fig4_qwen_s0_FINAL.log.gz`.*

**The run learned.** Countdown validation, every 10 steps:

```
0.104 0.114 0.188 0.233 0.261 0.341 0.403 0.418 0.463 0.523 0.578 0.602 0.628
0.644 0.656 0.672 0.668 0.671 0.674 0.675 0.700 0.691 0.704 0.707 0.702 0.702
```

0.104 → **0.702**, peak 0.707. For reference the C5 faithful run's baseline arm finished at
0.661. This is a healthy, converged RL run, not a failed one — which is what makes the
behavioural null below worth anything.

**Judge, 8 bins × 32 traces, 256 judged, 0 failures:**

| steps | Q&A | shift | conflict | reconciliation | **n_personas** |
|---|---|---|---|---|---|
| 0–31 | 0.800 | 0.741 | 0.296 | 0.089 | **1.00** |
| 32–63 | 0.403 | 0.681 | 0.222 | 0.083 | **1.00** |
| 64–95 | 0.301 | 0.312 | 0.022 | 0.043 | **1.00** |
| 96–126 | 0.400 | 0.426 | **0.000** | 0.040 | **1.00** |
| 127–157 | 0.529 | 0.397 | **0.000** | 0.033 | **1.00** |
| 158–188 | 0.493 | 0.385 | **0.000** | 0.062 | **1.00** |
| 189–219 | 0.488 | 0.535 | **0.000** | 0.047 | **1.00** |
| 220–250 | 0.473 | 0.458 | **0.000** | 0.059 | **1.00** |

**All four behaviours decline across training. None rises.**

- question–answering 0.800 → 0.473
- perspective shift 0.741 → 0.458
- conflict of perspectives 0.296 → **0.000**, and it is zero from step 96 onward
- reconciliation 0.089 → 0.059, negligible throughout

`n_personas` is 1.00 in every bin. **Across all four judged runs: 861 traces, 0 failures,
distribution `{1: 861}`.**

### One thing that only a deeper sample caught

At 15 traces per bin, perspective shift appeared to *rise* (0.692 → 0.975) and an earlier
draft of this section said so. At 32 per bin it declines (0.741 → 0.458). The apparent rise
was sampling noise in a rate computed over ~150 words per trace. **Do not read a per-bin
trend off n = 15 here.** Four of this project's errors have now been small-sample or
filtering artifacts that produced a clean-looking trend.

### The marker proxy is not usable on this question

On the same 256 traces the proxy reports perspective shift going 0.207 → 0.000 — a total
collapse where the judge sees a 38% decline — and reconciliation flat at ~0.6 where the
judge sees 0.06. Correlations against the judge across bins are ~0 or negative
(reconciliation r = −0.86). Taken with §4.5, surface-marker behaviour rates have now
disagreed with the judge on every configuration tested, in both directions and by up to two
orders of magnitude. **Report the judge; do not report marker rates.**

## 4.10 Three corrections to our own instrument, found while cross-judging (2026-10-08)

Re-scoring the published 256-trace sample with a second judge (Haiku 4.5) exposed three
things that were wrong in §4–4.9, all ours. Each is fixed in the code and every number
below §4.10 is on the corrected pipeline.

1. **The paper's judge prompts are published.** We said for three months that the Fig. 4
   instrument was unpublished (§4 "A design note", §6 Q3, the note to the authors, the
   literature review). The v1 HTML's *Supplementary Methods: LLM-as-Judge prompts* carries
   the behaviour-counting prompt and the persona-identification prompt in full. Our v1
   prompt paraphrased their definitions and folded the persona count into the same call.
   `rl/judge.py` now ships both prompts verbatim as `paper-v1` (two calls per trace), and
   the claim is withdrawn wherever it appeared. One consequence is load-bearing: their
   persona prompt lists *"however," "but," "alternatively," "wait," "let me check,"
   "actually"* as indicators of a perspective and instructs the judge to treat *"each
   identifiable shift as a boundary between perspectives."* Under that instrument a persona
   count is close to a marker-segment count by construction, which is the mechanism §3
   proposed for the marker curve.
2. **The log mixes two series, and the paper's is the one we were not reporting.** verl
   prints the greedy held-out validation generations into the same log at every 10th step.
   Our parser attributed them to the step as if they were temperature-1 training rollouts:
   **453 of 2,555 (18%)**, concentrated at steps 0, 10, 20, … The paper's Methods: *"we
   evaluate model performance on a held-out validation set of 1,024 Countdown problems at
   each training checkpoint (every 10 steps). For each checkpoint, we generate reasoning
   traces for all validation problems and measure … the frequency of conversational
   behaviours."* So Fig. 4b is a validation-generation curve. `parse_rollouts` now tags
   `source`; the two series are judged separately below.
3. **Normalisation.** §4.9 reported judge counts per 100 words. The paper reports
   frequency per trace, unnormalised. Traces double in length over training (105 → 211
   words), so per-100-words rates fall even where per-trace counts do not: under the v1
   judge, per-trace question-answering goes 0.84 → 1.00 and perspective shift 0.78 → 0.97
   (flat to rising), while conflict goes 0.31 → 0.00 and n_personas stays 1.00. "All four
   behaviours decline" was true per word and **false per trace** for two of the four.
   Conflict → 0 and personas = 1 hold under either normalisation.

Also stripped: the scorer's post-hoc verdict line (`Invalid equation` etc., 2,239 traces)
and the prompt's `Assistant:`; responses are now cut at `<|endoftext|>`.

### A judge-free series on the corrected parse

Share of traces containing the token, by bin, on the **validation** generations (the
paper's series) and the training rollouts:

| validation (greedy) | n | "we" | "I" | any of *wait/however/actually/alternatively* | words |
|---|---|---|---|---|---|
| steps 0–30 | 76 | **1.00** | 0.00 | 0.42 | 143 |
| 40–70 | 76 | 0.41 | 0.64 | 0.29 | 264 |
| 80–100 | 53 | 0.00 | 1.00 | 0.00 | 245 |
| 110–250 (four bins) | 42–60 | 0.00–0.02 | 1.00 | 0.00 | 169–212 |

| train (T=1.0) | n | "we" | "I" | markers | words |
|---|---|---|---|---|---|
| 1–32 | 376 | 0.52 | 0.31 | 0.17 | 109 |
| 33–63 | 253 | 0.55 | 0.56 | 0.31 | 175 |
| 64–94 | 239 | 0.13 | 0.95 | 0.08 | 238 |
| 95–249 (five bins) | 222–260 | 0.00–0.01 | 0.99–1.00 | 0.00–0.01 | 175–210 |

The paper's Fig. 4c–d narrative is that the step-40 model is "mechanical, enumerative"
and by step 120 "two distinctive simulated personas have appeared, recognizing their
collectivity with the pronoun 'we'." On this run the pronoun runs the other way: the base
model starts in a collective "we" voice (100% of greedy validation traces before step 40),
RL replaces it with a single "I" by step 80, and the transitional markers their persona
prompt keys on vanish at the same point. Question marks are essentially absent in every
bin (≤ 0.08 per trace). None of this needs a judge. What the paper's own judge makes of
it, with their prompts verbatim, is §4.11.

## 4.11 The paper's instrument, verbatim: Fig. 4 reproduces — from a monologue

Three Anthropic judges were asked to score two stratified samples (8 bins × 32) with the
paper's two prompts exactly as published: the validation generations (the paper's series)
and the training rollouts. **Claude Opus 5 refused every call** — `stop_reason: refusal`,
0 output tokens, on 100% of 256 + 256 traces, with the paper's behaviour prompt and nothing
else in the context. (A judge that will not read the instrument is a judge-change effect
of the kind Yang et al. describe, and it is reported, not papered over.) The other two
judges completed with 1 and 0 failures.

### Validation series (the paper's Fig. 4b)

*claude-sonnet-5* — mean count per trace, by training-step bin:

| steps | n | Q&A | shift | conflict | reconc. | personas | >1 persona | words |
|---|---|---|---|---|---|---|---|---|
| 0–30 | 32 | 0.53 | 1.62 | 1.41 | 0.19 | 1.28 | 0.25 | 161 |
| 40–70 | 31 | 5.58 | 6.84 | 6.16 | 0.32 | 1.16 | 0.16 | 245 |
| 80–100 | 32 | 4.91 | 11.47 | 11.06 | 0.47 | 1.22 | 0.22 | 249 |
| 110–130 | 32 | 9.59 | 10.28 | 4.62 | 0.69 | 1.50 | 0.50 | 193 |
| 140–160 | 32 | 9.22 | 9.97 | 7.44 | 0.75 | 1.62 | 0.62 | 196 |
| 170–190 | 32 | 9.25 | 9.06 | 6.84 | 0.62 | 1.69 | 0.69 | 174 |
| 200–220 | 32 | 7.84 | 8.50 | 7.41 | 0.56 | 1.66 | 0.66 | 169 |
| 230–250 | 32 | 12.25 | 12.44 | 8.53 | 0.69 | 1.81 | 0.81 | 225 |

*claude-haiku-4-5-20251001* — mean count per trace, by training-step bin:

| steps | n | Q&A | shift | conflict | reconc. | personas | >1 persona | words |
|---|---|---|---|---|---|---|---|---|
| 0–30 | 32 | 1.19 | 1.06 | 1.16 | 0.03 | 1.75 | 0.75 | 161 |
| 40–70 | 32 | 2.12 | 4.06 | 2.22 | 0.09 | 1.47 | 0.47 | 256 |
| 80–100 | 32 | 2.03 | 2.16 | 1.66 | 0.03 | 1.56 | 0.56 | 249 |
| 110–130 | 32 | 1.00 | 0.59 | 5.59 | 0.03 | 1.88 | 0.88 | 193 |
| 140–160 | 32 | 1.38 | 0.84 | 4.69 | 0.09 | 1.81 | 0.81 | 196 |
| 170–190 | 32 | 1.12 | 0.88 | 8.84 | 0.03 | 1.94 | 0.94 | 174 |
| 200–220 | 32 | 1.12 | 0.41 | 7.41 | 0.06 | 1.88 | 0.88 | 169 |
| 230–250 | 32 | 1.25 | 0.41 | 9.59 | 0.09 | 1.97 | 0.97 | 225 |

Agreement on the same traces (chance-corrected beside exact match):

| pair | field | n | exact | κ | κ_quad | ICC(3,1) | ρ | mean A | mean B |
|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | Q&A | 255 | 0.45 | 0.09 | 0.02 | 0.03 | 0.12 | 7.40 | 1.41 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | shift | 255 | 0.29 | 0.15 | 0.02 | 0.04 | 0.00 | 8.78 | 1.27 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | conflict | 255 | 0.55 | 0.43 | 0.53 | 0.54 | 0.50 | 6.69 | 5.12 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | reconc. | 255 | 0.52 | 0.10 | 0.10 | 0.18 | 0.23 | 0.54 | 0.06 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | personas | 255 | 0.61 | 0.23 | 0.23 | 0.28 | 0.29 | 1.49 | 1.78 |

### Training series

*claude-sonnet-5* — mean count per trace, by training-step bin:

| steps | n | Q&A | shift | conflict | reconc. | personas | >1 persona | words |
|---|---|---|---|---|---|---|---|---|
| 1–32 | 32 | 0.78 | 1.25 | 0.78 | 0.28 | 1.44 | 0.41 | 86 |
| 33–63 | 30 | 3.70 | 5.00 | 3.30 | 0.73 | 1.37 | 0.37 | 185 |
| 64–94 | 32 | 7.22 | 9.88 | 5.88 | 0.69 | 1.31 | 0.31 | 255 |
| 95–125 | 32 | 7.84 | 8.75 | 3.97 | 0.75 | 1.53 | 0.53 | 185 |
| 126–156 | 32 | 9.03 | 9.34 | 6.59 | 0.69 | 1.62 | 0.62 | 192 |
| 157–187 | 32 | 7.94 | 8.28 | 6.12 | 0.78 | 1.72 | 0.72 | 160 |
| 188–218 | 31 | 11.71 | 11.90 | 9.90 | 0.74 | 1.58 | 0.58 | 220 |
| 219–249 | 32 | 12.00 | 11.75 | 10.59 | 0.72 | 1.78 | 0.78 | 225 |

*claude-haiku-4-5-20251001* — mean count per trace, by training-step bin:

| steps | n | Q&A | shift | conflict | reconc. | personas | >1 persona | words |
|---|---|---|---|---|---|---|---|---|
| 1–32 | 31 | 1.03 | 1.32 | 0.90 | 0.03 | 1.58 | 0.55 | 89 |
| 33–63 | 32 | 2.38 | 3.81 | 2.81 | 0.19 | 1.47 | 0.47 | 197 |
| 64–94 | 32 | 3.28 | 5.16 | 2.62 | 0.12 | 1.47 | 0.47 | 255 |
| 95–125 | 32 | 1.03 | 2.16 | 4.44 | 0.03 | 1.75 | 0.75 | 185 |
| 126–156 | 32 | 1.00 | 1.38 | 5.00 | 0.00 | 1.81 | 0.81 | 192 |
| 157–187 | 32 | 1.06 | 1.16 | 5.59 | 0.09 | 1.97 | 0.97 | 160 |
| 188–218 | 32 | 1.00 | 1.00 | 8.69 | 0.06 | 1.81 | 0.81 | 225 |
| 219–249 | 32 | 1.00 | 0.91 | 7.34 | 0.09 | 1.84 | 0.84 | 225 |

Agreement on the same traces (chance-corrected beside exact match):

| pair | field | n | exact | κ | κ_quad | ICC(3,1) | ρ | mean A | mean B |
|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | Q&A | 252 | 0.45 | 0.15 | 0.06 | 0.09 | 0.21 | 7.57 | 1.44 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | shift | 252 | 0.32 | 0.23 | 0.10 | 0.14 | 0.04 | 8.31 | 2.06 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | conflict | 252 | 0.54 | 0.43 | 0.51 | 0.51 | 0.43 | 5.92 | 4.67 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | reconc. | 252 | 0.40 | 0.08 | 0.08 | 0.17 | 0.20 | 0.67 | 0.08 |
| claude-sonnet-5 vs claude-haiku-4-5-20251001 | personas | 252 | 0.66 | 0.30 | 0.26 | 0.28 | 0.29 | 1.55 | 1.71 |

### What this means

1. **Fig. 4b and 4e reproduce under the paper's instrument.** Conflict of perspectives
   rises roughly 1 → 8–10 per trace under both judges (κ = 0.43 between them, the only
   field on which they agree beyond chance), and so does the persona count: Sonnet 1.28 →
   1.81 with the share of traces at >1 persona going 0.25 → 0.81; Haiku 1.75 → 1.97. Per
   100 words, conflict still rises 0.87 → 3.79, so it is not trace length (persona count
   is uncorrelated with length, ρ = 0.02). Reconciliation stays low and flat, as the paper
   says it does. Question-answering and perspective shift are judge-dependent: Sonnet 0.5
   → 12 per trace, Haiku flat near 1 or falling; κ 0.09–0.23.
2. **Our §4–4.9 null was our prompt.** The v1 paraphrase asked for "distinct perspectives"
   with "a single undifferentiated voice has n_personas = 1" and gave no examples; the
   paper's persona prompt names transitional markers and cognitive-role shifts as
   perspective boundaries and says to treat "each identifiable shift as a boundary." Same
   traces, same judge, 1.00 personas under ours and 1.5–1.9 under theirs. We were wrong to
   call the persona result non-reproducing; we were measuring a different construct.
3. **What the paper's instrument is counting on these traces, judge-free (§4.10):**
   - *Question-answering*: **no validation trace contains a question mark** (0 of 255),
     yet Sonnet counts 1,888 Q&A instances; 220 traces score Q&A > 0 with no "?". The
     definition's own example, *"Let's try X…? This gives us Y"*, is satisfied by every
     line of an arithmetic enumeration.
   - *Conflict*: correlates with the count of `(not 29)`-style rejection lines (ρ = 0.40;
     late traces average 9.1 such lines and 7.6 conflicts). These are the "doesn't equal"
     template of §3, now scored by the paper's prompt rather than by our regex.
   - *Personas*: in every one of the 69 late two-persona traces, the second perspective is
     a "final answer presentation" role — the `<think>` / `<answer>` split the prompt format
     imposes. Sonnet's own labels: *"the exploratory calculation agent"* and *"the
     answer-presenting agent."*
   - Meanwhile "we" goes 100% → 0%, "I" 0% → 100%, and *wait/however/actually/alternatively*
     42% → 0% by step 80 (§4.10). The paper's Fig. 4c–d story — step 40 mechanical, step
     120 two personas saying "we" — runs backwards on this seed.
4. **So the answer to Fig. 4 is now sharper than "it depends on the instrument."** The
   paper's instrument reproduces the paper's curves on a run whose traces become a
   single-voice, question-free, marker-free enumeration. That is a content-validity
   problem in Norman et al.'s sense: reliable (κ = 0.43 on conflict across two judges;
   the paper's ICC ≈ .85) and not measuring dialogue. The positive control (§4: dialogue
   corpus 2.9 personas, monologue 1.0 under the v1 prompt) still shows a judge *can* tell
   the two apart when asked our way; the paper's prompt asks a different question.

Files: `results/emergence/cross_judge_validation.json`, `cross_judge_train.json`
(verdicts, agreement, curves, failures), `results/emergence/human/` (50-trace blind sheet
on the validation series, paper's definitions; **awaiting a human rater**).

## 5. Limits — read before quoting

- **Config.** §2–4 are on the claimA log (`rollout.n=1`, train batch 256). §4.5 and §4.9
  are on the paper's configuration (`n=4`, batch 128, `scripts/fig4_pod.sh`), complete at
  250 steps. The answer did not change between them.
- **One seed.** A co-author has told us the RL results are seed-sensitive and that they
  are running multi-seed for the revision. Two more seeds are costed (~$140) and sit
  below instrument validation in `docs/ROADMAP.md`: a seeded result on an unvalidated
  judge is a more precise version of the same uncertainty.
- **Judge identity.** The paper judges with Gemini-2.5-Pro; we judge with an Anthropic
  model. Their reported cross-judge ICC(3,1) ≈ .85 is why this is a declarable deviation
  rather than a different experiment — but it is a deviation, and Norman et al. (§7) show
  that agreement of that size coexists with large judge-specific bias. Our own null rested
  on one model family until §4.10; it still rests on one *vendor's* models, because no
  second-family key is available here. `rl/judge.py` takes `gemini/…` or `openai/…` specs
  the moment one is.
- **Persona extraction.** §4–4.9 used our own single-call count. §4.11 uses the paper's
  persona prompt (BFI-10 and expertise per perspective) verbatim; the segmentation
  follow-up prompt is not run.
- **Opus refusal.** One of three intended judges returned no output on the paper's
  prompt; the cross-judge result is two judges, one vendor.
- **Human validation.** Not yet done; the sheet exists.
- **n = 13–15 per control set**; 861 traces judged across all tables, 0 failures.
- **Countdown.** Arithmetic search by a 3B base model is the paper's own choice of task
  for this figure, but it is not where dialogue would be most expected.

## 6. What we would ask the authors

1. For Fig. 4b, is the LLM judge run on training rollouts, on held-out evaluation
   generations, or on checkpoint samples? Our traces are scraped from training rollouts at
   temperature 1.0 and that may not be comparable.
2. Does Fig. 4e's persona count rise above 1 on Countdown, and by how much? That single
   number would tell us immediately whether we are measuring the same thing.
3. Does the judge see a *conflict of perspectives* in a numbered enumeration of rejected
   candidates? Ours does not; a judge that did would reproduce the marker-count curve.

## 7. Reading this result after the 2026 H2 literature

Full review in `docs/literature_2026_H2.md`. Four things change how §4–6 should be read.

1. **Judge-dependence is now a documented, quantified phenomenon, not our suspicion.**
   Norman, Rivera & Hughes (arXiv 2606.19544; 21 judges, 541k judgments) find 33–41 point
   deflation from exact match to Cohen's κ, judge rankings that move up to 14 positions
   across benchmarks, and production judges with test–retest > .95 *and* position bias >
   .10 — reliability without validity. Yang, Hou & Yang (2607.08535): a judge score "can
   move even when the candidate responses stay fixed, simply because the evaluator has
   changed." In that vocabulary the paper's ICC ≈ .85 is a *reliability* number, and Fig.
   4b/4e are an instrument reading, and the instrument's prompts are in the v1
   supplement (see §4.10, correction 1 — we had said otherwise). The same standard
   applies to us, which is why §4.10 exists and why every table from here reports κ and
   ICC rather than exact match.
2. **A concession.** Boppana et al. (2603.05488, "Reasoning Theater") show that discourse
   inflection points — backtracking, "wait" — "occur almost exclusively in responses where
   probes show large belief shifts." Markers track internal state. The "markers are exhaust"
   reading this document has leaned toward is too strong, and we withdraw it in that form.
   What survives is narrower: the *dialogic* reading (distinct personas in conflict) does
   not reproduce under the paper's own instrument. Whether persona boundaries coincide
   with probe-detected belief shifts is a judge-free test and is the roadmap's Tier 2.
3. **The decline is a regularity, not an anomaly.** RL fine-tuning reduces chain-of-thought
   faithfulness and narrative structure over training in several independent reports
   (2602.12506, 2605.24286, 2607.23458), sometimes rise-then-fall. A behaviour curve that
   falls under accuracy-only RL on Countdown is the expected shape, which makes the paper's
   rising curve the thing that needs an explanation rather than ours.
4. **The C5 mechanism has a name.** Krishnamurthy, Huang & Rajaraman (2606.13125) describe
   strategy selection via SFT on diverse strategies as a post-training mechanism; it is
   the theory-side label for "priming installs the contract."
