# Fig. 4 (C4): the instrument decides the answer

*Status 2026-09-19. The faithful multi-seed run is in flight; everything below is from
`results/rl_ab/tz_train_claimA.log` — un-primed Qwen-2.5-3B, PPO on Countdown,
accuracy-only reward, 232 steps, 1,099 logged rollouts — which is the right experiment
under a **non-faithful config** (`rollout.n=1`, train batch 256) and a **single run**.
Read §5 before quoting any of it.*

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
counting convention (*"integer counts, 0 if none are present"*). On 60 stratified traces
from the same log:

| steps | Q&A | shift | conflict | reconciliation | **n_personas** |
|---|---|---|---|---|---|
| 1–23 | 0.876 | 0.438 | 0.000 | 0.000 | **1.00** |
| 98–119 | 0.311 | 0.000 | 0.000 | 0.000 | **1.00** |
| 210–232 | 0.230 | 0.000 | 0.000 | 0.000 | **1.00** |

Conflict of perspectives is **exactly zero in 7 of 10 bins**. `n_personas` is **1.00 in
every bin, first to last**. Nothing rises.

### The control that makes this credible

A judge that always answers 1 would produce this table on any input. It does not:

| trace set | n_personas | traces with >1 | shift | reconciliation |
|---|---|---|---|---|
| teacher **dialogue** corpus (known multi-persona) | **2.92** | **100%** | 2.15 | 0.92 |
| teacher **monologue** corpus (known single voice) | 1.00 | 0% | 0.20 | 0.07 |
| **RL traces, step > 200** | **1.00** | **0%** | 0.27 | 0.07 |

Same judge, same prompt, same session. It recovers ~3 personas when personas are present,
in 100% of traces. On late-RL traces it recovers one — and the whole row is
indistinguishable from the known-monologue corpus.

**So the two instruments disagree about Fig. 4, and the disagreement is not noise.** Where
marker-counting sees a 12× rise in dialogic behaviour, an LLM judge sees a single voice
that never becomes two.

### A design note on the judge

A parse failure is never read as zero. Traces lengthen under RL, so judge failures
concentrate in late training, and a fail-open judge would manufacture precisely the
decline being looked for. `parse_verdict` raises. The cache key carries both judge model
and prompt version, so two instruments cannot be silently mixed inside one curve.

## 5. Limits — read before quoting

- **Config.** This log is `rollout.n=1`, train batch 256. The paper's is `n=4`, batch 128.
  The faithful run is in flight (`scripts/fig4_pod.sh`).
- **One run.** No seeds. A co-author has told us the RL results are seed-sensitive and
  that they are running multi-seed for the revision; three seeds are queued here.
- **Judge identity.** The paper judges with Gemini-2.5-Pro; we judge with an Anthropic
  model. Their reported cross-judge ICC(3,1) ≈ .85 is why this is a declarable deviation
  rather than a different experiment — but it is a deviation.
- **Persona extraction.** The paper's judge characterises each perspective and answers
  BFI-10 items from its point of view. Ours returns a count. A richer procedure could
  segment a trace we score as one voice.
- **n = 13–15 per control set**, 60 traces in the main judge table (250-trace run pending).
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
