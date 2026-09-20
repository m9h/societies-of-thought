# Synopsis for the SoT authors — draft reply

*Drafted 2026-09-19 in reply to the co-author's note (code not yet released; journal review;
asking which result we tried to replicate; flagging seed sensitivity and arXiv 2503.01307).*

---

Thank you — and no apology needed on timing.

Happy to be specific. Short version first, then the numbers, then the handful of places
where your answer would change what we conclude.

## 1. Which results we have run

Two, and your guess was half right — so we went and ran the other one.

**Fig. 8** (the SFT-priming → RL experiment: no priming / dialogue-primed / monologue-primed,
Qwen-2.5-3B, PPO on Countdown, 250 steps). That is what our attendee was describing.

**Fig. 4** (base model developing conversational behaviour under accuracy-only reward) —
which is what you guessed, and you were right that we had looked at it. We had, in July,
and we have now redone it properly with your instrument rather than ours. See §4; it is
the part we would most like you to push back on.

We also ran the feature-30939 steering effect and the perspective-diversity ↔ correctness
relationship on an independent trace corpus, neither of which needs training.

## 2. Our setup, so you can tell us if we got something wrong

We rebuilt from the paper alone, so the first useful thing you can do is point at a mismatch.

Matched to Supplementary Tables 6 and 8: PPO in verl, 250 steps, actor LR 1e-6, critic LR
1e-5, KL 0.001, mini-batch 64, temperature 1.0, max response 1024, max prompt 1024, train
batch 128, val 640, eval every 10 steps, rollout **n=4**; SFT 500 train / 100 val, AdamW,
1e-5 cosine, 5 epochs, batch 64, context 2048, 10% warmup. Teacher **Qwen-2.5-32B-Instruct**,
your verbatim generation prompts. Priming problems drawn **out of domain**
(BBH / GPQA / MATH-Hard / MMLU-Pro / MUSR) with RL on Countdown — we reconstructed the pool
and got 8,262 problems, of which 7,738 are gradable, which matches your stated figure.

Two declared deviations: we can't recover your exact problem IDs, and our RL prompt uses the
TinyZero wrapper rather than a bare instruction.

*(An earlier attempt of ours primed on Countdown itself. That was our error, not a finding,
and we discarded it.)*

## 3. What we found

**(a) The RL experiment.** We reproduce your one published per-arm Qwen number. At step 40
you report dialogue 38% / monologue 28%; we get **37.7% / 30.4%**. Your baseline PPO reward
at 250 steps is 0.5665; ours is 0.661 on the same metric.

Where we differ is what happens after that. Dialogue leads monologue by **+0.043** averaged
over steps 10–60, is level by step 70, and at step 250 finishes **last** of the three:

| step 250 | baseline | dialogue | monologue |
|---|---|---|---|
| reward | 0.661 | 0.653 | 0.671 |

Both primed arms sit ~+0.04 over the unprimed baseline through mid-training. So in our run
the variable that persists is **priming**, and the dialogue-vs-monologue distinction stops
mattering after ~step 60.

We note this agrees with your own Fig. 8 caption ("*though both eventually converge*") and
not with the main text ("*reach higher asymptotic accuracy*"). We'd rather ask than assume
which you intend.

**(b) Steering.** Feature 30939, layer 15, DeepSeek-R1-Distill-Llama-8B. The feature we
recover is clearly yours — sparsity 0.00017 vs Neuronpedia's 0.00016, and it fires only on
surprise markers. Countdown accuracy does rise, but by **+10 points** (24.0% → 34.0%, n=200,
95% CI [+2.5, +18.0]) rather than your +27.7. The dose response is an inverted U: at the
higher reading of s=±10 the model collapses to 3.5% and 96% of traces are unparseable. On
MATH-Hard the same feature at the same dose costs **−22 points**.

Two things that cut in your favour and are worth recording. First, against six
sparsity- and max-activation-matched control features, 30939 is a clear outlier (control
accuracy averages 57.5% vs 62% baseline; 30939 gives 40%) — it is a real, specific lever,
not an arbitrary direction. Second, we **retracted** an earlier claim of ours that steering
produces a redundant society: that came from dropping traces with too few segments to
score, which discarded 47.5% of our baseline condition against 4% at the top dose. Under
your own convention (single voice = 0) normalised diversity *rises* with steering,
0.124 → 0.182.

**(c) Diversity and correctness.** On 6,689 QwQ-32B traces, correct traces are more diverse
than length-matched incorrect ones on GPQA: +0.0110 [+0.0078, +0.0143], 1,003 pairs. But
when we hold the **problem** fixed and compare a problem's own correct traces against its own
incorrect ones, the effect goes to **+0.0023 [−0.0032, +0.0078]** — and the between-problem
estimate sits 3.1 SE outside that interval, so it's excluded rather than merely unresolved.
On that corpus the relationship looks like between-problem structure rather than diversity
per se.

(Incidental, possibly useful to you: only 168 of 767 GPQA problems yielded both outcomes
across 6 samples at T=0.6. Independence would predict ~94% mixed; we see 22%. Per-problem
accuracy is bimodal, not binomial, which matters for how any per-trace analysis is powered.)

## 4. Fig. 4: two instruments, two different answers

We ran the un-primed arm — Qwen-2.5-3B base, PPO on Countdown, reward = accuracy only,
232 steps, 1,099 logged rollouts — and measured the behaviour frequency two ways.

**By counting surface markers, your result replicates and then some.** Conflict of
perspectives rises 12× across training, as a rate per 100 words rather than a per-trace
count, so it is not the length increase (166 → 357 words).

**We do not believe that number, and here is why.** Decomposing the matches by which
string produced them: in late training, **2,090 of 2,095 — 99.8% — are the single phrase
"doesn't equal"**, emitted once per line of a numbered enumeration:

```
1. 89 - 64 - 6 = 21 (Doesn't equal 19)
2. 89 - 64 + 6 = 31 (Doesn't equal 19)
3. 89 - 6 + 64 = 93 (Doesn't equal 19)   ...
```

Every genuinely dialogic marker moves the other way over the same steps: *however* 77 → 1,
*let's try another* 22 → 2, *nope* 26 → 0. The model is converging on a brute-force
enumeration template that a marker-counting instrument scores as self-disagreement.

**So we built your instrument instead** — an LLM judge over your four behaviours
(question–answering, perspective shift, conflict of perspectives, reconciliation) with
your counting convention. On stratified traces from the same run:

| steps | Q&A | shift | conflict | reconciliation | n_personas |
|---|---|---|---|---|---|
| 1–23 | 0.876 | 0.438 | 0.000 | 0.000 | **1.00** |
| 98–119 | 0.311 | 0.000 | 0.000 | 0.000 | **1.00** |
| 210–232 | 0.230 | 0.000 | 0.000 | 0.000 | **1.00** |

Conflict of perspectives is exactly zero in 7 of 10 bins. `n_personas` is 1.00 in every
bin, first to last. Nothing rises.

**With the control, because a judge that always says 1 would produce that table on any
input.** Same judge, same prompt, same session:

| trace set | n_personas | traces with >1 | shift | reconciliation |
|---|---|---|---|---|
| your dialogue prompt's own output (32B teacher) | **2.92** | **100%** | 2.15 | 0.92 |
| your monologue prompt's own output | 1.00 | 0% | 0.20 | 0.07 |
| **our RL traces, step > 200** | **1.00** | **0%** | 0.27 | 0.07 |

It finds ~3 personas when personas are there, in every trace. On late-RL traces it finds
one, and the whole row is indistinguishable from the known-monologue corpus.

**And the two instruments agree on three of your four behaviours**, which is why we think
the fourth is worth your attention rather than dismissible as judge noise:

| behaviour | judge, first → last | markers, first → last | same direction? | r |
|---|---|---|---|---|
| question–answering | 0.561 → 0.262 | 0.028 → 0.000 | yes | +0.60 |
| perspective shift | 0.236 → 0.060 | 0.085 → 0.000 | yes | **+0.90** |
| **conflict of perspectives** | 0.000 → 0.000 | 0.085 → **2.976** | **no** | **−0.39** |
| reconciliation | ~0, flat | ~0.35, flat | both flat | +0.79 |

Perspective shift tracks at r = +0.90 across the whole run. The two instruments part
company on exactly one behaviour, and it is the one whose marker count is 100% a single
repeated string.

Our honest summary: **on this run, the answer to Fig. 4 depends entirely on which
instrument you use**, and we cannot tell from the paper which way yours would fall on a
numbered enumeration of rejected candidates. That is a question only you can answer, and
it is the single thing we would most like to hear back about.

Caveats we hold against ourselves: this log is `rollout.n=1` and train batch 256, not your
config; it is one run; our judge is an Anthropic model rather than Gemini-2.5-Pro (your
cross-judge ICC ≈ .85 is why we think that is declarable rather than disqualifying); and
your persona extraction characterises each perspective and answers BFI-10 items from its
point of view, where ours returns a count — a richer procedure might segment a trace we
score as one voice. The faithful config is running now, three seeds, and we will send the
numbers either way.

## 5. On seeds — you're right, and here's exactly how far ours reaches

Ours is **n=1 per arm**. So we can't and don't claim your Fig. 8 result fails to replicate.
The late-window gaps (|Δ| ≤ 0.02) are well inside plausible run-to-run noise, and we say so
in the write-up. The one difference large and sustained enough for us to lean on is the
**early** dialogue lead, +0.043 across six consecutive checkpoints — which is your effect,
found where you found it.

Your pointer to Gandhi et al. fits our curve's shape well, and if your multi-seed runs show
the dialogue-vs-monologue gap is a transient that closes, that is what we see too — we'd
just read it as a statement about training *speed* rather than asymptote. We'd be glad to
see those curves.

That citation raises something we'd like your view on, because it cuts in two directions
and we can't tell which you intend.

**The defensive direction, which we accept.** Their §3.4 reports that "RL selectively
amplifies empirically useful behaviors while suppressing others" — backtracking and
verification are retained and strengthened while backward chaining and subgoal setting
diminish. If conversational markers behave the same way, a marker curve that rises and then
falls across seeds is a known dynamic rather than a failed effect, and we'd read Fig. 4
accordingly.

**The direction that worries us.** The same section reports that Qwen "transitions from
explicit verification statements in language ... to implicit solution checking." If RL
routinely moves a competence *off the token surface*, then marker frequency stops being a
valid proxy for the underlying behaviour partway through training — and a decline in the
curve becomes ambiguous between "the behaviour faded" and "the behaviour went implicit."
That's an instrument problem rather than a seed problem, and averaging over seeds won't
resolve it.

**And a prior-art question about Fig. 8 specifically.** Gandhi et al. is the same design:
Qwen-2.5-3B and Llama-3.2-3B, PPO on Countdown, behaviour-rich SFT before RL. They already
establish that priming Llama with behaviour-containing examples "enables substantial
improvements during RL, matching or exceeding Qwen's performance." So "priming helps, and
helps Llama most" is prior work. What Fig. 8 adds over it is specifically that *dialogue*
form beats *monologue* form — which is the contrast our run finds dissolving by step 70.

Their sharpest control seems directly applicable here: they find "the presence of reasoning
behaviors, rather than correctness of answers, proves to be the critical factor," with
models primed on *incorrect* solutions containing proper reasoning patterns performing
comparably. Since your teacher is a 32B instruct model, both your primed conditions are
behaviour-rich by their measure. We'd want to know whether your monologue traces and your
dialogue traces differ in backtracking/verification density, because if they do, the
dialogue arm may be winning on behaviour content rather than on dialogic form.

One confound we flag against ourselves: our dialogue priming traces are **1.75× longer**
than the monologue ones (252 vs 144 median words). We notice your Supplementary Methods
concatenate personas into one `<think>` block for Llama-3.2-3B specifically, to equalise
sequence length. We don't see the same control described for Qwen — question below.

## 6. Where your feedback would change our conclusions

1. **Fig. 8 endpoint.** Caption says converge, main text says higher asymptote. Which is the
   intended claim? Our whole read of (a) turns on this.
2. **Length matching on Qwen.** Was the persona-concatenation control applied to the Qwen
   dialogue condition too, or only to Llama? If only Llama, is the Qwen dialogue arm longer
   per example than the monologue arm in your runs as well?
3. **Format compliance.** Did you measure `<answer>`-emission rate for the primed vs unprimed
   checkpoints? Our leading mechanism for the early gap is mundane: unprimed Qwen-2.5-3B
   emits `<answer>` in only 48/64 completions, and a rollout without it scores exactly 0, so
   the unprimed policy starts with a large fraction of ungradable rollouts. That predicts a
   real early advantage that decays as RL teaches the format anyway. If your primed
   checkpoints are both at ~100% while the base model is not, the effect may localise there.
4. **Steering units.** Is `s = ±10` in units of the feature's max activation, in SAE
   activation units, or something else? We tried both readings; one gives +5 points, the
   other total collapse, and neither gives +27.7. A single sentence from you would settle it.
5. **Steering off Countdown.** Did you evaluate feature 30939 on any non-search benchmark?
   Our MATH-Hard reversal is the result we'd most like to be wrong about.
6. **Behaviour density across your two primed conditions.** Do your dialogue and monologue
   priming sets match on Gandhi et al.'s four behaviours (verification, backtracking,
   subgoal setting, backward chaining)? If not, that's an alternative explanation for the
   early gap that doesn't involve dialogue.
7. **Fig. 4's instrument, concretely.** Does your judge score a numbered enumeration of
   rejected candidates ("1. ... (Doesn't equal 19) 2. ...") as a conflict of perspectives?
   Ours does not. A judge that did would reproduce our marker-count curve exactly.
8. **Fig. 4e on Countdown.** Does the judge-inferred persona count rise above 1, and by how
   much? That one number would tell us immediately whether we are measuring the same thing.
9. **What the judge reads.** Is Fig. 4b run on training rollouts, on held-out evaluation
   generations, or on checkpoint samples? Ours are training rollouts at temperature 1.0,
   which may not be comparable.
10. **Rise and fall.** In your multi-seed runs, does the marker curve rise and then fall?
   And if it falls, how do you separate the behaviour fading from it going implicit, given
   the explicit-to-implicit transition Gandhi et al. report for Qwen on this exact task?
11. **Mediation, within-problem.** We know from the SI that you control for trace length and
   include problem fixed effects in several analyses. Our question is narrower: in the
   mediation model specifically, is the diversity → accuracy path estimated within-problem?
   Our (c) says that's where the effect lives or dies.

## 7. What would help most, and what we can offer

If any of it is shareable before the code release, the three things that would most reduce
guesswork on our side are: the **problem IDs** for the 8,262-task pool (or just the 600
sampled dialogues), the **steering scale convention**, and one **raw PPO curve** from Fig. 8
with its seed.

In return, everything of ours is public and we're happy to hand over anything useful: the
reconstructed pool builder, the priming corpora, per-step PPO metrics for all three arms,
the 6,689-trace QwQ corpus, and the steering sweep including the matched controls.

We'd also be glad to run something for you — if there's a control you'd like to see that
didn't fit in the paper, we have the harness standing and would run it and report the result
whichever way it came out.

Last thing, in the spirit of full disclosure: we've corrected three of our own results during
this work, including the retraction in (b). Anything above is offered as our current best
reading, not a verdict, and we'd rather be corrected now than in public.
