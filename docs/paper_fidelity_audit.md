# Fidelity audit: our Claim B vs the paper's Claim B

*Written 2026-07-26, reading arXiv:2601.10825v1 HTML directly (not from notes). Every
paper-side number below is quoted from the source. Companion to
`results/claimB/FINDINGS.md`, which this audit corrects in two places.*

## Summary

**Our C5 is not a faithful replication of the paper's C5.** It is a related but different
experiment. The single biggest reason is that the paper primes on **out-of-domain**
reasoning problems and RLs on Countdown, while we prime on **Countdown itself**. Several
smaller deviations compound it. The differences we measured are **not** attributable to
seed, and I should not have framed them as though seed were the main open question.

Two errors in our own write-up, corrected below: the paper **does** run a no-priming
baseline arm (we claimed it didn't), and the paper's own figure caption **already says the
arms converge** (we presented convergence as a finding against the paper).

---

## 1. The paper's claims, as stated

Quoted from the source.

- **D (descriptive).** Reasoning models "exhibit much greater perspective diversity than
  baseline and merely instruction-tuned models."
- **C1 (causal, steering).** Steering feature 30939 raises Countdown accuracy.
- **C2 (mediation).** SEM: direct effect β=.228, indirect effect β=.066 (N=2048).
- **C3 (diversity).** LLM-judge-inferred personas, spread measured by an embedding
  dispersion metric.
- **C4 (RL emergence).** "the base model can spontaneously develop conversational
  behaviours … when rewarded solely for reasoning accuracy."
- **C5 (RL scaffolding).** Three conditions: "(1) Baseline (RL only, no priming), (2)
  Conversation fine-tuning …, and (3) Monologue fine-tuning."
- **C6 (cross-domain transfer).** Countdown-primed models evaluated on PolitiFact
  misinformation detection, "23,299 fact-checked claims" (Extended Data Fig. 9).

## 2. The paper contradicts itself on C5's endpoint

This matters more than anything we measured, so it goes first.

**Extended Data Fig. 8 caption** (their own figure, describing their own curves):

> Models initially fine-tuned with multi-agent dialogues (red) reach high accuracy faster
> than models fine-tuned with monologue-style reasoning (blue), **though both eventually
> converge**. The base model without fine-tuning (default; light green colors) learns more
> slowly.

**Main text, same experiment:**

> conversation-fine-tuned models consistently improve faster and **reach higher asymptotic
> accuracy**.

**Abstract:** "fine-tuning models with conversational scaffolding **substantially
accelerates** reasoning improvement."

The caption says converge; the main text says higher asymptote. These cannot both be true.
Our Qwen result (dialogue 0.621, monologue 0.618 at step 250) **agrees with their figure
caption** and disagrees with their main text.

> **Correction to `results/claimB/FINDINGS.md`.** We presented convergence as a result
> *against* the paper. It is more accurately a result *confirming the paper's own figure*
> and contradicting the sentence its abstract is built on. That is a sharper and more
> defensible framing, and it is the one to use.

> **Correction 2.** FINDINGS.md said "our version adds a third arm the paper does not run
> — baseline, no priming at all." **That is wrong.** Condition (1) is exactly that arm.
> Our baseline is a replication of their baseline, not an addition.

## 3. Where we match

Recovered from the resolved verl config printed in our own PPO logs, against their
Supplementary Tables 6 and 8.

| | paper | ours | |
|---|---|---|---|
| base models | Qwen-2.5-3B **and** Llama-3.2-3B | same two | ✅ |
| RL algorithm / library | PPO, Verl | PPO, Verl (TinyZero fork) | ✅ |
| training steps | 250 | 250 | ✅ |
| actor LR | 1e-6 | 1e-6 | ✅ |
| critic LR | 1e-5 | 1e-5 | ✅ |
| KL coefficient | 0.001 | 0.001 | ✅ |
| PPO mini-batch | 64 | 64 | ✅ |
| rollout temperature | 1.0 | 1.0 | ✅ |
| max response length | 1024 | 1024 | ✅ |
| arms | 3 (baseline / conversation / monologue) | 3 (same) | ✅ |
| SFT set size | 500 train / 100 val | 500 / 100 | ✅ |
| SFT optimizer / LR / schedule | AdamW, 1e-5, cosine | AdamW, 1e-5, cosine | ✅ |
| RL task | Countdown | Countdown | ✅ |

**Reward is effectively identical.** Theirs: `accuracy × 0.9 + format × 0.1`, format = has a
`<think>` block *and* an `<answer>` block. TinyZero's stock scorer: `1.0` correct, `0.1` if
an `<answer>` tag is present but wrong, `0` otherwise. Since a correct answer implies
correct format, both schemes give 1.0 / 0.1 / 0 — the same three values. **One asymmetry:**
their format term requires a literal `<think>`; ours does not. Our dialogue traces contain
`<persona1>`/`<think1>`/`<group_consensus>` and **no `<think>`** (verified: 500/500 traces),
so under the paper's reward our dialogue arm would forfeit the 0.1 format point on every
rollout. **Our scorer is more generous to dialogue than theirs** — and our dialogue arm
still only tied on Qwen.

Also worth recording: `val/test_score/countdown` is the **reward**, not pure accuracy. The
paper reports its Qwen baseline PPO reward at 250 steps as **0.5665**. Ours is **0.597**.
Same metric, ~3 points apart — the baseline arm is a genuine quantitative replication.

## 4. Where we deviate

### 4.1 The one that matters: priming domain

> we prompt Qwen-2.5-32B-IT to produce multi-agent-like dialogues … solving **8,262
> reasoning tasks** (see Methods: Data), and sample 600 instances that reach correct answers

And those 8,262 tasks are:

> BigBench Hard (BBH) …; GPQA …; MATH (Hard) …; MMLU-Pro …; IFEval …; and MUSR

**Their priming data is general reasoning — not Countdown.** Supplementary Table 7's
examples confirm it: a ball-swapping BBH problem and a GCD number-theory problem. RL then
runs on Countdown, a task the priming data never contains.

**Our priming data is Countdown** (`rl/data/*.json` records are `{pid, numbers, target,
…}`, and our tests assert every SFT response scores 1.0 under the Countdown grader).

This is a different experiment:

- **Their C5 asks:** does conversational *structure*, learned on unrelated problems,
  transfer and accelerate RL on a new task?
- **Our C5 asks:** does dialogue vs monologue *format* matter when both arms already carry
  in-domain task knowledge?

Ours is the harder test of format, and the weaker test of *their* claim. It also explains
nearly every quantitative gap:

- our absolute numbers are far higher (their Llama dialogue is 11% at step 70; ours is
  41.9% at step 75) — in-domain priming teaches Countdown-solving, not just a format;
- our monologue arm is far stronger than theirs, because our monologue priming also
  teaches Countdown while theirs teaches general CoT;
- **the Qwen convergence follows directly** — if both arms are handed the task knowledge,
  there is little left for format to differentiate.

### 4.2 Everything else

| | paper | ours | risk |
|---|---|---|---|
| **teacher model** | Qwen-2.5-32B-IT | **Qwen-2.5-72B-IT** | OpenRouter does not host the 32B; applied identically to both primed arms, so controlled *between* arms, but inflates our absolutes |
| **rollouts per prompt (`n`)** | **4** | **1** | fewer samples per prompt → weaker advantage estimates, slower/noisier learning |
| train batch size | 128 | 256 | |
| val batch size | 640 | 1312 | |
| **max prompt length** | **1024** | **256** | ours truncates a long prompt; our prompt is short so likely inert, but unverified |
| eval cadence | every 10 steps, 1,024 problems | every 25 steps, 1,312 problems | coarser curve; we cannot resolve their step-40 and step-70 claims |
| **SFT epochs** | **5** | **3** | our priming is weaker than theirs |
| SFT batch size | 64 | 32 (4 × grad-accum 8) | |
| SFT context window | 2048 | 1536 | |
| SFT warmup | linear, 10% of steps | 3% | |
| **RL prompt** | bare instruction, no chat wrapper | TinyZero wrapper ending `"Assistant:"` | ours is the more controlled choice for the arm comparison, but it is not their prompt |
| dialogue answer tag | `<group_solution>` | `<group_consensus>` → `<answer>` | cosmetic; normalised before grading |
| seeds | not stated | 1 | |

### 4.3 The Llama arm has a specific, disqualifying deviation

> For the conversation condition, reasoning content from multiple personas was
> **concatenated into a single block (`<think> </think>`) to ensure comparable sequence
> lengths across conditions** in Llama-3.2-3B.

**They length-matched their Llama dialogue condition. We did not.** Our Llama dialogue arm
carries the full 1.8× length advantage (220 vs 123 median words) that theirs was explicitly
constructed to remove.

So our Llama dialogue-vs-monologue gap (+0.081) is measured under a confound the paper
controlled for. **It is not comparable to their Llama result**, and it is the weakest number
in our set. Note the paper controlled length on Llama but *not* on Qwen — worth flagging in
any write-up.

## 5. So: is the discrepancy seed?

**No.** Ranked by how much each could plausibly move the result:

1. **Priming domain (in- vs out-of-domain)** — different experiment. Almost certainly the
   dominant cause of both our inflated absolutes and our Qwen convergence.
2. **Llama length-matching** — they removed the length confound, we didn't. Our +0.081 is
   not their +22-point gap.
3. **Rollouts n=1 vs 4** — a real change to the PPO gradient signal, on every arm.
4. **Teacher 72B vs 32B** — stronger traces for both primed arms.
5. **SFT epochs 3 vs 5**, batch 32 vs 64, context 1536 vs 2048 — our priming is uniformly
   weaker than theirs.
6. **Seed** — genuinely unknown, and still worth measuring, but it is now item 6, not the
   headline.

I previously told you seeds were "the binding gap." That was wrong on this evidence.
Seeds bound how precisely we can state *our own* numbers; they do not close the gap to the
paper. Fidelity does.

## 6. What our result still supports

Stated at the width the evidence actually carries:

- **The baseline arm replicates quantitatively.** Their 0.5665, our 0.597, same metric.
- **On Qwen, with in-domain priming, dialogue and monologue converge by 250** (+0.003) —
  consistent with their figure caption, inconsistent with their abstract.
- **Our Llama baseline plateaus at 0.176 @150, where their Llama *monologue* arm plateaus
  (~18% @150).** With §4.1 in hand this has a mechanism rather than being a coincidence:
  their monologue priming was out-of-domain general CoT, which teaches Countdown little, so
  it behaves much like no priming. We cannot assert this about *their* baseline, since they
  report no Llama baseline number — only that the base model "learns more slowly."

## 7. What to run next, in priority order

1. **Rebuild the priming sets from out-of-domain problems** (BBH / MATH-Hard / MMLU-Pro, as
   they did) and rerun Qwen C5. This is the only change that makes our experiment *their*
   experiment. Everything else is a rounding error next to it.
2. **Length-match the Llama dialogue condition** by concatenating personas into one
   `<think>` block, exactly as their Supplementary Methods describe.
3. **Set `rollout.n=4`**, train batch 128, val 640, max_prompt_length 1024, eval every 10
   steps; SFT epochs 5, batch 64, context 2048, warmup 10%. Cheap, mechanical, removes six
   deviations at once.
4. Only then seeds.

Note for `briefs/transfer_misinformation_scope.md`: **the paper already runs the
cross-domain transfer experiment** (C6, PolitiFact, 23,299 claims, Extended Data Fig. 9).
That brief was scoped as novel work; it is a replication target. Their transfer comparison
is conversation-primed **vs baseline** — they do not appear to run a monologue arm there,
which is a genuine gap our design would close.
