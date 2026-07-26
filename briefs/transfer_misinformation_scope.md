# Cross-domain transfer: does Countdown-dialogue priming help misinformation detection?

*Scope + design. Written while Claim B (C5) trains. Companion to
`faithful_rl_replication_scope.md`.*

## The claim under test

The SoT paper's most ambitious claim, and the least tested by anyone external: a base
model **primed on multi-agent Countdown dialogues** then improves *faster* on
**political misinformation detection** (PolitiFact, ~23,299 claims) than a
monologue-primed control — despite neither priming set containing a single
misinformation example. If true, it is the claim that matters most, because it is the
only one that separates two very different readings of the whole paper:

> **"dialogue format is a better fit for Countdown"**  (a narrow, task-shaped effect)
> vs
> **"dialogue format is a better fit for reasoning"**  (a general, transferable effect)

Every other result in the paper is consistent with the narrow reading. This one is not.
So it is the single highest-information experiment left — and the one where our
controls-first stance has the most to add, because a raw "dialogue transfers" result has
at least three cheaper explanations the paper does not rule out.

## Why this is cheap to run: it reuses the whole harness

Misinformation detection with a gold label is a **verifiable-reward** task, exactly like
Countdown. The model reads a claim, emits reasoning + a verdict, and the reward is
`1.0 iff predicted label == gold`. That means the *entire* Claim A/B machine transfers
with two swaps and nothing else:

| Countdown (Claim A/B)              | Misinformation transfer            |
|------------------------------------|------------------------------------|
| dataset = Countdown-Tasks-3to4     | dataset = LIAR / PolitiFact        |
| reward = equation evaluates to target | reward = verdict matches gold label |
| **prompt template, PPO recipe, priming checkpoints, verl config — all identical** |

The primed checkpoints are the *same artifacts* Claim B produces (`/workspace/ckpt/
{dialogue,monologue}`). No new priming, no new model. Only a dataset loader and a
label-match reward are new code — and both are testable offline the way `claimB_data.py`
was.

## Dataset

Use **`liar`** (Wang 2017, HF `liar`): 12.8k PolitiFact statements, 6 truthfulness labels
(pants-fire → true), with speaker/context metadata. The paper's "23,299 claims" is a
larger PolitiFact scrape; LIAR is the reproducible, citable subset and is enough to
measure a learning-*rate* difference. Recorded deviation, verified-loadable at Tier 0.

- **Binary collapse** (primary): {pants-fire, false, barely-true} → FALSE;
  {half-true, mostly-true, true} → TRUE. Cleaner reward, less label noise. The 6-way
  boundary ("half-true" vs "mostly-true") is noisy even for humans and would inject
  variance that swamps a priming effect.
- **6-way** (secondary, only if binary shows a gap): the paper's harder setting.

A `data_source="misinfo"` reward key keeps it separate from the countdown scorer, so both
can coexist in one verl run if ever wanted.

## Arms and controls — the part the paper skips

The paper reports dialogue-primed vs monologue-primed, n=1. A bare gap there has **three
cheaper explanations than "reasoning transfers"**, and each gets a control:

1. **More priming tokens.** Dialogue priming is 1.8× the tokens of monologue
   (220 vs 123 median words — measured, `test_claimB_data.py`). A dialogue advantage could
   be sheer priming compute. → **length-matched monologue** arm: monologue traces padded
   with extra verified reasoning to match dialogue token count. If the dialogue edge
   survives length-matching, it is not just tokens.
2. **More test-time compute.** Dialogue-primed models emit longer outputs at inference
   (more tokens = more chances to be right). → **report accuracy at matched output
   length**, and include per-token-budget accuracy, not just final accuracy.
3. **Format-matching to a verbose task.** → the **baseline** (unprimed) arm anchors the
   no-priming curve; if dialogue only matches baseline once you account for 1+2, there is
   no reasoning transfer.

So the arm set is **baseline / dialogue / monologue / monologue-length-matched**, all RL'd
identically on LIAR. Seeds per our standing critique (the paper's n=1-vs-n=1 gap is inside
plausible seed noise): 1 seed at Tier 1, 3 seeds at Tier 2.

## Our sharper prediction (falsifiable, and it splits from the paper's)

From the steering + HSE + Claim-A-emergence results, our model of what Countdown RL/priming
actually installs is **systematic search — enumerate, verify, backtrack — not a society of
voices.** That yields a concrete, testable prediction that *differs* from the paper's:

> If any transfer to misinformation exists, it tracks **verification/backtracking**
> behaviour (does the model check a claim against its stated reasoning before committing?),
> **not persona-diversity.** Concretely: regress per-example transfer accuracy on (a) our
> marker/verification rate and (b) HSE persona-diversity. We predict (a) carries the signal
> and (b) does not — the same dissociation the steering and RL-trace analyses already found.

If instead diversity carries it, our whole "search not society" thesis is wrong here, and
we say so. That is the experiment being adversarial to *our own* claim, not just the paper's.

## Tiers and cost (against ~$150 balance)

- **Transfer Tier 0 — zero/few-shot eval, ~$5, inference only.**
  Take the four Claim B checkpoints (and their post-Countdown-RL versions) and evaluate
  them *without any misinformation training* on the LIAR test set. Does priming alone move
  misinformation accuracy at all? This is nearly free and can already support or kill the
  claim before any RL spend. Also validates the loader + reward end-to-end.

- **Transfer Tier 1 — the paper's actual claim, ~$50.**
  RL each arm on LIAR (label-match reward, same PPO recipe), 1 seed, compare learning
  curves. Produces the dialogue-vs-monologue transfer gap once, *with* the length-matched
  control the paper lacks. Gate: only if Tier 0 shows priming perturbs misinfo behaviour,
  or if Claim B itself showed a real Countdown gap worth chasing.

- **Transfer Tier 2 — robustness, ~$150.**
  4 arms × 3 seeds. Turns "there is a transfer gap" into "the gap exceeds between-seed
  variance." Only if Tier 1 shows a gap.

## New code (all offline-testable, like `claimB_data.py`)

1. `rl/liar_data.py` — load LIAR, binary/6-way collapse, build the shared prompt
   (claim + "Assistant:" terminator, reused verbatim so priming is in-distribution), emit
   SFT-style eval parquet + verl RL parquet with `data_source="misinfo"`.
2. `rl/misinfo_reward.py` — extract the verdict from `<answer>`, map to a label, `1.0` iff
   it matches gold. Tests: gold TRUE/FALSE both directions, unparseable → 0, label
   synonyms ("false"/"fake"/"pants on fire") mapped correctly, no reward for emitting the
   *claim text* back (the analogue of the Countdown empty-skeleton exploit).
3. A one-line `data_source` registration so verl routes to the new scorer.

The reward-hacking surface is different and must be pre-empted: on a binary task a model can
farm 50% by always answering FALSE. So Tier 0 must **report the majority-class baseline and
per-class accuracy**, and the reward/eval must refuse a constant-answer degenerate the way
`attempt_reward` refused `<answer>1</answer>` on Countdown.

## One-paragraph summary for the proposal

*We test the SoT paper's boldest claim — that multi-agent priming transfers reasoning gains
to an unrelated domain (political-misinformation detection) — by reusing the exact faithful
RL harness with two swaps (LIAR dataset, label-match reward). Unlike the paper we run it
controls-first: a length-matched monologue arm separates "reasoning transfers" from "1.8×
more priming tokens," matched-output-length accuracy separates it from "more test-time
compute," and ≥3 seeds separate it from n=1 noise. We register in advance the prediction
that any transfer tracks verification/search behaviour rather than persona-diversity — the
same dissociation our steering and RL-trace analyses already found — making the experiment a
test of our own thesis as much as the paper's.*

---

## ⚠ SCOPE CORRECTION 2026-07-26 — this is a replication target, not novel work

A direct read of the paper (see `docs/paper_fidelity_audit.md`) shows **the paper already
runs this experiment.** Claim C6, main text:

> We further test whether conversational scaffolding transfers across domains. Models
> fine-tuned on multi-agent dialogues for the Countdown task are evaluated on a
> qualitatively different task: political misinformation detection, where models
> discriminate between true and fabricated headlines from **23,299 fact-checked claims from
> PolitiFact**. Despite never encountering this domain during fine-tuning, conversation-
> primed models achieve faster accuracy gains than baseline models (see Supplementary
> Methods: Cross-domain reasoning transfer and **Extended Data Fig. 9**).

This brief was scoped as a novel extension. It is not. Two consequences:

1. **Use their dataset and framing**, not a separately-designed LIAR/PolitiFact setup —
   23,299 PolitiFact claims, true vs fabricated headlines.
2. **The real gap is their comparison, not the task.** They compare conversation-primed
   **vs baseline**. They do not appear to run a **monologue-primed** arm on transfer — so
   their transfer result cannot distinguish "conversational structure transfers" from "any
   priming transfers." Our three-arm design (plus the length-matched monologue) closes
   exactly that hole. That is the contribution here; the task itself is theirs.
