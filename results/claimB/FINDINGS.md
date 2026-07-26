# Claim B / C5 — result

*Completed 2026-07-26. Data: `curves.json`. Code: `rl/claimB_data.py`, `rl/sft_prime.py`,
`scripts/claimB_pod.sh`. Raw verl logs on the Spark under `~/claimB_logs/`.*

## What was tested

The SoT paper's central causal experiment: prime a base model with SFT on **dialogue**
traces vs **monologue** traces, then run **identical** RL and compare. If the dialogic
society is what drives reasoning gains, dialogue priming should train better.

Our version adds a third arm the paper does not run — **baseline**, no priming at all —
and enforces two controls the paper's setup does not:

1. **One prompt for all arms**, ending at `"Assistant:"`. TinyZero's stock prompt ends
   `"...\n<think>"`, which pre-opens the monologue container and would push a dialogue
   model's `<persona1>` opening out-of-distribution. That alone could manufacture a
   dialogue-vs-monologue gap.
2. **Answer-container normalisation** — dialogue traces state answers in
   `<group_consensus>`, the PPO scorer reads `<answer>`. Unnormalised, the dialogue arm
   would be scored as wrong for being right in the wrong tag.

Every arm then runs the *same* PPO config, recovered verbatim from Tier-0, capped at 250
steps for a matched-step comparison. Only the starting weights differ.

## Result

**Final val accuracy at step 250:**

| base model | baseline | dialogue | monologue | dialogue − monologue |
|---|---|---|---|---|
| Qwen2.5-3B | 0.597 | 0.621 | 0.618 | **+0.003** |
| Llama-3.2-3B | 0.196 | 0.568 | 0.487 | **+0.081** |

**The two models give different answers.** This is the headline, and it is not the answer
we expected from the Qwen run alone.

### Qwen2.5-3B — the dialogue advantage is transient

Dialogue leads by **+0.117** over baseline at step 25 and by **+0.057** over monologue.
By step 250 both gaps have collapsed: **+0.024** over baseline, **+0.003** over monologue.
Monologue crosses dialogue around step 110–175 and the three curves are indistinguishable
at the end.

Priming buys *speed of convergence*, not a better endpoint — and dialogue priming buys
essentially no more speed than monologue priming does. Note also that dialogue priming
produces **1.8× longer** responses (220 vs 123 median words), so what advantage exists
early is confounded with length and is not attributed to dialogue structure.

### Llama-3.2-3B — priming matters enormously, and the gap does *not* close

The baseline arm **fails to learn**: it plateaus at ~0.19 from step 75 onward and never
moves again. Both primed arms take off and reach 0.49–0.57. Priming is worth **+0.29 to
+0.37 absolute** here — far more than on Qwen, where the baseline gets there on its own.

Between the primed arms, dialogue holds a **+0.06 to +0.08** lead that persists to 250.
Caveat: the monologue curve is noisy at the tail (0.505 → 0.420 → 0.487), so the endpoint
gap sits close to that arm's own oscillation. At n=1 this is suggestive, not established.

## What this means for the paper's claim

Two things, pulling in opposite directions — we report both.

**Against the paper's interpretation.** On the paper's *own base model*, the effect it
reports does not survive to matched-step 250 once the prompt is shared and answers are
normalised. The dialogue-vs-monologue gap is +0.003. And a *no-priming* baseline gets to
the same place, which means the comparison the paper actually runs — dialogue vs monologue
— is not the comparison that matters. The interesting variable is "any priming vs none,"
and even that washes out on Qwen.

**Complicating our own conclusion.** On Llama the dialogue arm is genuinely and durably
ahead of monologue. We cannot say "dialogue priming does nothing" — we can say it does
nothing *on the model the paper used*, and something *on a model the paper did not test*.
Whether that something is dialogue structure or the 1.8× length confound is exactly what
the length-matched-monologue arm (designed in `briefs/transfer_misinformation_scope.md`,
not yet run) is for.

### The reframing of the paper's numbers

⚠ **The paper-side figures below are from notes taken in an earlier session, not re-read
from the PDF for this write-up. Re-verify against the source before citing them anywhere.**

As recorded, the paper's monologue arm plateaus around **~18%** while its dialogue arm
reaches ~38%. Our Llama **baseline** plateaus at **0.196** — i.e. the paper's monologue
arm performs like a model that was never usefully primed at all.

If that holds up, their contrast is not "dialogue priming beats monologue priming." It is
"priming that worked beats priming that did nothing," and the causal weight they place on
*dialogue* is carried by their monologue arm being broken rather than by their dialogue
arm being special. Our monologue arms reach 0.487–0.618 on identical data, which is what a
monologue arm looks like when the priming lands.

## Limits — read before using any of this

- **n=1 per arm per model.** No seeds, no error bars. The Qwen +0.003 and the Llama +0.081
  both need ≥3 seeds before either is load-bearing. This is the single biggest gap and it
  costs ~$50–60 to close on Qwen.
- **Our teacher is stronger than theirs** (72B vs Qwen-2.5-32B-Instruct). This is applied
  *identically to both primed arms*, so it is a controlled variable rather than a
  confound between arms — but it does mean our absolute accuracies are not comparable to
  the paper's, only our between-arm differences are. It plausibly explains why our arms
  land near 0.6 where theirs land at 0.18–0.38.
- **Length confound unresolved.** Dialogue priming = 1.8× tokens. Not yet separated from
  dialogue structure.
- **250 steps is the paper's own horizon**, not a truncation of it. We are not reporting a
  gap that closes past where they stopped looking; it closes inside their window.

## Connection to the theory

`docs/related_work_2026.md` §2: SDRL (arXiv 2601.22297), on this same Qwen2.5-3B, finds
debate gains "peak quickly" and accumulate only logarithmically. Our Qwen curve is that
shape — large early advantage, converged by the end. Two independent methods, same result.
