# Claim B / C5 — result

*Completed 2026-07-26. Data: `curves.json`. Code: `rl/claimB_data.py`, `rl/sft_prime.py`,
`scripts/claimB_pod.sh`. Raw verl logs on the Spark under `~/claimB_logs/`.*

## What was tested

The SoT paper's central causal experiment: prime a base model with SFT on **dialogue**
traces vs **monologue** traces, then run **identical** RL and compare. If the dialogic
society is what drives reasoning gains, dialogue priming should train better.

> ⚠ **Read `docs/paper_fidelity_audit.md` before using anything below.** A direct read of
> the paper (2026-07-26) established that our priming data is built from **Countdown**
> while theirs is built from **out-of-domain** reasoning benchmarks. That makes this a
> related but *different* experiment, and it — not seed — is the main reason our numbers
> differ from theirs. Two claims in an earlier version of this file were wrong and are
> struck through below.

~~Our version adds a third arm the paper does not run — baseline, no priming at all~~ —
**wrong: the paper runs exactly three conditions, "(1) Baseline (RL only, no priming), (2)
Conversation fine-tuning …, (3) Monologue fine-tuning." Our baseline replicates theirs.**
We do enforce two controls the paper's setup does not:

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

**Against the paper's *stated* interpretation — but with its own figure.** On the paper's
own base model the dialogue-vs-monologue gap is +0.003 at step 250. That agrees with the
paper's Extended Data Fig. 8 caption — *"though both eventually converge"* — and
contradicts its main text (*"reach higher asymptotic accuracy"*) and its abstract
(*"substantially accelerates"*). The paper contradicts itself here, and our result picks a
side. That is the correct framing; an earlier version of this file presented convergence
as a finding against the paper, which undersells it.

Caveat that cuts hard: with **in-domain** priming (§audit 4.1) both arms are handed the
task knowledge, so there is little left for format to differentiate. Our convergence may
be a consequence of our deviation rather than a test of their claim.

**Complicating our own conclusion.** On Llama the dialogue arm is genuinely and durably
ahead of monologue. We cannot say "dialogue priming does nothing" — we can say it does
nothing *on the model the paper used*, and something *on a model the paper did not test*.
Whether that something is dialogue structure or the 1.8× length confound is exactly what
the length-matched-monologue arm (designed in `briefs/transfer_misinformation_scope.md`,
not yet run) is for.

### The reframing of the paper's numbers — now verified against the source

Verified 2026-07-26 by direct read. The paper states: *"By step 150, conversation-fine-tuned
Llama models achieve 40% accuracy while monologue-fine-tuned models plateau around 18%."*
Our Llama **baseline** sits at **0.176 at step 150** — where their *monologue* arm plateaus.

The audit supplies the mechanism. Their monologue priming is **out-of-domain** general
chain-of-thought over BBH/GPQA/MATH-Hard problems, which teaches Countdown very little —
so it behaves much like no priming at all. On that reading their contrast is closer to
"priming that transferred vs priming that didn't" than to "dialogue beats monologue."

Limit on this claim: the paper reports no Llama *baseline* number, only that the base model
"learns more slowly." So we can say their monologue arm lands where *our* baseline lands;
we cannot say it lands where *theirs* does.

## Limits — read before using any of this

- **Priming domain differs from the paper's — this is the big one.** They prime on 8,262
  out-of-domain reasoning problems; we prime on Countdown. Different experiment. See
  `docs/paper_fidelity_audit.md` §4.1. Fixing this, not seeds, is the top priority.
- **Our Llama dialogue arm is not length-matched, theirs is.** The paper explicitly
  concatenates personas into one `<think>` block for Llama "to ensure comparable sequence
  lengths." We didn't, so our +0.081 carries the full 1.8× length confound and is **not
  comparable to their Llama result.** Weakest number in the set.
- **`rollout.n=1` vs their 4**, train batch 256 vs 128, eval every 25 steps vs 10, SFT 3
  epochs vs 5, SFT batch 32 vs 64, context 1536 vs 2048. Six mechanical deviations, all
  cheap to fix.
- **n=1 per arm per model.** No seeds, no error bars. Real, but *item 6* — seeds bound how
  precisely we can state our own numbers; they do not close the gap to the paper.
- **Our teacher is stronger than theirs** (72B vs Qwen-2.5-32B-IT — OpenRouter does not
  host the 32B). Applied *identically to both primed arms*, so controlled between arms, but
  it inflates our absolutes relative to theirs.
- **250 steps is the paper's own horizon**, not a truncation of it. We are not reporting a
  gap that closes past where they stopped looking; it closes inside their window.

## Connection to the theory

`docs/related_work_2026.md` §2: SDRL (arXiv 2601.22297), on this same Qwen2.5-3B, finds
debate gains "peak quickly" and accumulate only logarithmically. Our Qwen curve is that
shape — large early advantage, converged by the end. Two independent methods, same result.
