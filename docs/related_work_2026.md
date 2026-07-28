# Related work: the debate literature has converged on our finding

*Written 2026-07-25, while the Llama Claim B arms train. Companion to
`replication_and_tools.md` (what the SoT paper argues) and `results/steering/FINDINGS.md`
(what we measured).*

## Why this note exists

Our results read as a set of isolated nulls — the steering effect doesn't generalize,
the induced society is redundant, the dialogue priming advantage is transient. Framed
that way they are easy to dismiss as a failed replication.

They are not isolated. A parallel literature on **multi-agent debate (MAD)** has, in the
same six months, produced the *theory* that predicts exactly what we measured — and
does so without any contact with the mechanistic-interpretability line the SoT paper
belongs to. This note records that convergence, because it changes the framing of our
contribution from "we could not replicate a paper" to "we supply the missing empirical
bridge between mechanistic claims about internal societies and the formal limits on what
societies can do."

**Provenance discipline.** This project has been bitten by paraphrase-drift before (the
stitched-quote incident). So each entry below marks whether the claim was read directly
from the source, or is relayed at second hand and still needs first-hand verification.

**What SoT is, and is not.** SoT is **not a multi-agent-debate paper**, and nothing below
should be read as placing it in that literature. MAD studies *N model instances with
separate contexts exchanging messages over rounds*. SoT studies *one model, one context
window, one autoregressive stream*, where the "voices" are stylistic segments of a single
generation; its instruments are SAE steering, an SEM mediation model, and an LLM judge.
The paper's own abstract hedges accordingly -- "multi-agent-**like** interactions", "implicit
simulation". The two literatures touch at exactly one point: SoT's SFT priming corpus
*is* literally multi-agent dialogue (a teacher prompted to simulate 2-4 personas arguing to
a `<group_solution>`). So the training signal is multi-agent even though the phenomenon
studied is not.

This distinction is load-bearing for the argument in S1. The MAD theorem is about
*information asymmetry between agents*. Mapping it onto SoT is an **analogy** unless the
theorem's assumptions are shown to hold for segments within one trace -- see S5.1.

---

## 1. The martingale result — the theory behind our redundancy finding

**[Diverse Evidence, Better Forecasts: Multi-Agent Deliberation Under Information
Asymmetry](https://arxiv.org/abs/2607.01661)** — CMU, arXiv 2607.01661v1, 2 July 2026.
*(read directly)*

It states, citing **Choi et al. 2026b**:

> standard multi-agent debate under homogeneous input behaves as a **martingale** whose
> expected accuracy does not improve over rounds

and the mechanism:

> when all agents share identical information, deliberation fails to improve collective
> accuracy: they reach the same conclusions independently, and iterative exchange merely
> reinforces the shared prior

**Proposition 3.1** makes it quantitative: with identical evidence, inter-agent error
correlation is **1.0**; partitioning evidence into public/private subsets drops it to
~**0.5** at a 50/50 split. Empirics: *PolyGym*, 375 binary forecasting questions from
Polymarket; their InfoDelphi method reaches Brier 0.178 / 77.9% accuracy. Notably,
**information asymmetry alone, without any deliberation, already reaches 74.1%** — i.e.
most of the benefit is the diversity, not the conversation.

> ⚠ The martingale theorem itself is **cited**, not proved, in this paper. We have read it
> second hand. Before relying on it in a submission, read Choi et al. 2026b directly and
> confirm the exact conditions (belief model, independence assumptions, what "identical
> inputs" means formally).

### Why this matters to us

Our judge-free HSE analysis (`analysis/hse.py`, FINDINGS §7) found that steering the
SoT "conversational surprise" feature produces **more voices but LOWER normalised
diversity** — segments become more alike, not more different — while accuracy falls. We
called that a *redundant society*.

If the theorem transfers, it is the formal statement of why a redundant society **cannot**
help: zero information asymmetry ⇒ error correlation 1.0 ⇒ expected accuracy flat under
exchange. And SoT's internal society is the *maximal-correlation* case by construction --
one set of weights, one context, so the segments cannot hold different evidence.

> ⚠ **This is an analogy, not a theorem application.** Their result is information-theoretic
> and *between models*; ours is mechanistic and *within* one model. Segments of a single
> autoregressive trace are not agents: they are not independently sampled, they condition
> on each other by construction, and "exchange" is just continued decoding. Whether the
> martingale conditions hold for that object is exactly what S5.1 and S5.3 are for. Until
> then: suggestive convergence, not derivation.

This is the strongest available answer to the natural defence of the SoT paper — "you
induced surface markers, not real diversity." The reply is: correct, and there is a
theorem saying surface markers without genuine differentiation is precisely the case
where a society does nothing.

---

## 2. Diminishing returns under RL — the same curve shape as our Claim B

**[Prepare Reasoning Language Models for Multi-Agent Debate with Self-Debate
RL](https://arxiv.org/abs/2601.22297)** — arXiv 2601.22297v1, January 2026. *(read directly)*

SDRL trains one model to be both a solo reasoner and a debate participant, using
verifiable rewards — methodologically the closest sibling to our RL work, and on
**Qwen2.5-3B**, the *same base model* as SoT's Claim B and ours. Their theory:

- debate benefits *"improve in early rounds … but peak quickly"* as answer correlation rises
- **Lemma 4.5**: improvement accumulates only **logarithmically** across rounds —
  diminishing per-round gains
- some outright degradation in baselines (DAPO on MATH500 with Qwen3) that SDRL mitigates
  but does not eliminate

### Why this matters to us

This is the same curve shape as our Claim B result, arrived at independently:

| | their finding | our Qwen Claim B |
|---|---|---|
| early | debate helps, gains largest in first rounds | dialogue leads (+4 to +10 at step 25–50) |
| later | gains accumulate only logarithmically, peak quickly | monologue catches up ~step 110, **ties at 250** |

An "early advantage that does not persist" is now reported by two independent groups
using different methods. That is much harder to wave away as a replication artifact.

**They do not cite the SoT paper at all.** The mechanistic-interpretability line and the
MAD line are running in parallel without contact — which is the gap our work sits in.

---

## 3. Invisible reasoning — the sharpest threat to the paper's inference

**[Not All LLM Reasoning is Visible in the Chain-of-Thought](https://arxiv.org/abs/2607.22925)**
— Baherwani, Goldstein & Panda, arXiv 2607.22925, 24 July 2026. *(abstract read directly;
⚠ full text NOT yet read — verify before citing specifics)*

> We demonstrate a concrete failure mode where frontier models exhibit invisible reasoning
> by leveraging **semantically irrelevant filler tokens** to improve performance on
> synthetic reasoning tasks. We evaluate 13 frontier language models across three tasks
> and find that many models benefit significantly from filler tokens, with accuracy
> improvements of **up to 13 percentage points**.

And, critically for us:

> **neither RL nor supervised fine-tuning produces a filler token benefit that persists at
> test time**

### Why this is the sharpest challenge to SoT

The SoT paper's inference runs: reasoning traces *look* dialogic → dialogic structure
*causes* the reasoning gain. Every instrument it uses reads the **semantic content** of the
trace: an LLM judge inferring personas, an SAE feature for "conversational surprise", an SEM
over conversational behaviours.

This result says content is not reliably where the computation lives. If **semantically
irrelevant** tokens buy up to 13 points, then observing structure in a trace and
correlating it with accuracy cannot establish that the structure is doing the work. The
trace can be a place computation *happens near*, not a description of what it *is*.

### It promotes our length confound from caveat to candidate mechanism

We have measured, on our own reconstructed corpus, that dialogue traces run **1.75×** longer
than monologue traces (252 vs 144 median words). We have been reporting that as a confound
to control. On this result it is better described as a **rival explanation with independent
empirical support**: more tokens can be causal while carrying no semantic load.

That reframes the paper's own asymmetry. It length-matches the dialogue condition for
Llama-3.2-3B ("concatenated into a single block ... to ensure comparable sequence lengths")
and **does not** for Qwen-2.5-3B. Under a filler-token account, the un-matched Qwen arm is
exactly where a spurious dialogue advantage would appear.

### And it predicts the decay we measured

Our faithful out-of-domain Claim B run shows the dialogue arm leading baseline by +0.049 to
+0.081 through step 100, then decaying to **−0.008 by step 250**. Their finding that a
filler benefit does **not persist** through RL or SFT is the same shape: a token-quantity
effect that training washes out. Two independent setups, one prediction.

> ⚠ This is a *convergence*, not a derivation. Their filler tokens are deliberately
> meaningless; our dialogue tokens carry real content. The claim we can support is that a
> length/quantity account is live and unexcluded — not that it is established.

### The experiment it hands us

A **length-matched filler control**: prime a third variant on monologue traces padded to
dialogue length with semantically empty tokens. Three outcomes, all informative:

| filler arm behaves like | conclusion |
|---|---|
| dialogue (early lead) | the early advantage is **token quantity**, not dialogic structure |
| monologue (no lead) | the advantage is **content-bearing** — SoT's mechanism survives |
| neither | something else; the design needs rethinking |

This is cheap — it reuses the existing corpus and needs no new teacher generation — and it
is the single most decisive arm we could add. It also fills the gap the paper's own Llama
control gestures at but does not close: their concatenation equalises length *while keeping
the dialogic content*, which is the complement of this test, not a substitute for it.

## 4. Also tracking (not yet read first-hand)

> ⚠ Everything in this section is from search snippets only. Read before citing.

- **[Demystifying Multi-Agent Debate: The Role of Confidence and
  Diversity](https://arxiv.org/abs/2601.19921)** (arXiv 2601.19921) — directly on the
  diversity question; snippets indicate confidence-modulated debate *breaks* the
  martingale symmetry and strictly improves correctness in expectation. If so, this is
  the sharpest statement of *what would have to be true* for a society to help — a
  natural predictor to test against our HSE measurements.
- **[Multi-Agent Debate with Memory Masking](https://iclr.cc/virtual/2026/poster/10010659)**
  — ICLR 2026 poster.
- **Choi et al. 2026b** — the martingale source. Highest priority to locate and read.
- Huot, Kaisers & Lapata, **arXiv 2607.09197** — the Hierarchic Social Entropy paper our
  `analysis/hse.py` already implements ("accuracy and meaningfulness can sharply
  diverge"). Same convergence, and we are already using their instrument.

---

## 5. How to frame our contribution

Three literatures, and nobody is joining them:

1. **Mechanistic (SoT)** — claims internal "societies of thought" explain reasoning gains.
   Measures diversity with an LLM judge; reports single runs without error bars.
2. **Theoretical (MAD)** — proves societies *without* information asymmetry cannot improve
   expected accuracy, and that gains are logarithmic and early-peaking even when they exist.
3. **Ours** — the only work measuring whether the internal society claim (1) identifies
   actually has the differentiation theory (2) requires. It does not.

The one-paragraph version:

> The claim that reasoning models improve by generating internal "societies of thought"
> requires those societies to be genuinely differentiated — a requirement made precise by
> recent MAD theory, where debate among agents with identical information is a martingale
> whose expected accuracy cannot improve. We test that requirement directly. Using a
> judge-free diversity instrument (Balch's Hierarchic Social Entropy) we find the induced
> society is *redundant*: steering produces more voices but lower normalised diversity,
> and accuracy falls rather than rises. In genuine PPO training we find no dialogic
> society emerges at all — conversational markers *decline* 20% while systematic search
> intensifies. And in a faithful three-arm replication of the paper's own scaffolding
> experiment, the dialogue-priming advantage is transient: monologue priming catches up
> and ties by step 250, matching independent reports that debate gains peak early and
> accumulate only logarithmically.

## 6. Open follow-ups this suggests

1. **Read Choi et al. 2026b** and state the martingale conditions precisely. If its
   assumptions are met by the within-model case, the connection is rigorous rather than
   analogical — that is the difference between a nice framing and a real result.
2. **Test the confidence-modulation predictor** (2601.19921): if confidence-modulated
   exchange breaks the martingale, does the SoT feature carry any confidence signal? We
   have the SAE machinery to check.
3. **Measure error correlation directly** between segments of a single trace, the
   quantity Proposition 3.1 is about — we currently measure cosine distance. Error
   correlation is the theoretically-licensed measure and we can compute it from the
   graded rollouts we already have.
4. **Cite the parallel-literature gap explicitly** in the write-up: SDRL uses the same
   base model as SoT and does not cite it.
