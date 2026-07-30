# Related work: the debate literature has converged on our finding

*Written 2026-07-25, while the Llama Claim B arms train. Companion to
`replication_and_tools.md` (what the SoT paper argues) and `results/steering/FINDINGS.md`
(what we measured).*

## Why this note exists

Our results read as a set of isolated nulls — the steering effect doesn't generalize,
the dialogue priming advantage is transient, diversity does not predict correctness. Framed
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

> ⚠ **CORRECTION 2026-07-30.** This section was built on our finding that the induced
> society is *redundant*. **That finding is retracted** — it was a filtering artifact, and
> corrected analysis shows steering makes the society genuinely MORE diverse
> (`results/steering/RECHECK_length_and_filtering.md`). The martingale connection was
> already flagged below as an analogy rather than a derivation; it has now also lost its
> main empirical anchor. Measured segment diversity is not inter-agent error correlation,
> and we have never measured the latter. Treat everything in §1 as motivation only until
> that measurement exists.

## 1. The martingale result — ⚠ its empirical anchor has been withdrawn

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
called that a *redundant society* — **now retracted, see the correction above**.

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

## 3. Invisible reasoning — the sharpest challenge to the paper's inference

**[Not All LLM Reasoning is Visible in the Chain-of-Thought](https://arxiv.org/abs/2607.22925)**
— Baherwani (NYU), Goldstein (UMD) & Panda (TogetherAI), arXiv 2607.22925, 24 July 2026.
*(FULL TEXT read directly — this section was previously written from the abstract and
overstated one thing; corrected below.)*

Frontier models gain accuracy from **filler tokens**: fixed, semantically irrelevant
sequences ("1 2 3 4 …", "cat dog bear …") that are identical for every problem and so
carry no information about any particular question. Claude Opus 4.5: **+11.2** on
multi-step arithmetic and **+10.0** on 4-digit multiplication. Gemini 3 Flash: **+10.7**
on arithmetic. Opus 4.6 shows **+30.0**, though its API forbids assistant prefilling so
the authors flag possible selection effects.

**Correction to what I wrote from the abstract.** I said the result shows trace *content*
does not matter. That is too strong and misses their point. Their criteria 2 and 3 are
that performance *does* depend on which filler tokens are used, and that preferences
*differ across models*. The claim is sharper than "content is irrelevant": the tokens are
**semantically** empty yet **representationally** consequential. What matters is not what
the tokens mean but what they do to the residual stream of a particular model. Their
mechanistic section supports this — activation patching recovers >90% of the gap between
a strong and a weak filler type when applied at layers 0–30, and linear probes on the
filler span decode task-relevant information from layer 15 onward.

### The passage that matters most for us

Section 6.1, on what RL over filler tokens is actually doing:

> RL training improves pass@8 but yields little gain in pass@1, suggesting that filler
> tokens primarily affect the **diversity of model outputs** rather than the accuracy of
> any single forward pass. … We speculate that filler tokens allow the model to
> **internally explore multiple candidate answers within a single forward pass, with the
> prefilled span acting as a workspace** over which different latent computations can be
> conditioned. … models may use filler tokens as an **internal search mechanism**.

Read that against the SoT thesis. SoT argues that reasoning improves because the model
simulates *a society of differentiated perspectives* that explores the solution space, and
it locates that society in the **dialogic content** of the trace. This paper reaches the
same functional description — a workspace supporting parallel exploration of candidate
answers — using tokens with **no dialogic content whatsoever**.

If the functional role SoT attributes to internal voices is reproducible with counting
sequences and animal names, then observing personas in a trace does not establish that the
personas are the mechanism. The society may be the *readable shadow* of a search process
that does not require any society to run. That is precisely the account we reached
independently from the steering and HSE results (`FINDINGS.md` §Interpretation), and this
is the first external evidence for it.

> ⚠ The authors mark the workspace reading as **speculative** and say they provide no
> direct evidence that distinct candidate answers are simultaneously represented. Cite it
> as a converging interpretation, never as an established mechanism.

### It also predicts our Claim B result, twice over

**First, the decay.** Our faithful out-of-domain run has dialogue leading baseline by +0.049
to +0.081 through step 100, decaying to **−0.008 by step 250**. They report that RL
"does not produce a filler token advantage that persists at test time."

**Second, and more specifically, the SFT failure.** SoT's Claim B *is* an SFT-then-RL
design. Appendix G:

> Across all configurations, SFT fails to transfer invisible reasoning. … The model learns
> to reproduce filler token sequences from training data, but any accuracy gain is also
> present without filler tokens. … The computation that makes filler tokens useful for
> Opus 4.5 occurs in latent space. **The tokens themselves carry no transferable signal, so
> imitating them provides no benefit.**

If the dialogue advantage is a latent-workspace effect rather than a content effect, then
imitating dialogue *tokens* by SFT should fail to install it durably — which is the shape
our curve has.

### The experiment, upgraded — use their ablation, not my filler arm

I previously proposed padding monologue traces with meaningless filler. Their **trace
ablation** is better because it is already validated in this paper: remove 
r
1
 or replace it
with random tokens and test whether downstream accuracy depends on its *content*. Their
result under a monitor penalty is the sharpest possible version —

> replacing 
r
1
 with a single random token fully restores accuracy … indicating the model
> relies only on the **presence of a token** in the 
r
1
 position as a cue rather than its content.

Applied to us: take the dialogue-primed checkpoint and, at evaluation, replace the persona
and conversation content with length-matched random tokens.

| result | conclusion |
|---|---|
| accuracy preserved | the dialogic content is inert; length/position is the mechanism |
| accuracy collapses | the content is load-bearing and SoT's mechanism survives |

This needs **no new training** — it runs on checkpoints we already have.

### An opening they leave explicitly

Their stated limitations include:

> we do not evaluate fillers that repeat the question or **natural CoT text used as
> filler**; both are natural extensions of our experiments.

Our dialogue traces *are* natural CoT text, and our corpus is length-matched by
construction against a monologue control. We are positioned to run the extension they
name, on data that already exists.

## 3b. Prime Intellect: "Better Initializations for RL" — independent support for *priming, not dialogue*

**INTELLECT-MATH** (Prime Intellect) — dataset card read directly at
`PrimeIntellect/NuminaMath-QwQ-CoT-5M`. Subtitle:

> *Frontier Mathematical Reasoning through **Better Initializations for Reinforcement Learning***

> We demonstrate that the quality of our SFT data can impact the performance and **training
> speed of the RL stage**: Due to its better synthetic SFT dataset that encourages the model
> to imitate the reasoning behavior of a strong teacher model, INTELLECT-MATH … matches its
> performance with **10x faster training**.

This is our Claim B conclusion arrived at independently and at larger scale. They attribute
the RL speed-up to **SFT-data quality and teacher strength** — not to any dialogic or
multi-agent structure, which they never invoke. Their framing of pre-RL SFT as an
*initialisation* problem is the same account we reached: priming changes how fast RL gets
going, not where it ends up.

Worth noting against SoT specifically: SoT holds teacher and problems fixed and varies only
trace *form*, concluding that form is causal. Prime Intellect varies teacher *quality* and
finds a large effect on RL speed. Two studies, one axis each; the one that moved a big lever
moved teacher quality, not conversational structure.

### The dataset is also our cheapest remaining experiment

`PrimeIntellect/NuminaMath-QwQ-CoT-5M` — **MIT**, **5,138,102** traces, 43.8 GB, fields
`problem_id / prompt / response / ground_truth / correct`.

Those are **QwQ** traces. QwQ-32B is one of the two models SoT builds its descriptive claim
(D) and diversity claim (C3) on, and SoT measures diversity with an LLM judge over **8,262**
traces. Our HSE instrument (`analysis/hse.py`) is judge-free and needs only trace text, so
this corpus supports the same measurement at **~600× the sample size on the paper's own
subject model**, on CPU.

The `correct` boolean is what makes it decisive. SoT's C2/C3 claim is that perspective
diversity *accounts for* the accuracy advantage. With correctness labels we can test the
within-model version directly:

> Among traces from a single model on a single task, is HSE diversity higher in the traces
> that reach the right answer?

If diversity does not separate correct from incorrect traces at n = 5M, the mediation claim
has no room left to hide in sample noise. If it does, the claim gains its strongest support
yet — and either way it is an answer, obtained judge-free, for the price of embeddings.

⚠ Caveat: NuminaMath is **math-only**, whereas SoT's pool spans BBH/GPQA/MATH/MMLU-Pro/MUSR.
This tests the claim within a domain, not across the paper's mix.

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
> society is genuinely more differentiated at every dose — normalised diversity rises
> monotonically with steering — and accuracy still collapses, so diversity and accuracy are
> decoupled rather than linked. In genuine PPO training we find no dialogic
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
