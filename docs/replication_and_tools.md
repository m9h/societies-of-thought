# Societies of Thought, Measured

**What a faithful replication, the controls the paper skipped, and two new instruments
reveal about "Reasoning Models Generate Societies of Thought"** (Kim, Lai, Scherrer,
Agüera y Arcas & Evans, [arXiv:2601.10825](https://arxiv.org/abs/2601.10825)).

Everything below is reproducible from two public repositories — `github.com/m9h/
societies-of-thought` (the steering replication, controls, HSE, and RL harness) and
`github.com/m9h/jacobian-lens` (the J-space analysis). No frontier-model access is
required, by design; the paper's model is the open DeepSeek-R1-Distill-Llama-8B.

---

## 1. The paper's argument, decomposed

The paper makes one descriptive observation and builds five load-bearing claims on it. It
is worth separating them, because they do not stand or fall together, and our results land
differently on each.

- **Observation (D).** RL-trained reasoning models produce traces that *read* as multiple
  interacting voices — self-questioning, disagreement, reconciliation. Not in dispute.
- **C1 — Causal mechanism.** Steering a single "conversational surprise" SAE feature
  (30939, layer 15) *doubles* Countdown accuracy (27.1% → 54.8%). The dialogic structure
  therefore *causes* better reasoning.
- **C2 — Mediation.** A structural-equation model shows conversational behaviour
  *mediates* the accuracy gain.
- **C3 — Diversity.** The internal voices constitute a genuinely *diverse* society — an
  LLM judge infers distinct personas and measures their spread.
- **C4 — RL emergence.** Under PPO rewarding *only* answer correctness, conversational
  behaviour arises anyway; by step 120 the trace contains two personas calling themselves
  "we."
- **C5 — RL scaffolding (their main event).** Priming a base model on multi-agent
  *dialogue* traces, versus *monologue* traces over identical problems with identical
  answers, makes the dialogue-primed model learn faster under RL.

C1 is the keystone — it is the only *causal* claim, and it rests entirely on one task.

## 2. We reproduced the paper faithfully — first, and on the record

A critique is worthless if it cannot first reproduce the thing it criticises. We did.

| | ours | paper |
|---|---|---|
| Countdown baseline (unsteered) | 24.0% | 27.1% |
| feature 30939 sparsity | 0.00017 | 0.00016 |
| SAE reconstruction (layer 15, resid_post) | 52.5% EV | — |

The feature is the right one: it fires only on surprise markers (` Oh`), reads exactly
0.000 on non-conversational text, and steering it **does** raise Countdown accuracy. The
effect is real. We reproduced the paper's headline direction before we touched a single
control — and we found six silent traps in doing so (hook point, JumpReLU threshold, BOS
attention-sink, dataset-wise scaling, a Neuronpedia metadata error that steers the wrong
layer, and an activation-scale mismatch), each documented and tested so the next person
does not pay for them again.

**This is the part that earns the rest.** We are not strawmanning a result we could not
obtain.

## 3. The experiments the paper did not run — which we ran, and released

The paper's causal claim (C1) is measured on Countdown alone. It steers nowhere else; it
only *observes* traces on GPQA and MATH. Its controls are random features, not matched
ones. And it reports a single dose. Four experiments close those gaps. All are in the
repository, with tests.

**(a) The dose-response curve the single dose hides.** Steering is not monotone. It helps
up to α≈1.0 (+10 points, CI excludes zero) and then the model *collapses* into degenerate
babble — literally `"...cyclo-oh, no, wait, no, wait..."`, 3.5% accuracy, 96% of traces
unparseable. **An inverted U.** A single reported dose cannot see this, and it is the whole
shape of the phenomenon.

**(b) The missing causal cell — steering on the paper's own benchmarks.** We ran C1 where
the paper declined to, at the dose optimal on Countdown, with controls **matched on
sparsity and max-activation** so the comparison isolates *meaning* rather than
perturbation size:

| benchmark | the paper's feature, same dose | direction |
|---|---|---|
| Countdown | **+10.0** | helps |
| MATH-Hard | **−22.0** (CI excludes zero) | **reverses** |
| GPQA-Diamond | negative (secondary; truncation-limited) | reverses |

**The same feature, at the same dose, that gives +10 on Countdown costs −22 on MATH-Hard.**

**(c) The mediation test the SEM cannot do.** C2 says dialogue mediates accuracy. Steering
tests it directly, and the two come apart:

| MATH-Hard, steered α=1.0 | vs baseline |
|---|---|
| self-interruption (`wait`/`hmm`/`oh`) | ↑ 15.6 → 25.1 |
| contradiction (`but`/`however`) | ↑ 15.4 → 21.5 |
| **accuracy** | **↓ 62.0% → 40.0%** |

The intervention *works* — the traces become measurably more dialogic, exactly as the
paper predicts — and the model gets *dumber*. You can induce the society of thought and
lose the reasoning. Mediation requires the two to move together; they move apart.

**(d) The diversity measure the LLM-judge cannot give.** More on this in §5 — it needs a
new instrument.

## 4. The story is not all positive — including for us

Two honesties belong here, because a replication that only ever confirms its own thesis is
as suspect as the paper it audits.

**The descriptive observation (D) survives, and we say so.** We do not dispute that R1's
traces are dialogic. Our account (§Interpretation in FINDINGS) is that RL on verifiable
correctness selects for *verification and backtracking* — a search property — and that
English renders self-audit in dialogic form: a mind checking itself *sounds like* two
people arguing. The conversational structure is a **stylistic signature of self-correction,
not its cause.** The paper measured the shadow and steered it. That is a smaller, and we
think truer, claim than "societies of thought" — but it concedes the paper saw something
real.

**We could not adjudicate the RL claims (C4, C5) — and we report that as a limitation, not
a refutation.** C5 is the paper's main event. To test it we built the full harness: the
paired dialogue/monologue SFT data (verified to solve identical problems with identical
answers — the invariant the whole design rests on), GRPO (TRL removed PPOTrainer, which the
paper used), LoRA (full fine-tuning OOMs an 80GB A100 at 3B). Under that forced
substitution the setup **does not reproduce the paper's learning signal at all**: a $5
probe showed the model format-hacks — reward rises by getting shorter and cleaner while
accuracy stays flat near baseline. So we cannot say C5 is false; we can only say it does
not survive the GRPO+LoRA regime the post-PPO tooling forces on an independent replicator,
and we stopped before spending $88 comparing three arms that do not learn. That is a
finding about the reproducibility of the RL claims, and an honest boundary of ours.

So: **C1 replicated then reversed. C2 fails on its own terms. C3 refuted (below). C4/C5
untested by us, with the reason documented. D conceded.** Not a hit piece — a map of which
claims carry weight.

## 5. The tools the paper needed — and the better metaphors they generate

The paper's errors are not carelessness; they are the errors you make when your only
instrument is *steer-one-feature-and-read-the-accuracy*. Two instruments the paper did not
use turn its metaphors into measurements — and the measurements suggest better metaphors.

### Instrument 1 — Hierarchic Social Entropy (a judge-free diversity measure)

C3's "perspective diversity" is an LLM that first *infers* the personas and then *scores
their spread* — a judge grading constructs it invented. Balch's Hierarchic Social Entropy
needs no judge: segment a trace at the paper's own perspective-shift cues, embed the
segments, integrate the diversity from the clustering dendrogram. We ran it on 1,200
steered Countdown traces:

| α | segments/trace | normalised diversity | accuracy |
|---|---|---|---|
| 0 | 21.4 | **0.236** | 15.2% |
| 1.0 | 44.6 | 0.190 | 31.5% |
| 1.693 | 54.7 | **0.190** | 3.6% |

**Steering makes the society bigger and proportionally *more redundant*.** Segment count
more than doubles; normalised diversity *falls 20%*. The paper's own metaphor — a diverse
society of voices — is measurably backwards under its own intervention.

> **Better metaphor.** Not a council of distinct voices deliberating toward an answer.
> Steering harder produces **a louder crowd saying more of the same thing** — an echo
> chamber, not a debate. "Society of thought" implies differentiation the diversity
> measure shows evaporating exactly as you push the knob.

### Instrument 2 — the Jacobian lens (locating the feature in the workspace)

Where does the conversational feature *live*? Anthropic's Jacobian lens defines a model's
"J-space" — a global-workspace-like subspace of what a layer is poised to report. We
pre-registered a prediction (the feature is *off*-workspace noise) and it was **wrong**:
the conversational feature is *more* workspace-resident than the average SAE feature (0.803
vs 0.728). The knob is genuinely wired into the machine.

That reframes the whole steering result. The feature is not off-workspace dialogic garnish;
it is a real workspace direction, and steering **over-drives** it. Combined with the HSE
collapse and the inverted-U, the mechanism is not "add voices, get reasoning" — it is
"saturate one real channel until the workspace's diversity collapses into redundancy and
then noise."

> **Better metaphor.** Cranking a real knob until the signal clips — not adding
> instruments to an orchestra. The feature is load-bearing, which is *why* over-driving it
> degrades multi-step reasoning rather than enriching it. The paper found a real knob and
> mistook "it moves the output" for "more of it is better."

### And a cross-project corroboration the paper could not have had

On AI2's fully-open OLMo-3 ladder, the same Jacobian-lens machinery shows that
post-training installs a genuine *viewpoint* in J-space that is **decoupled from
capability** — a 31% representational shift with flat MMLU. Two unrelated instruments
(SAE steering here, lens geometry there) reach one structural conclusion: **the "viewpoint"
layer of a model is separable from its competence.** The society-of-thought paper's core
error is exactly this conflation — reading a real representational/stylistic direction as a
lever on reasoning capability. It is not one; the capability and the style come apart, and
now there are two independent ways to show it.

## 6. What this is, in one paragraph

We reproduced the paper's headline, then ran the four experiments its instrument could not
run and the two instruments it did not have. The causal claim is a Countdown artifact; the
mediation claim fails when you can move dialogue and accuracy independently; the diversity
claim inverts under its own intervention; and the "society of thought" is better understood
as the dialogic exhaust of a search process — a real workspace channel that breaks the
machine when over-driven, not a chorus that improves it. The RL claims we could not test
under the tooling available to an outsider, and we say so. Every number, every control, and
every retraction is in a public repository, runnable without privileged access — which is,
in the end, the only thing that lets a claim like "societies of thought" be believed or
disbelieved at all.
