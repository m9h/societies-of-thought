# Why do researchers think particular layers matter?

This document exists because of a question we should have asked on day one: **why layer 15?**

The paper locates its entire mechanistic claim there. We rebuilt its harness around layer
15, ran a dose-response there, ran matched controls there, and argued with the results —
all without ever interrogating the choice. This is that interrogation.

---

## 1. The paper's stated reason is convention, not evidence

The paper's justification, in full, is that SAEs trained on middle layers "including Layer
15, are known to capture key behavioural and semantic features in models," citing prior
interpretability work. That is a *norm of the field*, not a finding about this mechanism in
this model.

And there is a tooling reason underneath it. The pure-SlimPajama SAE the paper uses —
`15-llamascope-slimpj-res-32k` — **exists only for layer 15.** All 32 layers are published
for the *mixed* SlimPajama+OpenR1 suite, but the one they chose is a layer-15 singleton.
So the claim is anchored where the artifact happened to be.

That is a thin foundation for "reasoning works by simulating a society of thought." But the
underlying norm is not arbitrary, and it is worth understanding properly, because it is
*mostly right* — just not automatically right here.

---

## 2. The canonical result: layers really do take on roles

### Vision — the textbook case

The clearest demonstration is **Zeiler & Fergus (2014)**, visualising what convolutional
layers respond to. The hierarchy is unmistakable and reproduces across architectures:

| depth | what the layer detects |
|---|---|
| early | oriented edges, Gabor-like filters, colour blobs |
| middle | textures, motifs, object *parts* (eyes, wheels) |
| late | whole objects, class-discriminative structure |

**Olah et al.'s Circuits** work (Distill, 2020) made this mechanistic rather than
suggestive — naming actual units in InceptionV1 (curve detectors, high–low frequency
detectors) and showing how they compose. Layers are not interchangeable; depth buys
abstraction.

### Language models — where "the middle" comes from

The intuition transfers, and has been made precise:

- **Induction heads** (Olsson et al., 2022). In-context learning is implemented by a
  specific two-layer circuit: a *previous-token head* in an early layer feeding an
  *induction head* in a later one. These heads form in an abrupt phase change during
  training. Specific layers, specific job.

- **The IOI circuit** (Wang et al., 2022) — the most complete circuit ever traced in a real
  LM (GPT-2 small, indirect-object identification). Duplicate-token heads *early*,
  S-inhibition heads *middle*, name-mover heads *late* (layers 9–10). A clean division of
  labour by depth.

- **The logit lens / tuned lens.** Decode the residual stream at each layer and you watch
  the prediction *form*: early layers are token/syntax-bound, the middle is where the most
  abstract, task-general representation lives, and late layers rotate that into
  output-specific, logit-shaped form.

That last point is the real reason interpretability reaches for the middle. **Early layers
are too close to the tokens; late layers are too close to the answer.** The middle is where
you find representations that are *about the content* rather than about the input encoding
or the output vocabulary — which is exactly where a "conversational surprise" concept would
have to live if it lived anywhere.

So the convention is defensible. It is just not a substitute for checking.

---

## 3. Layers that become *instrumental* — where the payoff is

The strongest form of "this layer matters" is when locating it lets you *intervene*.

### ROME / MEMIT — the gold standard

**Meng et al. (2022), "Locating and Editing Factual Associations in GPT."** They use
**causal tracing** — corrupt the input, restore individual hidden states, and see which
restoration recovers the correct answer — and find factual recall is mediated by a
**mid-layer MLP bottleneck** (roughly layers 3–8 in GPT-2 XL). They then *edit the weights
of exactly those layers* (a rank-one update) to change a fact, and it works: the model now
believes the Eiffel Tower is in Rome, consistently, across paraphrases.

This is the template the society-of-thought paper is implicitly following: **localise, then
intervene.** The crucial difference is that ROME *earned* its layer by causal tracing.
Layer 15 in our paper was inherited from convention and artifact availability.

### Steering vectors and the linear representation hypothesis

Activation addition (Turner et al.), and the whole SAE-steering literature this paper sits
in, typically operate mid-stack for the reason above. **Arditi et al. (2024)** found refusal
in chat models is mediated by a *single direction* — and ablating it removes refusal,
adding it induces it. Again: a specific place, a causal handle.

### Steering *training*, not just inference

- **Surgical fine-tuning** (Lee et al., 2023): which block you should tune depends on the
  kind of distribution shift — input-level shift favours early layers, label shift favours
  late. Tuning *the right layer only* can beat full fine-tuning.
- **LoRA placement**: where you insert adapters materially changes what the model learns.
- **HTSR / WeightWatcher** (Martin & Mahoney): the eigenvalue-tail exponent α of each
  layer's weight matrix predicts model quality *without test data*, and α ≈ 2 marks a
  well-trained layer — which turns "which layers are healthy" into a trainable objective
  (see `m9h/wwj`'s differentiable `alpha_loss`).

---

## 4. What our own weight analysis says about layer 15

`DeepSeek-R1-Distill-Llama-8B` is `Llama-3.1-8B` with reasoning distilled into it — a
matched pair. So we can ask directly: **where did reasoning get written?**

**Not layer 15 in particular. Not any layer in particular.**

```
mean relative weight change ||W_r1 - W_base|| / ||W_base||, by layer:
  L02  0.1976   <- most changed
  L15  0.1905   <- rank 21 of 32, BELOW the median (0.1917)
  spread across all 32 layers: ~4%
```

Distillation rewrote every layer almost identically. There is no depth-wise hotspot, and
the paper's layer sits slightly below average.

**The structure is in the projections instead, and it is dramatic:**

| projection | relative change | what it does |
|---|---|---|
| **`o_proj`** | **0.424** | writes attention's result *into* the residual stream |
| `up/down/gate_proj` | ~0.17 | MLP |
| `v_proj` | 0.160 | what information is carried |
| `q_proj` | 0.127 | where to attend |
| **`k_proj`** | **0.086** | where to attend |

Reasoning-distillation **barely changes where the model looks, and massively rewrites what
it writes back**. That is a clean, interpretable finding, and it is *depth-uniform*.

It also reframes the SAE result. The residual stream at layer 15 is precisely the channel
`o_proj` writes into and the SAE reads out of. So the conversational features exist and fire
*because* that channel was rewritten — but nothing privileges layer 15 as the seat of the
mechanism.

---

## 5. The consequence for this project

Our steering result (+10 on Countdown, −22 on MATH-Hard, negative difference-in-differences
against matched controls) is **all at layer 15**. That is exactly right as a refutation of
*the paper's claim*, which is about layer 15.

It does **not** license the broader claim "conversational features don't cause reasoning
anywhere." The mechanism could live at layer 22 and we would never have seen it.

**Hence the layer sweep** (`scripts/run_stages.sh layers`): the same experiment — matched
controls, one dose, MATH-Hard — across depth, using the mixed SAE suite that exists for all
32 layers. Layer 15 appears in *both* suites, so it doubles as a check that the effect
isn't an artifact of the SAE's training mixture.

That is the experiment that turns "the paper's claim fails" into "the claim fails *and we
looked everywhere it could have been true*."

---

## 6. A related dissociation, at the level above

Everything above is about locating a mechanism *inside* one model. Huot, Kaisers & Lapata,
*[When is Routing Meaningful?](https://arxiv.org/abs/2607.09197)* (2026), show the same
trap one level up, between models: a router can hit high accuracy while operating over a
**redundant society** — actors that are not actually differentiated. They conclude
"accuracy and meaningfulness can sharply diverge," and propose **Hierarchic Social Entropy**
as a judge-free measure of whether a society is real.

The parallel to the layer question is exact. *Where* a mechanism lives, and *whether* it is
doing work, are separate questions — and a metric that conflates them (accuracy for them;
weight-change for us) will mislead. This is why the weight analysis above is a **prior**,
not an answer, and why the causal sweep is the experiment that settles it.

## References

- Zeiler & Fergus (2014), *Visualizing and Understanding Convolutional Networks*
- Olah et al. (2020), *Zoom In: An Introduction to Circuits*, Distill
- Olsson et al. (2022), *In-context Learning and Induction Heads*, Anthropic
- Wang et al. (2022), *Interpretability in the Wild* (the IOI circuit)
- Meng et al. (2022), *Locating and Editing Factual Associations in GPT* (ROME)
- Meng et al. (2023), *Mass-Editing Memory in a Transformer* (MEMIT)
- Arditi et al. (2024), *Refusal in LLMs is Mediated by a Single Direction*
- Lee et al. (2023), *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts*
- Martin & Mahoney (2021), *Predicting trends in the quality of state-of-the-art neural
  networks without access to training or test data*, Nature Communications
- Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*
- Huot, Kaisers & Lapata (2026), *When is Routing Meaningful? Diversity and Robustness in
  Language Model Societies* (arXiv:2607.09197)
