# Brief: what the HT-SR literature says, and the three holes this project sits in

Related work and methodology for the weight-spectrum arm of this project
(`analysis/weight_spectra.py`, and the α measurements that Experiment 2 will produce).

This is a *positioning* brief, not a build brief. It exists because the weight-spectrum
work has a real related-work problem: the HT-SR literature is contested in ways the
headline papers do not advertise, and a referee who knows it will ask three specific
questions. Two of them we already answer by construction. One we should say out loud.

It also records something worth knowing before Experiment 2 runs: **the experiment as
designed already sits in two verified-empty niches of this literature.** That is not a
reason to change the design. It is a reason to measure α while we are in there.

---

## The short version

- HT-SR's α is **best supported as a cheap, data-free *allocation* signal** (which layer
  to prune, how many LoRA experts, what learning rate) — not as a predictor of
  generalisation. The allocation cluster is peer-reviewed with real numbers. The
  generalisation claim is contested from four directions, including by its own authors.
- Our layer-15 measurement is a **paired, same-matrix difference**, which is exactly the
  form that survives the two strongest methodological critiques. Say so explicitly.
- Experiment 2 lands in **two niches nobody has published in**: α under style-controlled
  fine-tuning, and α transfer across SFT→RL.

---

## What is actually established, and what is not

Sorted by how much weight it will bear.

| Claim | Status |
|---|---|
| Weight ESDs are heavy-tailed and phase-structured | **Solid.** JMLR 22(165), consistent with the independent RMT literature |
| α̂ ranks models within one architecture series, no data needed | **Narrow.** Strong Kendall-τ within series; never independently replicated |
| α *alone* drives that result | **Contested by Martin & Mahoney themselves** — α̂ is shape × scale and the scale term carries much of it |
| α transfers across architectures / objectives / depth | **Contested.** Reverses under large LR; degrades on pretrained Transformers; uninformative when only depth varies |
| α is a reliable *per-matrix* estimate | **Contested.** Aspect-ratio bias; power-law-vs-lognormal indistinguishability; WeightWatcher's own docs concede fit failures |
| α is causal, not an optimiser-hyperparameter readout | **Open.** Heavy tails provably arise from step size and batch size |
| α-guided *allocation* helps (pruning, LoRA, LR, quantisation) | **Best-supported cluster.** NeurIPS/EMNLP/ICML, concrete numbers — but every paper is by an HT-SR-aligned group |
| α relates to SAE / interpretability quality | **Untested.** The two literatures have zero crossover |

The honest one-line summary: *the strongest case for α is not "it predicts generalisation"
— it is "it is a cheap data-free allocation signal."*

---

## The three questions a referee will ask

### 1. "Aspect-ratio bias makes your α values incomparable."

Real, and recent: FARMS (ICML 2025) shows α estimates are systematically biased by the
matrix aspect ratio Q = m/n. We measure across `q/k/v/o/gate/up/down`, which have
*different* shapes. A raw cross-projection α comparison would be unsafe.

**Why we survive it:** we never compare α across projections. We report

```
d_alpha = alpha_distill - alpha_base
```

on the *same matrix* — same layer, same projection, same shape. Whatever bias the aspect
ratio induces, it is the same constant in both terms and cancels in the difference. The
paired design is doing real work here; it is not incidental.

**Action:** one sentence in the writeup stating this. Do not leave it implicit.

### 2. "Shape metrics are uninformative when only depth varies."

Yang et al. (KDD 2023) document this specifically, and we vary depth — 32 layers. If we
claimed *"layer 15 has a low α,"* we would be squarely in the flagged regime.

**Why we survive it:** that is not our claim. Ours is *"layer 15's α **changed least**
under distillation."* A per-layer paired difference is a different object from a
cross-layer α level, and the critique does not reach it.

**Action:** phrase the finding as a change, never as a level. The current results support
this: layer 15 ranks 21/32 on relative weight delta and 26/32 on |Δα| — below average on
both, i.e. distillation *barely touched it*. That is the "evidence against" branch of the
pre-registered prediction, and independent structural corroboration of the steering
result.

### 3. "Why not just use WeightWatcher?"

Because a power-law exponent you cannot justify fitting is a number you should not
interpret. Clauset, Shalizi & Newman (SIAM Review 2009) show that log-log fitting gives
"substantially inaccurate estimates … and no indication of whether the data obey a power
law at all," and that power-law vs log-normal is undecidable without very large samples.
WeightWatcher's own documentation concedes the single-PL fit sometimes fails, and that at
Q = 1 a *random* matrix ESD can look heavy-tailed.

`wwj` gives a calibrated posterior over α plus Bayes-factor model comparison (is the tail
a power law *at all*?). That is a methodological advantage, not a preference. The docstring
in `analysis/weight_spectra.py` already says this; keep it.

---

## The three holes — and we are in two of them

Searched for and **verified absent** from the literature:

**(a) α under style-controlled fine-tuning.** No work compares models fine-tuned on
*differently-styled but semantically identical* data via weight spectra. The nearest
neighbour (Spectral Signatures of LLMs, 2607.03377) does not control style at fixed
semantics.

This is a literal description of Experiment 2's design: dialogue vs monologue traces over
**identical problems with identical correct answers**. The control that makes the
behavioural comparison meaningful is the same control that makes the spectral comparison
novel. We get it for free.

**(b) α transfer across SFT and RL.** No paper tests whether α behaves consistently
between supervised fine-tuning and RL/RLHF. Genuinely empty. (Adjacent search hits on
"catastrophic Goodhart in RLHF" concern heavy-tailed *reward error* — unrelated.)

Experiment 2 runs SFT-priming *then* GRPO on both arms, so the SFT→RL α trajectory falls
out of checkpoints we are already saving.

**(c) Adversarial α-Goodharting.** No hostile-manipulation study exists. Every published
α intervention regularises *toward* balanced/lower α and reports gains — and all are by
HT-SR-aligned groups. (This one is covered by the companion `wwj` work, not here.)

**What to actually do about (a) and (b):** nothing to the experimental design. Just take
`wwj` α on the saved checkpoints for both arms, at the same steps, across seeds. If the
two arms diverge spectrally, that is a result. **If they do not, that is also a result** —
and it is the one that matters, because it would say the dialogue/monologue distinction
is not written into the weights at all, which is the same dissociation the steering
experiment found from the other direction.

Pre-register the direction before looking, as we did for layer 15.

**One caution.** α is a layer-level scalar and RL fine-tuning moves weights very little
compared to distillation. Expect |Δα| well below the distillation deltas measured here
(mean |Δα| per layer is already only ~0.008–0.014 for a *full reasoning distillation*).
Power may simply not be there. Check that the effect is above the seed-to-seed spread
before interpreting anything — the same discipline the ≥3-seeds requirement enforces on
the accuracy side.

---

## The unclaimed question next door

The HT-SR and SAE literatures are **disjoint** — no crossover papers, despite shared
vocabulary. They fit different matrices: HT-SR fits the ESD of WᵀW for a *weight* matrix;
SAE geometry work fits covariance eigenvalues of *decoder dictionaries* or activation
clouds.

The open question: **does a layer's α predict how well an SAE trained on that layer
performs?** This project is one of very few with both halves already working — `wwj` α per
layer, and a validated SAE steering rig at layer 15.

Two honest caveats before anyone gets excited. α is a single layer-level scalar while SAE
quality varies with width, k, and training data, so it may explain little variance even if
the mechanism is real. And the sign is not predictable a priori: heavier tails mean more
concentrated dominant directions, which could help SAEs (clearer structure) or hurt them
(more superposition in fewer directions). That unpredictability is a reason to *run* it,
not to assume the answer.

Adjacent and worth reading: Li, Michaud, Baek, Engels, Sun & Tegmark, "The Geometry of
Concepts: Sparse Autoencoder Feature Structure" (2410.19750, **Entropy 27(4):344, 2025**,
peer-reviewed) — feature point clouds are anisotropic with a power-law eigenvalue spectrum
tested against a Wishart/RMT null, slope steepest in *middle* layers (≈ −0.47 at layer 12
vs ≈ −0.24/−0.25 early/late). Middle layers are exactly where production SAEs are trained.

---

## Citation hygiene

Three errors are in circulation. Do not reproduce them.

1. **"Post-mortem on a deep learning contest" (2106.00734) is NOT JMLR.** It is a preprint;
   its arXiv page carries no journal-ref. The JMLR 22(165) entry is the *different*
   "Implicit Self-Regularization" paper. Secondary sources conflate the two.
2. **Do not cite "108 models / 17 architectures"** for the Nature Communications paper.
   Unverifiable — it appears only in secondary summaries. The paper says "hundreds"
   (≥500 CV, ≈100 NLP).
3. **"Power Laws in Empirical Eigenvalue Spectra," Entropy 28(4):418 — wrong paper.** The
   DOI is real but it concerns neural criticality / phenomenological RG, not weight
   matrices. It is not an HT-SR critique.

Also note: **"Evaluating NLP models with generalization metrics…" and "Test Accuracy vs.
Generalization Gap…" are one paper**, not two (arXiv title vs KDD title, identical
authors). It is **KDD 2023**, not NeurIPS.

---

## Verified bibliography

Every entry below was checked against a primary source. Venue is stated exactly; preprints
are marked as such.

**Foundational**
- Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks*, **JMLR 22(165):1–73** (2021), arXiv:1810.01075
- Mahoney & Martin, *Traditional and Heavy-Tailed Self Regularization*, **ICML 2019**, PMLR 97:4284–4293, arXiv:1901.08276
- Martin, Peng & Mahoney, *Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data*, **Nature Communications 12:4122** (2021), arXiv:2002.06716
- Martin & Mahoney, *Post-mortem on a deep learning contest*, **preprint**, arXiv:2106.00734
- Martin & Hinrichs, *SETOL: A Semi-Empirical Theory of (Deep) Learning*, **preprint** (2025), arXiv:2507.17912

**α as an allocation signal (the strong cluster)**
- Zhou et al., *TempBalance*, **NeurIPS 2023 Spotlight**, arXiv:2312.00359
- Lu et al., *AlphaPruning*, **NeurIPS 2024**, arXiv:2410.10912 — LLaMA-7B @70% sparsity: 23.86 PPL vs 85.77 uniform
- Qing et al., *AlphaLoRA*, **EMNLP 2024 Main**, arXiv:2410.10054 — 80 experts beat MoLA's 160
- Liu et al., *Model Balancing Helps Low-data Training and Fine-tuning*, **EMNLP 2024 Oral**, arXiv:2410.12178
- Hu et al., *FARMS: Eigenspectrum Analysis without Aspect Ratio Bias*, **ICML 2025**, arXiv:2506.06280

**Critiques**
- Jiang et al., *Fantastic Generalization Measures and Where to Find Them*, **ICLR 2020**, arXiv:1912.02178 — >40 measures, >10,000 CNNs; norm measures *negatively* correlate. **Note: α itself was never entered in this study** — it tests spectral norms, not exponents. Cite it as "the closest family fails," not as a refutation of α.
- Yang et al., *Test Accuracy vs. Generalization Gap*, **KDD 2023**, 3011–3021, arXiv:2202.02842 — the most specific documented α failures, from an HT-SR-friendly group
- Dziugaite et al., *In Search of Robust Measures of Generalization*, **NeurIPS 2020**, arXiv:2010.11924 — rank-correlation-over-a-model-zoo can exploit non-causal correlations
- Gastpar et al., *Fantastic Generalization Measures are Nowhere to be Found*, **ICLR 2024**, arXiv:2309.13658
- Clauset, Shalizi & Newman, *Power-law distributions in empirical data*, **SIAM Review 51(4)** (2009), arXiv:0706.1062

**Causal confound (α may read the optimiser, not the model)**
- Hodgkinson & Mahoney, **ICML 2021**, arXiv:2006.06293 — heavy tails from multiplicative SGD noise
- Gürbüzbalaban et al., **ICML 2021**, arXiv:2006.04740 — tail index is a function of step size and batch size

**Weights-only, adjacent**
- Unterthiner et al., *Predicting Neural Network Accuracy from Weights*, **arXiv-only**, arXiv:2002.11448 — R² > 0.98 from weight moments; caveat: zoo has a wide accuracy spread
- Schürholt et al., *Self-Supervised Representation Learning on Weight Spaces*, **NeurIPS 2021**, arXiv:2110.15288
- Navon et al., *DWSNets*, **ICML 2023**, arXiv:2301.12780
- White et al., *NAS-Bench-Suite-Zero*, **NeurIPS D&B 2022**, arXiv:2210.03230 — #params and FLOPS are competitive with every zero-cost proxy
- Dinh et al., *Sharp Minima Can Generalize*, **ICML 2017**, arXiv:1703.04933 — ReLU scale invariance; the essential caveat for any raw-magnitude weight metric

**SAE side**
- Li, Michaud, Baek, Engels, Sun & Tegmark, *The Geometry of Concepts*, **Entropy 27(4):344** (2025), arXiv:2410.19750
- Gao et al., *Scaling and evaluating sparse autoencoders*, **preprint**, arXiv:2406.04093 — all quality metrics activation-based; no spectral metric anywhere (direct evidence for the gap)
- Cunningham et al., **ICLR 2024**, arXiv:2309.08600
- Dunefsky, Chlenski & Nanda, *Transcoders Find Interpretable LLM Feature Circuits*, **NeurIPS 2024**, arXiv:2406.11944

**False friend:** Hodgkinson, Wang & Mahoney, *Models of Heavy-Tailed Mechanistic
Universality* (arXiv:2506.03470) — "mechanistic" here means statistical-physics
universality (Kesten–Goldie, α-stable Lévy), **not** mechanistic interpretability.
