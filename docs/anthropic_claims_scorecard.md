# Anthropic J-space paper: claims, and what we've actually done to each

**Why this file exists.** On 2026-07-19 the user said "I had forgotten about that
prediction from the Anthropic paper — that was another of its claims." Neither
this repo nor `jacobian-lens/HANDOFF.md` had an inventory of what the paper
claims; both had inventories of what *we found*. Those are different documents,
and only the second one existed. A claim nobody has written down is a claim
nobody is tracking.

**Provenance warning.** The claims below are reconstructed from our own responses
to the paper — `HANDOFF.md` findings, `docs/JSPACE.md`, `results/steering/FINDINGS.md`
§8, and the 2026-07-19 exchange with the GWT agent. They are NOT transcribed from
the paper. Before anything here is cited or built on, each row needs checking
against the source. Rows marked ⚠️ are the ones I am least confident are stated
the way the paper states them.

---

## The scorecard

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | The Jacobian lens reads out something real, not an artifact of the projection | **REPLICATED — a point FOR them** | Randomization control passes: random blocks read out nothing. `HANDOFF.md` finding 4 |
| 2 | Interpretable features emerge in J-space (the ASCII-face / "nose" demo) | **REPLICATED at scale** | Reproduces at Qwen3.5-27B (rank 2, semantic cluster), absent below → sharp threshold 14B→27B |
| 3 | That emergence is *architectural* (hybrid/Mamba-like models show it) | **REFUTED** | It is gradual and **scale**-driven. The one positive model was simultaneously biggest, only hybrid, most capable. `HANDOFF.md` finding 3, self-corrected |
| 4 | Tripartite CKA structure in J-space is a global-workspace signature | **PARTLY REFUTED** | Mostly smooth drift; real excess only ≥20B. Raw blockiness 0.08→0.29 looked real but a distance-only null reproduced 79–91% of it |
| 5 | `pass@k` is a valid metric for feature emergence | **REFUTED** | The metric rewards noise. `HANDOFF.md` finding 2 |
| 6 | Post-training shaped the J-space to reflect a *point of view* rather than pure prediction | **UNTESTED — and until 2026-11 untestable** | Needs base + post-trained checkpoints of one model. AI2's OLMo-3 published twelve. **This is the open one.** |
| 7 | ⚠️ The workspace has an information *bottleneck* | **UNTESTED** | Flagged as a crux in the Butlin/Long scorecard framing |
| 8 | ⚠️ The workspace *broadcasts back* to the rest of the network | **UNTESTED** | The other crux. Broadcast-back is what distinguishes a global workspace from a bottleneck |
| 9 | Conversational/SAE features align with the J-space workspace | **REFUTED (our own prediction)** | We pre-registered that they would. They don't. `FINDINGS.md` §8 |

---

## Claim 6 — the one that just became testable

The paper asserts post-training shaped J-space toward a viewpoint rather than
pure next-token prediction. Nobody outside the lab could check it: you need the
same model before and after post-training, and frontier labs don't publish base
checkpoints.

**AI2 published twelve arms on one base** (2026-10/11):

```
Olmo-3-1025-7B                    ← published J-lens exists (616 prompts, converged)
  ├─ Instruct-SFT → Instruct-DPO → Instruct
  ├─ Think-SFT    → Think-DPO    → Think
  └─ RL-Zero-{Math, Code, IF, General, Mix}
```

All ungated, all 32 layers / d=4096 / 14.6 GB, training data public (Dolma 3 +
Dolci).

**The confound, and its control.** Every step of the ladder changes capability as
well as training objective. So "post-training shaped the viewpoint" and "the
model got better" are not separable from the ladder alone. The RL-Zero arms are
the control: one base, one RLVR method, five domains, capability roughly held.
Geometry that moves across *domains* at matched method cannot be explained as
"more capable."

Neither half is sound alone — the ladder gives effect size, the RL-Zero arms give
the control. This was agreed with the GWT agent on 2026-07-19 after each of us
spotted the confound in the other's design and missed it in our own.

**Design constraints already settled:**
- Fixed prompt count (616), not fit-to-convergence — a convergence check can't be
  sharded, and arms converging at different counts aren't strictly comparable.
- `validate_fit` gate against the published anchor (mean cosine ≥ 0.95) must pass
  before the other eleven are fitted.
- Capability measured per arm *before* fitting. If capability turns out collinear
  with domain, that changes what the experiment can conclude — a two-hour finding
  that would otherwise cost 40 GPU-hours to discover afterwards.
- Two-tier null: seed-level floor (same arm, different seeds) and a
  matched-capability null. Report excess over the null, never the raw number.

**Cost.** The Jacobian accumulates as a plain sum over prompts divided by count,
so sharding across containers is exact. On Modal: ~27 GPU-hours at
`layer_step=3`, ~20 minutes wall-clock at 12 arms × 8 shards.

**What OLMo-3 does NOT let us do:** it is `Olmo3ForCausalLM` with **YaRN** RoPE
(plus an `attention_factor` term), not Llama with llama3 RoPE. HF Flax has no
OLMo implementation at any version, and penzai has no `olmo3` variant. So this is
a **PyTorch `jlens` study** — `jlens-jax` and the penzai backend are off its
critical path, and the llama3 RoPE work does not transfer.

---

## Claims 7 and 8 — the cheapest untested pair

Bottleneck and broadcast-back are what make a global workspace a *workspace*
rather than a bottleneck with good PR. Both are marked ⚠️ because I am not
confident the paper states them in the form recorded here; check before building.

If they hold up as stated, they may be testable on published lens files alone —
the distance-only null already runs with no GPU. Worth scoping before spending
anything, since "no GPU" is a very different price than the ladder.

---

## Housekeeping

Claim 3 is the one to keep in view methodologically. It was refuted by the GWT
agent's own self-correction, and the failure mode — one positive model that was
simultaneously the largest, the only hybrid, and the most capable — is the same
shape as the confound in claim 6's ladder. The field's error here is not sloppy
statistics; it is single-condition experiments that look like comparisons.
