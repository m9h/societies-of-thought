# Scope: a faithful replication of the paper's RL claims

**Goal.** Reproduce the Societies-of-Thought RL results *with the paper's actual setup* --
PPO (not GRPO) + full fine-tuning (not LoRA) via **verl** (the paper's own library) -- to
either reproduce Claim A/B or show they don't reproduce even faithfully. This closes the
one gap our critique currently has to concede: "we could not test the RL claims."

## Why this is the right experiment, not just a bigger one

Our GRPO+LoRA attempt hit a **greedy-mode collapse / training instability** (train correct
rate climbed to 0.099 while greedy eval degenerated -- see rl_replication.md). The paper's
own words explain why: it reports it *"chose PPO over GRPO for hyperparameter stability."*
The paper anticipated exactly the instability we hit. So switching to PPO is not only
faithful -- it is the **specific predicted fix** for our failure mode. This experiment
tests that prediction directly.

## The three substitutions we are undoing

| ours (cheap) | paper (faithful) | why we substituted |
|---|---|---|
| GRPO | **PPO** (verl) | TRL 1.8 removed PPOTrainer |
| LoRA | **full fine-tuning** | 3B full FT OOMs one 80GB A100 |
| shaped reward | **0.9·acc + 0.1·format** (theirs) | GRPO made the paper's reward farmable; PPO's value baseline is far less exploitable |

## Hardware and cost (verified 2026-07-22)

Full-FT PPO of a 3B model needs actor + **critic** + frozen reference + fp32 Adam states +
gradients + vLLM rollout cache ≈ **110+ GB**, so it does NOT fit one 80GB card. verl shards
with FSDP across a node.

RunPod community pricing (verified via API): A100 PCIe 80GB **$1.19/hr**, H100 PCIe 80GB
**$1.99/hr**, H100 SXM **$2.69/hr**.

- **4× A100 80GB** (320 GB, comfortable): **~$4.76/hr**
- 2× A100 80GB (160 GB, tight with aggressive sharding): ~$2.38/hr -- risk of OOM, not recommended for a first run

Throughput is the biggest unknown: verl 3B PPO at 250 steps on 4× A100 is estimated
**8–12 h** (not benchmarked here -- flag as the largest cost-uncertainty).

## Tiered plan -- each tier gates the next

**Tier 0 -- the decisive single run (~$75–100).**
Claim A only: Qwen-2.5-3B *base*, PPO + full FT via verl, the paper's reward, 250 steps,
1 seed. One question: **does the paper's exact setup learn Countdown where our GRPO+LoRA
collapsed?** ~10 h × $4.76 ≈ $48 compute, +~$25–50 for verl setup/debug on a fresh node.
- If it learns cleanly (greedy eval climbs, no collapse) → the instability WAS the GRPO+LoRA
  substitution, the paper's stability reasoning is vindicated, and Tier 1 is worth it.
- If it *also* collapses → a much stronger finding: the paper does not reproduce even
  faithfully, which needs the paper's exact undocumented hyperparameters to resolve.

**Tier 1 -- Claim B, once (~$150–250).**
Only if Tier 0 learns. 3 arms (baseline / dialogue-SFT / monologue-SFT) × 1 seed, 250 steps.
The paired SFT data already exists and is verified (identical problems + answers). Produces
the dialogue-vs-monologue learning gap once -- the paper's main event.

**Tier 2 -- Claim B, robust (~$450–750).**
Only if Tier 1 shows a gap. 3 arms × 3 seeds = 9 runs. This is the experiment our OWN
critique demands: the paper's headline is a step-40 gap from n=1 vs n=1, and we argue that
is inside seed noise. Three seeds per arm is what turns "there is a gap" into "the gap
exceeds between-seed variance" -- or shows it doesn't.

## Risks, stated up front

- **verl setup on RunPod is non-trivial** (multi-GPU, Ray, config). Budget real debug time;
  the ~$25–50 setup overhead in Tier 0 is for this. verl is the paper's tool, actively
  maintained, but not plug-and-play.
- **The paper is underspecified** -- exact LR, KL coefficient, batch size, GPU count, and
  reward details are not all published. Some config is educated guessing, and a Tier-0
  collapse could be our guess rather than the paper. That ambiguity is itself reportable:
  "the paper cannot be exactly reproduced from its published details."
- **Throughput estimate is unbenchmarked.** If 250 steps takes 20 h not 10, every tier
  doubles. Tier 0 de-risks this before any larger spend.

## Recommendation

Run **Tier 0 only**, as a gate. ~$75–100 answers the single most important question -- does
the paper's own algorithm reproduce the learning our substitutions broke -- and directly
tests the paper's stability claim. It is the cheapest experiment that can either rescue the
RL replication or turn "we couldn't test it" into "we tested it faithfully and here is what
happened." Do not commit to Tier 1/2 until Tier 0's result and measured throughput are in.

Against the current RunPod balance ($165.76), Tier 0 is comfortable; Tier 2 is not, and
would need a top-up or a funded run -- which is exactly what the Longview grant would cover.
