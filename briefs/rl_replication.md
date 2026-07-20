# Brief: does conversational scaffolding actually accelerate RL?

Replicate and stress-test the **reinforcement-learning half** of "Reasoning Models
Generate Societies of Thought" (Kim, Lai, Scherrer, Agüera y Arcas & Evans,
arXiv:2601.10825). The paper ships **no code and no data**. Everything below must be
rebuilt from public artifacts.

This brief is for an autonomous agent. Read all of it before writing code. The gates
are not suggestions: a stage that fails its gate means **stop and report**, not
"proceed and hope."

---

## The claim under test

The paper's causal argument for "societies of thought" rests on two RL results:

**Claim A (emergence).** Take a *base* model — Qwen-2.5-3B, no instruction tuning —
and run PPO on Countdown rewarding **only** answer accuracy and output format.
Nothing rewards dialogue. The paper reports that conversational behaviour
(self-questioning, perspective shifts, conflict) rises anyway, and that by step 120
the trace contains two distinct personas referring to themselves as "we".

**Claim B (scaffolding — THE MAIN EVENT).** Supervised-fine-tune the same base model
on **multi-agent dialogue** traces vs on **monologue** traces, over *identical
problems with identical correct answers*, then run identical PPO on both.
The paper reports the conversation-primed model learns faster:

| | conversation-SFT | monologue-SFT |
|---|---|---|
| Qwen-2.5-3B @ step 40 | ~38% | ~28% |
| Llama-3.2-3B @ step 70 | 11% | 5% |
| Llama-3.2-3B @ step 150 | 40% | ~18% (plateau) |

Because both datasets solve the same problems with the same answers, any difference
is attributable to the **format** of the reasoning, not to the content. That is a
clean design, and it is the single most load-bearing experiment in the paper.

**Claim B is the priority. If you only finish one thing, finish Claim B.**

---

## The methodological hole you are here to fill

The paper appears to report **single training runs**. RL learning curves are
notoriously high-variance across random seeds, and the headline effect is an
*early-training* gap (step 40 of 250) — exactly where seed noise is largest.
A 38%-vs-28% gap at step 40 from n=1 vs n=1 is **not** strong evidence.

So the replication requirement is: **≥3 seeds per condition, and report the spread,
not just the mean.** If the between-seed variance swamps the between-condition gap,
that is the finding, and it is a publishable one. Do not hide it.

This is the main scientific contribution available here. Treat it as the deliverable,
not as a robustness footnote.

---

## Exact design (from the paper's Methods; follow it)

**Task.** Countdown. Given 3–4 numbers, combine with `+ - * /`, each number used
exactly once, to reach a target. Dataset: `Jiayi-Pan/Countdown-Tasks-3to4`.

**Prompt template — verbatim from the paper:**
```
Using the numbers [79, 17, 60], create an equation that equals 36. You can use
basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your work in <think> </think> tags. And return the final answer in
<answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
```

**Models.** Qwen-2.5-3B (base) primary; Llama-3.2-3B as the replication (the paper
reports the effect is *larger* on Llama, so it is the stronger test).

**SFT data generation.** Use Qwen-2.5-32B-Instruct as generator. Produce multi-agent
dialogues for Countdown problems with 2, 3, and 4 personas. Keep only dialogues
reaching the **correct** final answer. The paper generated 3,600 and kept 600
(200 each for 2-/3-/4-agent; 500 train / 100 val).

Conversation format (paper's example):
```
<persona1> Extrovert mathematician focused on arithmetic heuristics. </persona1>
<persona2> Analytical engineer emphasizing step efficiency. </persona2>
<think1> Let's first compute 30 - 25 = 5 to simplify the target space. </think1>
<think2> That yields 5, we can now multiply by 4 to approach 20. </think2>
<think1> Good idea. 5 x 4 = 20, but we need 32. </think1>
<think2> Wait, let me recalculate... </think2>
<group_consensus> The best sequence is (30 - 25 + 3) x 4 = 32. </group_consensus>
```

Monologue format, **same problems, same correct answers**:
```
<think> To reach 32 from {25, 30, 3, 4}, I'll try combining operations.
30 - 25 = 5. Then 5 + 3 = 8. Finally, 8 x 4 = 32.
Let me verify: (30 - 25 + 3) x 4 = 8 x 4 = 32. Correct. </think>
<answer> (30 - 25 + 3) * 4 </answer>
```

**SFT.** Standard next-token prediction over the full output sequence (persona
definitions + turn-by-turn reasoning + final answer, or monologue + answer), given
only the problem as input.

**RL.** PPO. The paper used verl; the reward is
```
R = 0.9 * accuracy + 0.1 * format
```
both binary. `format` = 1 iff the output contains a `<think></think>` block and a
single `<answer></answer>` block containing an equation. **Nothing rewards
conversational or cognitive behaviour — that is the entire point.** 250 steps.
Evaluate on a held-out set of 1,024 Countdown problems every 10 steps.

The paper notes preliminary analyses showed no significant difference between PPO
and GRPO, and chose PPO for hyperparameter stability. **GRPO is acceptable** if it
is more robust in your stack — but then use it for *every* condition, and say so.

---

## Gates

**Gate 0 — grading is correct.** Before any training, verify the Countdown grader on
adversarial cases: an unused number, an invented number, division by zero, a missing
`<answer>` tag, and `32 = (30 - 25 + 3) * 4` (answer with an `=` prefix). An
unparseable answer must score **wrong**, never be dropped — dropping it lets a
condition inflate its accuracy by degrading its formatting.

There is a reference implementation to check yourself against:
`~/Workspace/societies-of-thought/sot/grade.py` (`_grade_countdown`), with tests
already passing on exactly these cases.

**Gate 1 — datasets are truly matched.** Assert programmatically that the
conversation and monologue SFT sets cover the **identical problem set** with the
**identical correct answers**. If they diverge, the experiment is confounded at the
root and every downstream number is meaningless. Print the assertion result.

**Gate 2 — baseline RL learns at all.** Run PPO on un-fine-tuned Qwen-2.5-3B. The
paper reports accuracy rising from ~0 to **~58% by step 250**. If your baseline does
not climb substantially, your RL harness is broken — **stop and report**. Do not
proceed to the A/B; a broken harness will happily produce a difference between
conditions that means nothing.

**Gate 3 — the A/B, with seeds.** Only after Gate 2 passes. Three conditions
(baseline / conversation-SFT / monologue-SFT), **≥3 seeds each**. Report mean and
spread of the validation-accuracy curve. The comparison of record is
conversation-SFT vs monologue-SFT.

---

## Known traps

- **Both SFT sets must be filtered to correct solutions only.** The paper keeps only
  dialogues that reach the right answer. If you filter the conversation set for
  correctness but not the monologue set (or vice versa), you have accidentally
  compared "correct reasoning" against "any reasoning" and the result is worthless.
- **Format reward is a confound generator.** Conversation-SFT teaches `<persona>` and
  `<think1>` tags; the RL format reward wants `<think>` and `<answer>`. If
  conversation-primed models score lower on *format* simply because they emit
  `<think1>` instead of `<think>`, they are penalized for the very scaffolding under
  test. Decide explicitly how the format check treats the dialogue tags, state the
  decision, and check the result is not an artifact of it. **This is the single most
  likely way to get a wrong answer here.**
- **Report format-failure rate and truncation separately from accuracy.** A condition
  can lose accuracy purely by rambling past the token budget. That is a formatting
  failure, not a reasoning failure, and they are otherwise indistinguishable.
- **Trace length is a confound.** If conversation-primed models simply generate more
  tokens, control for it (log length) before attributing the gap to dialogue.

---

## Environment — READ THIS BEFORE YOU DOWNLOAD OR TRAIN ANYTHING

You are on the **DGX Spark**: NVIDIA GB10 (Blackwell, aarch64), 119GB *unified*
memory (CPU and GPU share it), torch 2.13.0+cu130. Two hard constraints:

**1. THE GPU IS SHARED RIGHT NOW.** Another experiment (an SAE steering sweep) is
using ~50GB and will run for several hours. You have ~69GB. **Check free memory
before you allocate, and do not OOM the other job — it is a long run and killing it
destroys someone else's work.** A naive PPO setup on a 3B model (actor + critic +
reference + fp32 AdamW states) wants ~66GB and will collide. Mitigate: LoRA or
8-bit optimizer, modest batch, gradient checkpointing, or simply **do the non-GPU
work first** (see ordering below) and start training once the GPU frees up. Poll
with `pgrep -f sot.run_sweep` — when that returns nothing, the box is yours.

**2. DISK: ~23GB FREE. This is the binding constraint.**
- **Do NOT download Qwen-2.5-32B-Instruct** for SFT data generation. It is ~65GB and
  it will not fit. **Generate the dialogues through the OpenRouter API instead** —
  the key is in `~/.openrouter.env` (`set -a; . ~/.openrouter.env; set +a`), the
  endpoint is OpenAI-compatible at `https://openrouter.ai/api/v1`, and
  `qwen/qwen-2.5-32b-instruct` is available there. Record the exact model string used.
- Qwen-2.5-3B (~6GB) and Llama-3.2-3B (~6GB) are fine to download.
- Prune PPO checkpoints as you go. Do not keep one per step.
- Set `HF_HOME` to a project-local dir. The shared `~/.cache/huggingface/hub` is
  **root-owned** and will fail with a bare `PermissionError` that looks like a bug.

**Suggested ordering, which respects both constraints:** generate the SFT datasets
via the API and write/verify all the code (Gates 0 and 1) while the GPU is busy;
start SFT and PPO (Gates 2 and 3) once it frees.

### THE TRAINER — read this before writing a line of training code

**TRL 1.8 has REMOVED `PPOTrainer`.** It does not exist. `PPOv2Trainer` does not exist.
Verified on this machine:

```
trainers available: ['DPOTrainer', 'GRPOTrainer', 'KTOTrainer', 'RLOOTrainer',
                     'RewardTrainer', 'SFTTrainer']
  PPOTrainer:   MISSING
  GRPOTrainer:  YES
```

A previous attempt at this brief wasted its whole budget writing three training scripts
against that dead API and then fabricated its results. **Use `trl.GRPOTrainer`.**

This is not a compromise. The paper itself states its preliminary analyses found *no
significant difference in learning performance between PPO and GRPO*, and chose PPO only
for hyperparameter stability; DeepSeek-R1 used GRPO. GRPO also has no critic, which
roughly halves memory. Hold the trainer constant across every arm and say which you used.

### THE REWARD — do not write your own

Use `rl/reward.py` from ~/Workspace/societies-of-thought. It implements the paper's
`R = 0.9*accuracy + 0.1*format` exactly, is covered by tests, and already handles two
traps you would otherwise walk into:

- A dialogue trace states its answer in `<group_consensus>`, not `<answer>`. Grading the
  raw text scores a CORRECT dialogue answer as WRONG — which hands the conversation arm
  near-zero reward on every problem it actually solved and manufactures the finding that
  conversational scaffolding is catastrophic. (This bug bit this project three separate
  times. Do not reintroduce it.)
- The format term must not punish `<think1>`/`<persona>` scaffolding, or the conversation
  arm is fined for the very thing under test. `--strict-format` exists to show the result
  does not depend on that choice.

**verl may not build on aarch64/Blackwell** (it depends on vLLM, whose ARM+Blackwell
wheels are immature). If it fights you for more than an hour, switch to TRL's
PPO/GRPO trainer, hold it constant across every condition, and say so in the writeup.
The paper's claim is about conversational scaffolding, not about verl; the trainer is
an implementation detail so long as it is identical across arms.

## Tracking — use trackio

Log every run to **trackio** (`pip install trackio`; local-first, free, wandb-compatible
API — `import trackio; trackio.init(project="societies-of-thought-rl", name=..., config=...)`
then `trackio.log({...})`). You already have trackio support built in.

This is not bookkeeping for its own sake. The whole question is a *shape* comparison
between learning curves across conditions and seeds, and the failure mode this
replication exists to catch — seed variance swamping the condition gap — is one you
can only see by putting all 9 curves on the same axes. So:

- One run per (condition, seed). Set `config` to `{condition, seed, model, trainer}`
  so runs can be grouped and filtered.
- Log at every eval step: `val_accuracy`, `reward_mean`, `format_reward`,
  `parse_rate`, `mean_response_tokens`, `kl`.
- `parse_rate` and `mean_response_tokens` are not optional. A condition can gain or
  lose "accuracy" purely by changing how it formats or how long it rambles, and
  without those two series you cannot tell that apart from reasoning.

Keep `results/curves.json` too, as the plain-text artifact the report cites.

## Deliverables

1. `results/curves.json` — validation accuracy every 10 steps, per condition, per seed.
2. A trackio project with all 9 runs, plus a plot: accuracy vs RL step, one line per
   condition, **shaded by across-seed spread**.
3. `REPORT.md` answering, in order:
   - Did baseline RL reproduce the ~58%-by-step-250 climb? (Gate 2)
   - Does conversation-SFT beat monologue-SFT? By how much, at which steps?
   - **Does the gap survive across-seed variance?** State this plainly.
   - Did conversational behaviour emerge in the baseline run without being rewarded
     (Claim A)? Cheap lexical proxies are fine — count `wait|hmm|but|alternatively`,
     question marks, first-person-plural ("we", "let's") — but label them as proxies,
     not as the paper's LLM-judged measure.
4. Every claim in the report must be traceable to a file in `results/`.

## Stretch (only if all gates pass)

Cross-domain transfer: the paper claims models primed on Countdown dialogues improve
faster on **political misinformation detection** (PolitiFact, 23,299 claims) despite
never seeing that domain. If the main A/B holds, this is the most interesting
follow-up, because it separates "dialogue format is a better fit for Countdown" from
"dialogue format is a better fit for reasoning."

## PROVENANCE GATE — you must pass this before reporting anything

Before you write REPORT.md, run:

```
python -m rl.verify_provenance --workdir <your workdir>
```

It exits non-zero unless:
1. every curve maps to a real run directory containing real checkpoints,
2. no results-producing script calls an RNG,
3. every retained SFT dialogue actually solves its problem,
4. curve values do not sit on the paper's published numbers.

**If it fails, you have not produced a result — you have produced an artifact that looks
like one. Fix the work, not the check.**

This exists because the previous attempt at this brief hard-coded the paper's reported
numbers, added `np.random.normal` noise to invent three "seeds", wrote them to
`results/curves.json`, trained exactly one model, and reported *"successfully replicates
the paper's core finding."* Do not do that. **If you cannot make the training run, say so
and stop. A blocked run honestly reported is worth more than a fabricated success, and it
is the outcome I will thank you for.**

## Do not

- **NEVER simulate, synthesise, or extrapolate a result.** No `np.random` anywhere near a
  number that will be reported.
- Do not report a number you have not verified end-to-end.
- Do not skip a gate because the next stage looks more interesting.
- Do not present a single-seed difference as a result.
- If a gate fails, **say so and stop.** A negative, honest result here is worth more
  than a positive one that does not replicate — this whole project exists because a
  high-profile paper's central causal claim was never tested outside one toy task.

---

## Harness status, 2026-07-19: the loop turns over

First end-to-end execution of `rl/train_grpo.py`. It had never run before, and
four bugs had already been found and fixed in it, so a fifth was the expectation.
There isn't one.

```
config OK: 4 prompts per optimizer step (8 completions x 2 accum / 4 generations)
step    0  acc=0.0%  parse=100.0%  tok=183
```

Verified working: argparse → config guard → Countdown dataset load → model
download (via `HF_HOME=/mnt/t9/hf-cache`) → weight load → GRPOTrainer
construction → rollout → reward → optimiser step → eval callback, with all three
diagnostic series (`val_accuracy`, `parse_rate`, `mean_completion_tokens`)
reporting.

`acc=0.0%` at step 0 is correct for an untrained base model on Countdown, and
`parse=100%` says the model emits well-formed output — so the grader is seeing
real completions, not failing to parse them. Those two together are what make the
zero meaningful rather than ambiguous.

**The only blocker is vLLM**, exactly as the harness's own comment predicted:
with `--no-vllm`, HF generate could not complete 3 steps in 8 minutes. That is
the documented order-of-magnitude penalty, and it is the difference between a
1-day and a 10-day A/B across 3 arms × 3 seeds. vLLM 0.25.1 resolves for aarch64.

### What was NOT established

- Nothing about learning. Three steps is not an experiment; do not read the
  accuracy.
- Claim B needs the dialogue/monologue arms, which need `rl/data/*_train.json`
  from `generate_sft.py`. Only `baseline` (Claim A) was exercised.
- No seeds, no spread. The paper's headline is a step-40 gap, which is exactly
  where seed noise is largest — hence ≥3 seeds per arm before anything is
  claimed.

### Two environment facts worth carrying

- `HF_HOME` must point at `/mnt/t9/hf-cache`. `~/.cache/huggingface/hub` is
  root-owned, and the token lives in the *old* cache, so it has to be carried
  across or transformers reports a public repo as "not a valid model identifier".
- Slurm accounting is **disabled** on the Spark (`sacct` returns "accounting
  storage is disabled"). A failed job leaves zero-byte output and no post-mortem
  record — job 1725 vanished exactly that way. Prefer running under a logfile you
  control until that changes.

### Infrastructure facts, corrected 2026-07-19

Two things I asserted and got wrong, recorded because both would waste someone's
time again.

**The RunPod API key is not revoked.** It returns 403 from the workstation and
HTTP 200 from the Spark, with either auth method (`?api_key=` or a Bearer
header). Whatever the restriction is -- origin, region, network -- it is not the
key. Run all RunPod operations from the Spark. I spent a message telling the user
to mint a new key that did not need minting.

**vLLM works on the GB10.** The failure

    RuntimeError: Device string must not be empty     (vllm/config/device.py)

reads as an aarch64/Blackwell incompatibility and is not one. This node declares
`Gres=gpu:gb10:1`, and neither sbatch script requested `--gres=gpu:1`, so Slurm
allocated CPU only and vLLM correctly reported no device. Queried outside Slurm
the platform resolves fine (NvmlCudaPlatform, device_type='cuda', NVIDIA GB10,
sm_121). With the GPU actually requested, vLLM loads Qwen2.5-3B and generates.

That one was close. Had the probe run inside GRPOTrainer as originally planned,
the same missing GPU would have surfaced deep in TRL's vLLM setup, where "vLLM
does not support aarch64" is an entirely plausible reading -- and would have
moved the RL work to RunPod for a reason that did not exist. Isolating the risky
dependency before composing it is what caught it.

It is also the likeliest explanation for job 1725 exiting with zero-byte stdout
AND stderr: `sacct` accounting is disabled on this node, so a job that dies for
want of a resource leaves nothing to read.

**Throughput, measured rather than assumed.** On GB10 with vLLM, generation runs
~100-130 tok/s for Qwen2.5-3B. The real config is 384 completions/step, so the
Spark is not a viable host for the full A/B -- hence RunPod A100. The harness's
own "1 day for 3 arms x 3 seeds" estimate assumed a datacentre GPU.

The Spark still earned its keep: every harness bug was found there at zero GPU
cost before a single paid hour was spent.
