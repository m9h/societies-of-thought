# Handoff: next work on the DDM pilot

*State as of 2026-08-18, branch `analysis/ddm-pilot`, commits cf4847e + b729e25.
Read `RESULTS.md` first — it has the full findings and caveats. Everything below
is runnable with `uv venv && uv pip install hssm`; every fit here converged in
seconds-to-minutes on a laptop CPU.*

## 1. ~~The discriminating experiment~~ — DONE 2026-08-05, see `RESULTS_math.md`

> **RESOLVED.** Drift falls under steering on MATH-Hard: dV = −0.336 (P(v↑)=0.001) flat,
> −0.282 (P=0.010) with near-ceiling dropped, −0.460 (P=0.000) hierarchical. The paper's
> story predicted the opposite sign. The boundary half of the prediction is NOT established
> (Δa swings +0.156 → +0.049 → +0.271 across the three fits) — only drift is.
>
> **Correction:** the traces are in the repo. `results/steering/main_rg.jsonl` has 800
> MATH-Hard rows (baseline + seven features at α=1.0) and carries real `pid`, so the MATH
> design is exactly paired — 100 shared problems — and needs no position-as-problem
> reconstruction.
>
> Next on this thread: the six matched-control features at α=1.0 on MATH are also in that
> file, unfit. They answer whether the drift decline is specific to the conversational
> feature or generic to steering anything.

### original text


Fit `fit_ddm.py`'s model to the **MATH-Hard steered traces** (the raw
`results/steering/*.jsonl` sweep outputs — 5,664 attempts — live on the
pod/workstation, not in this repo). Per-trace fields needed: condition
(feature/alpha), pid, correct, trace word count.

Why it discriminates where Countdown could not: on Countdown, steering raised
BOTH boundary (deliberation) and drift (evidence quality) — kinder to the paper
than the exhaust story. On MATH the two accounts finally separate:

- **exhaust story** (dialogic markers are the style of search, not its cause):
  boundary UP, drift DOWN at alpha=1.0
- **paper's story** (conversational feature drives reasoning): drift UP on MATH too

Pre-registered here, before anyone looks at those traces.

## 2. Censored-likelihood DDM (substantial local build, no new data needed)

Ceiling shares on the ladder are dose-dependent (7% baseline → 46% at
alpha=1.0), and the drop-ceiling sensitivity moves real credit between a and v
(RESULTS.md). The correct fix: treat near-ceiling traces as right-censored —
likelihood contribution P(RT > budget) via the WFPT survivor function — instead
of dropping or scoring them. Route: custom likelihood in HSSM/PyMC (analytic
WFPT density + numerically integrated CDF). This also likely explains the one
PPC misfit (fast-correct/slow-error asymmetry), since sv ~ 0.04 ruled out drift
heterogeneity (`fit_ddm_sv.py`).

Do this BEFORE trusting fine-grained a-vs-v credit on MATH, where baseline
truncation is 38%.

## 3. Cheap extensions, in descending value

- **Dose-dependent lapse**: p_outlier ~ C(alpha) (the alpha=1.693 condition is
  96% unparseable — partly a lapse process, currently fixed at 0.05).
- **BBH within-problem replication** of `fit_gpqa_diversity.py --mixed-only
  --center-within` (BBH has 2,666 traces in
  `results/qwq/hse_domains_v2.json` per_trace; check samples-per-problem first).
- **RL claim-B checkpoints as LaRT rows**: LaRT's regime is N ~ 100; the PPO
  eval checkpoints (25 per arm x 3 arms) get closer than the N=6 dose mapping
  ever could. Caution recorded in RESULTS.md: LaRT's latency channel misranks
  degenerate-length conditions.

## 4. Known sharp edges (do not rediscover these)

- `hse_recheck.json` vs `hse_recheck_rg.json`: only the `_rg` (re-graded) file
  has the corrected 24% baseline. Same per_trace order, different `correct`.
- pid on the ladder is reconstructed from row position; `build_dataset.py`
  asserts the alignment. Re-export gate_dose.jsonl with pid/sample to retire it.
- The all-problems GPQA fit does NOT converge (r_hat 3-4) because 79% of
  problems are always-right/always-wrong (complete separation of problem
  intercepts). Use `--mixed-only`. Grand-mean diversity centering leaks
  between-problem structure into the drift coefficient (+0.100 P=1.000 that is
  NOT real); use `--center-within`.
- Posterior `.nc` files are gitignored (30-50 MB, regenerable in ~20 s).
