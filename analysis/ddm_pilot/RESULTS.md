# DDM pilot on the Countdown steering dose ladder

*2026-08-17. Data: `per_trace` from `results/steering/hse_recheck_rg.json` (societies-of-thought
repo) — 1,200 traces, 6 doses × 200 problems. RT proxy = kilowords of trace; choice =
correct/error (accuracy coding). Fit with HSSM 
(analytical DDM likelihood), v and a per condition (log link on a), t ~ U(0, 0.105), z free.
Problem identity reconstructed from row order (blocked 6×200, same problem list per
condition per `run_sweep.py`; validated by pid-aligned cross-condition correlations
φ ≈ +0.15–0.39 vs +0.03 under shuffled alignment).*

## Result — full data (flat and hierarchical fits agree; r̂ = 1.00, 0 divergences)

| α | boundary a | drift v | accuracy |
|---|---|---|---|
| 0.0 | 0.827 [0.788, 0.864] | −0.746 [−0.891, −0.593] | 24.0% |
| 0.25 | 0.958 | −0.637 | 23.0% |
| 0.5 | 1.011 | −0.428 | 31.0% |
| 0.678 | 1.138 | −0.415 | 29.0% |
| 1.0 | 1.150 [1.105, 1.195] | −0.314 [−0.405, −0.220] | 34.0% |
| 1.693 | **5.264** [4.805, 5.742] | **−2.666** [−2.906, −2.420] | 3.5% |

Contrasts vs baseline: P(a↑) = 1.000 at every dose; P(v↑) = 0.83 / 0.996 / 0.999 / 1.000
at α = 0.25–1.0; at α = 1.693, P(v↑) = 0.000.

**Reading.** Steering *always* raises the boundary (more deliberation per problem, monotone
in dose). Through α = 1.0 it also modestly raises drift — evidence quality per kiloword
genuinely improves on Countdown, not just time-on-task. At α = 1.693 the picture inverts
catastrophically: the boundary explodes (5.3× baseline) while drift collapses to −2.67 —
in DDM terms the babble condition is not "more thinking", it is a model driven hard toward
the error bound while refusing to stop. The inverted-U in accuracy decomposes into a
monotone boundary effect plus a rise-then-collapse drift effect.

## Censoring sensitivity (near-ceiling traces >2,400 words dropped)

Ceiling shares are dose-dependent (7% at baseline → 46% at α = 1.0), so unmodeled
right-censoring is real. Dropping near-ceiling traces shifts the decomposition toward
drift: at α = 1.0, dA = +0.025 (P = 0.76, n.s.) while dV = +0.927 (P = 1.000), with
v turning positive (+0.211). The two fits bracket the truth (dropping long traces
mechanically deflates a and inflates v); a proper censored likelihood is the fix. The
α = 1.693 collapse (a ≈ 5, v ≈ −2.5) is invariant to the sensitivity.

## Posterior predictive check (HSSM simulator at posterior means)

Marginal accuracy and mean RT track the data at all six doses, including sim accuracy
0.000 at α = 1.693 (obs 3.5%). One systematic misfit: observed correct traces are much
*shorter* than predicted (0.38 vs 0.57 kw at baseline) and errors longer — the classic
fast-correct/slow-error asymmetry, which a plain DDM cannot produce and inter-trial drift
variability (sv, or richer problem-level structure) can. Expected, and it tells us the
next model term.

## What this means for the SoT critique

1. The pilot *works*: 19-second fits, clean convergence, interpretable decomposition of
   the inverted-U that raw accuracy cannot provide.
2. On Countdown the decomposition is *kinder to the paper* than the exhaust story
   predicts: moderate steering raises drift, not just boundary. On a search task,
   "trying more candidates" plausibly IS evidence accumulation, so this doesn't
   discriminate the hypotheses on its own.
3. **The discriminating experiment is the same fit on the MATH-Hard steered traces**
   (not in the public repo; on the pod/workstation). Exhaust story predicts: a↑, v↓ at
   α = 1.0 on MATH. Paper's story predicts v↑ there too. This is now a sharp,
   pre-registrable prediction.
4. Fixed lapse p = 0.05 should become dose-dependent (or estimated) — the degenerate
   condition is partly a lapse process (96% unparseable), and letting lapse soak it up
   would sharpen the v/a estimates at lower doses.

## Caveats

- pid alignment inferred from row order (validated statistically, not from raw jsonl —
  re-export `gate_dose.jsonl` with pid to make this exact).
- RT = words, not tokens; censoring unmodeled in the primary fit; z free (0.565, slight
  bias toward the correct bound); p_outlier fixed at 0.05 with U(0,20) outlier RT.
- Construct validity: this is a measurement model — parameters named "drift/boundary"
  earn their meaning from fit + prediction, not from any claim that transformers
  literally accumulate evidence.

## Files

`countdown_ladder.csv` (extracted dataset) · `fit_ddm.py` (model) · `idata_flat.nc`,
`idata_hier.nc`, `idata_flat_noceil.nc` (posteriors) · `*_summary.csv`, `*_params.csv`.
LaRT companion fit (`fit_lart.py`, dose-as-model mapping, N=6 caveat): `lart_fit.json`.

## LaRT companion fit (doses as "models", N=6 — outside LaRT's regime; illustrative only)

| α | θ (ability) | τ (speed; higher = shorter traces) | accuracy |
|---|---|---|---|
| 0.0 | −1.51 | +1.38 | 24.0% |
| 0.25 | −0.23 | +1.13 | 23.0% |
| 0.5 | +0.54 | +0.28 | 31.0% |
| 0.678 | +1.25 | −0.66 | 29.0% |
| 1.0 | +1.25 | −0.94 | 34.0% |
| 1.693 | **−0.63** | −1.38 | **3.5%** |

ρ(θ, τ) = −0.48 (slower ↔ more able, across these six rows).

**The instructive artifact:** LaRT ranks the degenerate α = 1.693 condition (3.5% accuracy)
*above* baseline (24%) on ability, θ = −0.63 vs −1.51. With only 6 rows, the negative
ability–speed correlation is dominated by the one extreme condition, and the latency channel
credits the babble condition's slowness as ability. The DDM, fit on the same data, attributes
the same trace length correctly: boundary 5.26 with drift −2.67. This is exactly the kind of
mechanism-agnostic-vs-mechanism-committed divergence worth reporting if either instrument is
used at scale — and a caution against LaRT-style latent-speed coupling in regimes with
degenerate/adversarial length distributions. (Not a criticism of LaRT in its intended
N ≈ 100 regime.)

## Reproducing

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python hssm
.venv/bin/python build_dataset.py          # regenerates countdown_ladder.csv (validates pid alignment)
.venv/bin/python fit_ddm.py                # flat fit, ~20 s on an M-series Mac
.venv/bin/python fit_ddm.py --hier         # + (1|pid) on v
.venv/bin/python fit_ddm.py --drop-ceiling # censoring sensitivity
```

Posterior `.nc` files (~30-50 MB each) are not committed; each fit regenerates
its own in under a minute. LaRT companion: `uv pip install` the clone of
github.com/Toby-X/Latency-Response-Theory-Model, then `python fit_lart.py`
(slow: ~20 min of SAEM; output in lart_fit.json).
