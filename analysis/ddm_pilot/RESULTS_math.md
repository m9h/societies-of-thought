# The pre-registered MATH-Hard fit: drift falls under steering

*Resolves `NEXT.md` §1. Data: `results/steering/main_rg.jsonl`, MATH-Hard, baseline
(feature −1) vs the paper's feature 30939 at α=1.0, 100 problems each, **fully paired**
(100 shared pids). Model specification identical to `fit_ddm.py` on Countdown — only
`--data` was added, so this is the pre-registered model, not a variant chosen afterwards.*

## The prediction, as written before these traces were touched

> - **exhaust story** (dialogic markers are the style of search, not its cause):
>   boundary UP, drift DOWN at alpha=1.0
> - **paper's story** (conversational feature drives reasoning): drift UP on MATH too
>
> — `NEXT.md` §1

Drift is the discriminator. The two accounts disagree about its sign.

## Result

| fit | n | boundary Δa | drift Δv | P(v↑) |
|---|---|---|---|---|
| flat | 200 | +0.156 (P=0.936) | **−0.336** | **0.001** |
| flat, near-ceiling dropped | 154 | +0.049 (P=0.686) | **−0.282** | **0.010** |
| **hierarchical, (1\|pid)** | 200 | +0.271 (P=0.969) | **−0.460** | **0.000** |

All three converged (r̂ = 1.00).

**Drift falls under steering on MATH-Hard, decisively and robustly.** The paper's story
predicted the opposite sign. Accuracy corroborates it directly: 62.0% → 40.0%.

## Why the censoring sensitivity matters here

On Countdown, dropping near-ceiling traces *shifted credit toward drift* (dV = +0.927 at
α=1.0). That is the direction which would rescue the paper's account, so it is the
sensitivity that could overturn this result. MATH censoring is dose-dependent in the same
way — 15% of baseline traces exceed 2,400 words versus 31% when steered.

It does not overturn it. Dropping those traces moves dV from −0.336 to −0.282 and P(v↑)
from 0.001 to 0.010: the drift decline survives the correction that works against it.

## What does NOT survive

**The boundary half of the exhaust prediction is not established.** Δa runs +0.156
(P = 0.936) on the flat fit but +0.049 (P = 0.686) once near-ceiling traces are dropped.
Under censoring correction the boundary effect is not distinguishable from zero. The
hierarchical fit puts it back at +0.271 (P = 0.969), so the three fits disagree, and the
honest summary is that **only the drift direction is established**.

That is enough. The pre-registration made drift the discriminator precisely because both
accounts predict more deliberation; only the paper's account predicts better evidence.

## Reading

On Countdown the decomposition was kinder to the paper — steering raised both boundary and
drift, and the pilot's author flagged that a search task cannot separate the hypotheses,
since "trying more candidates" plausibly *is* evidence accumulation.

MATH-Hard has no such escape. Steering the conversational-surprise feature makes traces
longer and their evidence *worse* per kiloword. The dialogic behaviour is induced — our
earlier work confirmed markers rise — while the quality of what the trace accumulates
falls. That is the exhaust account stated in DDM coordinates: the markers are the style of
the search, and driving them harder degrades the search.

It also sharpens the existing MATH result. `results/steering/FINDINGS.md` reports the same
feature at the same dose costing −22 points on MATH-Hard while gaining +10 on Countdown.
The DDM says *what* was lost: not deliberation, which if anything increased, but evidence
per unit of deliberation.

## Matched controls: the decline is specific to the conversational feature

`main_rg.jsonl` also holds six matched-control features at α=1.0 on MATH (n=100 each),
selected in the original sweep to match feature 30939 on sparsity and max-activation. Same
fit, same spec:

| feature | Δ drift | P(v↑) | accuracy |
|---|---|---|---|
| **30939 (the paper's)** | **−0.336** | **0.001** | **40%** |
| 26919 | −0.141 | 0.084 | 53% |
| 20402 | −0.128 | 0.112 | 54% |
| 10126 | −0.097 | 0.160 | 56% |
| 3114 | −0.050 | 0.321 | 59% |
| 5993 | −0.014 | 0.457 | 61% |
| 22600 | +0.006 | 0.516 | 62% |

**Not one control reaches significance.** Control Δv averages −0.071 (sd 0.061, range
−0.141 to +0.006); the paper's feature sits **4.4 control-SD below the control mean**.
Control accuracy averages 57.5% against a 62% baseline — a mild generic cost of steering
anything — while the paper's feature drops to 40%.

This matters, and it cuts both ways:

- **For the paper:** feature 30939 is *not* an arbitrary direction. Steering it does
  something specific and large that matched controls do not do. The authors found a real
  causal handle, which our earlier matched-control work (DiD −2.9% on GPQA) understated.
- **Against the paper:** what that handle specifically does on MATH-Hard is degrade
  evidence quality per unit of deliberation. The feature is special, and its special
  effect is to make multi-step reasoning worse.

The honest one-line version: *the conversational feature is a real, specific lever, and on
MATH it is a lever in the wrong direction.*

## Caveats

- **n = 100 per condition**, one feature, one dose. Countdown had six doses × 200.
- **The boundary estimate is unstable across fits** (see above) — do not quote Δa.
- **Censoring is unmodeled in the primary fit.** `NEXT.md` §2 (censored likelihood via the
  WFPT survivor function) applies here at least as strongly as on Countdown: baseline
  truncation is 15% and steered 31%. The two fits bracket rather than resolve it.
- **RT = words, not tokens**; lapse fixed at 0.05; z free (0.58).
- **This is a measurement model.** "Drift" and "boundary" earn their meaning from fit and
  prediction, not from any claim that transformers literally accumulate evidence.
- **Control comparison is across-feature, not a formal test.** The "4.4 control-SD" figure
  treats six point estimates as a reference distribution; it is a descriptive contrast, not
  a hierarchical model over features. It is enough to say the paper's feature is an outlier
  among its own matched controls, not to put a p-value on it.

## Correction to `NEXT.md`

§1 states the MATH traces "live on the pod/workstation, not in this repo". They are in the
repo: `results/steering/main_rg.jsonl` holds 800 MATH-Hard rows across baseline and seven
features at α=1.0. It also carries **real `pid`**, so the position-as-problem reconstruction
that `build_dataset.py` has to validate statistically is unnecessary here — the design is
paired exactly.

## Reproducing

```bash
uv pip install hssm
python build_math_dataset.py                                   # writes math_pair.csv
python fit_ddm.py --data math_pair.csv --tag math              # flat
python fit_ddm.py --data math_pair.csv --tag math --drop-ceiling
python fit_ddm.py --data math_pair.csv --tag math --hier       # paired on 100 pids

for F in 3114 5993 10126 20402 22600 26919; do            # matched controls
  python build_math_dataset.py $F
  python fit_ddm.py --data math_pair_f$F.csv --tag math_f$F
done
```
