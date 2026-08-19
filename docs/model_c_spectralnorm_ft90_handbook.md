# Production emulator and its ninety-day fine-tune

A single operator, `F_theta: [x_t, S] -> x_{t+10}`, trained once from random
initialization and then continued for a short second stage at a longer
autoregressive horizon. This document describes the architecture, the
training / validation / inference protocols, and the measured results on both
sides of that second stage.

Lineage:

```
model_c_production_1in_1out_spectralnorm_v1        (parent, from scratch)
        |  selected.pt @ step 7,680   sha e7595168...
        v
model_c_production_1in_1out_spectralnorm_ft90_v1   (child, 90-day fine-tune)
           selected.pt @ step 1,440   sha 4acb7633...
```

The parent is frozen and stays published. The child changes four things —
initialization, rollout length, learning rate, step count — and **nothing about
the architecture**.

---

## 1. The map

```
x_t = [U_1:15, V_1:15, Theta_1:15, eta]     46 channels   prognostic state
S   = [tau_x, wet, f, dx, theta_clim]        5 channels   physical statics
                                            --
external input                              51 channels
+ Bire sine/cosine position encoder           2 channels  (added inside model)
                                            --
lifting input                               53 channels
output x_{t+10}                             46 channels
```

Grid `62 x 62`, one degree, MITgcm `tutorial_baroclinic_gyre` double gyre.
Prediction interval `dt = 10 days`. The output is the **state**, not a residual.

The state is deliberately *closed*: all fifteen vertical levels of `U`, `V` and
`Theta` plus the free surface. The purpose of this emulator is a self-contained
prognostic map whose Jacobian can serve as a tangent/adjoint operator, which a
reduced three-level state would not give.

### Static channels

| channel | source |
| --- | --- |
| `wind_stress_x` | trajectory store, pooled wet-cell standardization |
| `wet_mask` | trajectory store, raw 0/1 indicator |
| `coriolis_parameter` | derived, `f = 2 omega sin(phi)` from the store's latitude |
| `zonal_grid_spacing` | MITgcm's own `DXF`, read not recomputed |
| `sst_relaxation_target` | MITgcm's `thetaClimFile`, wet-cell standardized |

The three derived channels are built from the simulation's own inputs, each
pinned by SHA-256, and the `data` namelist is parsed so a setup change (an
f-plane override, a different `thetaClimFile`) is a hard failure rather than a
silently wrong channel.

`sst_relaxation_target` is standardized on its own rather than in the surface
temperature channel's pointwise coordinates. The elegant alternative would make
the relaxation difference exactly representable by one lifting weight pair, but
the pointwise scale falls to ~0.011 degC where SST is pinned, which would put
that channel two orders of magnitude above every other input into the same 1x1
lifting.

---

## 2. Architecture

A 32x32-mode, width-128, three-block FNO with a parallel local correction.

```
input 51ch
  |
  +-- Bire position encoder  ->  53ch
  |
  lifting  53 -> 256 -> 128
  |
  [ FNO block ] x 3
  |     h -> LayerNorm -> GELU( SpectralConv(h) + W h ) -> LayerNorm
  |       -> ChannelMLP 128 -> 512 -> 128  (+ skip)
  |
  projection 128 -> 256 -> 46
  |
  + local 3x3 conv, bias-free, 51 -> 46      (parallel path from raw input)
  |
output 46ch
```

| setting | value |
| --- | --- |
| Fourier modes | `32 x 32` (`ky x kx`) |
| hidden width | 128 |
| blocks | 3 |
| lifting / projection width | 256 (`ratio 2`) |
| ChannelMLP expansion | `4C` -> 512 |
| pointwise LayerNorms | 6 (two per block, over channels at each grid point) |
| domain padding | 10 % |
| local branch | one bias-free `3x3`, zero-initialized at construction |
| dropout | 0.0 |
| precision / factorization | dense float32, none |
| **parameters** | **27,297,960** |

### Where the parameters live

| component | parameters | share |
| --- | ---: | ---: |
| spectral convolutions | 26,739,072 | 97.95 % |
| ChannelMLPs | 395,136 | 1.45 % |
| FNO pointwise skips `W` | 49,152 | 0.18 % |
| lifting | 46,720 | 0.17 % |
| projection | 44,846 | 0.16 % |
| local `3x3` | 21,114 | 0.08 % |
| LayerNorms | 1,536 | 0.01 % |

Almost the entire model is the Fourier operator, which is why the one
constraint applied to it matters as much as it does.

### Per-mode spectral normalization

Inside a `SpectralConv` the channel mixing at Fourier mode `k` is a dense
complex matrix `R_k` in `C^{128x128}`; there are `32 x 17 = 544` per block,
**1,632** in total. Each is capped:

```
R_k  <-  R_k * min(1, 1 / sigma_max(R_k)),        rho = 1
```

following McCabe et al., *Towards Stability of Autoregressive Neural
Operators* (arXiv:2306.10619). Properties that matter:

- **One-sided.** A mode already at or below one is left exactly alone, and its
  scale is then a constant, so no gradient flows through it.
- **Estimated, not solved.** A full SVD of 1,632 matrices per step is out of the
  question, so each mode carries a persistent pair of singular-vector estimates
  advanced by two power iterations per forward, with a 400-iteration warmup at
  construction. Power iteration converges from below, so the residual error is
  always an overshoot: measured post-cap `sigma_max` is 1.0350 after 24
  iterations, 1.0050 after 100, 1.00009 after 400.
- **It does not bound the emulator Jacobian.** Each block is
  `GELU(K_R h + W h)` with its own skips, a ChannelMLP and the local branch, so
  `sigma_max(R_k) <= 1` does **not** imply `sigma_max(J_F) <= 1`. Legitimate
  transient amplification stays representable. This is a constraint on one
  constituent, not a declaration that perturbation growth is always wrong.
- **Materialized on write.** Checkpoints bake the normalized weights into the
  tensor, so a published checkpoint loads into a plain `ProductionFNO` and the
  inference layer is exactly `y_hat(k) = R_k_tilde x_hat(k)`, whose adjoint is
  `R_k_tilde^H` — no clipping, no stochastic branch, nothing that would corrupt
  the sensitivities.

Adds 0 parameters.

### Deliberately rejected

| alternative | why not |
| --- | --- |
| 32 -> 24 modes | the day-2000 high-`k` power fraction is already ~0.004–0.014 against truth 0.005–0.028; the operator is too smooth, not contaminated. Truncating would stabilize by deleting structure already missing. |
| 15 -> 3 levels | Bire's reduced state suits a forecasting problem; this one needs a prognostically closed state for the adjoint interpretation. |
| change `dt = 10 d` | the strongest point of agreement with Bire: shorter intervals need many more recursive calls, longer ones misrepresent wave evolution. |
| `tanh` stabilizer | saturating regions drive `tanh' -> 0`, buying bounded 2,000-day rollouts by destroying the sensitivities the programme exists to measure. |
| tighten cap to `rho = 0.99` | clean and cheap, and the declared *next* experiment — but bundling it with the training-protocol change would leave neither attributable. |

---

## 3. Data and split

Trajectory store `trajectories_v3.zarr`: 3 regimes (S0/S1/S2) x 9,000 days x 46
channels x 62 x 62. The Bire Section 3.2 arrangement is applied in memory:

| block | days | role |
| --- | --- | --- |
| train | 0 – 5,999 | training pairs, normalizers, climatology |
| validation | 6,000 – 7,199 | checkpoint selection |
| inference | 6,200 – 7,199 | held evaluation (nested inside validation) |
| evaluation truth only | 7,200 – 8,999 | lead-matched MITgcm truth, never a model input |

The model sees nothing at or beyond index 7,200. Selection starts are drawn from
the 200 validation days *outside* the inference block, so no checkpoint is
selected on a start it is later scored on. There is no independent third test
split — Bire's protocol has none either; this is a nested validation/inference
arrangement and is labelled as such everywhere.

### Normalization

```
x_hat_c(y,x) = (x_c(y,x) - mu_c(y,x)) / sigma_c(y,x),      land set to zero
```

Pointwise `(46, 62, 62)` mean and scale, computed from **training days only**.
The scale is floored per channel at its own 5th percentile over wet cells, so a
point where a field barely varies (pinned surface temperature) cannot divide by
an almost-zero standard deviation. A per-channel RMS of the normalized ten-day
increment is also computed; it is the divisor of the objective's increment term.

---

## 4. Objective

Eight terms, every one dimensionless, every one giving the four physical groups
(`U`, `V`, `Theta`, `eta`) equal status so no group's dimensional scale decides
its own optimizer weight.

```
L = L_state
  + 0.001  L_increment
  + 0.50   L_rollout
  + 1e-5   L_spectral
  + 0.065  L_boundary
  + 0.05   L_pressure_gradient
  + 0.05   L_continuity
  + 0.05   L_barotropic_transport
```

| term | definition |
| --- | --- |
| `state` | group-balanced masked relative L2 at 10 days, `E_10` |
| `increment` | group RMSE of the 10-day increment error, scaled by the training-only per-channel increment RMS |
| `rollout` | mean of `E_{10k}` over the remaining calls |
| `spectral` | 12-bin radial amplitude relative L2 of 10-day increments, on the exact wet rectangle after a Hann taper |
| `boundary` | relative L2 on the first 4 wet cells east of the western wall |
| `pressure_gradient` | relative L2 of the horizontal gradient of PHIHYD reconstructed from `THETA` and `ETAN`, 15 levels, equal x/y weight |
| `continuity` | truth-referenced residual `d(eta)/dt + div(Q)`, chained exactly like the rollout |
| `barotropic_transport` | truth-referenced relative L2 of the 10-day tendency of the depth-integrated transport |

The three physics terms are **truth-referenced**, not penalties on a vanishing
residual: ten-day sampling only approximates MITgcm's discrete operators, so the
exactly-balanced residual is not zero, and asking the prediction to reproduce
the *truth* residual keeps the term honest about that discretization gap. They
add no output channels.

All eight are active from optimizer step one in both stages. The only thing the
fine-tune changes is the length of the sequence they are evaluated over.

---

## 5. Training

### Parent — from scratch

Nothing inherited: random initialization, both normalizers recomputed over
training days 0–5,999, Adam cold, no parent checkpoint, no staged fine-tuning.

| | |
| --- | --- |
| rollout | 6 calls, **60 days**, no teacher forcing after `x_t` |
| `L_state` / `L_rollout` | `E_10` / mean of `E_20..E_60` |
| optimizer | Adam, betas (0.9, 0.95), weight decay 0, no gradient clipping |
| learning rate | 5e-4 for steps 1–5,760, then 1e-4 |
| batch | 8 = microbatch 4 x accumulation 2 |
| steps | 7,680 |
| state transitions | 7,680 x 8 x 6 = **368,640** |
| training sequences | 17,820 (5,940 starts per regime) |
| seed | 20260724 |
| wall time | 3.21 h on one V100-32GB |

### Child — ninety-day fine-tune

| | parent | child |
| --- | --- | --- |
| initialization | random | parent `selected.pt` @ 7,680 |
| rollout | 6 calls / 60 d | **9 calls / 90 d** |
| learning rate | 5e-4 -> 1e-4 | **5e-5, constant** |
| steps | 7,680 | **1,920** |
| microbatch x accumulation | 4 x 2 | 2 x 4 |
| state transitions | 368,640 | **138,240** (37.5 %) |
| training sequences | 17,820 | 17,730 (5,910 per regime) |
| seed | 20260724 | 20260817 |
| wall time | 3.21 h | **1.67 h** |

Everything else is identical: architecture, static channels, modes, width,
blocks, local branch, spectral cap at `rho = 1`, dataset, split, normalization,
batch size, Adam betas, weight decay, gradient clipping, validation protocol,
selection rule.

```
L_state   = E_10
L_rollout = (1/8) sum_{k=2}^{9} E_{10k}          # 20, 30, ..., 90 days
```

**Why 90 days.** Bire et al. estimate the double gyre decorrelates in roughly 90
days. Past that the emulator is not expected to reproduce a particular MITgcm
trajectory, only to stay on the same statistically stationary attractor —
so training against exact `x_{t+500}` would optimize an impossible deterministic
target whose cheapest minimizer is *suppressed variability*, the opposite of
what is wanted. Ninety days is the longest horizon at which pointwise truth is
still a defensible objective.

**Why 5e-5, constant.** The parent already knows the ten-day propagator; this
stage adjusts where its autonomous trajectory settles. A rate able to relearn
the propagator would dissolve the thing being fine-tuned. Constant because 1,920
steps at a rate already five times below the parent's terminal value has nothing
left to anneal, and a decay would confound "converged" with "stopped moving".

**Why cold Adam.** The parent's second moments were accumulated at 5e-4;
inheriting them would set the effective first-step size from the parent's
schedule instead of from this contract.

**Why the normalization is reused byte-for-byte.** The operator was trained in
those coordinates. Recomputing them — even from the same days to the same recipe
— would move the coordinate system the trained operator lives in, for no gain.
The published child artifact is a byte copy of the parent's and hashes to the
same value (`fe424b37...`).

**Why 30 starts per regime are lost.** Nine calls need the full 90-day target
sequence inside the training block, so the latest usable start moves from day
5,939 to day 5,909 (whose final target is day 5,999). 5,910 divides by the
microbatch of 2 without remainder, which is what makes gradient accumulation
arithmetically identical to a single batch of 8.

**Function-preserving load, verified.** The parent checkpoint is materialized, so
it loads into a plain model that *is* the published operator. Reinstalling the
cap on already-capped weights moved the forward map by
`8.6e-4` of its own output RMS (tolerance `1e-2`), so the fine-tune demonstrably
starts from the published function. Per-mode `sigma_max` at load: 1.0025 /
1.0020 / 1.0006 across the three blocks.

---

## 6. Validation and checkpoint selection

Every checkpoint is autoregressed **360 days** at ten-day steps from all 102
pooled validation starts (34 per regime, days 6,000–6,198, stride 6) and scored
against lead-matched MITgcm truth, with persistence and a train-only climatology
as the two reference curves.

Selection rule (identical in both stages):

1. compute the 10–90-day RMSE-AUC of each primary field
   (`surface_speed`, `sst`, `phihyd_surface`);
2. keep checkpoints within 5 % of the best short AUC in **every** field;
3. keep only those whose measured perturbation growth rate is `<= 1.0`;
4. among those, minimize the worst 90–360-day RMSE-AUC ratio to climatology.

Exactly one checkpoint is published as `selected.pt`.

Condition 3 exists because an earlier arm was blind to it: all four of its
checkpoints had growth rates above one and an RMSE-only rule still returned one
of them, which then amplified 28x over the 200-call inference. *A criterion that
cannot see the failure cannot avoid selecting it.*

`lambda_hat` is the composite per-call growth rate of a finite twin
perturbation, fitted on calls 51–200 of `log||x_twin - x||` from a 1e-2 relative
initial perturbation.

### Conditions added for the fine-tune

Three inherited, two new:

| condition | ceiling |
| --- | --- |
| short AUC within 5 % of this run's best, every field | 1.05 |
| **short AUC within 5 % of the parent, every field** | 1.05 |
| worst 90–360 d ratio to climatology | 0.85 |
| **90–360 d curve flattening**, `(E_360-E_270)/(E_180-E_90)` | 1.0 |
| perturbation growth rate `lambda_hat` | 1.0 |

The flattening ratio compares two 90-day secant slopes of the mean RMSE curve.
Below one is a curve bending toward a plateau; above one is a curve still
accelerating away. It is a statement about the *shape* of the approach to
climatology — the thing the fine-tune targets.

---

## 7. Inference protocol

Held evaluation only; no training, no selection, no promotion. 15 members drawn
from the inference block by a fixed seed (starts 6,263 … 6,979), autoregressed
to **day 2,000** at ten-day steps. Every member has lead-matched MITgcm truth
through day 2,000. Six figures: streamfunction structure at days 0–40, RMSE
0–200 d, single-member RMSE, ACC 0–200 d, streamfunction at day 60 and day
2,000, RMSE 0–2,000 d.

The child's figure package uses the **same 15 members, same seed, same leads,
same fields** as the parent's — a fine-tune scored on different starts would not
be comparable to the model it fine-tuned — and pins the parent's sealed figure
summary as a digest-verified artifact so the comparison is against published
bytes rather than a remembered number.

A companion anomaly package removes MITgcm's own two-dimensional time-mean
barotropic streamfunction over training days 0–5,999:

```
psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)
```

applied identically to truth and prediction. The model's own mean is never
subtracted, so a bias in the stationary circulation cannot hide inside the
anomaly. Both packages remove the **same** field, which is what makes the two
arms' transient amplitudes comparable.

---

## 8. Results

Selected checkpoints: parent step **7,680**, child step **1,440**.

### 8.1 Forecast skill, 10–90 days (RMSE-AUC, lower is better)

| field | parent | fine-tuned | ratio |
| --- | ---: | ---: | ---: |
| surface speed | 0.0808 | **0.0710** | 0.879 |
| SST | 0.8210 | **0.7360** | 0.896 |
| surface pressure | 0.5791 | **0.2671** | 0.461 |

Short-horizon skill *improved* in all three fields. The 5 % degradation budget
was never approached — this is the trade the gate exists to refuse, and it was
not made.

### 8.2 Day-200 anomaly correlation

ACC here is an **uncentered pattern correlation over all wet surface cells**,
taken about the pointwise train-only *time-mean* climatology, unweighted by cell
area, averaged over the 15 inference members. Three properties of that
definition have to be stated before the numbers mean anything.

**Its no-skill value is not zero.** Because the anomaly is taken about a
time-mean rather than a day-of-year climatology, whatever stationary spatial
structure survives that subtraction is counted as signal. Figure 6 therefore
carries two truth-only references: each member's persistence, and fifteen
independent states drawn from the inference block — a forecast that knows the
climate but nothing whatever about the ocean state.

| field | parent | fine-tuned | persistence | **no-information floor** | skill above floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| surface pressure | 0.9810 | **0.9977** | 0.9467 | **0.8914** | 0.979 |
| SST | 0.9466 | **0.9595** | 0.2668 | −0.0095 | 0.960 |
| surface `u` | 0.9596 | **0.9741** | 0.1113 | −0.0961 | 0.976 |
| surface `v` | 0.9434 | **0.9671** | 0.4463 | −0.0999 | 0.970 |

(15-member S0 inference protocol, matching figure 6. The 102-member pooled
S0/S1/S2 validation rollouts give closely similar values.)

**Surface pressure is the one number not to quote unqualified.** Two entirely
unrelated states already correlate at 0.891, because **90.5 % of the surface
pressure anomaly power about the training mean is a fixed spatial offset** —
a standing pattern plus slow drift between the training and inference blocks —
and only 9.5 % is transient. The usable range is 0.891 → 1.0, not 0 → 1. The
`skill above floor` column rescales onto that range. For SST, `u` and `v` the
floor is ≈ 0 (only 13.4 % of SST anomaly power is a fixed offset), so those
values are genuine large skill: persistence manages 0.11 – 0.45 at the same lead.

**These curves are flat because S0 is not chaotic.** The S0 SST anomaly field
pattern-correlates with itself at **0.992 after 171 days** (0.899 at 342, 0.980
at 513): the trajectory is on a near-periodic attractor and nearly repeats every
~171 days. This is visible directly in figure 6, where persistence collapses to
below zero by day 50 and then *recovers to ≈ 0.95 at day 170* as the cycle comes
back into phase. There is consequently no butterfly-effect predictability
horizon in S0 — a model that learns the limit cycle can stay phase-locked
indefinitely, and all error is model error rather than chaotic divergence.

**So this is not comparable to Bire's Figure 6.** Their ACC decays over ~90 days
because their configuration is genuinely eddying and decorrelates; ours stays
flat because the system itself is predictable, not because the emulator is
better. Read the fine-tune's gain here as "improved phase-locking to a periodic
attractor", not as forecast skill against a chaotic ocean. A Bire-comparable
decay can only be measured on the turbulent 0.25° S0/S1/S2 campaign.

A secondary caveat: no `cos(phi)` area weighting is applied, so northern cells
are over-weighted by up to 3.61x relative to an area-weighted score.

The improvement over the parent is real in every field and largest exactly where
the floor is lowest, but note that flat ACC at 200 days sits alongside a
day-2,000 RMSE that is still rising (§8.5) — the two are consistent: phase-locked
over hundreds of days, slowly drifting in amplitude over thousands.

### 8.3 Long horizon, 90–360 days

| metric | parent | fine-tuned |
| --- | ---: | ---: |
| worst RMSE-AUC / climatology | 0.309 | **0.274** |
| flattening, pressure | 0.955 | **0.892** |
| flattening, SST | 1.729 | **0.876** |
| flattening, speed | 1.648 | **1.077** |

The parent's curve was steepening in two of three fields. The child's decelerates
in two of three; surface speed at 1.077 is the one that still does not.

### 8.4 Stability

| metric | parent | fine-tuned | ceiling |
| --- | ---: | ---: | ---: |
| `lambda_hat` per call | 1.01322 | **1.00831** | 1.0 |
| max normalized amplitude, 360 d | 5.257 | 5.398 | — |

Improved, but still above one, so both runs select through the declared fallback
branch (`no checkpoint met the growth rate ceiling`) rather than the primary
rule. Growth by child checkpoint: 480 -> 1.0127, 960 -> 1.0068,
1,440 -> 1.0083, 1,920 -> 1.0097.

### 8.5 Day-2,000 held inference

| metric | parent | fine-tuned | target |
| --- | ---: | ---: | ---: |
| max normalized magnitude | 10.328 | **8.434** | <= 8 |
| RMSE / climatology, pressure | 0.478 | **0.248** | ~1 or below |
| RMSE / climatology, SST | 1.425 | **0.924** | ~1 or below |
| RMSE / climatology, speed | 1.764 | **1.349** | ~1 or below |
| spatial std ratio to truth | 0.9972 | 0.9954 | 0.80–1.25 |
| streamfunction minimum (Sv) | -29.39 | -29.30 | >= -33 (truth -30.01) |
| all values finite | yes | yes | yes |

Absolute day-2,000 RMSE: speed 0.00633 -> 0.00484, SST 0.05765 -> 0.03740,
pressure 0.03941 -> 0.02045.

SST crossed from *worse* than climatology to below it. Pressure roughly halved.
Amplitude fell 18 % but still exceeds the ceiling of 8.

### 8.6 Day-2,000 streamfunction anomalies

| metric | parent | fine-tuned | truth |
| --- | ---: | ---: | ---: |
| anomaly RMS ratio, day 60 | 1.145 | **1.064** | 1.0 |
| anomaly RMS ratio, day 2,000 | 1.795 | 1.738 | 1.0 |
| western boundary / interior | 5.06 | **8.16** | 23.10 |
| high-`k` power fraction, zonal | 0.0122 | 0.0141 | 0.0051 |
| high-`k` power fraction, meridional | 0.0038 | 0.0022 | 0.0276 |

The western boundary is markedly better resolved — 5.06 -> 8.16 against a truth
value of 23.10 — though still under-represented. Transient amplitude at day
2,000 remains ~1.7x too large, essentially unchanged.

### 8.7 Acceptance gate

| half | parent | fine-tuned |
| --- | --- | --- |
| validation conditions | fail (growth 1.0132 > 1) | fail (growth 1.0083 > 1; speed flattening 1.077 > 1) |
| 2,000-day conditions | fail (magnitude 10.33 > 8) | fail (magnitude 8.43 > 8) |
| finite / std ratio / streamfunction min | pass | pass |
| gyre identifiable, western boundary sharp | pass | pass |

---

## 9. What this experiment did and did not establish

**Established.** A short, cheap second stage — 37.5 % of the parent's exposure,
1.67 h — that changes only the autoregressive horizon improves essentially every
measured quantity at once, and does so *without* the usual trade: short-horizon
skill got better, not worse. Day-2,000 SST moved from worse-than-climatology to
below it, pressure error halved, amplitude fell 18 %, the western-boundary /
interior ratio rose from 5.06 to 8.16 against a truth of 23.10, and the
90–360-day error curve began bending over in two of three fields. The double
gyre at day 2,000 remains clean and identifiable, with sharp western boundary
intensification and a spatial std ratio of 0.995.

**Not established.** The emulator still does not *plateau*. Figure 8 shows RMSE
rising through day 2,000 in all three fields, with surface speed crossing
climatology around day ~1,400. Three declared conditions still fail:
`lambda_hat` 1.0083 > 1, speed flattening 1.077 > 1, and max normalized
magnitude 8.43 > 8. The day-2,000 anomaly RMS ratio barely moved (1.795 ->
1.738), so the transients are still ~1.7x too energetic.

The distinction that matters: the goal is **not** `x_hat_2000 ~ x_2000`
pointwise, which is not expected past the ~90-day decorrelation time. It is that
the error curve settles near the error between two independent states drawn from
the same climate. The fine-tune moved the model a long way toward that and did
not reach it.

**Declared next experiment.** The same ninety-day protocol with the spectral cap
tightened to `rho = 0.99` — a small, adjoint-clean modification, deliberately
not bundled with this run so the two remain attributable.

---

## 10. Reproducing

```bash
# stage 1: fine-tune the published parent
sbatch slurm/models/c/train_production_1in_1out_spectralnorm_ft90_v1.sbatch

# stage 2: S0 figures 3-8 and the 2,000-day gate
sbatch slurm/models/c/figures_production_1in_1out_spectralnorm_ft90_v1.sbatch

# stage 3: streamfunction anomaly companions
sbatch slurm/models/c/anomaly_production_1in_1out_spectralnorm_ft90_v1.sbatch
```

Each stage runs `finalize` (where applicable), `preflight`, then `run`. Every
contract pins its inputs by SHA-256 and every stage refuses to start against a
changed source, a half-written artifact, or a checkpoint whose recorded lineage
does not match.

### Code layout

| module | role |
| --- | --- |
| `train.py` | from-scratch production training, selection |
| `finetune.py` | the ninety-day staged fine-tune |
| `figures.py` / `anomaly.py` | the parent's held evaluation |
| `figures_ft90.py` / `anomaly_ft90.py` | the child's, as lineage adapters |

The three `_ft90` modules **import** their numerics from the production modules
and reimplement only provenance — which arm produced the checkpoint, which
sealed package feeds the next stage. `train.py`, `objective.py`, `figures.py`,
`anomaly.py` and `plots.py` are byte-identical to the hashes the parent's frozen
contracts pin, so the parent remains reproducible and its day-2,000 numbers stay
a fixed baseline rather than a moving one. Two test modules assert exactly that.

### Artifacts

```
outputs/af_fno/C/
  model_c_production_1in_1out_spectralnorm_ft90_v1/              training report, arrays, selection plot
  model_c_production_1in_1out_spectralnorm_ft90_v1_s0_figures_v1/
      S0/                                                        figures 3-8, arrays, summary, CSV
      ..._acceptance_gate.json                                   both gate halves + parent comparison
  model_c_production_1in_1out_spectralnorm_ft90_v1_s0_anomaly_v1/
      S0/                                                        figures 3a, 7a, reference mean
```

Contracts in `config/`, launchers in `slurm/models/c/`, tests in
`tests/test_finetune.py` and `tests/test_finetune_evaluation.py`.
