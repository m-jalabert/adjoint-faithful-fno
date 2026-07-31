# Model C long-term stability handoff

Date: 2026-07-29

## Decision

The selected Model C remains the best deterministic model for 10–30-day work
and is skillful through day 200 under the S0 control wind, but it is rejected
as a long-term emulator. Its 2,000-day ensemble develops a reproducible,
boundary-amplified global runaway.

The next study should pause the complete adjoint campaign and first establish
a reduced ten-output, Bire-facing forward baseline. The 46-versus-10-output
bottleneck is the most important major hypothesis not yet tested directly.

This does not mean a 2,000-day forecast must beat persistence or climatology
in trajectory RMSE. After decorrelation, the correct long-term target is a
bounded and statistically plausible ensemble. Persistence/climatology skill is
required over the deterministic 10–90-day interval; day 200 can remain a
reported extension.

## Saved final result

Slurm job 304754 completed normally in 1m06s. All nine manifest-listed
artifacts verify, and the scratch/project arrays and reports are byte-identical.

- Output package:
  `outputs/af_fno/C/anomaly_direct_s0_reflected_spectral_gate_control_v1/`
- Contract SHA-256:
  `69b0100aae2730473f96d9fb3f3bea660d6822e2f5fc25e1c45613832e2db8e7`
- Arrays SHA-256:
  `74827af0d9e627767a0f1ca1c6c5528936f510a120e3b97e14ca2545f036e164`
- Report SHA-256:
  `2b739a495fc81e9a6290164162a520324763c4b5d2621e05f08ef7eee243ec53`
- Summary SHA-256:
  `0e2f11d7fba351e2d1d0ba35c99d092de77c37c99d3e83f9a70379d61fc94a36`
- Report-content SHA-256:
  `84e7b3521a5436a49cc8d894281c385327f62ae0c00516463b72e96a238685c3`
- Manifest-content SHA-256:
  `018af80f1942c0a191a969ea256ea6ed04c67b6aa325b1394dfaeb940de0154d`

No spectral-gate strength passed the training-only gate, so the fixed
15-member inference phase remained sealed. At the weakest nonzero strength
(`alpha=0.25`):

- day-1,000 normalized amplitude/source: 0.4365;
- day-1,000 worst high-k power/source: 0.0638;
- day-1,000 worst RMSE/climatology relative to source: 0.6178;
- absolute normalized amplitude/truth: 3.65;
- absolute worst RMSE/climatology: 4.41;
- worst short RMSE/source: 6.49;
- SST day-10/30/90 RMSE/source: 5.30/6.49/5.94.

Removing 93.6% of the source high-band power therefore does not restore a
climate-like state and destroys useful short dynamics. High-k excess is a
coupled symptom and amplifier, not the sole cause or a safely removable
instability.

## Bire architecture versus selected Model C

| Choice | Bire recovered paper/archive configuration | Selected Model C |
|---|---|---|
| Grid | 248×248 at 0.25° | 62×62 including land rim; 60×60 wet at 1° |
| Reduced dynamic outputs | 10 | 46 |
| Dynamic variables | U surface/mid; V surface/mid; T surface/mid; PHIHYD surface/mid/bottom; barotropic streamfunction | all 15 U, 15 V, 15 temperature levels, plus SSH |
| Recovered input store | 10 dynamic channels plus wind | 46 dynamic plus wind, longitude, latitude, mask, wall distance = 51 |
| FNO blocks | 3 | 4 |
| Fourier modes | 64×64 | 24×16 |
| Fractional bandwidth | 64/248 = 0.258 in both directions | 24/62 = 0.387 and 16/62 = 0.258 |
| Latent width | 128 | 128 |
| Lift/projection width | 256/256 | 256/256 |
| Channel MLP | 4C = 512 | 4C = 512 |
| Latent/output ratio | 128/10 = 12.8 | 128/46 = 2.78 |
| Padding | none reported/archive default none | 10% |
| Block normalization | paper reports LayerNorm; executable archive is ambiguous | none |
| Channel-MLP dropout | paper silent; archive default 0.5 | effectively zero |
| State representation | pointwise temporal anomalies | pooled S0–S2 pointwise temporal anomalies with robust scale floors |
| Forecast parameterization | direct normalized future state | direct normalized future state |
| Objective | comparatively simple one-step, then two differentiable steps; MSE+MAE reported | three-step multi-term state/increment/rollout/spectral/boundary objective |

Bire's archived commands and experiment logs are incomplete, so paper/archive
differences in LayerNorm, dropout, learning rate, batch size, and MAE weight
must remain declared ambiguities rather than being blended into one
"faithful" recipe.

## What has been tried

### 1. Dense residual-state Model C and rollout-oriented losses

Channel-balanced state/increment losses, pushforward duration, truncated
unrolls, forecast-age supervision, bias/constraint terms, and
rollout-conditioned loss variants improved some 10–90-day metrics but did not
remove slow SST/SSH feedback. Fixed-bias removal explained only about 27–33%
of day-90 slow-field error, identifying feedback amplification rather than a
single static bias.

### 2. Bire-derived pointwise anomalies and direct future-state prediction

This was the largest positive change. All three seeds passed the training
gate, and seed 20260724 at step 13,440 became the selected Model C. Under S0,
the 15-member ensemble beats persistence and climatology for surface speed,
surface pressure anomaly, and SST through day 200. Day-200 RMSE is
0.00270 m/s, 0.0151 m²/s², and 0.0241 °C, respectively.

The result proves that representation/parameterization was a major source of
the earlier slow-field problem. It does not establish a stable invariant
measure.

### 3. Deep-pressure spectral diagnosis and regularization

The day-360 pressure failure was localized to a tiny high-wavenumber tail,
not excess integrated energy. A bounded spectral fine-tune improved
mid/bottom ratios from 13.37/5.57 to 6.19/2.97 and protected primary skill,
but did not pass the prospective gate.

### 4. Bire regularization and architecture controls

- LayerNorm reduced the later runaway substantially, but degraded short
  skill and remained 5.54/6.48/10.12 times climatology at day 2,000 for
  speed/pressure/SST.
- Dropout and LayerNorm+dropout were worse under the frozen training gate.
- A three-layer/no-padding model did not solve the pressure tail or long
  stability.
- Group-specific heads improved the mid/bottom pressure ratios to 9.21/4.40
  with only 5.5% worst primary degradation, but still failed.

These controls reject layer count, padding, LayerNorm, dropout, or output
heads as individual magic fixes. Group heads provide limited evidence of
multitask interference, but they retain the shared 46-channel burden.

### 5. Bire-style S0 long evaluation

Continuous MITgcm S0 truth was extended and six Figure-3–8 analogues were
generated for 15 fixed inference starts at a ten-day step.

- Day 200: the selected model beats both baselines for all three primary
  fields; U/V/pressure/SST ACC is 0.784/0.887/0.972/0.833.
- Day 2,000: speed/pressure/SST RMSE is 0.426/2.593/5.339.
- Fixed-member streamfunction RMSE reaches 96.2 Sv.

This separates good deterministic forward skill from failed long-term
statistical stability.

### 6. Boundary/checkpoint mechanism

All seven checkpoints run away. Earlier checkpoints cross the normalized
amplitude threshold sooner, so shorter training is not the cure. At the
selected checkpoint, the four-cell boundary band occupies 24.9% of wet cells
but contains 72.8/85.2/72.7% of day-2,000 Qx/Qy/streamfunction squared error.
The pattern recurs across all 15 starts, while boundary growth does not
consistently precede interior growth.

The best classification is a boundary-amplified global transport mode, not a
single bad initialization and not a proven boundary-to-interior cascade.

### 7. Stability and tangent diagnostics

During days 300–600, speed/pressure/SST RMSE grows by
1.0332/1.0397/1.0456 per ten-day call; normalized maximum amplitude grows by
1.0252 per call. All states can remain finite while being scientifically
catastrophic, so finiteness is not a useful stability gate.

The earlier residual model is vastly more unstable, reaching RMSE of order
10^25–10^27. LayerNorm selectively damps high-k tangent response but has
larger full/mid-band ten-call gains. The evidence does not support uniform
contraction or a blanket Jacobian penalty.

### 8. Fixed high-k postprocessing

A weak reflected 3×3 binomial smoother lowered day-1,000 high-k power,
amplitude, and RMSE. Its weakest strength missed the short guard narrowly
(1.0658 worst short ratio), and stronger smoothing damaged skill rapidly.

The scale-selective reflected spectral gate then removed almost all diagnosed
high-band excess while causing factors-of-6–21 short degradation and still
failing absolute long-term bounds. Fixed filtering is now closed.

## Positive conclusions

- The dataset, split, baselines, 15-member ensemble protocol, 2,000-day S0
  truth, and hash-bound reporting pipeline are in place.
- Pointwise anomaly normalization plus direct-state prediction is clearly
  superior to the earlier residual/global-normalization direction.
- The selected checkpoint is reproducibly excellent for short deterministic
  forecasts and remains useful as a reference.
- The long failure is ensemble-wide, checkpoint-wide, and mechanistically
  characterized; it is not an evaluation artifact.
- High-k content, boundaries, and normalization affect growth, but none alone
  explains or cures it.
- Tests now distinguish deterministic skill from long-term statistical
  stability, which prevents false promotion based on finite day-200 runs.

## Remaining challenges and ranked hypotheses

### 1. Output-channel capacity and conditioning — highest priority

The same width 128 represents ten Bire outputs but 46 Model C outputs. The
latent/output ratio is 4.6 times smaller. The dense model jointly learns
fast surface dynamics, very slow deep levels, and hydrostatic/barotropic
couplings. Quiet deep channels also require robust scale floors and can feed
small normalized errors back at every call.

This is plausible but not yet proven. Width 128 improved dense training fit,
and group heads helped somewhat, but neither test removes the 46-output task.
A direct ten-output comparison is required.

### 2. Short unroll versus invariant-measure learning

Three differentiable ten-day steps expose only 30 days, while instability
emerges after hundreds of days. Excellent local loss can coexist with a
slightly expansive off-manifold map. Training currently has no direct
requirement to saturate to a climate-like attractor over 200 calls.

### 3. Boundary/global transport coupling

Errors concentrate near walls and corners, but removing padding and changing
depth did not cure them. The boundary is an amplifier, not yet a demonstrated
root cause.

### 4. Resolution and unresolved physics

Bire trained at 0.25° on 248×248 fields; Model C uses 1° fields. Fractional
Fourier bandwidth is comparable, so simply adding modes is not motivated.
However, the coarse model has a different closure problem and may distort
wave/boundary propagation. This should be tested only after the channel
hypothesis at fixed 1° resolution.

### 5. Loss/regularization mismatch

The current multi-term loss strongly optimizes dense short-range skill.
Bire's simpler loss may reduce competing gradients, but Bire's own long runs
are not universally stable. Loss simplification should be tested within the
reduced model, not assumed to be a cure.

## Recommended next experiment

Use the same 1° MITgcm archive first so channel count can be tested without
changing the underlying truth.

### Arm R: reduced-channel causal control

Keep selected Model C's pointwise normalization, direct-state
parameterization, width 128, four blocks, 24×16 modes, padding, optimizer,
and three-step objective. Replace the 46-channel state by the ten recovered
Bire channels:

1. U surface;
2. U mid (vertical index 7);
3. V surface;
4. V mid;
5. T surface;
6. T mid;
7. PHIHYD surface;
8. PHIHYD mid;
9. PHIHYD bottom;
10. barotropic streamfunction.

This arm isolates output count and reduced-state conditioning.

### Arm B: Bire-facing architectural baseline

On the same reduced channels and data, use three FNO blocks, width 128,
lift/projection 256, Channel MLP 512, no padding, and 16×16 modes as the
1° fractional-bandwidth analogue of Bire's 64×64/248 setup. Use pointwise
training-only anomalies and direct future-state prediction. Start with the
declared simple MSE+MAE objective; keep paper LayerNorm and archive dropout as
separate bounded controls because their provenance is ambiguous.

Comparing Arm R with Arm B separates the channel hypothesis from the
remaining architecture/loss differences.

### Forward gates before adjoints

1. Training-only selection with exact reload and 10/30/90-day rollout checks.
2. Fixed 15-member S0 evaluation through day 200:
   primary RMSE/ACC for surface speed, surface pressure anomaly, and SST;
   require aggregate 10–90-day skill better than persistence and climatology.
3. Fixed 15-member S0 evaluation through day 2,000:
   require no sustained multiplicative runaway, normalized amplitude near
   truth, climate-like mean/variance and spectra, and realistic mean
   circulation/streamfunction.
4. Do not require 2,000-day trajectory RMSE below persistence or climatology.
5. Only after a reduced model passes both deterministic and statistical
   forward gates should the learned-physics and complete adjoint study reopen.

If the ten-output control is stable, add channels back in declared groups or
use a stable reduced dynamical core plus a diagnostic/conditional decoder for
the full 46-channel state. If it is still unstable, the channel hypothesis is
weakened and the next targets become long-unroll/invariant-measure training
and boundary-aware dynamics, followed by the 0.25° resolution difference.
