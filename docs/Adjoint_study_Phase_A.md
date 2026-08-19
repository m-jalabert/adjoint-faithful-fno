# Adjoint study, Phase A — the production emulator against MITgcm

Compare the sensitivity of a scalar SSH objective to the initial sea surface, taken
through the **published production emulator** and through **MITgcm/TAF**, on the same
window, the same trajectory and the same functional, out to a ninety-day lead.

```
S_L[j,i]  =  dJ_L / d eta(j, i, day 7200)              L in {10, 20, 30, 90} days
```

Two independent producers of `S_L`:

| | producer | cost |
| --- | --- | --- |
| reference | MITgcm c68j differentiated by TAF 6.8.11 | ~20 min per lead per objective |
| test | `model_c_production_1in_1out_spectralnorm_ft90_v1`, `selected.pt` @ step 1,440, `4acb7633…` | ~seconds |

This document supersedes `docs/fno_adjoint_plan.md` (since deleted) and extends
`docs/mitgcm_adjoint_ground_truth_plan.md`, which remains valid for what it produced and
is not edited here.

---

## Status — 2026-08-19

Steps 0-5 are implemented and run; steps 6-13 are staged and queued. Three things in the
plan below were wrong and are corrected in place, because two of them fail *silently*.

**1. The emulator's "float64" adjoint was single precision. This was the big one.**
`neuralop` 2.0.0's `SpectralConv.forward` allocates its Fourier working buffer from a
hard-coded dtype:

```python
if self.fno_block_precision in ["half", "mixed"]:
    out_dtype = torch.chalf
else:
    out_dtype = torch.cfloat          # not derived from the input or the weights
out_fft = torch.zeros([...], dtype=out_dtype)
out_fft[slices_x] = self._contract(x[slices_x], weight, ...)
```

`out_dtype` does not depend on the model's dtype, so with `fno_block_precision = "full"`
the buffer is **complex64 however the model is cast**. The contraction is computed in
complex128 and truncated on assignment; `irfftn` runs from there; the bias addition
afterwards promotes the *type* back to float64 without recovering a digit. §4's
"float64 throughout" was therefore false for **97.95 % of the operator** — the spectral
convolutions hold 26,739,072 of 27,297,960 parameters.

Measured, all on one `SpectralConv` where the exact answer is computable because the
layer is linear, plus the whole model:

| quantity | as shipped | buffer promoted to complex128 |
| --- | ---: | ---: |
| adjoint identity `<v, J u>` vs `<J^T v, u>` | 1.30e-06 | **1.15e-13** |
| round-off noise in the scalar cost `J` | 5.5e-10 m | **3.1e-16 m** |
| forward-mode vs reverse-mode AD, whole model | 3.6e-11 | **8.3e-17** |
| gate F2, `|FD − adjoint| / max|adjoint|` | 4.6e-06 | **2.65e-10** |

The tell was gate F4: a "float64" path that differs from float32 by only 1.7e-07 is not
float64 — a genuine one differs from float32 by float32's *own* error. Same shape as
`s0-twin-float32-floor`, one level deeper.

`scripts/fno_adjoint_ft90.py` carries `DoublePrecisionSpectralConv`, which re-classes each
`SpectralConv` in place, and `verify_double_precision_spectrum()` asserts both that the
buffer is complex128 and that the adjoint identity holds — verified, not trusted. **This
does not change the operator**: the weights are untouched and the forward map moves by
6.0e-08, below float32 epsilon. The project's own inference runs in float32, so the
deployed map carries this precision anyway; the adjoint study wants the derivative of the
operator the weights define, not of one particular rounding of it.

**2. §6's finite-difference epsilon range was on the wrong branch.** With the round-off
floor at 5.5e-10 the F2 error *grew* as epsilon shrank — `|FD − adj| ≈ noise/(2ε)` — so
the declared `{1e-2 … 1e-5}` sat entirely on the round-off side and the "plateau" the gate
asks for did not exist to be found. With the floor at 3.1e-16 the optimum moves to
ε ≈ 1e-5 and the sweep runs `{1e-2 … 1e-6}`, spanning both branches so the minimum is
visible rather than assumed. **The plan's own prediction was right**: "if F2 is loose, it
is a bug in the pipeline, not physics." It was a bug, in the pipeline.

**3. Two gates compared against constants that were unreachable, and now compare against
computed floors.** F1's flat 1e-10 is unreachable for the mean-only functional, whose sum
cancels to eight figures (`<η>_A` ≈ 3.4e-09 m, condition number 1.06e+08) — the same trap
the MITgcm side hit at gate G5 and settled the same way. F1 now tests against
`eps_64 × condition_number` per objective. The forward-vs-reverse check tests against a
measured floor rather than a guessed 1e-12; the thread-reordering probe returns exactly
zero here because PyTorch's CPU reductions are bit-deterministic, so machine precision on
the largest entry of the map is the floor instead.

**4. A forward run was missing from §3.** `V10` starts at day 7280, and no archive segment
holds a pickup there; `data.diagnostics` sets `dumpAtLast=.FALSE.`, so no adjoint run dumps
day 7290 either. `F90` — the forward executable, day 7200 → 7290 plus one tail day,
`pChkptFreq` 10 days — supplies both, plus the daily snapshots gate G0 needs across the
whole window. It is now the first job in the order of execution. **It has run, and
gate G0 passes: all 91 days, `max|diff| = 0.0` exactly**, so both sides provably evolve
along the same trajectory across the entire window.

**5. `--exclusive` could not be a `#SBATCH` directive.** §3.2 asks for exclusive nodes on
the strength of `mitgcm-cg2d-needs-exclusive-nodes`. A hard directive cannot be turned off
from the command line, and on 2026-08-19 fifty of the sixty-two batch nodes were allocated:
an exclusive request sat in `(Resources)` indefinitely while nine nodes had free cores. It
is now an opt-in flag documented in the launcher, and the Phase A runs go on shared nodes.
That is a defensible downgrade — these are 62², an eighth the linear size of the case that
motivated the finding, and the v1 runs completed on shared nodes — but the wall time is
worth watching: a ninety-day adjoint taking hours rather than tens of minutes would be the
same reduction contention showing up at this grid size, and is worth recording either way.

### Confirmed as planned

| | |
| --- | --- |
| no TAF resubmission | `72 × 200 × 1 = 14,400 ≥ 6,480` steps, asserted by a test; `mitgcmuv_ad` reused byte-for-byte |
| the kernel stencil | all five cells wet, sums to 1.0, centroid displacement **exactly 0** |
| the rejected isotropic 5×5 | 15 of 25 cells usable, centroid displaced **0.504 cells** east — reproduced, not quoted |
| gate F5 | `S_forced` = `S_free` at lead 10 to **0.0** exactly |
| F2 against MITgcm | **2.65e-10** against MITgcm's 2.0e-06, as §6 predicted for a branch-free operator |

## Results — the emulator side, steps 0-5 complete

`outputs/af_fno/adjoint/fno_ft90_s0_adjoint_v1/`, 19.2 min on 8 CPU threads. **Every gate
passes at every lead.** Nothing below has been differenced against MITgcm; step 8 (gate
G1-90) is still ahead of any comparison. The conservation probe is the exception and is
final, because its reference is analytic.

| gate | worst over all leads | verdict |
| --- | ---: | --- |
| F1, `J` on truth vs numpy | 1.12e-08 against a 2.35e-08 computed floor | pass |
| F2, `\|FD − adjoint\|/max\|adjoint\|` | 3.14e-09, plateau at 6-7 of 7 wet probes | pass |
| F2, forward vs reverse mode | 8.33e-17 absolute | pass |
| F3, plain operator + complex128 spectrum | adjoint identity 1.15e-13 | pass |
| F4, float32 vs float64 | 6.13e-07 relative L2 | pass |
| F5, `S_forced` = `S_free` at lead 10 | **exactly 0** | pass |
| F6, weight digests | 3 fields | pass |
| backward-sweep `sigma` conversion | **exactly 0** | pass |

### The headline result: the emulator annihilates the basin-mean mode

MITgcm's answer for the mean-only functional is `w_mean` itself at **every** lead, exactly,
because the area integral of eta is conserved. The emulator:

| lead (d) | `‖S − w‖/‖w‖` | amplitude ratio | pattern corr. |
| ---: | ---: | ---: | ---: |
| 10 | 0.5945 | 0.4107 | 0.9269 |
| 20 | 0.8367 | 0.1676 | 0.8258 |
| 30 | 0.9352 | 0.0680 | 0.7160 |
| **90** | **1.0013** | **0.0062** | **0.0392** |

The exact answer is 1.000 and 1.000 on both right-hand columns. At ten days the emulator
retains 41 % of the correct sensitivity to a uniform sea-level offset with the pattern
still largely right; by ninety days it retains **0.6 %** and the pattern is uncorrelated.
A uniform offset is a null mode of the true dynamics — conserved, and driving no flow
because it changes no pressure gradient — and the emulator destroys it. For any
adjoint or assimilation use this says the operator is blind to the gravest barotropic mode.

This needed no MITgcm run to establish, exactly as §5.4 predicted.

### Amplitude: decay, not runaway

| lead (d) | `‖S_forced‖` | ratio to lead 10 |
| ---: | ---: | ---: |
| 10 | 1.5834e-01 | 1.000 |
| 20 | 3.7779e-02 | 0.239 |
| 30 | 2.1788e-02 | 0.138 |
| 90 | 1.0022e-02 | **0.063** |

§8.2 asked whether the ninety-day map would be dramatically *larger* than the ten-day one,
since `lambda_hat` = 1.00831 compounds to only 1.077 over nine calls. It is not — it is
**sixteen times smaller**. The failure mode on this axis is over-damping, not runaway, and
it is consistent with the conservation result. Whether that is right is the question P90
answers: in a non-chaotic S0 the true adjoint should decay far less.

### Trajectory drift is small, which is good news for the design

| lead (d) | pattern corr. | relative L2 | amplitude ratio |
| ---: | ---: | ---: | ---: |
| 20 | 0.99978 | 0.0211 | 1.0029 |
| 30 | 0.99838 | 0.0617 | 0.9746 |
| 90 | 0.99556 | 0.0942 | 0.9986 |

`S_free` and `S_forced` agree to a pattern correlation of 0.996 even at ninety days. The
decomposition of §5.3 was worth building and its answer is that the two are nearly the
same object here: the eventual comparison against MITgcm will be dominated by **Jacobian
error**, not by the emulator's own trajectory drift. That was not knowable in advance.

### No period-2 computational mode

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `corr(S(10k), S(10(k+1)))` | +0.135 | −0.008 | −0.067 | +0.038 | +0.025 | +0.132 | +0.388 | +0.759 |

§8.1's probe comes back **negative**: no strong anticorrelation at any lead, in either
chain. The mode `local-branch-release-lr` warns about — which passes every other check in
this project — does not appear in the sensitivities. A clean null result that nothing else
here could have delivered. The rise toward +0.76 at large `k` is the adjoint field
smoothing as it propagates backward, which is expected.

### Structure

| lead (d) | western band / interior | e-folding (cells) |
| ---: | ---: | ---: |
| 10 | 25.14 | 50.6 |
| 20 | 7.63 | 38.7 |
| 30 | 4.87 | 37.0 |
| 90 | **1.87** | 72.8 |

The sensitivity starts strongly concentrated in the western boundary band and ends nearly
uniform. `western-boundary-ratio-degrades` records the same signature in the day-2000
*anomaly* field, unscored by the gate; here it is visible in the *sensitivity*, and it is
the sharpest thing to test against P90. Unmasked land leakage is small: max 1.23e-05,
0.36 % of the wet-cell norm.

The radial spectra roll off steeply above the operator's 4.63-cell cutoff at every lead —
drawn on the figure — so whether the emulator is *missing* fine-scale sensitivity or
correctly has none there is a question only MITgcm can settle.

### The two objectives

| lead (d) | pattern corr. | relative L2 | amplitude ratio |
| ---: | ---: | ---: | ---: |
| 10 | 0.8994 | 0.4877 | 0.6832 |
| 20 | 0.9502 | 0.3777 | 0.7373 |
| 30 | 0.9313 | 0.4324 | 0.6991 |
| 90 | 0.7277 | 0.8063 | 0.3038 |

`J_kernel` behaves as a smoothed `J_point` at short lead and diverges from it at ninety
days — so the kernel is not redundant, and the robustness check §1.2 declared it for is a
real one.

---

## 0. What changed, and why this is not a re-run of the old plan

Four things are different, and one of them removes the largest interpretive caveat in the
whole earlier study.

**1. The model is Markov now.** `docs/fno_adjoint_plan.md` differentiated
`model_c_2in_1out_new_channels_p_cont_BT_loss_v1`, a **two-input** operator
`F(x_{t-10}, x_t, S) -> x_{t+10}` with 97 external channels. The production model is
**one-input**, `F(x_t, S) -> x_{t+10}`, 51 external channels, 27,297,960 parameters.

That deletes the entire conceptual asymmetry of §4 of the old plan. There is no history
slot, so:

- experiment **E2** (history-slot sensitivity) does not exist and is removed;
- the FNO's `dJ/d eta(t0)` is a **total** derivative with respect to a complete initial
  condition, exactly as MITgcm's `adxx_etan` is — not a partial derivative holding a
  second, dynamically-linked input fixed;
- the *history-slot* off-manifold problem is gone: the old model's two inputs were
  dynamically linked on the real trajectory, so perturbing one and not the other produced
  an input pair the operator never saw. A one-input operator has no such pair;
- E1 and E3 collapse into a single chained lead sweep.

Both models are now Markov in the same 46-channel state, on the same grid, at the same
cell centres. The comparison is as clean as it can be made.

**Correction, 2026-08-19.** An earlier draft of this section said the derivative is "no
longer taken off the training manifold" and that the old plan's warning about
off-manifold behaviour no longer applies. That was wrong, and the results in §Results
are what showed it. The Markov property removes the *history-slot* asymmetry; it does
**not** make an eta-only perturbation on-manifold. Raising eta while holding `U`, `V` and
`Theta` fixed produces a state that is not in geostrophic or hydrostatic balance, and no
state the emulator was ever trained on looks like that. MITgcm answers the question anyway
— it radiates the imbalance away as barotropic gravity waves within hours — and the
emulator, whose timestep is ten days, cannot represent that at all. So the old plan's
warning stands in full: this is a test of **off-manifold** behaviour, which is strictly
harder than the forecast skill the acceptance gate measured, and is exactly why it is
worth doing.

**2. The results in `outputs/af_fno/adjoint/fno_s0_adjoint_v1/` are stale.** They were
produced on 2026-08-13 from the two-input checkpoint `bf3ccc70…`, which is not the
published operator. They are kept for provenance and are not read by this plan.

**3. The ninety-day horizon introduces a new asymmetry that did not exist at ten days,
and it is decomposable.** MITgcm's adjoint is, by construction, linearized about the
**true** trajectory. The emulator's autoregressive chain is linearized about **its own**
trajectory, which has drifted from truth by day 7290. A disagreement at 90 days therefore
conflates two different errors:

```
   wrong Jacobian                +      right Jacobian, wrong linearization point
   (the operator is inexact)            (the emulator's trajectory has drifted)
```

§5.3 separates them with a teacher-forced chain, at the cost of nine extra forward passes.
This is the single most important new piece of design in this plan.

**4. No TAF submission is needed.** `af_fno/mitgcm/code_ad/tamc.h` was deliberately sized
for 200 days, not for the first experiment's 20:

```
nchklev_1 * nchklev_2 * nchklev_3  =  72 * 200 * 1  =  14,400  >=  6,480   (90 days)
```

Nothing in §2's "requires a new TAF submission" column is touched: no `.F` file, no CPP
flag, no header, no package, and **not `tamc.h`**. `mitgcmuv_ad`
(`787246365e96a93b…`) is reused byte-for-byte, and its gates G1–G5 carry over as
provenance for the executable itself. The generous sizing was a deliberate call in the
ground-truth plan and it is now cashed in.

## Results — the comparison, steps 6-13 complete

All eleven MITgcm runs plus the F90 forward run and a 35-task gradient-check sweep are
done. **Every gate on both sides passes.**

| gate | result |
| --- | --- |
| G0, forward re-run vs the archive | 91 days, `max\|diff\| = 0.0` exactly |
| **G1-90**, `grdchk` at the ninety-day window | **7/7 points, worst `\|1 − fd/adj\|` = 2.10e-06** against 1e-4 |
| G2a, `adxx_etan` vs `ADJetan` at day 7200 | relative L2 **exactly 0** |
| G2b, V10 vs P90's day-7280 adjoint state | relative L2 **exactly 0** |
| G3, mean-only over 91 dumps | worst 3.57e-08 vs 1e-5 |
| G4, land and finiteness | 244/244 land cells exactly 0 |
| G5, `fc` vs numpy, all 9 runs | worst 3.79e-08 against a computed 8.94e-08 float32 floor |

The G1-90 sweep needed a wider epsilon range than the ten-day study: at ninety days
`|S|` at p⋆ is 5.27e-06 against the ten-day study's 9.26e-05, seventeen times smaller, so
the same epsilon buys seventeen times less signal against the same `fc` round-off floor.
The declared `{1e-3, 1e-4, 1e-5}` put the minimum at its own top edge, and the sweep was
extended upward to bracket it. `|1 − fd/adj|`:

| point | 1e-1 | 1e-2 | **1e-3** | 1e-4 | 1e-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p⋆ (2,17) | 1.0e-04 | 5.0e-04 | **7.0e-07** | 1.0e-06 | 1.1e-05 |
| WBC (2,14) | 1.2e-04 | 7.3e-04 | **1.2e-06** | 5.4e-06 | 5.6e-06 |
| WBC (2,11) | 1.2e-03 | 6.5e-03 | **2.1e-06** | 2.2e-05 | 4.5e-04 |
| offshore (4,17) | 2.2e-04 | 3.3e-04 | 1.1e-05 | 1.9e-06 | **1.2e-06** |
| interior (31,17) | 3.4e-04 | 6.1e-05 | **2.8e-08** | 1.6e-07 | 1.5e-06 |
| eastern (61,17) | 3.3e-05 | 1.8e-05 | **4.2e-07** | 1.3e-06 | 4.9e-06 |
| northern (31,55) | 1.3e-04 | 4.6e-07 | **1.1e-08** | 4.8e-06 | 3.7e-05 |

A clean V at **six of the seven** points: the response leaves the linear range above
1e-2, round-off takes over below 1e-4, and the plateau sits at 1e-3. §12.3 asked that the
linear range be reported rather than claimed — at ninety days it is roughly 1e-4 to 1e-3,
narrower than the ten-day study's 1e-3 to 1e-5, which is what nine times the propagation
should do.

Two bugs in the gate itself were found and fixed here, both of the silent kind this study
keeps turning up. `grdchk` prints its `grad-res` block from the rank owning the tile that
contains the test point, so reading `STDOUT.0000` alone dropped the eastern and northern
points and left the gate reporting 5 of 7 with no error — the same shape as the v1 `nbeg`
trap. And the epsilon substitution applied a `e`→`d` exponent fix to the whole namelist
line, rewriting the key as `grdchk_dps`, which MITgcm would have ignored in favour of the
default epsilon.

### The headline: the emulator is not an adjoint of this system

Both derivatives are independently gate-validated — MITgcm's against a finite difference
at 2.10e-06, the emulator's at 2.65e-10 — so what follows is a **model disagreement, not
a pipeline artifact**.

| lead (d) | pattern corr. | relative L2 | amplitude ratio | sign agreement |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.0594 | 37.89 | **37.94** | 0.431 |
| 20 | −0.0106 | 16.52 | 16.48 | 0.522 |
| 30 | 0.0084 | 9.69 | 9.65 | 0.513 |
| 90 | 0.0213 | 6.62 | 6.57 | 0.483 |

Pattern correlation is **indistinguishable from zero at every lead**, sign agreement is
coin-flip, and the emulator is **38 times too sensitive** at ten days.

### Why, and it is not subtle

The barotropic gravity-wave speed here is `sqrt(gH)` with `H = 1800 m`, i.e. 133 m/s,
which crosses the ~5,500 km basin in **11.5 hours**. An eta-only perturbation is
dynamically unbalanced, so MITgcm radiates almost all of it away within hours and what
survives at ten days is the small projection onto slow geostrophic and Rossby modes:
`S[p⋆] = 8.36e-05`, and the map's peak is not at the target at all but at (j=28, i=1),
*upstream* in the boundary current, which is advection.

The emulator's timestep is ten days. That entire adjustment happens **inside one of its
calls**, so it cannot represent it. Its map peaks exactly at p⋆ with
`S[p⋆] = 1.42e-01` — it treats an unbalanced sea-surface perturbation as **1,700 times
more persistent than it is**.

### It is not grid-scale noise sitting on a correct large-scale field

The obvious hope is that the emulator has the right basin-scale sensitivity underneath
contamination from its 3x3 local branch. It does not. Both maps low-pass filtered to the
same scale:

| filter | ≥2 cells | ≥4 | ≥6 | ≥8 | ≥12 | ≥20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| correlation, lead 10 | 0.061 | 0.084 | 0.095 | 0.102 | 0.107 | 0.212 |
| correlation, lead 90 | 0.022 | 0.025 | 0.029 | 0.034 | 0.081 | 0.146 |

and the amplitude ratio restricted to wavelengths ≥ 8 cells is still 13.8 at lead 10.
The large-scale component is wrong in both pattern and amplitude.

Where the variance sits is itself the diagnostic:

| lead (d) | fraction of variance below the operator's 4.63-cell cutoff — MITgcm | emulator |
| ---: | ---: | ---: |
| 10 | 0.044 | **0.670** |
| 30 | 0.005 | 0.555 |
| 90 | 0.002 | 0.312 |

MITgcm's answer is almost purely large-scale. Two thirds of the emulator's ten-day
sensitivity lives at scales its own spectral operator **cannot represent** — so it comes
from the bias-free 3x3 local branch, 21,114 parameters, 0.08 % of the model. The
sensitivity of a 27-million-parameter operator is being set by its smallest component.

### Decay, and the error decomposition

| lead (d) | ‖S‖ MITgcm | ‖S‖ emulator | ratio | MITgcm decay | emulator decay |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 4.173e-03 | 1.583e-01 | 37.9 | 1.000 | 1.000 |
| 20 | 2.292e-03 | 3.778e-02 | 16.5 | 0.549 | 0.239 |
| 30 | 2.258e-03 | 2.179e-02 | 9.7 | 0.541 | 0.138 |
| 90 | 1.527e-03 | 1.002e-02 | 6.6 | **0.366** | **0.063** |

MITgcm's adjoint decays by 2.7x over ninety days; the emulator's by 16x. §8.2 asked
whether the emulator's amplitude would run away — it does the opposite at long lead, and
the opposite of *that* at short lead. Both errors have the same cause: the fast adjustment
it cannot resolve is missing at day 10, and its own dissipation then removes what MITgcm
keeps.

The §5.3 decomposition earns its keep by ruling something out. Jacobian error dominates
trajectory drift by two to three orders of magnitude at every lead (37.9 against 0.000 at
lead 10; 6.62 against 0.094 at lead 90), so **none** of this is attributable to the
emulator's rollout having drifted from truth. It is the operator.

### What this does and does not say

It does **not** contradict the forecast skill in the model card. §8 there measures
propagation of *balanced* states, and the emulator is good at it. This measures the
Jacobian with respect to an *unbalanced* perturbation, which is a different and harder
question, and the honest reading is that a ten-day-timestep emulator cannot be an adjoint
for a quantity whose adjustment time is hours.

The smooth-kernel objective was declared in §1.2 as a robustness check, and it confirms
that none of this is an artifact of asking for a delta:

| lead (d) | pattern corr. | relative L2 | amplitude ratio |
| ---: | ---: | ---: | ---: |
| 10 | 0.082 | 26.15 | 26.22 |
| 20 | −0.011 | 12.37 | 12.32 |
| 30 | 0.016 | 6.86 | 6.80 |
| 90 | 0.028 | 2.20 | **1.99** |

Smoothing along the jet improves the amplitude substantially at long lead — 6.57 to 1.99
at ninety days — and does nothing whatever for the pattern. That is consistent with the
scale decomposition above: the kernel suppresses the grid-scale power that inflates the
norm, and there is no correct large-scale pattern underneath for it to reveal.

The declared Phase B follows directly: perturb eta **together with** the geostrophically
consistent `U` and `V`, so the perturbation stays on the manifold both models live on.
The ingredients are already on disk — MITgcm's `ADJuvel`/`ADJvvel`/`ADJtheta` were dumped
by every Phase A run, and the emulator's `lambda_k` carries all 46 channels.

---

## 1. The two objectives

The user requirement is a scalar SSH objective, evaluated both at exactly one grid cell
and over a small smooth kernel around it. Both are **linear in eta**, which matters more
than it looks: `grad_x J = (0, …, 0, w)` exactly, independent of the state, so the only
nonlinearity anywhere in the study is in the propagator. Nothing about the functional has
to be linearized.

Both keep the frozen target `p* = (i=2, j=17)` (0-based `(j=16, i=1)`), 30.5N, the first
wet column, mean surface speed 0.8414 m/s — the core of the western boundary current.
`config/mitgcm_adjoint_s0_target_v1.json` is immutable and is read, never re-derived.

### 1.1 J_point — primary

```
J_point  =  eta(p*, T)  -  <eta(T)>_A
w_point  =  delta_{p*}  -  rA * maskC / A_wet
```

`A_wet = 3.0046402806e13 m^2`, `w[p*] = +0.99964553`. This is **bit-identical** to
`work/costWeight_ssh_anomaly.bin` (`8b073515…`), the field the validated 10-day study
used. It is declared primary because it is the only objective with a passed `grdchk` on
this executable, and because a delta is the sharpest thing that can be asked for.

### 1.2 J_kernel — secondary, the smoothed variant

**Chosen form: a five-point meridional Gaussian along the wall, sigma = 1 cell, radius 2,
at the same `i*`.**

```
g_j  =  N(j - j*; sigma=1),  j in {j*-2 … j*+2}      normalized to sum 1
     =  [0.0545, 0.2442, 0.4026, 0.2442, 0.0545]

J_kernel = sum_j g_j eta(i*, j, T)  -  <eta(T)>_A
```

All five cells are wet (verified against `costWeight_mean_only.bin`), so no land
renormalization is needed and the kernel's centroid is exactly `p*`.

**Why this and not an isotropic box or 2-D Gaussian.** Three independent arguments point
the same way, which is why the anisotropy is a decision and not an accident.

| | |
| --- | --- |
| **The Munk layer is one grid cell wide.** | `(A_h/beta)^(1/3)` ~ 63 km against ~79–96 km zonal spacing. Measurement in the ground-truth plan: 0.84 m/s at `i=2`, 0.17 m/s at `i=3`. Any zonal smoothing replaces "SSH in the boundary current" with "SSH in a mixture of the current and the interior" — it changes what is being measured, physically. Meridional smoothing averages **along** the jet, which does not. |
| **`p*` sits against the land rim.** | A symmetric 5x5 stencil at `(j=16, i=1)` reaches `i=-1` (off grid) and `i=0` (land): only 15 of 25 cells are usable. Renormalizing over wet cells displaces the effective centroid **0.504 cells east**, off the jet into water five times slower. Measured, not estimated. The 1-D meridional kernel has no land contact at all. |
| **Band-limiting is the point, and it is bought honestly.** | See below. |

**What the smoothing actually buys, stated with numbers.** The operator's spectral path
keeps 32 modes in `ky` and 17 in `kx` on a **74x74** domain-padded grid (verified: the
`SpectralConv` weight is `(128, 128, 32, 17)` per block, padded shape `(1, 53, 74, 74)`).
So the cut is isotropic at `|k| <= 16` of 37 available — the shortest representable
wavelength is **~4.6 grid cells**, in either direction. Anything finer reaches the output
only through the bias-free 3x3 local branch (21,114 parameters, 0.08 % of the model) and
the pointwise `W` skips.

A delta functional therefore puts weight on modes the spectral operator structurally
cannot carry. The Gaussian's transfer function is `H(k) = exp(-(sigma k)^2 / 2)` with `k`
in rad/cell, which is padding-invariant:

| wavelength | `k` (rad/cell) | `H`, sigma=1 | `H`, sigma=1.5 |
| ---: | ---: | ---: | ---: |
| 12 cells | 0.524 | 0.87 | 0.74 |
| 8 cells | 0.785 | 0.74 | 0.50 |
| **4.63 cells (cutoff)** | **1.359** | **0.40** | **0.13** |
| 3 cells | 2.094 | 0.11 | 0.01 |
| 2 cells (Nyquist) | 3.142 | 0.01 | 0.00 |

sigma = 1 attenuates the first unrepresented mode to 0.40 — real, but partial, and the
zonal delta is left intact **by design**. This is stated as a measurement rather than sold
as a fix: the definitive answer to "does the emulator miss high-`k` sensitivity" comes from
the radial power spectra of §7.3, not from the kernel choice.

**The kernel is a runtime input, so sigma is cheap to sweep.** §4.4 of the ground-truth
plan pushed the whole functional into a binary weight field read at run time. A different
sigma, radius or shape costs one ~20-minute MITgcm run and one backward pass — no rebuild,
no licence. If the sigma dependence turns out to matter, measuring it is a half-day, not a
redesign.

### 1.3 The mean term is free diagnostics

Because `int eta dA` is conserved exactly by `implicitFreeSurface` + `exactConserv` in a
closed basin with no freshwater flux, the adjoint of the mean functional is **constant in
time and equals `w_mean` itself** at every lead. That is Gate G3 on the MITgcm side and it
is now tested over 91 dump times and 90 level-2 tape records instead of 11 and 10 — a
materially stronger test of the checkpointing, for one extra run.

On the FNO side the same run is not a gate but a **measurement**: the emulator conserves
nothing, so `S_fno_mean_only(L) - w_mean` is a spatially resolved, lead-resolved map of its
global sea-level conservation error, against an analytically exact reference and with no
MITgcm run required to interpret it. It is the cheapest publishable number in the study and
should be produced first.

---

## 2. The window, frozen

`iter(d) = 2,592,000 + 72 d`, 1200 s timestep, 72 steps/day, 360-day year.

| trajectory day | iteration | role |
| ---: | ---: | --- |
| 7,200 | 3,110,400 | **source**, and the sole initial condition. Pickup **already exists** at `…/mitgcm_long_truth_v1/S0/production/years_120_126/pickup.0003110400.*` |
| 7,210 | 3,111,120 | cost time, lead 10 |
| 7,220 | 3,111,840 | cost time, lead 20 |
| 7,230 | 3,112,560 | cost time, lead 30 |
| 7,290 | 3,116,880 | cost time, lead 90 — **the limit** |

Day 7,200 is the first day of the `evaluation truth only` block (7,200–8,999): never
trained on, never validated on, never a rollout start. Both models see it cold. Day 7,290
is inside the store (`state` is `(3, 9000, 46, 62, 62)`), so lead-matched truth exists at
every lead.

**Why 90 days is the limit, stated precisely.** It is a property of the *emulator's
training protocol*, not of S0's predictability. The fine-tune's `L_rollout` runs
`E_20 … E_90`, and 90 days is Bire et al.'s decorrelation estimate — past it, pointwise
truth stops being a defensible target and only statistics survive. S0 itself is far more
predictable than that: `s0-not-chaotic` and handbook §8.2 record the S0 SST anomaly
pattern-correlating with itself at **0.992 after 171 days**. So do not read the 90-day
limit as a chaotic predictability horizon; read it as the longest lead at which this
operator was ever asked to match a particular trajectory.

That has a concrete, testable consequence: **in a non-chaotic regime the tangent-linear
approximation should hold over 90 days**, so the finite-difference plateau of §6 should be
found, not merely hoped for. If it is not, that is itself the result.

### Two lead sweeps, and which is primary

There are two distinct things "lead" can mean, and conflating them is the standard way to
produce an uninterpretable figure.

```
 (B)  FIXED SOURCE, moving cost       <-- PRIMARY
      source day 7200, always
      J at 7210 / 7220 / 7230 / 7290
      lead = elapsed model time = how far the emulator has propagated

 (A)  FIXED COST, moving source       <-- free companion
      J at day 7290, always
      dJ/d eta(t) for t = 7200 … 7290
      lead = how far back the target's sensitivity reaches
```

**(B) is primary.** It holds the initial condition fixed and varies only the propagation
length, which is the axis this study is about: the emulator's drift grows with elapsed
time from a fixed start, exactly as in the fine-tune's own rollout objective. (A) varies
the initial condition together with the lead and conflates the two.

**(A) comes free from (B)'s lead-90 run on both sides**, and is reported as a companion:
MITgcm's `ADJetan` dumps give all 91 daily maps of the day-7290 target's backward
sensitivity, and the FNO's single backward chain gives the same object at all nine 10-day
multiples (§5.2). It is the classic backward-in-time sensitivity movie and costs nothing.

---

## 3. The MITgcm side — Phase A runs

Everything reuses the existing, gate-validated executable and staging. Only the window,
the weight field and `nTimeSteps` change.

| run | weight | `nIter0` | `nTimeSteps` | cost at | products |
| --- | --- | ---: | ---: | ---: | --- |
| `P10` | `w_point` | 3,110,400 | 720 | 7,210 | `adxx_etan` = `S_10` |
| `P20` | `w_point` | 3,110,400 | 1,440 | 7,220 | `S_20` |
| `P30` | `w_point` | 3,110,400 | 2,160 | 7,230 | `S_30` |
| `P90` | `w_point` | 3,110,400 | **6,480** | 7,290 | `S_90` + 91 `ADJetan` (sweep A) + `ADJuvel/vvel/theta` |
| `K10…K90` | `w_kernel` | as above | as above | as above | the same four maps for `J_kernel` |
| `C90` | `w_mean_only` | 3,110,400 | 6,480 | 7,290 | Gate G3 over 91 dumps |
| `V10` | `w_point` | **3,116,160** | 720 | 7,290 | Gate G2b — day-7280 cross-check against `P90` |
| `G90` | `w_point` | 3,110,400 | 6,480 | 7,290 | `grdchk`, `cg2dTargetResidual = 1e-12` |

Eleven jobs. All with `adjDumpFreq = 86400.`, `writeBinaryPrec = 64`, `inAdExact = .TRUE.`,
`useGrdchk = .FALSE.` except `G90`, 4 ranks, `srun --mpi=pmix -n 4`.

Settings that must **not** move, each for a recorded reason:

| setting | value | why |
| --- | --- | --- |
| `cg2dTargetResidual` | **1e-7** in every production run | it is what every S0 forward segment was integrated with and what the FNO was trained on. Tightening it produces the adjoint of a model the emulator never saw. |
| `ivdc_kappa` | **1.** | same reason; the convective switch is part of the physics being emulated (§9.1) |
| `writeBinaryPrec` | **64** | `DUMP_ADJ_XY` honours it, and the default 32 put a float32 floor directly under G2/G3 while `adxx_etan` came back at float64. `data.diagnostics` sets `fileFlags='R'`, so forward snapshots stay float32 and stay comparable to the archive. |
| `ALLOW_GENARR2D_CONTROL` | on, `xx_genarr2d_file(1)='xx_etan'`, `grdchkvarindex = 101` | `ALLOW_ETAN0_CONTROL` is dead code inside `#ifdef ECCO_CTRL_DEPRECATED` and builds a complete, entirely zero adjoint. Do not re-open that door. |
| `xx_genarr2d_weight` | non-blank, exact ones | blank means the control is never registered — a second silent route to a zero gradient |
| `xx_genarr2d_preproc(1,1)` | `'noscaling'` | without it `ctrl_map_genarr.F:142` divides the control by `sqrt(w_genarr2d)` and rescales the returned gradient |

`nbeg = 0` in `data.grdchk`, or `GRDCHK_GET_POSITION` is never called and the sweep
silently tests the first wet cell while printing a plausible `grad-res` block. `nend` is an
**offset**, not an absolute index.

### 3.1 Code changes required

Small, and none of them touch the differentiated source.

1. `scripts/stage_adjoint_run.py` — the `MODES` table hard-codes `start_day` and `days`.
   Replace with `--start-day` / `--days` / `--mode {forward,adjoint,grdchk}` and keep the
   existing five named modes as presets so the v1 runs stay reproducible from the same
   script. This is the only structural edit.
2. `scripts/make_cost_weight.py` — add `--qoi ssh_anomaly_kernel` with `--sigma`,
   `--radius` and `--axis {meridional,zonal,isotropic}`. Emit `w_kernel` to
   `work/costWeight_ssh_anomaly_kernel_sigma1.bin`. Assert all stencil cells are wet for
   the meridional default and refuse silently-renormalizing forms unless `--allow-land`
   is passed.
3. `af_fno/mitgcm/input_ad/data.grdchk` — `grdchk_eps` becomes a job parameter (§6).
4. Nothing in `af_fno/mitgcm/code_ad/`. Nothing in `tamc.h`. No `staf`.

### 3.2 Resources

Measured: 28 ms/step forward on 4 ranks; adjoint 3–5x plus tape I/O. `runB` (20 days) used
450 MB of scratch, `runA` (10 days) 250 MB — about 200 MB fixed plus ~12 MB/day.

| | per run |
| --- | --- |
| 90-day adjoint wall time | ~15–30 min |
| 90-day scratch | ~1.3 GB (plus ~130 MB of `ADJ*` dumps at float64) |
| eleven Phase-A jobs | ~4 h and ~12 GB, against 12 TB free |

**Request exclusive nodes.** `mitgcm-cg2d-needs-exclusive-nodes` records that shared Slurm
nodes made the 248^2 runs ~100x slower because the `cg2d` solve is reduction-bound. These
are 62^2 and the v1 runs completed on whatever they landed on, but a 90-day adjoint is 9x
`runB` and dominated by the same reduction. Ask for `--exclusive`; the cost is queue time,
the alternative is a job that looks hung.

---

## 4. The FNO side — what is loaded and what is asserted

| | |
| --- | --- |
| checkpoint | `…/models/C/model_c_production_1in_1out_spectralnorm_ft90_v1/selected.pt` |
| sha256 | `4acb7633d85a4df3925843cc833d248e86fcd5d2569ba0300c9c58b022537806` |
| optimizer step | 1,440 |
| normalizer | `…_train_only_normalization.npz`, `fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf` (byte-identical to the parent's) |
| parameters | 27,297,960, asserted before the first forward pass |
| load | `strict=True`, `map_location='cpu'`, `eval()`, every parameter `requires_grad_(False)` |

**Why this checkpoint is adjoint-clean, and why that is not an accident.** The per-mode
spectral cap is *materialized on write*: the published checkpoint carries normalized
weights baked into the tensor, so it loads into a plain `ProductionFNO` and the inference
layer is exactly `y_hat(k) = R_k_tilde x_hat(k)`, whose adjoint is `R_k_tilde^H`. There is
no clipping, no `min(1, ·)` branch, no power iteration and no persistent buffer in the
inference graph. Assert this rather than trust it: **fail if any spectral-norm hook,
buffer or wrapper is live on the loaded module.** Handbook §2 also records that `tanh`
stabilization was rejected precisely because saturation drives `tanh' -> 0` and destroys
the sensitivities this study exists to measure — that decision is now being cashed in too.

Precision: **float64 throughout**. Casting float32 weights to double does not change the
function, only the arithmetic evaluating it. `s0-twin-float32-floor` is the standing reason:
this project has already lost an entire signal to a float32 quantisation floor. The
float32-vs-float64 gradient difference is reported as Gate F4, not chosen between.

### 4.1 The deployed map, exactly

The graph must reproduce what `ProductionStepper` actually deploys, including both masks:

```
eta_phys(7200)  (leaf, float64, requires_grad)
   |  insert into the 46-channel physical state x_phys(7200) from the zarr
   v
x_norm = (x_phys - mu) / sigma ;  x_norm[:, ~wet] = 0        <- inside the graph
   |  concat 5 static channels (dataset.static_block, S0)  -> 51
   v
G(x) = ProductionFNO(cat(x, S)) * wet                        <- output mask inside the graph
   |  x9  (or x1 / x2 / x3 for the shorter leads)
   v
y_norm(T)  ->  y_phys = y_norm * sigma + mu                  <- inside the graph
   |  J = sum_ij w[j,i] * y_phys[45, j, i]
   v
J.backward()  ->  eta_phys.grad  =  S_fno   [dimensionless]
```

Three non-negotiables carried over from the old plan and still correct:

- **Do not apply the `1/sigma_45` correction by hand.** `sigma` is a *field*, `(46,62,62)`,
  not a scalar. Make the physical field the leaf and let autograd carry every factor,
  including the output-side one. Gate F2 catches a missing or mis-shaped `sigma`.
- **Read `w` from the same binary the MITgcm run staged.** Never rebuild it. The SHA-256
  cross-check against the MITgcm run manifest is an **assertion**, not a note. Rebuilding
  `w` silently turns the comparison into a convention test.
- **Read the truth state from the zarr**, not from a re-run.

### 4.2 A correction to the old plan: land-cell leakage is vacuous as it was defined

The old plan's headline structural metric was `max|S_fno|` over the 244 dry cells, against
MITgcm's exact zero. But `normalized_state` sets `value[:, :, ~wet] = 0.0` and `step`
multiplies the output by `wet`. With both masks inside the graph — which is required, or
the differentiated map is not the deployed map — **land eta is not an input degree of
freedom of the emulator either**, and `dJ/d eta(land) = 0` exactly, by construction, on
both sides. The metric would report a guaranteed zero and prove nothing.

Replace it with the version that measures something: the gradient of `J` with respect to
the **unmasked normalized input** at land cells, i.e. the same backward pass with the input
mask lifted. That quantifies how much the global spectral path *wants* to read land, which
is a genuine property of the operator. Report it as an FNO-only diagnostic; it has no
MITgcm counterpart and must not be differenced against one.

---

## 5. The FNO experiments

All four leads on both objectives, both linearization trajectories, plus the conservation
probe. Every one of them is forward-plus-backward passes measured in seconds.

### 5.1 The chain

Write `G(x) = mask(F(x, S))` for the deployed one-step map, `x_k` for the state at day
`7200 + 10k`, and `n = L/10` calls for lead `L`.

```
free:      x0 = truth(7200),   x_k = G(x_{k-1})            k = 1 … n
forced:    x_k = truth(7200 + 10k)                          k = 0 … n-1
```

Because `J` is linear in eta, the seed of the backward pass is `lambda_n = (0,…,0, w)`
exactly, with no dependence on the final state. The adjoint recursion is then

```
lambda_{k-1}  =  DG(x_{k-1})^T lambda_k
S_L           =  ( lambda_0 )_{eta}  / sigma_45           [converted inside the graph]
```

### 5.2 Every intermediate lead is free, on both sides

`lambda_k` **is** `dJ_L / dx(7200 + 10k)`, so one backward chain from a cost at day 7290
yields sweep (A) at all nine 10-day multiples — the exact structural analogue of MITgcm's
`ADJetan` dumps, which give the same object at all 91 daily leads from one run. The
symmetry is not a coincidence: MITgcm's adjoint *is* a chain of vector-Jacobian products
along its forward trajectory, and so is this.

`lambda_k` also carries all 46 channels, so `dJ/dU`, `dJ/dV` and `dJ/dTheta` at every lead
come out of the same pass at no cost. MITgcm dumps the matching fields —
`addummy_in_stepping.F` writes `ADJtheta`, `ADJuvel` and `ADJvvel` under
`ALLOW_AUTODIFF_MONITOR`, which is already enabled. **The full 46-channel adjoint state is
therefore available on both sides.** Phase A reports the eta comparison and *archives* the
other three groups; the multi-field comparison is declared Phase B rather than smuggled in
here.

### 5.3 A1 / A2 — the two trajectories, and what their difference measures

| | linearized about | answers |
| --- | --- | --- |
| **A1 `S_forced`** | the MITgcm truth trajectory, states read from the zarr at days 7200, 7210, … | **is the emulator's ten-day Jacobian right, composed nine times?** |
| **A2 `S_free`** | the emulator's own autoregressive trajectory from day 7200 | **what does the deployed emulator believe?** |

**A1 is the primary comparison against MITgcm**, because it is the matched object: MITgcm's
adjoint is linearized about the truth trajectory by construction, so `S_forced` puts the
FNO's Jacobian at exactly the same points. **A2 is the primary operational answer**, because
it is what a data-assimilation or sensitivity user would actually get. Both are declared
now, before either is computed.

```
S_mitgcm  vs  S_forced      ->  Jacobian error          (the operator is inexact)
S_forced  vs  S_free        ->  trajectory error        (the linearization point drifted)
S_mitgcm  vs  S_free        ->  the total, which the two above decompose
```

At lead 10 the two chains are **identical by construction** (one call, from the same truth
state). That is a free correctness check on the implementation: `||S_forced_10 - S_free_10||`
must be exactly 0, not small. Make it an assertion.

Implementation note: `S_forced` is nine independent VJPs
(`torch.autograd.grad(y_k, x_k, grad_outputs=lambda_k)`, each at a truth state) chained
backward. `S_free` is one ordinary `backward()` through the nine-call rollout, with the
intermediate `lambda_k` captured by hooks. Nine forward and nine backward passes each.

### 5.4 A3 — the conservation probe, at every lead

Both chains re-run with `w = w_mean_only`. MITgcm's answer is `w_mean` itself at every
lead, exactly. The emulator's departure from it, `||S_fno_mean(L) - w_mean|| / ||w_mean||`,
is its global sea-level conservation error, resolved in space and in lead. **This needs no
MITgcm run to interpret and should be produced first**, before any TAF-side job is
submitted.

A useful identity falls out: for MITgcm, `S_anomaly = S_point_only - w_mean` exactly, at
every lead, because the mean term's adjoint is constant. For the emulator that identity
*fails*, and the residual is the same conservation error seen from the other side. Compute
both ways and check they agree — it is a free consistency test of the FNO pipeline.

---

## 6. Gates

### MITgcm side — what carries over and what must be re-earned

| gate | status | Phase A action |
| --- | --- | --- |
| **G0** bit-for-bit forward vs the zarr | passed, days 7200–7220 | **extend to 7200–7289** from run `P90`'s own forward dumps, free. Day 7290 is never dumped (`dumpAtLast=.FALSE.`) and is covered by G5 instead. Use the dataset's own C-grid-to-centre operator (`0.5*(u + roll(u,-1,-1))`), or the comparison is about the C grid, not the restart. |
| **G1** `grdchk` plateau | passed at 10 days, worst 2.0e-06 | **re-run at the 90-day window** — see below |
| **G2** cross-run consistency | passed, relative L2 exactly 0 | **two forms, see below** |
| **G3** mean-only conservation | passed, worst 3.57e-08 over 11 dumps | re-run as `C90` over **91 dumps and 90 level-2 tape records** |
| **G4** land exactly 0, all finite | passed, 244/244 | re-assert on every Phase A map |
| **G5** `fc` vs `sum(w·eta)` in numpy | passed, 1.54e-08 vs a 1.26e-07 float32 floor | re-assert at day 7290, for **both** objectives, against the **computed** float32 half-ulp bound of that particular weighted sum — never a constant tolerance |

**G2 has no free cross-run form in Phase A, so it is re-earned two ways.** The v1 check
worked because Run A and Run B shared a cost time; sweep (B) deliberately does not, so
`P10 … P90` are four genuinely different objects with no identity between them. Instead:

- **G2a, free.** Within `P90`, `adxx_etan` (the control gradient at `nIter0`, through the
  `ctrl`/`GENARR2D` path) must equal `ADJetan` dumped at day 7200 (through
  `addummy_for_etan.F`). Two independent code paths, one number, relative L2 `< 1e-6`.
- **G2b, three minutes.** Add one verification run `V10`: `nIter0 = 3,116,160` (day 7280),
  `nTimeSteps = 720`, cost at day 7290, `w_point`. Its `adxx_etan` must equal `P90`'s
  `ADJetan` at day 7280 to solver tolerance. This is the exact v1 G2 structure — an
  end-to-end statement that a 90-day tape returns the same adjoint state as a fresh 10-day
  run — and it is the strongest single check that the longer checkpointing did not corrupt
  anything.

**G1 at 90 days needs its epsilon chosen, not guessed.** The established finding
(`grdchk-limited-by-cg2d`) is that at the production `cg2dTargetResidual = 1e-7` the
*finite difference* is the noisy party, not the adjoint: the error was flat in epsilon,
1e-2 to 4e-2, with no plateau, and 1e-12 moved `p*` from 1e-2 to 1.5e-08. So `G90` runs at
**1e-12, diagnostic only**, archived separately, exactly as `grdchk_cg2d1em12` was.

The 90-day window makes this sharper, because the sensitivity magnitude at 90 days is
unknown in advance and the `fc` noise floor is not. Procedure:

```
1. run P90 first, read |S_90| at each test point from adxx_etan
2. estimate the fc noise floor from two identical reruns at cg2d = 1e-12
3. choose eps_p = max(1e-5, 100 * fc_noise / |S_90(p)|)  per point
4. sweep {eps_p/10, eps_p, 10*eps_p} and require a plateau, not a single hit
```

Same seven points as the validated sweep — `p*` (2,17); WBC upstream (2,14) and (2,11);
offshore (4,17); interior (31,17); eastern (61,17); northern (31,55) — noting that
`GRDCHK_GET_POSITION` only ever lands on wet cells, so land is covered by G4 over all 244
cells, which is the stronger statement anyway.

**Gate G1-90: `|FD/adjoint - 1| < 1e-4` at all seven points at the central epsilon, with a
visible plateau.** If it fails, §9.1 (convection) is the first suspect and the eastern /
interior points are the discriminator: a uniform failure across the basin is a noise floor,
a failure confined to convecting northern columns is `ivdc_kappa`.

### FNO side

| gate | condition | method |
| --- | --- | --- |
| **F1** | `J` from the FNO pipeline on **truth** `eta(T)` matches numpy `(w*eta).sum()` to `< 1e-10` relative, at all four leads and both objectives | same `w` binary, same zarr slice |
| **F2** | finite difference: `\|FD/adjoint - 1\| < 1e-6` at 8+ cells over eps in {1e-2 … 1e-5} m, **with a visible plateau**, at **every** lead | central difference, physical units, float64 |
| **F3** | the loaded module is a plain `ProductionFNO`: 27,297,960 parameters, no live spectral-norm hook/buffer/wrapper, both masks in the graph | assertion before any gradient |
| **F4** | float32 and float64 gradients agree to `< 1e-4` relative | otherwise round-off dominated |
| **F5** | `S_forced_10 == S_free_10` **exactly** | the two chains coincide at one call |
| **F6** | `w` SHA-256 equals the MITgcm run manifest's | assertion, not a note |
| **F7** | conservation probe: report `||S_fno_mean(L) - w_mean|| / ||w_mean||` at every lead | **no threshold — a measurement** |

F2 is the emulator's `grdchk` and deserves the same discipline. The operator is smooth
(GELU, spectral convolutions, LayerNorm, no branches, and — critically — no `tanh`
saturation and no live clipping), so the plateau should be **wide** and the agreement far
tighter than MITgcm's 1e-4, where `ivdc_kappa` sets the floor. **If F2 is loose, it is a
bug in the pipeline, not physics.** There is no convective-adjustment excuse on this side.

F2 must be run at lead 90, not just lead 10. That is where a nine-call chain could develop
a genuine nonlinearity — and where a *finite* perturbation stops predicting a *linear*
response is a real, reportable property of the emulator, distinct from a pipeline bug.
Distinguish them: a pipeline bug fails at every lead; a nonlinearity fails only at long
lead and only at large epsilon.

**No pass/fail threshold is declared on any comparison metric.** F1–F6 guard the pipeline;
G0–G5 guard the reference; the science is reported, not graded. This is the first
measurement of this quantity for this model and inventing a threshold before seeing the
number would be reverse-engineering a verdict.

---

## 7. Comparison metrics

`scripts/compare_adjoint_maps_phase_a.py`, reusing `scripts/adjoint_metrics.py` unchanged
(it is model-agnostic and already carries the project's conventions).

### 7.1 Primary, per lead, per objective, per trajectory

| metric | function | reads |
| --- | --- | --- |
| pattern correlation over wet cells | `pattern_correlation` | is the sensitivity in the right places |
| relative L2 | `relative_l2` | overall error |
| amplitude ratio | `amplitude_ratio` | systematic over/under-response |
| sign agreement | `sign_agreement` | structural fidelity |

Reported as a 4 (leads) x 2 (objectives) x 2 (trajectories) table against MITgcm, plus the
`S_forced` vs `S_free` decomposition of §5.3.

### 7.2 Structural — where this connects to what is already known

| metric | function | why it matters here |
| --- | --- | --- |
| **western band / interior split**, 4 wet cells | `boundary_interior_split` | `western-boundary-ratio-degrades`: the day-2000 boundary/interior anomaly ratio fell across the input and architecture arms, the physics-loss arms reversed it to 4.25, and **nothing in the gate scores it**. Handbook §8.6 puts the fine-tuned model at 8.16 against a truth of 23.10. `p*` sits **inside that band**. If the sensitivity maps disagree most there, the adjoint has localised a defect the forecast gate is structurally blind to — which is the reason this study exists. |
| radial decay from `p*` | `radial_decay` | e-folding distance of `|S|`; effective range of influence, and how it grows with lead |
| unmasked land leakage | §4.2 | FNO-only; how much the global spectral path wants to read land |

### 7.3 Spectral — with the architectural cutoff drawn on the plot

`radial_power_spectrum`, 12 radial bins, Hann-tapered on the exact wet rectangle — the
project's existing convention, reused so the numbers are comparable to the spectral loss
term and to the anomaly packages.

**Report absolute power per bin, never the fraction.** `local-branch-gamma-ablation` is the
standing reason: halving the 3x3 branch cut the spurious day-2000 anomaly 24x while the
high-`k` *fraction* moved the wrong way. The fraction misleads.

Annotate every spectrum with the operator's `|k| <= 16`-of-37 cutoff (§1.2). The question
"does the emulator's sensitivity map lack the fine structure MITgcm's has, and is that the
mode truncation or the learned dynamics?" is answered by whether the deficit sits **above**
that line.

---

## 8. Three things only a ninety-day adjoint can measure

These are the reasons to do Phase A at 90 days rather than repeat the 10-day study on a new
checkpoint. Each is a declared experiment with a declared reading.

### 8.1 The period-2 computational mode, made visible

`local-branch-release-lr` records that the radius cap bounds only the local branch, so a
**period-2 computational mode — a negative real eigenvalue of the composite recurrence —
passes every check currently in the project.** Nothing in the forecast gate, the growth
rate, the spectra or the anomaly packages can see it, because an alternating mode has the
right amplitude statistics.

A lead sweep of the adjoint sees it directly. If the composite ten-day map carries a
negative real eigenvalue, the adjoint state alternates sign call to call, so

```
rho_k  =  corr( S_free(lead 10k), S_free(lead 10(k+1)) )        k = 1 … 8
```

is **strongly negative** where a healthy operator gives a smoothly decaying positive
sequence. Compute `rho_k` for both `S_free` and `S_forced`; the mode is a property of the
operator, so it should appear in both. This is the sharpest new diagnostic in the study and
it costs one extra line over the maps already being produced.

Report alongside: the per-lead norm `||S_free(L)||` and its ratio to `||S_mitgcm(L)||`.

### 8.2 Amplitude growth, independently of `lambda_hat`

`lambda_hat = 1.00831` per call compounds to only 1.077 over nine calls, so **the 90-day
adjoint should not be dramatically larger than the 10-day one.** If `||S_90||` exceeds
`||S_10||` by orders of magnitude, something is wrong that `lambda_hat` did not catch.

This is not a redundant check. `production-model-amplitude-runaway` records that per-mode
spectral normalization cut the day-2000 anomaly ratio 8.9 -> 1.8 **without moving
`lambda_hat`** — the two are decoupled. The adjoint norm is a third, independent handle on
amplitude, and one measured on the linearization rather than on a finite twin.

MITgcm supplies the reference: `||S_mitgcm(90)|| / ||S_mitgcm(10)||` is what the true system
does. In a non-chaotic S0 both should be modest, and the adjoint growth rate is then an
independent confirmation of `s0-not-chaotic` derived from the linearization instead of from
twin runs.

### 8.3 Where in the 90 days the agreement breaks

Both sides deliver the adjoint state at every intermediate lead (§5.2), so the divergence
is resolvable in time rather than reported as one number at day 90. Plot
`relative_l2(S_forced(L), S_mitgcm(L))` against `L`. A curve that is flat and small until
lead 40 and then rises localises the failure; a curve that rises linearly from lead 10 says
the ten-day Jacobian is uniformly slightly wrong and the error simply compounds. Those are
different defects with different fixes, and only the sweep distinguishes them.

---

## 9. Caveats — read before interpreting any map

### 9.1 `ivdc_kappa = 1.` is still not differentiable

The convective switch flips vertical diffusivity between 1e-5 and 1.0 on the sign of the
local stratification. TAF differentiates the branch actually taken, which is correct almost
everywhere and wrong on a measure-zero set — and near it the finite difference and the
adjoint legitimately disagree because the perturbed run flips a cell's convective state.

Over 90 days there are 9x more opportunities to sit near a flip than over 10. Quantify it:
count convecting columns per timestep from `P90`'s forward output and state the number,
rather than leaving it an unbounded worry. Keep `ivdc_kappa = 1.` in production. If G1-90
fails on northern columns only, a diagnostic-only `ivdc_kappa = 0.` rebuild isolates it and
is reportable — it does not become the deliverable.

### 9.2 The comparison is between two different discretizations of time

MITgcm's derivative is with respect to a continuous-in-time trajectory sampled at day
boundaries; the emulator's is with respect to its own 10-day discrete map, nine times.
Agreement is the scientific result. Note that handbook §2 lists `dt = 10 d` as the
strongest point of agreement with Bire — this is not a knob that will be adjusted to
improve the comparison.

### 9.3 What is *not* being addressed

Phase A is a 10-to-90-day sensitivity study. It does not settle the day-2000 question, and
`gate-long-horizon-is-90-360` records that `worst_long_ratio_to_climatology` never looks
past day 360 anyway, so day-2000 RMSE worse than climatology passes unscored. The emulator
side of a day-2000 adjoint is 200 backward passes and entirely feasible; the MITgcm side is
not, and would need a checkpointing redesign. If the long-horizon question becomes the
focus, the emulator-only sweep has to stand alone, and the honest framing is then "what the
emulator believes", not "what is true".

Phase A is also S0-only. `s0-not-chaotic` means S0 is a **friendly** case for an adjoint
study — the linearization should hold, which is why it is the right first experiment and
also why it is not a hard test of error growth. The turbulent 0.25-degree campaign
(`turb-s0-s1-s2-campaign`) is where a Bire-comparable, genuinely chaotic adjoint comparison
would live. Not here.

### 9.4 One structural confound is absent by construction

Both models have global instantaneous domain of dependence — the spectral convolutions are
global, and MITgcm's implicit free surface solves an elliptic problem over the whole basin
every timestep. So no part of any disagreement can be attributed to one model being local
and the other global. Worth stating in every figure caption, because it is the first thing
a reader will otherwise assume.

---

## 10. Deliverables

```
docs/            Adjoint_study_Phase_A.md                        (this file)

config/          adjoint_phase_a_v1.json                         frozen window, leads,
                                                                 both weight definitions,
                                                                 checkpoint + normalizer SHAs

work/            costWeight_ssh_anomaly.bin                      (existing, reused)
                 costWeight_mean_only.bin                        (existing, reused)
                 costWeight_ssh_anomaly_kernel_sigma1.bin        new

scripts/         make_cost_weight.py            + kernel QoI
                 stage_adjoint_run.py           + --start-day / --days
                 extract_mitgcm_adjoint.py      + multi-lead, multi-objective
                 fno_adjoint_ft90.py            new; replaces the two-in fno_adjoint.py
                 compare_adjoint_maps_phase_a.py new
                 adjoint_metrics.py             unchanged

slurm/mitgcm/    af_s0_adjoint_run.sbatch       parameterised, --exclusive
                 af_s0_adjoint_grdchk.sbatch    parameterised epsilon

outputs/af_fno/adjoint/
    mitgcm_s0_adjoint_v1/           frozen, not touched
    fno_s0_adjoint_v1/              stale (two-in model), kept for provenance, not read
    mitgcm_s0_adjoint_v2/           Phase A reference: .npz, report.json, figures
    fno_ft90_s0_adjoint_v1/         Phase A test:      .npz, report.json, figures
    comparison_phase_a_v1/          the tables, the lead curves, the spectra
```

Array keys, keyed identically on both sides so one loader reads both:

| array | shape | content |
| --- | --- | --- |
| `S_point` | (4, 62, 62) | leads 10/20/30/90, `J_point`, sweep (B) |
| `S_kernel` | (4, 62, 62) | leads 10/20/30/90, `J_kernel`, sweep (B) |
| `S_backward` | (91, 62, 62) MITgcm / (9, 62, 62) FNO | sweep (A) from the lead-90 run |
| `S_mean_only` | (4, 62, 62) | conservation probe |
| `lead_days` | (4,) | 10, 20, 30, 90 |
| `wet_mask`, `rA`, `target_ij` | — | copied from the shared contract, never recomputed |

FNO-side only: `S_forced` and `S_free` variants of each of the above, plus
`land_leakage_unmasked`.

`report.json` on both sides carries the executable or checkpoint SHA-256, the normalizer
SHA-256, the **weight-field SHA-256 which must match across the two**, TAF/torch versions,
dtype, the full gate table, and the frozen contract.

---

## 11. Order of execution

```
  0. freeze config/adjoint_phase_a_v1.json                          no compute
  1. make_cost_weight.py --qoi ssh_anomaly_kernel                   seconds
     assert all 5 stencil cells wet, centroid == p*
  2. FNO A3, conservation probe, all four leads, both chains        minutes
     ^ a complete, interpretable, publishable result with no MITgcm run at all
  3. FNO gates F1, F3, F4, F5, F6                                   minutes
  4. FNO A1 + A2, all leads, both objectives; gate F2 at every lead ~1 h incl. FD sweeps
  5. write fno_ft90_s0_adjoint_v1/                                  ---
     --- everything above needs no Slurm job, no TAF, no MITgcm ---
  6. stage + run F90 (forward, 90 d + 1 tail)     -> Gate G0, and the day-7280
                                                     pickup V10 needs
  7. stage + run C90 (mean-only, 90 d)            -> Gate G3
     stage + run P90                              -> Gate G2a, S_90, sweep A
     stage + run V10 (3 min)                      -> Gate G2b
  8. G90 grdchk, epsilon chosen from P90's |S|    -> Gate G1-90
     --- do not compare against MITgcm before G1-90 passes ---
  9. P10, P20, P30                                -> S_10, S_20, S_30
 10. K10 … K90                                    -> the kernel objective
 11. extract_mitgcm_adjoint.py                    -> Gates G4, G5
 12. compare_adjoint_maps_phase_a.py              -> §7 tables, §8 diagnostics
 13. report
```

Steps 0–5 are the majority of the interpretive value and need no licence, no Slurm job and
no MITgcm run. Start there. Step 2 in particular produces a finished number — the
emulator's global sea-level conservation error, resolved in space and in lead, measured
against an analytically exact reference — before a single job is submitted.

**Step 8 is a hard gate.** An unvalidated `adxx_etan` at a new window is not ground truth.
The executable's v1 gates certify the *build*; they do not certify a 90-day tape. Until
G1-90 clears, the emulator maps are self-contained results and the conservation probe is
the only thing that may be reported as final.

---

## 12. Decisions taken here, and the alternatives rejected

| # | decision | rationale |
| --- | --- | --- |
| 1 | primary objective = **`J_point`** | the only objective with a passed `grdchk` on this executable; sharpest possible ask; `p*` was frozen for it |
| 2 | smooth kernel = **5-point meridional Gaussian, sigma = 1, radius 2** | one-cell Munk layer forbids zonal smoothing; the land rim displaces any symmetric 2-D stencil 0.504 cells off the jet; all five cells wet; sigma is a runtime knob so the choice is cheap to revisit |
| 3 | lead sweep = **fixed source day 7200, moving cost** (B) | isolates propagation length from initial condition; matches the fine-tune's own rollout axis; sweep (A) comes free from the lead-90 run and is reported as a companion |
| 4 | primary MITgcm comparison = **`S_forced`** | matched object: MITgcm's adjoint is linearized about the truth trajectory by construction |
| 5 | primary operational map = **`S_free`** | what a deployed user actually gets; the difference from `S_forced` is the trajectory-drift term, reported explicitly |
| 6 | **float64** on the FNO side | `s0-twin-float32-floor`; the float32 difference is reported as F4, not chosen between |
| 7 | `cg2dTargetResidual` stays **1e-7** in production, 1e-12 for `grdchk` only | tightening it differentiates a model the emulator never saw; `grdchk-limited-by-cg2d` shows the FD, not the adjoint, is what 1e-7 limits |
| 8 | **no threshold** on any comparison metric | first measurement of this quantity for this model |
| 9 | full 46-channel adjoint **archived, not analysed** | available free on both sides; a multi-field comparison is Phase B, not a rider on this one |
| 10 | Phase A pins the **ft90** checkpoint | the handbook's declared next experiment is `rho = 0.99`; keeping the two attributable means the adjoint battery runs against a frozen operator and is re-run unchanged against the next one |
