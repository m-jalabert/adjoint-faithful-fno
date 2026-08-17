# FNO adjoint — generation and comparison plan

Companion to `docs/mitgcm_adjoint_ground_truth_plan.md`. That document produces the
reference sensitivity map from MITgcm via TAF. This one produces the same derivative
through the frozen FNO, and defines the comparison.

Target deliverable:

    S_fno[i,j] = dJ / d eta(i,j,t0)      through     x(t+10) = F(x(t-10), x(t), s)

with **J the identical scalar** the MITgcm side uses, on the identical window, in the
identical units, so that `S_fno` and `S` can be subtracted.

Model under test: `model_c_2in_1out_new_channels_p_cont_BT_loss_v1`, step 3,840,
checkpoint `bf3ccc70...`, 27,328,780 parameters. **Frozen** — no weights are trained,
loaded for fine-tuning, or modified anywhere in this plan.

---

## 0. Why this is cheap, and what that buys

Reverse-mode AD through the FNO is one `loss.backward()`. There is no TAF, no licence, no
source transformation, no checkpointing scheme — PyTorch already holds the adjoint of
every operator in the graph. A 62x62 map costs one forward and one backward pass:
milliseconds on the V100, seconds on CPU.

That asymmetry is the point. The MITgcm side is expensive and gated; the FNO side is
free. So the FNO side should be run **first and exhaustively** — every target cell, every
lead, every diagnostic — and the MITgcm runs reserved for the specific comparisons the
FNO survey says are interesting.

---

## 1. The contract with the MITgcm side

Any convention that differs between the two sides shows up as a difference in the maps
and will be mistaken for a model error. Five things must be shared, and four of them are
shared by *reading the same file* rather than by reimplementing.

| Quantity | Source of truth | How the FNO side gets it |
|---|---|---|
| Target cell p* | `config/mitgcm_adjoint_s0_target_v1.json` | read the JSON |
| Wet area `A_wet` | same JSON | read the JSON |
| Weight field `w` | `work/costWeight_ssh_anomaly.bin` | **read the same binary**, big-endian float32 |
| Wet mask | `trajectories_v3.zarr::wet_mask` | read the zarr |
| Window | days 7210 to 7220 | frozen below |

Reading `costWeight_ssh_anomaly.bin` rather than rebuilding `w` in the FNO script is
deliberate and non-negotiable. It removes at a stroke the whole class of errors where the
two sides disagree about the plain-vs-area-weighted mean, the value at p*, the float32
rounding of `rA/A_wet`, or the index order. Both sides then compute

    J = sum_ij  w[i,j] * eta[i,j](t = 7220)

from bit-identical `w`. Gate F1 checks this.

**Units.** `w` is dimensionless by construction and eta is in metres, so J is in metres
and `S_fno` is dimensionless — metres of J per metre of eta, directly comparable to
MITgcm's `adxx_etan`. A value of 0.2 means 1 cm at the source gives 2 mm at the target.

**Sign.** `S > 0` means raising eta at (i,j) raises the target anomaly.

**Grid.** Cell centres, `(j, i)` index order matching the zarr's spatial axes; land is
exactly 0 on the MITgcm side.

---

## 2. The gradient chain, and the one trap in it

The FNO does not operate on physical SSH. It operates on pointwise-normalized state:
`normalized = (physical - mu) / sigma`, with `mu`, `sigma` of shape `(46, Y, X)` from
`model_c_2in_1out_new_channels_p_cont_BT_loss_train_only_normalization.npz` (inherited
verbatim from the parent `ad7cc858...`, unchanged since arm 3).

So the derivative we want is not the derivative the network computes:

    dJ/d(eta_phys)  =  dJ/d(eta_norm)  *  1 / sigma_45

and `sigma_45` is a **field**, `(Y, X)`, not a scalar — the normalizers are pointwise.
Applying a scalar, or forgetting the factor entirely, produces a map that looks
qualitatively right and is quantitatively meaningless.

**Do not apply the correction by hand.** Build the graph so the physical field is the
leaf:

    eta_phys  (requires_grad=True, float64)
        |  normalize:  (x - mu)/sigma                 <- inside the graph
        v
    x_norm  ->  assemble 97 channels  ->  F(.)  ->  y_norm
                                                      |  denormalize: y*sigma + mu
                                                      v
                                                eta_hat_phys(7220)
                                                      |  J = sum(w * eta_hat)
                                                      v
                                                J.backward()
                                                      |
                                                eta_phys.grad  =  S_fno

Autograd then carries every factor of `sigma` automatically, including the output-side
one, and the answer is in physical units with no post-hoc rescaling. Gate F2 verifies this
against a finite difference taken in physical units, which is the check that catches a
missing or mis-shaped `sigma`.

### Channel indices — verify, do not assume

The dataset stacks `(x_{t-10}, x_t)` ahead of all five static fields
(`model.py::retained_two_in_features`), giving 97 input channels:

| Block | Channels | Contents |
|---|---|---|
| history `x_{t-10}` | 0 - 45 | U(15), V(15), theta(15), **eta = 45** |
| present `x_t` | 46 - 91 | U(15), V(15), theta(15), **eta = 91** |
| static | 92 - 96 | wind, wet mask, Coriolis, dx, SST target |

The two positional-encoding channels are appended inside the operator (99 lifting inputs)
and are not part of the external contract. A preflight assertion must confirm channel 91
is the present-state eta before any gradient is taken: perturb channel 91 by a known
amount and confirm only the SSH field of the denormalized present slot moved.

### Precision

Run in **float64**. The network was trained in float32, but casting the weights to double
does not change the function it represents — only the arithmetic that evaluates it. A
float32 gradient carries ~1e-7 relative noise, the same order as the finite-difference
signal at the epsilons where the FD check is most informative, and this project has
already been bitten once by a float32 quantisation floor swallowing a real signal
(`s0-twin-float32-floor`). Cast once, at load, and record that the cast happened.

Report the float32-vs-float64 gradient difference as a diagnostic (Gate F4). If it is
large, the map is dominated by round-off and the comparison needs re-scoping.

---

## 3. The experiments

Four, all on the same frozen model, all reading the same `w`.

### E1 — present-slot sensitivity  (primary; compares to MITgcm Run A)

| | |
|---|---|
| Inputs | history `x_7200`, present `x_7210`, both truth from the zarr |
| Differentiate w.r.t. | eta of the **present** slot (channel 91), physical units |
| Held fixed | U, V, theta of both slots; all of `x_7200`; the five statics |
| Output | 62x62 map `S_fno_present` |
| MITgcm counterpart | Run A `adxx_etan`, dJ/deta(day 7210) |

This is the headline number and the one the preliminary study specified.

### E2 — history-slot sensitivity  (FNO-only)

Same forward pass, differentiate w.r.t. channel 45 instead. **There is no MITgcm
counterpart** — see section 4. Report it separately; never sum it with E1 unless the sum
is explicitly the question being asked.

### E3 — two-call rollout  (compares to MITgcm Run B)

| | |
|---|---|
| Call 1 | `(x_7190, x_7200) -> x_hat_7210` |
| Call 2 | `(x_7200, x_hat_7210) -> x_hat_7220` |
| Differentiate w.r.t. | eta of `x_7200`, physical units |
| Output | 62x62 map `S_fno_20day` |
| MITgcm counterpart | Run B `adxx_etan`, dJ/deta(day 7200) |

`x_7200` enters **twice** — as the present slot of call 1 and the history slot of call 2 —
so autograd returns the total derivative through both paths. That is the right analogue
of MITgcm's 20-day adjoint, and it is why E3 is not E1 composed with itself.

Extend to 3, 6 and 20 calls for a lead sweep matching Run B's `ADJetan` dumps. Each
additional lead is one more backward pass.

### E4 — the conservation probe  (the sharpest single diagnostic here)

Re-run E1 with `w = costWeight_mean_only.bin`, i.e. J = -(area-weighted wet mean of eta)
alone.

MITgcm's answer is **analytically known**: because `implicitFreeSurface` with
`exactConserv` conserves the area integral of eta exactly in a closed basin with no
freshwater flux, the adjoint of that functional is constant in time, so the returned map
equals `w` itself. That is Gate G3 on the MITgcm side.

The FNO conserves no such thing. Its map for this functional therefore *is* a direct,
spatially resolved measurement of how badly the emulator violates global sea-level
conservation — and it needs no MITgcm run at all to interpret, because the correct answer
is known in closed form. Run this first; it is the cheapest meaningful result in the
study.

---

## 4. The conceptual asymmetry — read before interpreting E1 or E2

MITgcm is **Markov in its state**: the pickup at day 7210 is a complete initial condition,
and dJ/deta(7210) is a total derivative with respect to it. The FNO is **not**: its input
is a pair, and its present-slot derivative is a *partial* derivative holding `x_7200`
fixed.

Three consequences, all of which belong in any figure caption:

1. **The FNO derivative is taken off the training manifold.** On the real trajectory
   `x_7200` and `x_7210` are dynamically linked; perturbing one and not the other produces
   an input pair the operator never saw in training. Two emulators that agree perfectly on
   the manifold can disagree arbitrarily in this derivative. The comparison is therefore a
   test of *off-manifold* behaviour, which is strictly harder than the forecast skill the
   acceptance gate measured — and is exactly why it is worth doing.

2. **A non-zero E2 is not automatically an artifact.** MITgcm's own time stepping is
   Adams-Bashforth and its pickup carries the previous tendency, so the true discrete
   model also has two-time-level character. The analogy is loose — the FNO's lag is 720
   MITgcm timesteps, not one — so do not push it into a quantitative claim. But "the FNO
   uses history, therefore the FNO is wrong" does not follow.

3. **E1 + E2 answers a different question** than either alone: the response to an SSH
   offset applied at both input times. If that sum matches MITgcm's Run A map better than
   E1 does, that is a real and reportable finding about how the operator distributes
   dependence across its two slots — but it is a hypothesis to test, not the primary
   metric. Declare E1 as primary before looking.

One structural similarity that *helps*: both models have global instantaneous domain of
dependence. The FNO's spectral convolutions are global by construction, and MITgcm's
implicit free surface solves an elliptic problem over the whole basin at every timestep.
So no part of any disagreement can be attributed to one model being local and the other
global. That confound is absent by construction.

---

## 5. Validation gates — the FNO side

The MITgcm side has `grdchk`. The FNO side needs its own, and it is far cheaper.

| Gate | Condition | Method |
|---|---|---|
| **F1** | J from the FNO pipeline on **truth** `eta_7220` matches numpy `(w*eta).sum()` to < 1e-10 relative | same `w`, same zarr slice |
| **F2** | Finite difference: `abs(FD/adjoint - 1) < 1e-6` at 8 or more cells, over eps in {1e-2, 1e-3, 1e-4, 1e-5} m, with a visible plateau | central difference, physical units, float64 |
| **F3** | Channel preflight: perturbing input channel 91 moves only present-slot eta; channel 45 only history-slot eta | assertion before any gradient |
| **F4** | float32 and float64 gradients agree to < 1e-4 relative | otherwise round-off dominated |
| **F5** | E4 with `mean_only`: report `norm(S_fno - w) / norm(w)` | no threshold — a measurement, not a pass/fail |

F2 is the FNO's `grdchk` and deserves the same discipline: a plateau, not a single
epsilon. The FNO is smooth (GELU, spectral convolutions, layer norm — no branches), so the
plateau should be wide and the agreement far tighter than MITgcm's 1e-4, where
`ivdc_kappa` discontinuities set the floor. **If F2 is loose, it is a bug in the pipeline,
not physics** — there is no convective-adjustment excuse on this side.

F2's test points must include p*, a western-boundary cell, a mid-basin cell, and **a land
cell** (section 6).

---

## 6. Comparison metrics

Written into `scripts/compare_adjoint_maps.py`, reading the MITgcm `.npz` from section 10
of the ground-truth plan and the FNO `.npz` from section 7 below.

### Primary

| Metric | Definition | Why |
|---|---|---|
| Pattern correlation | Pearson `r` over wet cells | does the FNO put sensitivity in the right places |
| Relative L2 | `norm(S_fno - S) / norm(S)` over wet cells | overall error |
| Amplitude ratio | `norm(S_fno) / norm(S)` | systematic over/under-response |
| Sign agreement | fraction of wet cells with matching sign | structural fidelity |

### Structural — where this connects to what is already known

| Metric | Why it matters here |
|---|---|
| **Land-cell leakage**, `max(abs(S_fno))` over the 244 dry cells | MITgcm is **exactly** 0 there: eta on land is not a degree of freedom. Any non-zero FNO value is unambiguously spurious. The spectral convolutions are global, so leakage is expected — this quantifies it against a known-exact zero, with no modelling assumption. |
| Western-band vs interior split, via `dataset.western_boundary_mask(wet, 4)` | The project's existing convention. `western-boundary-ratio-degrades` records the day-2000 boundary/interior anomaly ratio moving 4.25 to 3.20 against a truth of 23.1, unscored by the gate. If the sensitivity maps disagree most in that band, the adjoint has localised a defect the forecast gate is blind to. |
| Radial decay from p* | e-folding distance of `abs(S)`; compares effective range of influence |
| Spectral power by radial wavenumber bin | reuse the 12-bin tapered convention from the spectral loss. `local-branch-gamma-ablation` warns the high-k *fraction* misleads, so report absolute power per bin, never the fraction |

### Deliberately not a gate

No pass/fail threshold is declared on the primary metrics. This is the first measurement
of this quantity for this model; inventing a threshold before seeing the number would be
reverse-engineering a verdict. Gates F1 to F4 guard the *pipeline*; the science is
reported, not graded.

---

## 7. Outputs

`scripts/fno_adjoint.py` writes `outputs/af_fno/adjoint/fno_s0_adjoint_v1/`, keyed to
mirror the MITgcm extractor so `compare_adjoint_maps.py` reads both with one loader.

| Array | Shape | Content |
|---|---|---|
| `S_fno_present` | (62, 62) | E1 |
| `S_fno_history` | (62, 62) | E2 |
| `S_fno_20day` | (62, 62) | E3 |
| `S_fno_lead` | (L, 62, 62) | E3 lead sweep |
| `lead_days` | (L,) | descending |
| `S_fno_mean_only` | (62, 62) | E4 |
| `wet_mask`, `rA`, `target_ij` | — | copied from the shared contract, not recomputed |

Plus `report.json`: checkpoint SHA-256, normalizer SHA-256, weight-field SHA-256 (**must
equal the MITgcm run manifest's**), torch version, dtype, the F1 to F5 table, and the
frozen contract.

---

## 8. Risks

**The weight field must not be rebuilt.** If `fno_adjoint.py` ever computes `w` itself
instead of reading `costWeight_ssh_anomaly.bin`, the comparison silently becomes a
convention test. The SHA-256 cross-check in `report.json` is the guard; make it an
assertion, not a note.

**The checkpoint must be loaded strictly.** `strict=True`, no partial loads, no
`map_location` surprises that silently drop buffers. Verify the parameter count is
27,328,780 before the first forward pass.

**Truth inputs come from the zarr, not from a re-run.** Days 7190, 7200, 7210 are read
from `trajectories_v3.zarr`. The MITgcm side's Gate G0 separately confirms the zarr and a
fresh MITgcm restart agree bit-for-bit, so both sides provably start from the same
numbers — but only if the FNO side reads the archive rather than the re-run directory.

**Do not compare against MITgcm before G1 passes.** An unvalidated `adxx_etan` is not
ground truth. Until the MITgcm gradient check clears, the FNO maps are self-contained
results — E4 especially, which needs no MITgcm run at all.

**Scope.** This is a 10-day and 20-day sensitivity. It does not address the day-2000
question, for the reasons in section 12.5 of the ground-truth plan. Extending the FNO side
to day 2000 is 200 backward passes and entirely feasible; extending the MITgcm side is
not. If the long-horizon question becomes the focus, the FNO-only lead sweep may have to
stand alone, and the honest framing is then "what the emulator believes", not "what is
true".

---

## 9. Order of execution

     1. preflight: load checkpoint, verify 27,328,780 params,
        verify channel 91 is present-slot eta                     -> gate F3
     2. J on truth eta_7220 vs numpy                              -> gate F1
     3. E4 (mean_only): conservation violation                    -> gate F5
        ^ a complete, interpretable result with no MITgcm run
     4. E1 in float64; finite-difference sweep at 8 points        -> gates F2, F4
     5. E2, E3, lead sweep
     6. write the .npz + report.json
     --- MITgcm Gate G1 passes ---
     7. compare_adjoint_maps.py

Steps 1 to 6 need no TAF licence, no Slurm job and no MITgcm run. They can start now.
Step 3 in particular produces a publishable number — the FNO's global sea-level
conservation error, resolved in space, measured against an analytically exact reference —
before the licence arrives.

---

## 10. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Primary FNO map: E1 alone, or E1+E2 | **E1 alone.** Declare before looking (section 4, point 3). Report the sum as a secondary hypothesis. |
| 2 | Lead sweep depth for E3 | 1, 2, 3, 6, 20 calls (10 to 200 days). Cheap; the MITgcm side only reaches 20 days at the present `nchklev`. |
| 3 | Additional target cells | Free on this side. Recommend the separation point (i=2, j=28) and a wall-excluded control (i=4, j=25) — the two runners-up from the p* decision. |
| 4 | float64 throughout | **Yes** (section 2). Record the float32 difference as F4 rather than choosing between them. |
