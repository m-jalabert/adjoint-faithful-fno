# MITgcm adjoint ground truth via TAF — generation plan

Target deliverable: the exact MITgcm sensitivity map

$$S^{10}_{ij} \;=\; \frac{\partial J}{\partial \eta(i,j,t_0)},
\qquad
J \;=\; \eta'(p^\star, t_0+10\ \text{days}),
\qquad
\eta' = \eta - \langle \eta \rangle_{\mathrm{wet}},$$

on the S0 (τ₀ = 0.1 N m⁻²) `tutorial_baroclinic_gyre` configuration, for direct comparison
against the same derivative taken through the frozen FNO
`model_c_2in_1out_new_channels_p_cont_BT_loss_v1`.

This plan produces the *reference* half of that comparison only. It loads no FNO weights,
reads no `outputs/af_fno/C/**`, and writes nothing into the FNO tree.

---

## Status — 2026-08-17

TAF is licensed and working (`staf`, TAF **6.8.11**). `mitgcmuv_ad` is built. Gate G0 passes.

Four things in the plan below were wrong and are corrected in place; this section records
what they were, because three of them fail *silently*.

**1. `ALLOW_ETAN0_CONTROL` is dead code in c68j — this was the big one.**
Every block of that flag — `ctrl_init.F:552`, `ctrl_map_ini.F:531`, `ctrl_pack.F:613`,
`ctrl_unpack.F:703`, `grdchk_getxx.F:601` — sits inside `#ifdef ECCO_CTRL_DEPRECATED`,
which is defined nowhere in this checkout. So §4.6 below was reading real line numbers in
code that never compiles. `CTRL_MAP_INI` reduced to an empty subroutine, `xx_etan_dummy`
never reached `etaN`, and TAF returned

```
 independent variable(s) =
TAF WARNING the independent variables have no influence on the variables : fc
```

— a complete, valid build whose gradient is identically zero. It cost a TAF submission to
find out. The live path in c68j is the generic-array control: `#define
ALLOW_GENARR2D_CONTROL`, `xx_genarr2d_file(1)='xx_etan'` in `CTRL_NML_GENARR`, and
`grdchkvarindex = 101` (= 100 + iarr). `CTRL_MAP_INI_GENARR` then calls
`CTRL_MAP_GENARR2D( etaN, ... )`, which does the `ACTIVE_READ_XY` against
`xx_genarr2d_dummy(1)`. After the switch TAF reports `independent variable(s) =
xx_genarr2d_dummy` and the generated source grows from 3.1 MB to 5.6 MB.

Two traps inside the generic path, both silent:

- `xx_genarr2d_weight` must be **non-blank** or `ctrl_init.F:807` never registers the
  control and `ctrl_map_ini_genarr.F:117` never matches it — a second route to a zero
  gradient. The named file is read unconditionally at `ctrlprec` (= 64).
- `xx_genarr2d_preproc(1,1)='noscaling'` is **required**. Without it
  `ctrl_map_genarr.F:142` divides the control by `SQRT(wgenarr2d)`, which rescales the
  returned gradient by the same factor. The staged weight is exact ones so that dropping
  `noscaling` is a no-op rather than a corruption.

**Pre-flight that would have caught it, for any future change:** `make ad_input_code.f`
needs no licence. Grep the payload for the chain `CALL CTRL_MAP_GENARR2D( etaN` →
`ACTIVE_READ_XY( ..., xx_genarr2d_dummy(iarr) )` → `fld = fld + xx_gen*mask2D` before
spending a submission.

**2. `etaH` is not a second control, and does not need to be.** §12.4 flagged that the
deprecated path perturbed both `etaN` and `etaH` while the FNO has no `etaH` counterpart.
The generic path perturbs `etaN` only, and that is strictly better: `INITIALISE_VARIA`
calls `INTEGR_CONTINUITY` *after* `PACKAGES_INIT_VARIABLES`, and with `implicDiv2Dflow = 1`
(the default here) `UPDATE_ETAH` sets `etaH = etaN` before the first timestep. `etaH` is a
dependent diagnostic. The caveat in §12.4 is withdrawn.

**3. The adjoint sbatches launched MPI with a bare `srun`,** which fails at `MPI_Init`
("PML add procs failed"). Every S0 forward segment was launched by `af_s0.py:270` as
`srun --mpi=pmix -n 4`. All three adjoint sbatches now match it.

**4. Gate G0 was comparing staggered against centred.** `trajectories_v3.zarr` stores U
and V at **cell centres** — `af_data.py:178-179` applies
`0.5*(u + roll(u,-1,-1))` and `0.5*(v + roll(v,-1,-2))` — while the raw MDS `dynState`
holds C-grid face values. Comparing the two made every wet velocity differ by O(1e-1)
while `THETA` and `ETAN` matched bit-for-bit, which was a statement about the C grid and
not about the restart. `verify_gate_g0.py` now applies the dataset's own operator.
Relatedly, `data.diagnostics` sets `dumpAtLast=.FALSE.`, so the state at the final
iteration is never dumped; the archive only holds day 7220 because that day is interior to
a six-year production block. The pickup run therefore steps one day past day 7220
(`tail_days = 1`, 1512 steps) so the day-7220 snapshot is produced by the same mechanism
that produced the archived one.

**5. `grdchk` ignored the requested test cell.** `grdchk_main.F:213` calls
`GRDCHK_GET_POSITION` *only* when `nbeg .EQ. 0`; with any other value `icomp` indexes the
packed control vector directly. The plan's `nbeg = 1` therefore tested the first wet cell,
(i=2, j=2), while printing a perfectly plausible `grad-res` block — it never touched p⋆.
With `nbeg = 0` the log reads `grad-res exact position met`. Note also that
`GRDCHK_GET_POSITION` does `nend = nbeg + nend`, so the namelist `nend` is an *offset*, not
an absolute index: `nend = 0` means exactly one point. And the plan's land test point is
unreachable here — the search only ever lands on cells with `wetlocal .NE. 0` — so land is
covered by G4 over all 244 land cells instead, which is the stronger statement.

**6. `ADJetan` was being dumped at float32.** `DUMP_ADJ_XY` writes through
`WRITE_FLD_XY_RL`, which honours `writeBinaryPrec`, default 32 — putting a float32 floor
directly under G2 and G3, while `adxx_etan` came back at float64 via `ctrlprec`. The two
halves of G2 were not even the same precision. The adjoint runs now set
`writeBinaryPrec = 64`; this is safe for the forward snapshots because `data.diagnostics`
sets `fileFlags='R'`, which `diagnostics_out.F:396` lets override `writeBinaryPrec`, so
`dynState`/`surfState` stay float32 and stay comparable to the archive.

**7. G5's 1e-10 tolerance is unreachable, for a reason worth stating.** `fc` is accumulated
from the model's float64 state; the archived η is the **float32** diagnostic snapshot. The
residual is 1.54e-08 against a computed float32 half-ulp bound of 1.26e-07 on this
particular weighted sum — 12% of the floor. The gate now compares against that computed
bound rather than a constant, which is the same discipline `s0-twin-float32-floor` forced
on the daily diagnostics.

## Results

| Gate | Result |
|---|---|
| **G0** | **PASS** — all 21 days, all 46 channels, `max\|diff\| = 0` |
| **G1** | **PASS at all 7 points**, once the finite difference is computed accurately enough to be a fair reference; worst 2.0e-06 against a 1e-4 tolerance, plateau visible. At the production solver tolerance the *finite difference* is too noisy to certify anything — see below |
| **G2** | **PASS** — relative L2 **exactly 0**: Run B's day-7210 `ADJetan` is bit-identical to Run A's `adxx_etan` |
| **G3** | **PASS** — worst relative L2 **3.57e-08** across all 11 dump times, tolerance 1e-5 |
| **G4** | **PASS** — exactly 3 600 non-zero cells, all 244 land cells exactly 0, everything finite |
| **G5** | **PASS** — 1.54e-08, against the 1.26e-07 float32 floor of the archived η |
| fc consistency | **PASS** — Run A and Run B both return `fc = 0.322547974637434` |

### G1 and the cg2d tolerance — §12.2, confirmed

At the production `cg2dTargetResidual = 1e-7`, all seven test points return `FD/adjoint` of
1 ± 0.01 to 0.04, and — the diagnostic detail — **the error does not shrink as ε shrinks**.
That is the signature of a noise floor in `fc`, not of nonlinearity: there is no plateau to
find because the finite difference itself is noisy.

The noise is in the finite difference, not the adjoint, and two independent lines of
evidence say so. First, G2 and G3 involve no finite differences at all and come back at 0
and 3.6e-08. Second, §12.2's prescribed test: rerunning the identical sweep with
`cg2dTargetResidual = 1e-12` makes **every point pass at every epsilon**, with the plateau
the gate asks for visible across 1e-3 → 1e-5 and degrading only at 1e-6 where round-off
takes over — exactly the predicted shape.

`FD / adjoint`, `cg2dTargetResidual = 1e-12`:

| test point | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|
| p⋆ (i=2, j=17) | 1.000000015 | 1.000000208 | 1.000000421 | 0.999982321 |
| WBC upstream (2, 14) | 0.999999999 | 1.000000019 | 1.000000088 | 1.000000644 |
| WBC upstream (2, 11) | 1.000000021 | 0.999999918 | 1.000001625 | 1.000009238 |
| offshore (4, 17) | 1.000000008 | 1.000000102 | 0.999999395 | 1.000006512 |
| interior (31, 17) | 0.999999852 | 1.000000072 | 0.999998590 | 1.000034010 |
| eastern (61, 17) | 1.000000014 | 1.000000040 | 0.999998026 | 1.000009385 |
| northern (31, 55) | 1.000000118 | 1.000000256 | 1.000000029 | 0.999999802 |

Worst deviation at the two gate epsilons is 2.0e-06, fifty times inside the 1e-4 tolerance;
at 1e-7 the same points were 1e-2 to 4e-2 away and flat in ε. The adjoint gradient itself
moves by only ~5e-06 relative between the two solver settings (p⋆: 9.262443e-05 →
9.262491e-05). The forward solve converging to 1e-7 was limiting how accurately `fc(±ε)`
could be differenced; it was never limiting the adjoint.

**The linear range** (§12.3, which asks that it be reported rather than claimed): flat to
within 2e-6 from ε = 1e-3 down to 1e-5 at every point, consistent with `s0-not-chaotic`.

**This does not change the deliverable.** The production maps keep `cg2dTargetResidual =
1e-7`, because that is what every S0 forward segment was integrated with and what the FNO
was trained on; tightening it would produce the adjoint of a model the FNO never saw. The
tightened runs are diagnostic only and are labelled as such in the run manifests
(`cg2d_target_residual`) and archived separately under `grdchk_cg2d1em12`.

Convection (§12.1) is ruled out as the cause: the discrepancy is uniform across the domain,
including the mid-basin interior and eastern boundary, rather than confined to columns that
convect.

---

## Status — 2026-08-12

Stages 1–5 of section 15 are **built and verified**; the build stops at the TAF
licence exactly as predicted, and nothing else is blocking.

| Step | Result |
|---|---|
| `scripts/select_adjoint_target.py` | p⋆ = **(i=2, j=17)**, 30.5N, mean surface speed 0.8414 m/s. `A_wet` = 3.0046402806e13 m². Frozen in `config/mitgcm_adjoint_s0_target_v1.json`. |
| `scripts/make_cost_weight.py` | `work/costWeight_ssh_anomaly.bin` and `work/costWeight_mean_only.bin`, 62×62 big-endian float32. w[p⋆] = +0.99964553; Σw over wet = −1.9e−08. |
| `af_fno/mitgcm/code_ad/` | 12 files written. |
| `af_fno/mitgcm/input_ad/` | 8 files written. |
| `tests/test_mitgcm_adjoint.py` | **28 passed.** |
| `genmake2` | Makefile generated, no warnings. TAF rule at `build/af_s0_ad/Makefile:1979`; `xx_etan_dummy` confirmed in the TAF `-input` list. |
| `make depend` | exit 0, 760 `.F` linked (626 forward + 134 AD). |
| `make ad_input_code.f` | **payload built: 220,723 lines, 7,951,073 bytes.** Verified to contain the expanded `cost_test.F` body, the `AF_COST_WEIGHT_C` common block, `xx_etan_dummy`, and `nchklev_1 = 72`. |
| `make adall` | `staf: No such file or directory` → `Makefile:1979: ad_taf_output.f Error 127`. **The licence wall, and nothing before it.** |
| `scripts/stage_adjoint_run.py` | 4 modes (`pickup`/`grdchk`/`runA`/`runB`). `pickup` staged and verified; PARM01/02/04/05 confirmed byte-identical to `af_s0.render_data`. |
| `scripts/verify_gate_g0.py` | written; runs after the pickup job. |
| `slurm/mitgcm/af_s0_adjoint_{pickup,build,grdchk,run}.sbatch` | written, `bash -n` clean. |

Not yet run: the pickup job itself (Slurm), and therefore Gate G0. It needs no
licence — submit `slurm/mitgcm/af_s0_adjoint_pickup.sbatch` whenever convenient.

Two findings changed the plan as written; both are folded into the sections below.

1. **The wall-adjacent exclusion in §3.1 was wrong for S0** and has been turned
   off. The Munk layer is (A_h/β)^⅓ ≈ 63 km against ~79 km zonal spacing at
   45N, so the western boundary current is one grid cell wide and lives
   entirely in i=2 (0.84 m/s, against 0.17 m/s at i=3). Excluding i=2 removed
   the jet rather than the wall artifact and left a flat field where the argmax
   was arbitrary — it returned the southwest corner. See the comment on
   `EXCLUDE_FIRST_WET_COLUMN` in the selection script.
2. **`MDSREADFIELD` is retired in c68j** — it hits `STOP 'ABNORMAL END: S/R
   MDSREADFIELD is retired'` unless `USE_OBSOLETE_MDS_RW_FIELD` is defined.
   `cost_test.F` therefore uses `READ_REC_XY_RL`, which is also the idiom
   `ctrl_map_ini_gen.F` uses for weight fields on the TAF path. It reads at
   `readBinaryPrec`, which must stay 32 for the tutorial's float32
   `bathy.bin` / `windx_cosy.bin` / `SST_relax.bin`, so the weight field is
   written float32.

---

## 0. Executive summary

| | |
|---|---|
| Method | TAF (FastOpt) reverse-mode source-to-source AD of MITgcm c68j |
| Blocker | `staf` client is **not installed and not licensed**. Everything else is in place. |
| Model window | day 7210 → day 7220 (720 timesteps), inside the never-trained `truth only` block |
| Cost | one scalar: area-weighted SSH anomaly at one frozen western-boundary cell |
| Control | initial `etaN` on every cell (`ALLOW_ETAN0_CONTROL`, `grdchkvarindex = 29`) |
| Primary product | one 62×62 map `adxx_etan`, dimensionless, plus `ADJetan` at every intermediate day |
| Verification | `pkg/grdchk` finite-difference test + an analytic invariant unique to this cost |
| Wall clock | TAF round trip and compile dominate (hours); each adjoint run is minutes |

The one non-negotiable prerequisite is a TAF licence. Section 2 is the critical path;
Sections 3–11 are ready to execute the moment `staf` runs.

---

## 1. Verified state of the repository

Everything in this table was read out of the working tree or the scratch products, not assumed.

### 1.1 MITgcm

| Item | Value | Source |
|---|---|---|
| Checkout | `external/MITgcm`, checkpoint68j, commit `f03a2f5e214bc57b8393f6201a6a1266dd1f53d6` | `doc/tag-index` |
| AD packages present | `autodiff`, `cost`, `ctrl`, `grdchk`, `ecco`, `smooth`, `openad` | `pkg/` |
| `staf` client | **absent** | `find . -name "staf*"` returns nothing |
| TAF server | `fastopt.net`, reachable from the login node on both :80 and :443 | verified by TCP connect |
| Forward optfile | `tools/build_options/linux_amd64_gfortran` | `build/af_s0/Makefile:10` |
| Compiler | `mpif77` under `gnu14/14.2.0` + `openmpi5/5.0.7` | `build/af_s0/Makefile:56`, `slurm/mitgcm/*.sbatch` |

`tools/adjoint_options/adjoint_default` already lists `xx_etan_dummy` in its TAF `-input` set,
so the SSH initial-condition control needs **no change to the AD option file**.

### 1.2 The S0 forward configuration being differentiated

Read from `archive/src/bire_repro/af_s0.py::render_data` and `af_fno/mitgcm/code/SIZE.h`.

| Parameter | Value |
|---|---|
| Grid | 62 × 62 × 15, `sNx=sNy=31`, `nPx=nPy=2`, `OLx=OLy=2`, 4 MPI ranks |
| `deltaT` | 1200 s → **72 steps/day**, 25 920 steps/model-year, 360-day year |
| Free surface | `implicitFreeSurface=.TRUE.`, `rigidLid=.FALSE.`, `exactConserv=.TRUE.` |
| Solver | `cg2dTargetResidual=1e-7`, `cg2dMaxIters=1000` |
| Viscosity / diffusivity | `viscAh=5000`, `viscAr=1e-2`, `diffKhT=1000`, `diffKrT=1e-5` |
| Convection | `ivdc_kappa=1.`, `implicitDiffusion=.TRUE.` — see §12.1 |
| EOS | `LINEAR`, `tAlpha=2e-4`, `sBeta=0`, `saltStepping=.FALSE.` |
| Sides | `no_slip_sides=.TRUE.`, `no_slip_bottom=.FALSE.` |
| Forcing | `windx_cosy.bin` (τ₀ = 0.100), `SST_relax.bin`, `tauThetaClimRelax=2592000` |
| I/O | `useSingleCpuIO=.TRUE.` (keep — it is what makes the products global, not per-tile) |
| Wet cells | 3 600 of 3 844 at k=1, one-cell land rim, single connected basin |

### 1.3 Time bookkeeping — frozen

Production starts at model year 100, iteration 2 592 000, which is **trajectory day 0**.

$$\texttt{iter}(d) = 2\,592\,000 + 72\,d$$

Confirmed against `dynState.0002592000` (day 0) and `dynState.0002592072` (day 1).

| Trajectory day | Iteration | Status |
|---|---|---|
| 7 200 | 3 110 400 | pickup **exists**: `/bigscratch/.../mitgcm_long_truth_v1/S0/production/years_120_126/pickup.0003110400.*` |
| 7 210 | 3 111 120 | must be generated — 720 forward steps from day 7200 |
| 7 220 | 3 111 840 | cost evaluation time |

Day 7200 is the first day of the `truth only` block: never trained on, never validated on,
never used as a rollout start. Both the FNO and MITgcm see it cold.

### 1.4 What the FNO channels correspond to

`af_fno/mitgcm/input/data.diagnostics.production` writes daily **snapshots**
(`frequency=-86400`, `timePhase=0`) of `UVEL,VVEL,THETA` → `dynState` and `ETAN` → `surfState`.
That is the 15+15+15+1 = 46-channel state exactly. Channel 45 is `ETAN`, and it is `etaN` in
`DYNVARS.h` that the adjoint differentiates. No interpolation or re-gridding sits between the
two sides of the comparison.

---

## 2. Stage 1 — obtain TAF (critical path, blocking)

TAF is commercial software from FastOpt GmbH. MITgcm ships only the *client hook*
(`TAF=staf` in `tools/adjoint_options/adjoint_default`); the client itself and a licence must
come from FastOpt. `staf` uploads the concatenated source to `fastopt.net`, which returns the
differentiated Fortran. Nothing is computed locally, so no local TAF installation is compiled —
but the licence is checked server-side per submission.

**Actions**

1. Request an academic TAF licence from FastOpt (`info@fastopt.com`, https://www.fastopt.de/).
   State: MITgcm checkpoint68j, ocean adjoint sensitivity, academic, single user, Linux x86-64.
2. Install the returned `staf` script somewhere on `PATH` on the build node, e.g.
   `~/bin/staf`, and register the licence key as FastOpt instructs.
3. Smoke test before touching MITgcm:
   ```bash
   printf '      subroutine f(x,y)\n      real x,y\n      y = x*x\n      end\n' > t.f
   staf -reverse -input x -output y -server fastopt.net t.f
   # expect t_ad.f containing adx = adx + 2*x*ady
   ```
   A failure here is a licence or network problem, not an MITgcm problem, and must be resolved
   before Stage 5.
4. Record the TAF version string in the run manifest. TAF version is part of the provenance of
   every gradient produced — a different TAF version is a different executable.

**Egress note.** `fastopt.net` accepts TCP on :80 and :443 from the login node. Compute nodes on
this cluster may not have the same egress. If the smoke test passes on the login node and fails
on a compute node, run `make adtaf` (the TAF step alone) on the login node and `make adall` on a
compute node — the Makefile splits cleanly at that boundary.

**How the licence works, and how many you need.** `staf` generates an SSH keypair on first use:
the private half lands in `~/.ssh`, the public half goes to FastOpt automatically, and FastOpt
registers it against a licensed account. Authentication is by that key, per submission,
server-side. There is no local licence file and no server to configure.

Critically, **TAF is consumed per build, not per run.** `ad_taf_output.f` is ordinary Fortran
sitting on your disk; `mitgcmuv_ad` never contacts FastOpt. One submission therefore supports an
unlimited number of adjoint datasets. A new submission is needed only when the *differentiated
source* changes:

| Requires a new TAF submission | Does not |
|---|---|
| Editing a `.F` file TAF differentiates | A new target cell p⋆ |
| Changing a CPP flag or header inlined into those files (`SIZE.h`, `CPP_OPTIONS.h`, `COST_OPTIONS.h`, `CTRL_OPTIONS.h`, `AUTODIFF_OPTIONS.h`) | A new start day t₀ |
| **Changing `nchklev_*` in `tamc.h`** — they appear in the `CADJ INIT comlev1 = COMMON,nchklev_1` tape-sizing directives (`the_main_loop.F:481–525`) | A different regime (S0/S1/S2) |
| Adding or removing a package | A different lead within the `nchklev` budget |
| | Any `grdchk` setting, any number of runs |

This is why §4.3 sizes `nchklev` generously and why §4.4 pushes the entire cost-function
definition into a runtime input file. The compile-time surface is the scarce resource; everything
that can be made a runtime knob should be.

**If the licence cannot be obtained.** The free alternative is **Tapenade** (MITgcm-AD v2, Gaikwad
et al. 2024), an open-source TL/adjoint framework built precisely because TAF's proprietary
status excludes most users. Note that Tapenade support postdates this checkout — c68j (2022) does
not have it, so this route means moving to a current MITgcm and re-validating that the forward S0
trajectory is unchanged, which is a substantial detour.

Do **not** plan on OpenAD. MITgcm support for it ended in July 2026; `verification/OpenAD` and the
`adjoint_oad` option file are present in this checkout but are now unmaintained upstream.

---

## 3. Stage 2 — freeze the target cell p⋆ and the wet-area constant

Two numbers must be frozen *before* any code is written, because both are compiled in.

### 3.1 p⋆

$$p^\star = \underset{p \in \mathcal{W}}{\arg\max}\ \overline{\sqrt{u_1(p,t)^2 + v_1(p,t)^2}}$$

Frozen rules, to be written verbatim into the contract JSON:

- Source: `trajectories_v3.zarr`, regime index for S0, **training days 0–5 999 only**. Same
  window as the normalizers. The target cell must not be chosen using data the FNO never saw.
- Level: k = 1 (surface).
- C-grid → cell centre before taking the speed:
  `u_c(i,j) = ½(U(i,j) + U(i+1,j))`, `v_c(i,j) = ½(V(i,j) + V(i,j+1))`.
  Taking the speed from the face values would bias p⋆ half a cell west.
- Search region `𝒲`: wet cells in the western third of the basin, `2 ≤ i ≤ 20` (1-based global),
  **including** the first wet column. The first draft excluded it; measurement showed that
  removes the entire boundary current (see Status, finding 1), so
  `EXCLUDE_FIRST_WET_COLUMN = False`.
  Also exclude the two northernmost and two southernmost wet rows so p⋆ cannot land in a corner
  where the sensitivity map is dominated by the sidewall rather than by the jet. This exclusion
  is retained and does not bind: the selected p⋆ at j=17 is far from both ends.
- Ties: lowest `j`, then lowest `i`.

Write `scripts/select_adjoint_target.py` to emit
`config/mitgcm_adjoint_s0_target_v1.json`:

```json
{
  "version": "mitgcm_adjoint_s0_target_v1",
  "dataset": "trajectories_v3",
  "regime": "S0",
  "selection_days": [0, 6000],
  "search_region": {"i_min": 3, "i_max": 20, "exclude_first_wet_column": true,
                    "exclude_rows_from_each_meridional_end": 2},
  "i_global": null,
  "j_global": null,
  "mean_surface_speed_m_s": null,
  "wet_cell_count": 3600,
  "wet_area_m2": null
}
```

Once written, this file is **immutable**. Every later stage reads `i_global`/`j_global` from it,
and the preflight in Stage 8 re-derives them and aborts on mismatch.

### 3.2 The wet-area constant

$$A_{\mathrm{wet}} = \sum_{i,j} rA_{ij}\, \mathrm{mask}C_{ij,k=1}$$

Compute it in the same script from `RAC.data` and the surface mask in any completed S0 run
directory (e.g. `/bigscratch/.../mitgcm_long_truth_v1/S0/production/years_120_126/RAC.data`,
62×62 float32, and the wet mask already carried as a static channel). Store it in the JSON to
full double precision. It is not compiled in — it enters only through the runtime weight field of
§4.4, so changing it never costs a rebuild.

### 3.3 Decision that must be made here, not later

`⟨η⟩_wet` can be the plain wet-cell mean or the **area-weighted** wet-cell mean. On a spherical
polar grid with `delY = 1°` the cell area varies as cos φ, so the two differ and the two
sensitivity maps differ.

**Recommendation: area-weighted.** Two reasons.

1. It is the physically meaningful "uniform sea-level offset". With `implicitFreeSurface`,
   `exactConserv`, no freshwater flux and a closed basin, `∫η dA` is conserved exactly by the
   discrete dynamics.
2. That conservation makes the mean term's contribution to the sensitivity map **analytically
   known**: the adjoint of a conserved functional is constant in time, so the second term
   contributes exactly `−rA_ij·maskC_ij / A_wet` at *every* time, including t₀. This is a free,
   exact test of the entire adjoint machinery (Gate G3, §11).

Whichever is chosen, the FNO-side diagnostic must use the identical weights, or the comparison
measures the convention rather than the model.

---

## 4. Stage 3 — the `code_ad` directory

New directory `af_fno/mitgcm/code_ad/`. It does **not** replace `af_fno/mitgcm/code/`; the
adjoint build stacks it on top so the forward physics stays byte-identical.

| File | Provenance |
|---|---|
| `SIZE.h` | copy of `af_fno/mitgcm/code/SIZE.h`, unchanged |
| `DIAGNOSTICS_SIZE.h` | copy, unchanged |
| `packages.conf` | forward list + `autodiff cost ctrl grdchk` |
| `CPP_OPTIONS.h` | from `model/inc/`, unchanged unless a build error demands otherwise |
| `AUTODIFF_OPTIONS.h` | from `pkg/autodiff/`, see §4.2 |
| `COST_OPTIONS.h` | from `pkg/cost/`, `#define ALLOW_COST_TEST` |
| `CTRL_OPTIONS.h` | from `pkg/ctrl/`, `#define ALLOW_ETAN0_CONTROL` |
| `CTRL_SIZE.h` | from `pkg/ctrl/`, defaults are fine |
| `tamc.h` | checkpoint levels for this window, §4.3 |
| `cost.h` | from `pkg/cost/`, plus a common block for `afCostWeightFile` |
| `cost_test.F` | **new** — overrides `pkg/cost/cost_test.F`, the QoI as a runtime weight field, §4.4 |
| `cost_readparms.F` | **new** — adds `afCostWeightFile` to `COST_NML`, §4.5 |

### 4.1 `packages.conf`

```
# AF--FNO adjoint packages.  Forward set plus the AD stack.
gfd
diagnostics

autodiff
cost
ctrl
grdchk
```

`diagnostics` is retained so the adjoint executable can also write the forward daily snapshots,
letting one run produce both the reference trajectory and the sensitivity.

### 4.2 `AUTODIFF_OPTIONS.h`

Start from `pkg/autodiff/AUTODIFF_OPTIONS.h` (which already has the right defaults) and confirm:

```
#define ALLOW_AUTODIFF_TAMC
#define ALLOW_TAMC_CHECKPOINTING
#define ALLOW_AUTODIFF_MONITOR        /* required for the ADJetan dumps */
#define ALLOW_AUTODIFF_WHTAPEIO       /* required: useSingleCpuIO is .TRUE. */
#undef  AUTODIFF_2_LEVEL_CHECKPOINT   /* we use the 3-level form with nchklev_3 = 1 */
#undef  ALLOW_DIVIDED_ADJOINT
#undef  ALLOW_CG2D_NSA                /* keep the self-adjoint cg2d — see §12.2 */
```

`ALLOW_AUTODIFF_MONITOR` is what enables `pkg/autodiff/addummy_for_etan.F`, which calls
`DUMP_ADJ_XY(... 'ADJetan.', ...)` at every `adjDumpFreq`. That routine is the source of the
time-resolved maps and it is already in the tree — no code needed.

### 4.3 `tamc.h`

The only edit to the stock header is the checkpoint block. The requirement enforced at runtime
by `pkg/autodiff/autodiff_check.F` is `nchklev_1 · nchklev_2 · nchklev_3 ≥ nTimeSteps`.

```fortran
      integer    nchklev_1
      parameter( nchklev_1      =   72 )   /* one model day  */
      integer    nchklev_2
      parameter( nchklev_2      =   20 )   /* twenty days    */
      integer    nchklev_3
      parameter( nchklev_3      =    1 )
```

72 · 20 · 1 = 1 440 ≥ 1 440, which covers the 20-day Run B and, a fortiori, the 10-day Run A.
`nchklev_1` sets the in-memory level-1 tape; at 31×31×15 per rank this is a few hundred MB per
rank, comfortable on one node. Levels 2 and 3 go to disk via `ALLOW_AUTODIFF_WHTAPEIO`.

If a longer lead is ever wanted, raise `nchklev_2` and rebuild — `nchklev_2`/`nchklev_3` cost
disk, not memory. **But that rebuild is a TAF re-submission** (§2), so size for the longest lead
the study might plausibly reach, not the first one. Going to `nchklev_2 = 200` now buys leads out
to 200 days for the price of some scratch space and costs nothing else.

### 4.4 `cost_test.F` — the QoI as a runtime weight field

Two facts drive this design.

1. `COST_FINAL` already sums `mult_test · objf_test(bi,bj)` over tiles and then globally
   (`pkg/cost/cost_final.F:148,183,195`). So anything written into the per-tile accumulator is
   assembled into `fc` exactly, with **no global sum inside the differentiated cost**.
2. TAF is consumed per build (§2). Anything hard-coded in `cost_test.F` — the target cell, the
   mean convention, the choice of QoI — costs a licence submission and a 90-minute recompile to
   change.

Both point the same way: write the cost as a **general weighted sum against a 2-D field read
from disk at runtime**.

$$J \;=\; \sum_{i,j} w_{ij}\, \eta_{ij}(T)$$

Every linear-in-η quantity of interest is then a choice of `w`, computed in Python and dropped
into the run directory as a binary file. The SSH anomaly at p⋆ is one such choice:

$$w_{ij} \;=\; \delta_{ij,p^\star} \;-\; \frac{rA_{ij}\,\mathrm{mask}C_{ij}}{A_{\mathrm{wet}}}$$

so p⋆, the plain-vs-area-weighted decision of §3.3, `A_wet`, regional averages, boundary-band
means and transport-like weightings all become runtime inputs. **One TAF submission covers the
entire study.**

`w` is read with `MDSREADFIELD` (passive, not `active_read_xy`), so TAF treats it as a constant —
it is a property of the functional, not a control.

```fortran
#include "COST_OPTIONS.h"

      SUBROUTINE COST_TEST( myThid )
C     *==========================================================*
C     | AF--FNO adjoint ground truth.
C     |   J = sum_ij w_ij * etaN_ij(T)
C     | w is read at runtime from afCostWeightFile, so any linear
C     | functional of the final SSH is a runtime choice.  For the
C     | SSH anomaly at p*:
C     |   w = delta_{p*} - rA*maskC/A_wet
C     | Accumulating into objf_test lets COST_FINAL's tile sum plus
C     | global sum assemble J exactly -- no global sum in the
C     | differentiated code.
C     *==========================================================*
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "GRID.h"
#include "cost.h"

      INTEGER myThid

#ifdef ALLOW_COST_TEST
      INTEGER bi, bj, i, j
      INTEGER itlo, ithi, jtlo, jthi
      _RL     afCostW(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)

      jtlo = myByLo(myThid)
      jthi = myByHi(myThid)
      itlo = myBxLo(myThid)
      ithi = myBxHi(myThid)

C--   Passive read: the weight field defines the functional and is
C--   not a control.  MDSREADFIELD, not ACTIVE_READ_XY.
      CALL MDSREADFIELD( afCostWeightFile, readBinaryPrec, 'RL',
     &                   1, afCostW, 1, myThid )

      DO bj = jtlo,jthi
       DO bi = itlo,ithi
        DO j = 1,sNy
         DO i = 1,sNx
          objf_test(bi,bj) = objf_test(bi,bj)
     &      + afCostW(i,j,bi,bj) * etaN(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
#endif /* ALLOW_COST_TEST */

      RETURN
      END
```

`objf_test` is zeroed by `cost_init_varia`, and `COST_TEST` is called once from `COST_FUNCTION`,
which `THE_MAIN_LOOP` invokes as `COST_FINAL` after the time loop
(`model/src/the_main_loop.F:747`). So `J` is evaluated on the **final** state — day 7220 — and no
`myIter` guard is needed. `myXGlobalLo`/`myBxLo` come from `EEPARAMS.h`, already included.

**Consequence for the gates.** With this form, Gate G3 (§11) becomes sharper and cheaper: run
once with `w = −rA·maskC/A_wet` alone and the resulting map must equal `w` itself at every dump
time, by conservation. No point term, no subtraction, nothing to disentangle. And Gate G5 is a
one-line Python check, `fc == np.sum(w * eta_7220)`.

### 4.5 `cost_readparms.F`

Copy `pkg/cost/cost_readparms.F` into `code_ad/` and add the weight filename to `COST_NML`:

```fortran
      NAMELIST /COST_NML/
     &         mult_atl, mult_test, mult_tracer,
     &         afCostWeightFile
```

with `afCostWeightFile = 'costWeight.bin'` as the default, declared as
`CHARACTER*(MAX_LEN_FNAM)` in a small common block added to `code_ad/cost.h`. It is a passive
character variable, so it costs nothing in the adjoint.

**Why this matters:** without it, every new target cell, every change of mean convention, and
every new QoI needs a fresh TAF round trip. With it, one build serves the whole study and a new
sensitivity map costs three minutes of compute and one Python array. This is the single highest-
leverage decision in the plan.

### 4.6 `CTRL_OPTIONS.h` and `COST_OPTIONS.h`

```
/* CTRL_OPTIONS.h */
#define ALLOW_ETAN0_CONTROL
```

`ALLOW_ETAN0_CONTROL` is not in the stock `#undef` list but is fully wired:
`ctrl_init.F:552` registers it as control id 29 (`'c','xy'`), `ctrl_map_ini.F:531–555` reads
`xx_etan.<optimcycle>` through `active_read_xy` with `xx_etan_dummy`, `ctrl_pack.F:613` writes
the gradient, and `grdchk_getxx.F:602` accepts `grdchkvarindex = 29`.

Note `ctrl_map_ini.F` adds the perturbation to **both** `etaN` and `etaH`. That is correct and
is what a physically consistent SSH initial-condition perturbation means — at `nIter0` the two
are equal. It does mean `adxx_etan = ∂J/∂etaN + ∂J/∂etaH` at t₀, which is the right thing for
"perturb the initial sea surface", and is exactly what `grdchk` validates.

```
/* COST_OPTIONS.h */
#define ALLOW_COST_TEST
#undef  ALLOW_COST_TSQUARED
#undef  ALLOW_GENCOST_CONTRIBUTION
```

---

## 5. Stage 4 — the `input_ad` directory

New directory `af_fno/mitgcm/input_ad/`, holding only the AD-specific namelists. The physics
namelist `data` is still rendered by `af_s0.render_data` so it cannot drift from the forward run.

### `data.pkg`

```
 &PACKAGES
 useMNC=.FALSE.,
 useDiagnostics=.TRUE.,
 useGrdchk=.FALSE.,
 &
```

`useGrdchk=.TRUE.` only for the Stage 7 gradient check.

### `data.cost`

```
 &COST_NML
 mult_test        = 1.,
 afCostWeightFile = 'costWeight.bin',
 &
```

`mult_test = 1.` exactly, so `fc = J` and `adxx_etan = ∂J/∂η` with no rescaling.

`costWeight.bin` is staged into the run directory by `scripts/make_cost_weight.py`: 62×62,
big-endian float64 (matching `readBinaryPrec`), built from the frozen p⋆, `RAC.data` and the
surface mask. Each distinct QoI in the study is one such file plus one entry in the run manifest.
Its SHA-256 goes into the report — the weight field *is* the definition of the quantity of
interest, so it is provenance, not configuration.

### `data.ctrl`

```
 &CTRL_NML
 xx_etan_file = 'xx_etan',
 &
 &CTRL_PACKNAMES
 &
 &CTRL_NML_GENARR
 &
```

### `data.autodiff`

```
 &AUTODIFF_PARM01
 inAdExact = .TRUE.,
 &
```

`inAdExact = .TRUE.` (the default) requests the exact adjoint with no approximations. Do not
relax it — the whole point of this dataset is that it is exact.

### `data.grdchk` (Stage 7 only)

```
 &GRDCHK_NML
 grdchk_eps     = 1.d-4,
 grdchkvarindex = 29,
 iGloPos        = <i_test>,
 jGloPos        = <j_test>,
 kGloPos        = 1,
 iGloTile       = 1,
 jGloTile       = 1,
 nbeg           = 1,
 nstep          = 1,
 nend           = 8,
 &
```

### `data.optim`

```
 &OPTIM
 optimcycle = 0,
 &
```

`optimcycle = 0` fixes the control filename to `xx_etan.0000000000.data` and the gradient to
`adxx_etan.0000000000.data`.

### Adjoint dump frequency

Added to `PARM03` of the rendered `data` for the adjoint runs only:

```
 adjDumpFreq = 86400.,
```

One `ADJetan` dump per model day. Over the 20-day Run B that is 21 maps — the complete backward
evolution of the sensitivity, at no extra cost.

---

## 6. Stage 5 — build the adjoint executable

```bash
module purge
module load gnu14/14.2.0 openmpi5/5.0.7
export PATH="$HOME/bin:$PATH"          # where staf lives

ROOT=/home/mjalabert314/bire_james25_repro
MIT=$ROOT/external/MITgcm
mkdir -p $ROOT/build/af_s0_ad && cd $ROOT/build/af_s0_ad

$MIT/tools/genmake2 \
  -rootdir $MIT \
  -mods    "$ROOT/af_fno/mitgcm/code_ad" \
  -of      $MIT/tools/build_options/linux_amd64_gfortran \
  -adof    $MIT/tools/adjoint_options/adjoint_default \
  -mpi

make depend
make adall            # serial: the TAF step is not parallel-safe
```

Produces `mitgcmuv_ad`.

Practical notes, all of which have bitten people on this exact path:

- **Do not** pass `-j` to `make adall`. `make -j8 depend` is fine.
- The TAF step concatenates the whole source into `ad_input_code.f` and ships it to
  `fastopt.net`. Expect 10–40 minutes. `taf_ad.log` holds the server's diagnostics and is the
  first place to look on failure.
- `ad_taf_output.f` is one very large Fortran file. gfortran needs **≥ 16 GB** and 20–90 minutes
  on it. Build on a compute node (`--mem=32G`), not the login node — subject to the egress
  caveat in §2.
- `make adtaf` stops after code generation. Useful for splitting the login-node and
  compute-node halves, and for inspecting the generated adjoint before committing to a compile.
- Keep `build/af_s0/` (forward) intact. The two builds must not share a directory.

Archive `taf_ad.log`, `ad_taf_output.f` (compressed), the `Makefile`, and the TAF version string
alongside the executable. These are the provenance of every number this plan produces.

---

## 7. Stage 6 — reference forward states

The adjoint run needs a pickup at its own `nIter0`, and the comparison needs the forward states
the FNO will be fed.

1. **Day 7210 pickup.** From `pickup.0003110400.*` (day 7200), run the *forward* executable
   `build/af_s0/mitgcmuv` for 720 steps with `nIter0 = 3110400`, `nTimeSteps = 720`,
   `pChkptFreq = 864000.` so a pickup lands at 3 111 120. Same 4-rank 2×2 decomposition, same
   node type, same modules — the twin study established that this matters and there is no reason
   to spend the credibility.
2. **Verify** the day-7210 and day-7220 `dynState`/`surfState` snapshots from this short run
   match `trajectories_v3.zarr` at those days to bit level. If they do not, the adjoint is being
   taken about a different trajectory than the FNO is being evaluated on, and everything
   downstream is invalid. This is Gate G0.
3. Stage the FNO input pair — day 7200 (history) and day 7210 (present) — from the zarr, not
   from a re-run, so the FNO side is fed exactly the archived truth.

---

## 8. Stage 7 — gradient check (acceptance gate, do not skip)

`pkg/grdchk` perturbs one control element by `±grdchk_eps`, runs the forward model twice, and
compares the centred finite difference to the adjoint gradient. This is the only thing standing
between "a 62×62 array of plausible numbers" and "ground truth".

Run with `useGrdchk=.TRUE.`, `nIter0 = 3111120`, `nTimeSteps = 720`.

**Test points** — at least eight, chosen deliberately, not by `nbeg/nstep/nend` sweeping alone:

| # | Location | What it probes |
|---|---|---|
| 1 | p⋆ itself | the self-sensitivity, largest value on the map |
| 2–3 | two cells 1–3 cells upstream of p⋆ in the boundary current | the advective signal |
| 4 | the first wet column at p⋆'s latitude | sidewall handling |
| 5 | mid-basin interior | small but non-zero, the hardest to get right |
| 6 | eastern boundary | should be near the noise floor at 10 days |
| 7 | a cell known to convect (from `ivdc_kappa` activity in the forward run) | §12.1 |
| 8 | a land cell | must be exactly zero |

**Epsilon sweep.** Run each point at `grdchk_eps ∈ {1e-3, 1e-4, 1e-5, 1e-6}` m. A correct
adjoint shows a plateau: the ratio (FD ÷ adjoint) sits at 1 across the middle of that range and
degrades only at 1e-6 where float round-off takes over. A *single* epsilon agreeing is weak
evidence; the plateau is strong evidence. Note the sibling result already in this project —
`s0-twin-float32-floor` — that daily diagnostics carry a quantisation floor. Here the comparison
is on `fc` in double precision from the log, not on the daily files, so that floor does not
apply, but the discipline of checking against a noise floor does.

**Gate G1:** `|FD/adjoint − 1| < 1e-4` at every wet test point, at both 1e-4 and 1e-5, with the
plateau visible. Land cell exactly 0.

Record the full table. If a point fails, §12 lists the three likely causes in order.

---

## 9. Stage 8 — production adjoint runs

Both use `mitgcmuv_ad`, 4 ranks, `useGrdchk=.FALSE.`.

### Run A — the primary map

| | |
|---|---|
| `nIter0` | 3 111 120 (day 7210) |
| `nTimeSteps` | 720 (10 days) |
| Cost evaluated at | day 7220 |
| `adjDumpFreq` | 86 400 s |

Primary product: `adxx_etan.0000000000.data`, 62×62, = **S¹⁰ᵢⱼ exactly**, dimensionless
(metres of J per metre of η). Companion: `ADJetan.*` at days 7210…7220.

### Run B — the history slot and the lead sweep

| | |
|---|---|
| `nIter0` | 3 110 400 (day 7200) |
| `nTimeSteps` | 1 440 (20 days) |
| Cost evaluated at | day 7220 |
| `adjDumpFreq` | 86 400 s |

Products: `adxx_etan` = ∂J/∂η(·, day 7200) — the 20-day map, and the direct analogue of the
FNO's *history slot* sensitivity ∂J/∂η_{t−10}. Plus `ADJetan` at all 21 days.

**The design point:** the adjoint state at time t inside a run whose cost sits at T is
∂J/∂η(·,t) for every t in [start, T], independent of when the run started. So Run B's `ADJetan`
at day 7210 must equal Run A's `adxx_etan` to solver tolerance. That is a free, strong,
end-to-end consistency check (Gate G2) *and* it hands you the full lead sweep — 1 day through 20
days — from a single three-minute job. Extend `nTimeSteps` and `nchklev_2` for longer leads when
the FNO comparison wants them.

Both runs also write the forward `dynState`/`surfState` snapshots, so each is self-documenting
about the trajectory it linearised around.

---

## 10. Stage 9 — extraction

`scripts/extract_mitgcm_adjoint.py` → `outputs/af_fno/adjoint/mitgcm_s0_adjoint_v1/`.

Reads the MDS `.data`/`.meta` pairs (big-endian float64 for `adxx_etan`; check the `.meta`
rather than assuming) and writes a single `.npz`:

| Array | Shape | Content |
|---|---|---|
| `S10` | (62, 62) | Run A `adxx_etan`, the primary map |
| `S20` | (62, 62) | Run B `adxx_etan` |
| `S_lead` | (21, 62, 62) | Run B `ADJetan`, days 7200…7220, lead descending |
| `lead_days` | (21,) | 20, 19, …, 0 |
| `wet_mask` | (62, 62) | surface mask, from the same run |
| `rA` | (62, 62) | cell areas, for the Gate G3 check and for area-weighted norms |
| `target_ij` | (2,) | frozen p⋆ |

Plus `report.json` with: TAF version, MITgcm commit, executable SHA-256, `fc` from each run's
log, the Gate G1 table, the Gate G2 and G3 residuals, the run manifests, and the frozen contract.
Plus `manifest.json` in the project's existing format.

**Conventions to state explicitly in the report**, because the FNO side must match all four:

- Sign: `S > 0` means raising η at (i,j) raises the target anomaly.
- Units: dimensionless. A value of 0.2 means 1 cm at the source gives 2 mm at the target.
- Grid: `S` is at cell centres, same (j,i) index order as the zarr's spatial axes.
- Land: exactly 0, not NaN.

---

## 11. Acceptance gates

| Gate | Condition | Where |
|---|---|---|
| **G0** | Day-7210 and day-7220 forward snapshots from the short re-run match `trajectories_v3.zarr` bit-for-bit | §7 |
| **G1** | `grdchk`: `\|FD/adjoint − 1\| < 1e-4` at all 8 points, at ε = 1e-4 and 1e-5, plateau visible; land cell exactly 0 | §8 |
| **G2** | Run B `ADJetan` at day 7210 equals Run A `adxx_etan`, relative L2 < 1e-6 | §9 |
| **G3** | A run with `w = −rA·maskC/A_wet` alone returns a map equal to `w` itself at every dump time, relative L2 < 1e-5 | §3.3, §4.4 |
| **G4** | `S` is exactly 0 on all 244 land cells and finite everywhere | §10 |
| **G5** | `fc` reported by the run equals `η(p⋆,7220) − ⟨η(7220)⟩_A` computed independently in Python from the archived day-7220 state, relative error < 1e-10 | §10 |

G3 deserves emphasis. Because `∫η dA` is exactly conserved by this configuration, the adjoint of
the mean functional is provably constant in time. Any drift in it is a direct, quantitative
measurement of error in the adjoint — spurious sources or sinks introduced by the AD, by the
cg2d adjoint, or by the checkpointing. It is the cheapest and sharpest diagnostic available
here, it requires no finite differences, and it is available at every dump time. If G3 passes at
all 21 leads and G1 passes, the map is trustworthy.

G5 is trivial but catches the whole class of "the cost function is not what I think it is"
errors, including a wrong `A_wet` and an off-by-one in p⋆. With the §4.4 weight field it reduces
to `fc == np.sum(w * eta_7220)`.

---

## 12. Known caveats — read before interpreting any map

### 12.1 `ivdc_kappa = 1.` is not differentiable

The forward configuration does convective adjustment by switching vertical diffusivity between
`1e-5` and `1.0` on the sign of the local stratification. That switch is a discontinuity. TAF
differentiates the branch actually taken, which is correct almost everywhere but wrong at
measure-zero sets — and near them, the finite difference and the adjoint legitimately disagree
because the perturbed run flips a cell's convective state.

This is a property of the physics being emulated, not a bug in the adjoint. Handle it as follows:

- **Keep `ivdc_kappa = 1.` in the production runs.** Changing it would produce the adjoint of a
  model the FNO was not trained on, which is not the ground truth being asked for.
- If G1 fails, first check whether the failing points are convecting columns. Test point 7 exists
  precisely to make that diagnosis fast.
- If they are, run a *diagnostic-only* build with `ivdc_kappa = 0.` and confirm G1 passes there.
  That isolates the discontinuity and is reportable. It does not become the deliverable.
- In the S0 double gyre — wind-driven, SST-relaxed, linear EOS with no salt — convection should
  be rare and confined to the northern cells. Quantify how rare from the forward run and state
  it in the report rather than leaving it as an unbounded worry.

### 12.2 The cg2d adjoint

The implicit free-surface solve is iterative. With `ALLOW_CG2D_NSA` undefined, MITgcm uses the
self-adjoint approximation: the adjoint solves the same symmetric system. That is accurate when
the forward solve converges tightly, and `cg2dTargetResidual = 1e-7` is tight. Leave it undefined.

If G1 or G3 fails and convection is ruled out, tightening to `1e-10` and re-testing distinguishes
solver-tolerance error from a genuine AD error. Enabling `ALLOW_CG2D_NSA` requires raising
`numItersMax` in `tamc.h` to at least `cg2dMaxIters`, which will make the tape much larger; treat
it as a last resort.

### 12.3 Linearity is an assumption to be tested, not asserted

The adjoint gives the exact derivative of the discrete model at this state. Whether that
derivative *predicts* the response to a finite perturbation is a separate question about the
flow, and this project already has a directly relevant finding: `s0-not-chaotic` — twin
perturbations at 1e-6 and 1e-3 both failed to grow over 25 years, with the response linear in ε.
At a 10-day lead in a non-chaotic regime, the linear range should be wide. The ε plateau in Gate
G1 measures it directly. Report the range; do not claim it.

### 12.4 What is and is not being compared

The FNO map holds `(U, V, θ)` at t fixed and varies η. `ADJetan` at time t is the partial
derivative of J with respect to that component of the state at that instant, with the others
held fixed. These are the same object. But two things differ and must be said out loud in any
figure caption:

- MITgcm's control also perturbs `etaH` (§4.6), which is the physically consistent choice and
  has no FNO counterpart because the FNO carries no separate `etaH`.
- MITgcm's derivative is with respect to a continuous-in-time trajectory sampled at day
  boundaries; the FNO's is with respect to its own 10-day discrete map. Agreement of the two is
  the scientific result. Disagreement localised to the western boundary would connect directly
  to `western-boundary-ratio-degrades` and `local-branch-gamma-ablation`.

### 12.5 Scope honesty

This plan produces the adjoint at 10- and 20-day leads. The unresolved question in the model
card is at **day 2000**, and `gate-long-horizon-is-90-360` records that the acceptance gate never
looks there. A 10-day adjoint map will not settle a day-2000 question. It is the right first
experiment — it is the one the FNO can be differentiated through cheaply and the one where the
comparison is cleanest — but the plan should not be read as addressing the long-horizon problem.
Extending to longer leads is a matter of raising `nTimeSteps` and `nchklev_2`; the adjoint cost
grows linearly and the tape grows with it, so a 200-day adjoint is feasible and a 2000-day
adjoint needs a checkpointing redesign and a serious look at §12.3.

---

## 13. Compute budget

| Stage | Resource | Estimate |
|---|---|---|
| TAF round trip | login node, network | 10–40 min |
| Compile `ad_taf_output.f` | 1 core, 32 GB | 20–90 min |
| Day-7210 pickup | 4 ranks | < 1 min (720 forward steps) |
| One `grdchk` point, one ε | 4 ranks | ~2 min (2 forward + 1 adjoint) |
| Full G1 sweep, 8 points × 4 ε | 4 ranks | ~1 h |
| Run A | 4 ranks | ~3 min |
| Run B | 4 ranks | ~5 min |

Derived from the measured forward cost: 10 model years (259 200 steps) in under 2 h on 4 ranks,
i.e. ~28 ms/step; the adjoint runs 3–5× that plus tape I/O. The build is the only slow part, and
it happens once, and it is the only step that touches the TAF licence. Every additional quantity
of interest — a new p⋆, a regional average, a different mean convention — costs one Run A, three
minutes, and a Python array, thanks to §4.4.

Slurm jobs to add, following the existing naming:

- `slurm/mitgcm/af_s0_adjoint_build.sbatch` (1 task, 32 GB, `--time=04:00:00`)
- `slurm/mitgcm/af_s0_adjoint_grdchk.sbatch` (4 tasks, `--time=02:00:00`)
- `slurm/mitgcm/af_s0_adjoint_run.sbatch` (4 tasks, `--time=00:30:00`, parameterised by
  `AF_ADJ_NITER0` / `AF_ADJ_NSTEPS` / `AF_ADJ_ILOC` / `AF_ADJ_JLOC`)

---

## 14. Deliverables

**New files**

```
af_fno/mitgcm/code_ad/          SIZE.h  DIAGNOSTICS_SIZE.h  packages.conf
                                CPP_OPTIONS.h  AUTODIFF_OPTIONS.h
                                COST_OPTIONS.h  CTRL_OPTIONS.h  CTRL_SIZE.h
                                tamc.h  cost.h
                                cost_test.F  cost_readparms.F
af_fno/mitgcm/input_ad/         data.pkg  data.cost  data.ctrl
                                data.autodiff  data.grdchk  data.optim
config/                         mitgcm_adjoint_s0_target_v1.json
                                mitgcm_adjoint_s0_run_v1.json
scripts/                        select_adjoint_target.py
                                make_cost_weight.py
                                stage_adjoint_run.py
                                verify_gate_g0.py
                                extract_mitgcm_adjoint.py
slurm/mitgcm/                   af_s0_adjoint_pickup.sbatch
                                af_s0_adjoint_build.sbatch
                                af_s0_adjoint_grdchk.sbatch
                                af_s0_adjoint_run.sbatch
tests/                          test_mitgcm_adjoint.py
docs/                           mitgcm_adjoint_ground_truth_plan.md   (this file)
```

**Tests** (`tests/test_mitgcm_adjoint.py`, runnable without MITgcm):

1. `iter(d) = 2592000 + 72d` on the three anchor days.
2. p⋆ selection is deterministic and reproduces the frozen JSON from the zarr.
3. `A_wet` from `RAC.data` × mask matches the frozen constant to 1e-12 relative.
4. The tile decomposition identity: for a synthetic η and a synthetic 2×2 tiling, summing the
   per-tile `Σ w·η` expression reproduces `η(p⋆) − ⟨η⟩_A` to machine precision. This is the one
   piece of `cost_test.F` logic that can be wrong silently, so it gets a test.
5. `make_cost_weight.py` output: `w` sums to ~0 over wet cells (the anomaly weight is
   mean-free by construction), is exactly 0 on land, and equals `1 − rA/A_wet` at p⋆.
6. The extractor's MDS reader round-trips a synthetic big-endian float64 `.data`/`.meta` pair.

**Products** (`/bigscratch/.../af_fno/mitgcm_adjoint_v1/`, mirrored to
`outputs/af_fno/adjoint/mitgcm_s0_adjoint_v1/`): the `.npz`, `report.json`, `manifest.json`,
`taf_ad.log`, and the two run logs.

---

## 15. Order of execution

```
 1. Request TAF licence                                    ← start today, it gates everything
 2. select_adjoint_target.py → freeze p* and A_wet         ← no TAF needed, do it in parallel
 3. make_cost_weight.py      → costWeight.bin              ← no TAF needed
 4. Write code_ad/ and input_ad/                           ← no TAF needed
 5. Write tests, run them                                  ← no TAF needed
 6. Generate the day-7210 pickup, verify Gate G0           ← forward executable only
 --- TAF arrives ---
 7. staf smoke test
 8. genmake2 + make adall                                  ← the only licensed step
 9. grdchk sweep → Gate G1
10. Run A, Run B → Gates G2, G3
11. extract_mitgcm_adjoint.py → Gates G4, G5
12. Report
```

Steps 2–6 are the majority of the work and none of them need TAF. Start them now; the licence
request runs in the background. After step 8, the study never touches TAF again unless the
compile-time surface of §2 changes.

---

## 16. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Plain vs area-weighted `⟨η⟩_wet` | **Area-weighted** — physically correct and buys Gate G3 (§3.3). Must match the FNO side. With §4.4 this is a runtime choice, so it is reversible without a rebuild. |
| 2 | Search region `𝒲` for p⋆ | **Resolved 2026-08-12.** `2 ≤ i ≤ 20`, two end rows excluded, wall-adjacent column *kept* — the WBC is one cell wide (§3.1, Status finding 1). p⋆ = (i=2, j=17). |
| 3 | QoI as a runtime weight field | **Yes** (§4.4/§4.5). One TAF submission then covers every linear-in-η quantity of interest for the whole study. Highest-leverage decision here. |
| 4 | Longest lead to build for | **Decide before the build, not after.** `nchklev` is compiled in, so raising it later is a TAF re-submission (§2). Sizing `nchklev_2 = 200` now costs scratch space and nothing else. |
| 5 | Fallback if no TAF licence | Tapenade (MITgcm-AD v2), which means moving off c68j and re-validating the forward S0 trajectory. **Not** OpenAD — upstream support ended July 2026. Do not silently substitute either. |
