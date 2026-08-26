# Adjoint-faithful FNO training from forward perturbation responses

**Study plan — approved for execution 2026-08-24; see "Implementation status
and amendments" below for what has actually been built and run.**

This document specifies the data, controls, loss, selection rule, blind
tests, and stop/go gates for a response-aware FNO. Sections below still use
the original prospective phrasing ("will," "is proposed") where the described
work has not happened yet; the amendments section is the source of truth for
what is done versus still ahead.

Evidence labels used throughout:

- **Verified** means checked in the current repository, current scratch products,
  or current generated reports.
- **Inferred** means a scientific interpretation of verified evidence.
- **Proposed** means a choice to freeze before the new study begins.
- **Unresolved** means the repository does not currently establish the detail;
  the specified gate must resolve it without using any adjoint result.

Unless a statement is explicitly labeled Verified, Inferred, or Unresolved,
prescriptive language in this prospective contract is **Proposed**. “Exact”
then means exact if the proposal is approved, not a claim that the new artifact
already exists.

The three required existing documents remain unchanged:
`docs/model_c_spectralnorm_ft90_handbook.md`,
`docs/Adjoint_study_Phase_A.md`, and
`docs/mitgcm_adjoint_ground_truth_plan.md`.

---

## Implementation status and amendments (2026-08-24)

**Verified.** The document was approved for execution. Before any MITgcm or
FNO compute ran, one design decision from the original draft was revised:

**Amendment — blind/adjoint isolation mechanism.** The original text (old
section 18.1/24) specified OS-level enforcement: separate development and
evaluator Unix identities, a mount namespace excluding adjoint/blind paths
from the development identity, ACL search-denial, and a "sanitized
development checkout" exported outside version control. Building that
required provisioning new accounts on a shared cluster, which is
disproportionate for a single-researcher project and was blocking Step 6 (a
step that never reads adjoint or blind data). It has been replaced throughout
this document with a lighter, code-level convention: a path/glob scan over
the training/selection code, configs, and logs actually used for a run,
checking for the same forbidden roots and patterns (`outputs/af_fno/adjoint/**`,
`af_fno/mitgcm_adjoint_v1/**`, `af_fno/mitgcm_adjoint_v2/**`, `**/ADJ*`,
`**/adxx_*`, `**/*TAF*.log`, `**/*grdchk*`), plus a write-once convention for
evaluator-only artifacts. The scientific property this protects — no adjoint
or blind-response value may influence amplitude, lambda, checkpoint, or
architecture selection — is unchanged; only the enforcement mechanism is
lighter. Sections 18.1, 22 (Gate D0), 24, and 25 (step 3) are edited in place
to describe the current mechanism rather than the retired one.

Removed as a consequence (all were untracked, never-committed files from the
initial implementation pass, so nothing is lost from history):
`config/adjoint_faithful_firewall_v1.json`,
`scripts/verify_adjoint_faithful_firewall.py`,
`scripts/freeze_adjoint_faithful_study.py`, and their two dedicated test
files, `tests/test_parent_contract_replay.py` and
`tests/test_response_blindness.py`. Section 23.1's file list is updated to
match.

**Verified — step 6 implemented and running.** The section-7.1 validation
pickup bank (one 320-day unperturbed continuation per regime, day 5,760 to
6,080, archiving a pickup every 10 days) is implemented as:

- `archive/src/bire_repro/af_response_pickup_bank.py` — the MITgcm driver
  (`prepare_segment`/`run_segment`), following the existing
  `af_s0`/`af_independent_wind_trajectories` pattern;
- `scripts/build_response_pickup_bank.py` — resolves and hash-verifies the
  day-5,760 source pickup against the trajectory-v3 source manifest, runs the
  driver, and verifies the three retained days' P32 projections against
  `trajectories_v3.zarr` (the Gate D0 check);
- `slurm/mitgcm/af_response_pickup_bank_segment.sbatch` and
  `scripts/submit_af_response_pickup_bank.sh`;
- `tests/test_af_response_pickup_bank.py` (15 tests, local fixtures only).

This superseded the originally planned filename
`scripts/build_response_validation_pickup_bank.py` in section 23.1, updated
below. All three regimes (S0, S1, S2) were submitted as independent Slurm
jobs on 2026-08-24 (job IDs 365180-365182) after an initial submission
failed fast on a real-data filename mismatch (`data.diagnostics`, not
`data.diagnostics.production`) that local fixture tests hadn't caught; fixed
and resubmitted. **All three completed in 71-86 seconds each (much faster
than the untimed estimate in section 7.1) and passed Gate D0's P32 check**:
32/32 archived pickups per regime, 320/320 daily diagnostics, and the day
6,010/6,050/6,080 pickups' P32 projections are bit-identical to
`trajectories_v3.zarr`. Each regime's full result is at
`outputs/af_fno/response/forward_response_v1/pickup_bank_verification_<regime>.json`.
The three regime-specific day-5,760-to-6,080 chains referenced throughout
section 7 now exist on scratch at
`${AF_SCRATCH_ROOT}/mitgcm_response_pickup_bank_v1/<regime>/bridge_5760_6080/`.

Steps 1-5 of section 25 (contract freeze against the production parent,
pickup-bank/direction-geometry design, and the generic pickup editor) were
implemented in the same pass; see `config/model_c_adjoint_faithful_*_v1.json`,
`scripts/build_forward_response_inventory.py`, and
`archive/src/bire_repro/af_s0_twin.py`'s generalized editor.

## Implementation status and amendments (2026-08-25)

**Verified — step 7 (amplitude pilot, section 7.2) implemented and run.**
`scripts/build_amplitude_pilot.py` (geometry/kernel/RMS-scale/pickup-edit
machinery) and `archive/src/bire_repro/af_pilot_segment.py` (the generic
MITgcm driver, parameterized start day/duration/checkpoint interval and,
later, `cg2dTargetResidual`) implement the pilot. The 24 pilot centres
themselves are solved by a scoped-down, exact shortcut
(`_select_pilot_centre` in `build_amplitude_pilot.py`) rather than the full
`allocate_centres_lexicographically`, justified because pilot is first in
`ROLE_ORDER` (no earlier role to separate from) and its two rows per
`(regime, family)` group always sit in disjoint regions by construction of
the section-7.2 anchor table — both the cross-role and within-role
components of the frozen leximax objective are provably vacuous for pilot
specifically, so the reduction to tie-break rule (iii) is exact, not an
approximation. All 154 pilot branches (144 signed − 2 correctly-recorded
SSH-cap failures at S0/day3600/alpha 0.10, + 6 nominal + 6 duplicates) ran
successfully; verified job-by-job against returncode, diagnostic counts, and
archived-pickup counts, plus one direct byte-level check of a staged edit.

**Verified — step 8 (Gate D2) complete; all four amplitudes frozen.**
`scripts/analyze_amplitude_pilot.py` implements the section-10.2/10.4
provisional-stage diagnostics (`Q_lin`, `Q_SNR` against the duplicate-nominal
floor, P32 realization/antisymmetry, adjacent-alpha centred-JVP) purely from
already-completed pilot output (no new compute). Result: U and V passed at
alpha 0.10, SSH at 0.05; **Theta had no passing candidate at any of
{0.025, 0.05, 0.10}** — both S1 Theta directions failed `Q_lin` by roughly
10x the threshold, traced (via direct comparison of each direction's
regime-local standard deviation against the *pooled* S0+S1+S2 normalizer
sigma at that cell) to S1's local variability there being ~15-20x smaller
than the pooled sigma implies, since the pooled normalizer is dominated by
S0/S2's stronger wind forcing at the same grid cells — a real, verified
regime/normalization interaction, not a bug.

Per section 22 Gate D2's own text ("a smaller-amplitude follow-up requires a
separately versioned pilot contract"), `config/forward_response_amplitude_pilot_theta_v2.json`
freezes a Theta-only follow-up (candidate alphas 0.01/0.005/0.001, same 6
frozen centres, nominal branches reused unchanged) via
`scripts/submit_amplitude_pilot_theta_v2.py` /
`scripts/analyze_amplitude_pilot_theta_v2.py`. Result: **Theta = 0.005**
passes every direction and gate.

Section 10.3's confirmatory duplicate/tight-CG controls
(`scripts/submit_amplitude_pilot_controls.py` /
`scripts/analyze_amplitude_pilot_controls.py`, using
`af_pilot_segment.py`'s new `cg2dTargetResidual` override) then ran on the 12
preassigned long directions at each family's provisional alpha: 24 duplicate
+ 24 tight-CG signed reruns + 6 tight-CG nominal reruns, all reusing the
already-staged edited pickups (no re-editing). 11 of 12 directions passed
every check against the combined floor
(`max(duplicate-nominal, perturbed-repeat, tight-CG disagreement)`); one,
S1/day720/V, exceeded the `q_cg<=0.01` threshold at exactly one of nine
leads (day 80: 0.0105/0.0107). Root cause verified directly (absolute
production-vs-tight-CG disagreement is flat across leads 60-90 while the V
response itself decays over the same window, so the ratio crosses threshold
as signal fades toward a fixed noise floor, at the same S1/day-720/eastern
location already known to be the pilot's tightest-margin direction) and
accepted as a documented exception
(`GATE_D2_EXCEPTIONS` in `analyze_amplitude_pilot_controls.py`, with the
full evidence inline) rather than treated as a defect requiring another
versioned pilot, since no amplitude choice fixes a late-time signal-decay
effect. **Final frozen amplitudes: U = 0.10, V = 0.10, Theta = 0.005,
SSH = 0.05** (`outputs/af_fno/response/forward_response_v1/amplitude_pilot_final_selection_v1.json`).

**In progress — finishing step 4 (joint train/validation/blind centre
materialization).** Step 9 (generating the production/train/validation
response dataset) needs real `(j, i)` centres for the 672 train + 216
validation + 216 blind directions, which were never solved (only pilot's 24
were, via the shortcut above). Attempting the full joint solve via the
existing `allocate_centres_lexicographically` exposed that it is
intractable at production scale as originally written — a single
`(regime, family)` group (94 rows) has up to ~49,000 (row, candidate) slots
because every eligible cell in a large region (e.g. "interior") is a
candidate, and the pairwise cross/within-role separation objective compared
every such pair, O(n^2) in the candidate pool. Fixed in
`scripts/build_forward_response_inventory.py`, in order of discovery:

1. `_nearby_same_region_pairs`: replaced the O(n^2) all-pairs distance
   comparison with a KD-tree nearest-neighbour query on the exact unit-sphere
   embedding (chord length is a strictly monotonic function of great-circle
   distance, so this is an exact reformulation, not an approximation) —
   `_build_centre_problem` went from hanging indefinitely to 1.7s.
2. `_reduce_candidate_pool`: the *raw* candidate pool per row (not just its
   pairwise comparisons) still made the MIP itself too large to re-solve
   repeatedly. This reduction is a heuristic (farthest-point-diverse subset,
   always keeping the SHA-tie-break-first and scalar-objective-extreme
   candidates), so `allocate_centres_lexicographically` now solves at an
   escalating pool-size ladder (150/300/600/1200/2400) and requires two
   consecutive levels to agree on every objective value before accepting the
   cheaper one, raising rather than guessing if the ladder is exhausted.
3. Two genuine pre-existing bugs surfaced once execution reached this far
   for the first time (neither caused by the two fixes above): (a)
   `_freeze_sorted_region_sums` unconditionally built sorting machinery for
   all four non-WBC regions even when some had zero real candidates,
   producing a degenerate MIP HiGHS could not solve — fixed to only include
   regions with actual candidates, matching an analogous fix already needed
   in `_leximax_region_minima` for the single-role (pilot-only) case; (b)
   `_freeze_linear_optimum` rounded its *entire* solution vector to the
   nearest integer before freezing it as an exact constraint, which is
   correct for the binary 0/1 selection variables but corrupts the
   continuous real-valued km-distance variables used elsewhere in the same
   solver to the nearest whole kilometre, making the immediate re-solve
   provably infeasible — fixed to round only genuinely integer variables
   (`model.integrality`).

Every fix that changes the objective's meaning (not just its computational
path) is guarded by a post-hoc correctness check
(`_verify_leximax_vectors_on_selection` recomputes the true brute-force
pairwise minima directly on the small *selected* subset and fails loudly on
any mismatch) rather than trusted blindly. With all four fixes, a single
pilot-scale group (2 rows, ~2,100 candidates) solves correctly in
~30-1,200s (all 12 pilot groups: ~34 minutes total).

**Verified — full reproduction check against the already-frozen 24 pilot
centres: 23/24 exact, 1 accepted precision exception.** Running the
corrected general-purpose `allocate_centres_lexicographically` (not the
pilot-specific shortcut) on the same 24 pilot rows matched the frozen
geometry bit-for-bit in 23 cases. The one exception, S0/day3600/V, picked
`(41,43)` instead of the frozen `(41,18)`; direct comparison showed both
candidates' `tertiary_distance_km` differ by ~2.6e-12 km (2.6 picometres) --
floating-point noise in the great-circle trig chain, below HiGHS's ~1e-9
feasibility tolerance, not a geographically meaningful difference. **Reviewed
and accepted as a documented limitation, not a defect** (see the docstring on
`_freeze_sorted_region_sums` in `build_forward_response_inventory.py`): the
already-completed pilot campaign, which used `(41,18)`, remains valid as-is,
and the same sub-nanometre-tie situation may recur during train/validation/
blind materialization with no scientific consequence either way.

**Resolved 2026-08-25 -- both design questions below were delegated ("make
the call yourself") and are now implemented and tested.**

1. *Train's distance-three exemption.* Section 9.3 step 4's rule is: "require
   non-WBC validation and blind centres to have ... distance at least three
   from every centre assigned to an earlier role." Train is never the
   *subject* of this rule (it never has to keep distance from anything), but
   it IS a valid *earlier-role target*: validation and blind must still stay
   >= 3 from train's centres, exactly as from pilot's. The only pair the rule
   never touches is (pilot, train), since neither role is ever "validation or
   blind". `_build_centre_problem`'s hard distance-three constraint
   previously applied full pairwise distance symmetrically via unconstrained
   `itertools.combinations`, which would *also* have required train to keep
   distance from validation/blind (not just the reverse) -- stricter than the
   plan's text. Fixed by building `roles_present` in `ROLE_ORDER` sequence
   and skipping any pair whose later role is `"train"`, which is exactly
   equivalent to "apply only when the later role is validation or
   blind_test" (train can only ever be paired as the later role against
   pilot). This part was correct on first landing. A companion
   mischaracterization was not: an earlier version of this note, and of the
   in-code comment, additionally claimed "nothing has [a distance-three
   obligation] to [train] either" -- i.e. that validation/blind need not
   avoid train's centres. That is wrong per the text above, and
   `prove_hard_capacity`'s post-hoc witness verification had been written to
   match the wrong claim (checking every cross-role pair including
   pilot/train, which the solver correctly never constrains) rather than the
   correct one. Running the fixed solver's witness check against the real
   672+216+216-row contract surfaced the mismatch directly: a real S0/Theta
   witness with pilot and train landing near each other was rejected by the
   verifier as a false violation. Fixed 2026-08-25 by gating the verifier's
   distance check on `later_role in ("validation", "blind_test")`, matching
   the solver exactly; the in-code comment was corrected at the same time.
2. *Fixing pilot's already-frozen centres.* Pilot's 24 centres are already
   frozen (not re-derivable -- they underlie the completed, verified
   amplitude-pilot campaign) and must be supplied to train/validation/
   blind's joint solve as fixed input, not re-decided. Added
   `load_pilot_fixed_centres` (reads `amplitude_pilot_geometry_v1.json`,
   keyed by `(regime, anchor_day, family)`) and `apply_fixed_centres`
   (collapses each fixed row's candidate list to a singleton containing only
   its frozen `(j,i)` `Candidate`, raising `ContractError` if that centre is
   no longer among the row's enumerated candidates -- a safety check against
   grid/mask drift since pilot was solved). No solver-internals change was
   needed: a row with one candidate is already pinned by the existing
   "exactly one candidate per row" constraint, while its y-variable stays in
   the model so region quotas, the (now train-exempt) distance-three
   exclusion, and the separation objective all still see pilot's real
   positions.

The two design-question fixes and the verification-logic correction above are
covered by new tests in `tests/test_forward_response_inventory.py`
(`test_train_is_exempt_from_the_distance_three_exclusion`,
`test_apply_fixed_centres_collapses_pilot_rows_to_their_frozen_choice`,
`test_apply_fixed_centres_rejects_a_frozen_centre_no_longer_enumerated`,
`test_load_pilot_fixed_centres_reads_the_frozen_geometry_file`,
`test_prove_hard_capacity_lets_pilot_and_train_share_close_centres`).

**Two more issues found and fixed while finishing Step 4, both only
reachable at real production scale (never previously exercised).**

1. *`materialize_inventory` still depended on the retired OS-identity
   firewall.* The 2026-08-24 amendment above replaced the firewall with a
   lighter write-once/separate-path convention and deleted
   `config/adjoint_faithful_firewall_v1.json` and
   `scripts/verify_adjoint_faithful_firewall.py`, but
   `build_forward_response_inventory.py`'s `materialize_inventory` (and its
   CLI) were never updated to match -- they still imported and required the
   deleted verifier, so `materialize` mode could not run at all. Fixed:
   removed `live_firewall_report`/`require_inventory_materialize`/
   `_load_firewall_verifier`/`FirewallError`/`DEFAULT_FIREWALL_CONTRACT`;
   `materialize_inventory` now writes public and blind manifests via the
   existing O_EXCL write-once helper at two fixed, distinct paths (blind
   mode 0400, public mode 0444), refusing outright if either already exists,
   with no live process-identity check. The one part of the old mechanism
   worth keeping (refusing to write through a symlinked path) was kept as a
   small standalone check. This also wired in `apply_fixed_centres`/
   `load_pilot_fixed_centres` from fix 2 above, which had no caller before
   this edit.
2. *`prove_hard_capacity` (the `audit` mode's read-only capacity witness)
   built its MIP from the raw, unreduced candidate pool.* This is the same
   scaling problem already fixed for the main optimizer
   (`allocate_centres_lexicographically`), missed here because `audit` mode
   had only ever been exercised at small scale before. Confirmed directly: a
   production `audit` run burned >90 CPU-minutes without finishing. Fixed
   with an escalating-cap strategy analogous to the main solver's, but
   simpler: since this function only needs *any* feasible witness, not an
   optimum, a solution found under a reduced candidate pool is automatically
   valid for the full problem, so it is safe to stop at the first cap that
   solves rather than needing the main solver's two-consecutive-caps
   convergence check. On the real S0/Theta group (94 rows, 16,370
   variables), this now solves in ~51s. A related crash
   (`_build_centre_problem` on an empty `(regime,family)` row group, the
   same failure mode as the two `_leximax_region_minima`/
   `_freeze_sorted_region_sums` empty-region crashes fixed earlier) was
   found and fixed the same way: skip the group when it has no rows.

Full suite: 38/38 passing. The remaining 1,104-row (672 train + 216
validation + 216 blind) allocation can now be run correctly, but the
orchestration driver itself (analogous to `build_amplitude_pilot.py`,
loading all rows, applying `apply_fixed_centres`, calling
`allocate_centres_lexicographically` per group, and writing public vs. blind
outputs separately) has not yet been written, and the full-scale timing of
the *optimization* path (as opposed to the pure-feasibility witness above)
is still being verified.

## Implementation status and amendments (2026-08-26)

**Verified — the 2026-08-25 note above ("orchestration driver ... has not
yet been written") was stale.** `materialize_inventory` in
`build_forward_response_inventory.py` already implements exactly that driver
(load all rows, `apply_fixed_centres`, per-group centre solve, write-once
public/blind manifests) and the 38-test suite already covers it. Re-running
`_prepare_inventory_context` (contract/grid/trajectory verification, all
1,128 rows, region-slot assignment, long-subset feasibility repair) against
the real production contract takes 9.6 s. That was not what blocked step 4.

**Verified — the real blocker was the centre-allocation MIP's scope, not
missing code.** `config/forward_response_dataset_v1.json`'s
`joint_spatial_allocator.joint_objective_scope.solve_unit` reads *"one
(regime,family) across all five regions and all four roles"* -- a broader
scope than this document's own section 9.3 step 3, which specifies
allocating centres "**jointly** within every `(regime,family,region)`
stratum." Measured directly against the real production contract: a single
`(regime,family)` group's frozen leximax objective (94 rows, up to ~49,000
(row, candidate) slots per the 2026-08-25 note above) builds a MIP of
15,868 binaries and 54,053 constraints even at the cheapest candidate-pool
cap (150) in the existing escalation ladder, with 287,923 cross-role and
158,017 within-role pairwise terms and 15,990 distinct distance thresholds.
The *first* of the ten leximax-vector solves this triggers (the k=0 order
statistic, i.e. the tightest, largest-minimum-separation feasibility probe)
did not return within 24 minutes on a 180k-310k-constraint model; the same
failure mode reproduced for S0/SSH. This is standard behaviour for a
near-maximum-independent-set MIP at this scale, not a bug in any one
function.

**Fix.** `config/forward_response_dataset_v2.json` restores the scope this
document's own section 9.3 step 3 already specifies:
`joint_spatial_allocator.joint_objective_scope.solve_unit` is corrected to
"one (regime,family,region) stratum across all four roles". `approved_plan.sha256`
is repinned to this amended document (v1 pinned the pre-amendment hash and is
kept as the historical record of the pre-fix, never-materialized contract --
nothing was lost superseding it, since v1's `materialize` mode never ran to
completion). `build_forward_response_inventory.py` gained:

- `allocate_centres_lexicographically_by_region`, which solves each
  `(regime,family,region)` stratum with the existing, unchanged
  `allocate_centres_lexicographically` (including its candidate-pool-cap
  ladder and convergence check), then reassembles the group-level frozen
  objective (`_merge_region_objectives`) by concatenating each region's own
  single-entry cross/within/negative-distance contributions and re-sorting
  ascending. This is an *equality*, not an approximation: every
  pair/distance/speed term in `_build_centre_problem` is already scoped to
  rows sharing one region, so a joint solve has no cross-region objective
  term to trade off, and independently maximizing each region's own
  component reproduces the identical ascending vector by componentwise
  dominance. Verified directly on a small synthetic two-region problem
  against the unmodified `allocate_centres_lexicographically` (which still
  exists and is still exact for a single stratum): byte-identical selected
  centres and objective vectors
  (`test_allocate_centres_lexicographically_by_region_matches_joint_solve_when_independent`).
- `_non_wbc_chebyshev_violations` / a solve-then-verify-then-repair loop,
  because the one genuine cross-stratum coupling this document's section 9.3
  step 4 states without a same-region qualifier is the non-WBC distance-
  three rule: a validation/blind centre in one region can sit within
  Chebyshev distance 3 of an earlier-role centre in a physically adjacent
  *different* region. A direct audit of the real grid finds 2,406
  non-WBC cell pairs within Chebyshev distance 2 that cross a region
  boundary, against 36,192 that do not (6.2%) -- rare but real, so it is
  checked and repaired (exclude the offending later-role centre, re-solve
  only that region), not assumed away. `_MAX_CROSS_REGION_REPAIR_ATTEMPTS`
  bounds the loop; exhausting it raises `CapacityError` rather than emitting
  an unverified geometry, matching the existing region-slot repair's
  "stop before inventory materialization" convention.
- Two non-scientific performance fixes applied inside the unchanged
  algorithm, verified to change wall time only, never the frozen tie-break
  or leximax semantics: `_MixedIntegerModel.fix` now tightens variable
  bounds instead of adding a constraint row (identical feasible set, no
  constraint-matrix growth across the SHA-tie loop's per-row fix attempts);
  `_add_order_statistic_threshold` now covers each region's below-threshold
  conflict graph with a greedy clique cover
  (`sum_{v in C} y_v - (|C|-1)*relax <= 1` per clique) instead of one row
  per conflicting pair -- the identical feasible integer set (a 2-vertex
  clique reduces to the original pairwise row byte-for-byte) with a
  strictly tighter LP relaxation. Benchmarked on the real S0/Theta/WBC
  stratum: 36.6 s to 12.3 s (3.0x) on a single representative probe.
- `materialize_inventory` now validates that *both* the public and blind
  output directories exist before writing either file (previously it wrote
  blind first, sealed write-once at mode 0400, and only then checked
  public's parent -- a real, verified bug: `outputs/af_fno/response/
  forward_response_blind_v1/` does not exist on this machine, so an
  unpatched run would have sealed the blind manifest and then failed on the
  public write, with every retry thereafter hitting "write-once inventory
  output already exists" and public never created). Covered by
  `test_materialize_inventory_requires_both_output_parents_before_writing`.

**Verified — proof of concept at full, unreduced candidate pool.** The
hardest single stratum in the study, S0/Theta/WBC (33 rows drawn from 103
eligible cells), solved its complete frozen lexicographic objective with no
candidate-pool reduction at all in 418 s / 529 MIP solves (cross-role
minimum separation 1337.53 km, within-role 64.57 km). Since the pool-cap
ladder's `fully_covers_pool` check is then trivially true, this run used no
heuristic reduction whatsoever -- exact by construction, not by the
ladder's weaker convergence argument.

Full 46-test suite (38 prior + 8 new) passes. `tests/test_forward_response_
inventory.py`'s `DATASET_CONTRACT` now points at v2.

---

## 1. Executive summary

**Verified.** The production parent
`model_c_production_1in_1out_spectralnorm_v1` is a strong one-input,
one-output, 10-day Markov emulator. It predicts the full 46-channel state
`[U(15), V(15), Theta(15), SSH(1)]`, uses five physical static inputs, and was
trained from random initialization with a six-call/60-day autonomous rollout.
Its architecture has 32 by 32 Fourier modes, width 128, three FNO blocks, a
zero-initialized bias-free local 3 by 3 branch, and per-mode spectral
normalization with `rho=1`. Its selected checkpoint is step 7,680 with SHA-256
`e75951681b1a...`.

**Verified.** The later child
`model_c_production_1in_1out_spectralnorm_ft90_v1` has the same architecture,
state, normalizer, loss coefficients, and spectral-normalization machinery. It
was initialized from the parent and fine-tuned for 1,920 updates using a
nine-call/90-day rollout. It improved most reported forward metrics, but it is
not the baseline or weight source for this study.

**Verified.** Phase A established a severe adjoint failure for the **ft90
child**, not yet for the parent. Against MITgcm/TAF SSH adjoints that passed
the scalar error, cross-run, land, and conservation gates, the child's
truth-forced adjoint has pattern
correlation near zero at 10, 20, 30, and 90 days and amplitude ratios of about
38, 16, 9.7, and 6.6 for the point objective. The mismatch is overwhelmingly
Jacobian error rather than trajectory drift. The parent has not yet been run
through the current one-input Phase-A adjoint package, so its quantitative
adjoint error is unresolved.

**Proposed.** The primary experiment will preserve the parent model design and
60-day training schedule while adding one new term: a forward perturbation-
response loss. The response-aware model will start from random initialization.
No parent or ft90 checkpoint will be loaded. Its matched nominal-only control
will use the same random seeds, optimizer budget, exact implemented production
split, normalization, architecture, rollout, optimizer schedule, parent loss,
and checkpoint selector.

The core finite-difference identity is

$$
\frac{M_k(x+\epsilon v)-M_k(x-\epsilon v)}{2\epsilon}
  = D M_k(x)v + O(\epsilon^2),
$$

with the analogous expression for the FNO. Matching these forward responses
constrains Jacobian-vector products without providing an adjoint label. For any
scalar-objective cotangent $w$,

$$
\left\langle
  (D F_k(x)^T-D M_k(x)^T)w,\,v
\right\rangle
=
\left\langle
  w,\,(D F_k(x)-D M_k(x))v
\right\rangle.
$$

Thus better forward JVPs imply better adjoint **projections onto the sampled
direction span**. They do not guarantee the full Jacobian in unsampled
directions; the final blind adjoint suite is needed to answer that question.

**Proposed data scale.** The production response dataset will contain, per
regime, 224 training directions at 14 annual pickup anchors, 72 validation
directions at three later anchors, and 72 blind-test directions at three still
later anchors. Every direction is run with both signs. Most runs stop at 10
days; a predeclared subset continues to the role-specific 60- or 90-day
horizon and writes endpoints every 10 days. Including the three-amplitude
pilot, validation pickup-bank setup, paired nominal branches, and all
repeat/solver controls, the exact total is **57,750 model-days = 160.417
model-years**. This is forward MITgcm only.

**Non-negotiable firewall.** No MITgcm adjoint, TAF output, existing adjoint
metric, new FNO adjoint map, or analytic adjoint probe will enter amplitude
selection, response-loss-weight selection, early stopping, checkpoint
selection, or architecture selection. Historical summary numbers are already
known, but no new-model adjoint evaluation or development-pipeline access is
permitted until model identity, selected checkpoints, amplitudes, response
weight, and all forward reports are frozen and hashed.

---

## 2. Current parent/child forward evidence

### 2.1 Implemented production map

**Verified from** `config/model_c_production_1in_1out_spectralnorm_v1.json`,
`src/oceanfno/model.py`, `src/oceanfno/dataset.py`,
`src/oceanfno/objective.py`, `src/oceanfno/spectral_norm.py`, and the current
parent report:

| Item | Implemented value |
| --- | --- |
| Dynamic input/output | `U_01:15`, `V_01:15`, `Theta_01:15`, `Eta`; 46 channels |
| Static input | wind stress, wet mask, Coriolis parameter, MITgcm `DXF`, SST relaxation target; 5 channels |
| Position encoding | 2 Bire sine/cosine channels added inside the model |
| Map | direct state, `x(t+10 d) = F_theta(x(t), S)`, not a residual |
| Grid | 62 by 62 by 15, with 3,600 wet tracer cells |
| Fourier operator | 32 by 32 modes, width 128, 3 blocks |
| Block normalization | six pointwise channel LayerNorms, two per block |
| Domain padding | 0.1 |
| Spectral weights/precision | dense, unfactorized (`factorization=null`); full FNO-block precision |
| Lifting/projection | ratio 2, hence width 256 |
| Channel MLP | expansion 4, no dropout |
| Local path | bias-free 3 by 3 convolution, zero initialized |
| Parameters | 27,297,960 |
| Spectral normalization | 1,632 per-mode complex matrices, `rho=1`, 400-iteration warm start, **2** power iterations per forward |
| Inference checkpoint | materialized spectral weights; no live clipping or power iteration |

The full architecture dictionary in the parent JSON—not this abbreviated
human table—is normative and must pass the field-by-field equality gate.

The parent objective implemented in `src/oceanfno/objective.py` is

$$
\begin{aligned}
L_{\rm parent}={}&L_{\rm state}
+0.001L_{\rm increment}
+0.50L_{\rm rollout}
+10^{-5}L_{\rm spectral}\\
&+0.065L_{\rm boundary}
+0.05L_{\rm pressure}
+0.05L_{\rm continuity}
+0.05L_{\rm transport}.
\end{aligned}
$$

All terms are dimensionless. The five state-space terms are explicitly group-
balanced over U, V, Theta, and SSH; pressure-gradient, continuity, and
barotropic-transport use their own implemented level/component balances. The
three physics terms are truth-referenced. No physics or adjoint channel is
added to the output.

### 2.2 Parent and ft90 optimization

**Verified.**

| Item | Parent | ft90 child |
| --- | ---: | ---: |
| Initialization | random | parent selected checkpoint |
| Autonomous rollout | 6 calls / 60 d | 9 calls / 90 d |
| Adam learning rate | `5e-4` through step 5,760, then `1e-4` | constant `5e-5` |
| Updates | 7,680 | 1,920 |
| Effective batch | 8 = 4 x 2 accumulation | 8 = 2 x 4 accumulation |
| State transitions | 368,640 | 138,240 |
| Selected step | 7,680 | 1,440 |
| Measured V100 wall time | 3.215 h | 1.671 h |
| Checkpoint SHA-256 | `e75951681b1a...` | `4acb7633d85a...` |

The child normalizer is byte-identical to the parent's, SHA-256
`fe424b37d74f...`. The new runs load no child/parent checkpoint; they
recompute the exact parent normalizer recipe and must hash-match that artifact
so the physical coordinate system is unchanged.

### 2.3 Forward metrics

**Verified selected pooled-validation metrics:**

| Metric | Parent | ft90 child |
| --- | ---: | ---: |
| 10-90 d RMSE-AUC, surface speed | 0.080820 | 0.071042 |
| 10-90 d RMSE-AUC, SST | 0.821037 | 0.735979 |
| 10-90 d RMSE-AUC, surface PHIHYD | 0.579126 | 0.267129 |
| 90-360 d AUC/climatology, speed | 0.309277 | 0.274442 |
| 90-360 d AUC/climatology, SST | 0.289608 | 0.269316 |
| 90-360 d AUC/climatology, PHIHYD | 0.182586 | 0.104063 |
| Twin growth per call | 1.013225 | 1.008315 |
| Maximum normalized amplitude through 360 d | 5.257 | 5.398 |

Both selected through the implemented fallback because no candidate met the
declared growth ceiling of 1.0.

**Verified held S0 day-2,000 metrics:**

| Metric | Parent | ft90 child | Reference |
| --- | ---: | ---: | ---: |
| PHIHYD RMSE | 0.039411 | 0.020448 | train climatology 0.082489 |
| SST RMSE | 0.057655 | 0.037402 | train climatology 0.040458 |
| Speed RMSE | 0.006326 | 0.004839 | train climatology 0.003587 |
| Maximum normalized magnitude | 10.328 | 8.434 | -- |
| Day-2,000 anomaly RMS ratio | 1.795 | 1.738 | MITgcm truth 1.0 |
| Day-2,000 WBC/interior anomaly ratio | 5.059 | 8.157 | MITgcm truth 23.098 |

**Inferred.** The child improves several forward RMSE/AUC and long-range
metrics but does not solve perturbation growth, the day-2,000 magnitude gate,
or the weak WBC concentration. It is valuable as a contextual forward
comparator, not as the starting point for the new response-aware model.

### 2.4 Why the primary schedule remains 60 days

**Verified.** The surviving 30-to-60-to-90 curriculum artifact used 11,520
updates and 599,040 state transitions, while also changing `rho` to 0.99. Its
selected checkpoint has growth 1.0216, all three 90-360-day flattening ratios
above 1.9, held maximum normalized magnitude 21.39, and day-2,000 RMSE ratios
to climatology of 3.82, 6.06, and 5.70. Its source config and launcher are not
present in the tracked current tree, and some artifact prose disagrees about
`rho`; it is evidence, not a reproducible baseline.

**Inferred.** The evidence is against automatically bundling a curriculum,
longer exposure, and a tighter spectral cap. It does not isolate which of
those changes caused the failure.

**Proposed.** Both new from-scratch primary arms retain the successful 60-day
schedule. V1 has no 90-day continuation, curriculum, or second training stage;
day 90 is a sealed evaluation horizon only.

### 2.5 Implemented selection nuance

**Verified.** The shared production selector in `src/oceanfno/validation.py`
enforces the within-run 5% short-AUC filter, attempts the growth ceiling, and
then applies its long-range/fallback rule. The ft90 code computes the handbook's
parent-relative short-skill and flattening conditions as acceptance diagnostics,
but those extra conditions are not filters passed into the implemented shared
checkpoint selector. The new contract below follows implemented code where it
differs from older prose, retains that selector unchanged, and implements any
new forward-preservation threshold only as a post-selection scientific gate.

---

## 3. Current adjoint failure evidence

### 3.1 What is established

**Verified.** Phase A is complete despite stale early status paragraphs that
say later steps are queued. The final comparison report is non-provisional,
and the focused tests currently pass (`65 passed` across
`tests/test_adjoint_phase_a.py` and `tests/test_mitgcm_adjoint.py`).

The MITgcm reference uses checkpoint68j commit `f03a2f5e...`, TAF 6.8.11,
source day 7,200, and leads 10, 20, 30, and 90 days. Its 90-day gradient check
passes the scalar `1e-4` error threshold at seven wet points, with worst
relative disagreement `2.10e-6`. The intended interior-epsilon-minimum
evidence holds at six of seven points; the current Boolean gate does not
enforce that plateau flag. Cross-run SSH adjoints agree exactly, the mean-
conservation gate is `3.57e-8`, and all 244 land cells are exactly zero. The
current v2 G0 trajectory check is ETAN-only, not a 46-channel trajectory gate.
These caveats must be resolved or explicitly carried by final Gate A0.

The **ft90 child's** truth-forced FNO adjoint versus MITgcm for the point SSH
anomaly objective is:

| Lead | Pattern correlation | Relative L2 | Amplitude ratio | Sign agreement |
| ---: | ---: | ---: | ---: | ---: |
| 10 d | 0.0594 | 37.895 | 37.941 | 0.431 |
| 20 d | -0.0106 | 16.523 | 16.482 | 0.522 |
| 30 d | 0.0084 | 9.694 | 9.651 | 0.513 |
| 90 d | 0.0213 | 6.620 | 6.565 | 0.483 |

For the five-point meridional kernel objective, smoothing reduces amplitude
error but does not recover the pattern:

| Lead | Pattern correlation | Relative L2 | Amplitude ratio |
| ---: | ---: | ---: | ---: |
| 10 d | 0.0821 | 26.154 | 26.217 |
| 20 d | -0.0109 | 12.367 | 12.316 |
| 30 d | 0.0163 | 6.861 | 6.804 |
| 90 d | 0.0282 | 2.198 | 1.985 |

The trajectory-drift component (`S_free` versus `S_forced`) is 0, 0.021,
0.062, and 0.094 relative L2 at the same leads, two to three orders smaller
than the total mismatch at short lead and still far smaller at 90 days.

The child also destroys the analytically conserved basin-mean SSH mode: the
adjoint amplitude ratio falls from 0.411 at 10 days to 0.0062 at 90 days, and
the 90-day pattern correlation is 0.039. At 10 days, 67% of child sensitivity
variance lies below the FNO's 4.625-cell spectral cutoff versus 4.4% for
MITgcm; low-pass filtering does not expose a hidden correct large-scale map.

### 3.2 What is not established

- **Unresolved:** no current one-input Phase-A result exists for the frozen
  production parent. The old `fno_s0_adjoint_v1` product is a stale two-input
  model and must not be used.
- **Verified limitation:** Phase A is S0-only and its primary control perturbs
  SSH while holding U, V, and Theta fixed. That direction is dynamically
  unbalanced and adjusts through barotropic gravity waves inside the FNO's
  10-day step.
- **Verified limitation:** packaged Phase-A arrays contain SSH adjoints. Raw
  P90 scratch has U/V/Theta adjoint dumps, but direct U/V comparison requires
  the adjoint of the face-to-centre map and is not yet validated.

**Inferred.** The failure is consistent with accurate state forecasts but an
inaccurate local derivative, particularly off the nominal trajectory manifold.
It motivates forward JVP supervision but does not predetermine that the method
will work for the parent or in every tangent direction.

---

## 4. Research hypothesis and identification strategy

MITgcm does not evolve the 46-channel FNO state in isolation. Let $Z_a$ be the
complete 108-record source pickup at anchor $a$, let $E_{a,q}(\eta)$ copy that
pickup and add the declared native-grid edit $\eta v_q$ to only the selected
`Uvel`, `Vvel`, `Theta`, or `EtaN` records, and let $\Phi_{a,k}$ be the full
MITgcm flow for $k$ days with the anchor's regime forcing. The trusted
projection $P$ extracts U/V/Theta/EtaN and centres the native U/V faces. The
conditional forward map actually sampled by this study is therefore

$$
M_{a,q,k}(\eta)=P\Phi_{a,k}[E_{a,q}(\eta)],
\qquad x_{a,q}(\eta)=P E_{a,q}(\eta).
$$

Salt, Adams-Bashforth/history records, and all other restart content in $Z_a$
are held fixed at the initial time, then evolve normally. This definition
does not pretend that an inverse normalization can reconstruct a complete
MITgcm restart from 46 channels.

Write the frozen pointwise FNO-state normalization as
$\hat x=\mathcal N_x(x)=S_x^{-1}(x-\mu_x)$. At an output lead write
$\mathcal N_k(y)=S_k^{-1}(y-\mu_k)$; in this study it is the same
time-independent state-normalizer artifact, so $S_k=S_x$ and $\mu_k=\mu_x$,
with the subscript retained only to distinguish endpoint from input roles. The learned map
$\hat F_\theta$ acts in those normalized coordinates. With a small
dimensionless standardized magnitude $\epsilon_q$, define the exact projected
initial states

$$
x_{a,q}^{s}=x_{a,q}(s\epsilon_q),\qquad s\in\{-1,+1\}.
$$

After the pilot, $\epsilon_q$ is the frozen family amplitude
$\alpha_{h(q)}$ used by the manifests and loaders.

Define oriented signed responses

$$
r_{M,a,q,k}^{s}
=\frac{\mathcal N[M_{a,q,k}(s\epsilon_q)]
       -\mathcal N[M_{a,q,k}(0)]}{s\epsilon_q},
$$

where $\mathcal N=\mathcal N_k$ is the one common parent-training scale used
for external comparison. The FNO response is

$$
r_{F,a,q,k}^{s}=
\frac{\hat F_\theta^{k/10}[\mathcal N_x(x_{a,q}^{s})]
      -\hat F_\theta^{k/10}[\mathcal N_x(x_{a,q}(0))]}{s\epsilon_q}.
$$

In the local regime,

$$
r_{M,a,q,k}^{+}\simeq r_{M,a,q,k}^{-}\simeq
\left.\frac{\partial}{\partial\eta}
\mathcal N[M_{a,q,k}(\eta)]\right|_{\eta=0},
$$

Define
$\hat v_{a,q}=\left.\partial_\eta\mathcal N_x[P E_{a,q}(\eta)]
\right|_{0}$. The FNO expression is the corresponding
$D\hat F_\theta^{k/10}(\hat x_a)\hat v_{a,q}$.
Because $E_{a,q}$ is an affine pickup edit, these are precisely conditional
Jacobian-vector products: $DM\,v_q$ with the unedited restart components held
fixed. Matching both signs therefore constrains the odd/JVP component and
exposes even, finite-amplitude contamination rather than hiding it in an
independently fitted perturbed state.

In implementation, each FNO branch starts from the exact canonical float32
projection of the corresponding edited pickup, not from a reconstructed Zarr
restart. The pilot's 1% magnitude/antisymmetry gates bound the tiny departure
from ideal $x\pm\epsilon v$ introduced by float32 projection; the denominator
remains the declared signed physical coefficient $s\epsilon_q$.

For an autoregressive lead $k=10n$ days,

$$
D\hat F_\theta^n(\hat x_0)
=D\hat F_\theta(\hat x_{n-1})\cdots D\hat F_\theta(\hat x_0).
$$

The sparse long responses constrain this Jacobian product along the forward
trajectory. Transposing reverses the order, exactly as in an adjoint sweep.
For a physical final-state cotangent $w$, the normalized reverse seed is
$S_k^T w$ and the physical initial gradient is
$S_x^{-T}[D\hat F_\theta^n(\hat x)]^T S_k^T w$. All final adjoint maps and metrics are
therefore transformed back to physical units; response training never
silently equates normalized and physical cotangents.

**Primary hypothesis.** Relative to a matched nominal-only from-scratch
control, forward-only response supervision will reduce held-out response error
and later reduce blind MITgcm-adjoint error while keeping nominal forward skill
inside a predeclared tolerance.

**Identification limits.** Finite directional supervision constrains the
Jacobian only on the sampled state/direction distribution. Improvement of a
blind adjoint projection is expected by duality, not guaranteed. The controls
below isolate response information with paired B/C runs that have identical
initialization, nominal minibatch sequence, nominal-loss definition/weight,
optimizer step count/schedule, architecture, and rollout horizon. The sole
algorithmic intervention is C's added response gradient; later numerical
nominal losses/gradients may differ endogenously because the weights have
diverged.

---

## 5. Experimental arms and controls

| Arm | Initialization | Data/objective | Rollout | Role |
| --- | --- | --- | ---: | --- |
| **A** frozen parent | existing random-init run | legacy production nominal data/loss | 60 d | Historical production reference; no retraining |
| **A90-context** ft90 child | parent checkpoint | legacy nominal data/loss | 90 d | Context only; never a baseline architecture or weight source |
| **B** exact parent-protocol replay | random | common response-study runner with response disabled; exact parent scientific contract | 60 d | Prospective from-scratch nominal control |
| **C** response-aware | random, paired seed with B | same runner and parent nominal contract as B, plus only `lambda_resp L_response` | 60 d for both nominal and response-gradient unrolls | Primary new model |

**Proposed names.**

- B: `model_c_adjoint_faithful_nominal_control_v1`
- C: `model_c_adjoint_faithful_response_v1`

**Proposed replication.** The primary paired B/C seed is the production
parent's exact seed, `20260724`. Two secondary paired replications use
`20260911` and `20260912`. No “best seed” is selected; the historical-seed arm
is primary and all three identities are frozen before blind evaluation. Within
each pair, initialization, nominal minibatch IDs, exact nominal-loss
code/coefficients, optimizer step indices, and learning-rate schedule are
identical. The numerical parent-loss value/gradient is not expected to remain
equal once weights diverge; the only code/data intervention causing that
divergence is the declared response-gradient contribution.

### 5.1 Single permitted primary delta

The following B/C quantities are frozen to the production implementation:

- trajectory store and active in-memory split;
- pointwise, increment, and static normalization algorithms and source days;
- 27,297,960-parameter architecture and spectral-normalization machinery;
- six-call/60-day autonomous unroll;
- `RolloutDataset`, regime-major records, `ChunkAwareBatchSampler`, effective
  batch eight, and nominal batch order for the paired seed;
- Adam, learning-rate schedule, update count, checkpoints, and lack of clipping;
- all eight forward-loss terms, coefficients, masks, and physics numerics;
- 360-day nominal validation code and production checkpoint selector.

The exact nominal randomness contract is also frozen: the global seed
initializes Python, NumPy, Torch CPU/CUDA, and the existing deterministic
runtime flags; records are regime-major and day-ascending; contiguous
four-record microbatches never cross regimes; only microbatch order is
shuffled each epoch with `random.Random(seed + epoch)`; and two consecutive
microbatches form one optimizer update.

C alone reads the separate response-training store and adds
`lambda_resp I_joint L_response` on predeclared optimizer updates. The response
sampler uses an isolated counter/hash stream and cannot consume Python, NumPy,
or Torch RNG state used by initialization or the nominal sampler. Persistent
spectral power vectors are snapshot/restored around auxiliary forwards as in
section 15.

There is **no** parent continuation, 90-day training continuation, curriculum,
ordinary perturbed-state loss, short-only arm, optimizer-budget extension, or
checkpoint initialization in v1. Those would introduce a second scientific
change and require a separately approved study and a new blind-response set.

---

## 6. Exact production-parent nominal training/validation/inference split

### 6.1 Immutable nominal source

**Verified.** `trajectories_v3.zarr` contains three independently equilibrated
regimes, each with 9,000 daily states and shape `(9000,46,62,62)`. S0/S1/S2
use wind amplitudes 1.0/0.75/1.25, corresponding to
0.100/0.075/0.125 N m^-2. The store carries its own three-block metadata, but
the production parent deliberately verifies that metadata and then overrides
it in memory with `src/oceanfno/dataset.py::store_codes`.

**Frozen proposal.** B and C use that implemented parent split exactly; the
Zarr store and its metadata remain byte-unchanged and response data live in a
separate store:

| Half-open days | Inclusive days | Exact parent role |
| --- | --- | --- |
| `[0,6000)` | 0-5,999 | nominal training; parent normalizers and climatologies |
| `[6000,7200)` | 6,000-7,199 | nominal validation |
| `[6200,7200)` | 6,200-7,199 | nested inference block inside validation |
| `[7200,9000)` | 7,200-8,999 | evaluation truth only; split code zero |

There are no nominal buffers and no independent third nominal test split.
The final-inference protocol is nested in validation because that is the
implemented production design; this study does not relabel any subset of
days 6,000-8,999 as an independent nominal test set.

### 6.2 Exact nominal records

- **Training:** all 5,940 valid six-call starts per regime, days 0-5,939;
  17,820 sequences pooled. The final target is day 5,999.
- **Nominal checkpoint validation:** exactly 34 starts per regime,
  $a_m=6000+6m$, $m=0,\ldots,33$, i.e. days 6,000, 6,006, ..., 6,198.
  Every checkpoint is rolled autonomously for 360 days, producing 102 pooled
  validation members, exactly as in `src/oceanfno/validation.py`.
- **Final production inference:** the existing S0 figure package uses seed
  `20260802`, draws from the admissible 2,000-day candidate-start window
  `[6200,7000)`, and uses exactly 15 starts:
  `6263, 6293, 6331, 6389, 6579, 6593, 6598, 6601, 6651, 6661,
  6694, 6707, 6711, 6968, 6979`. Each model receives only its initial state;
  lead-matched truth through day 2,000 may extend into days 7,200-8,999. These
  starts and their figure/anomaly definitions are reused unchanged after
  checkpoint freeze. They are called **nested final inference**, not a new
  nominal test split.

### 6.3 Normalization

**Frozen proposal.** Each B/C run invokes the production functions
`training_pointwise_normalizers`, `training_increment_scale`, and
`store_wind_normalization` on the production split, exactly as the parent did.
Static construction remains the existing `physical_static_block` followed by
`static_block`; no response-specific static feature is introduced.
The pointwise state recipe uses all 18,000 S0/S1/S2 snapshots on days 0-5,999:

$$
\hat x_c(j,i)=\frac{x_c(j,i)-\mu_c(j,i)}{\sigma_c(j,i)},
$$

with float64 accumulation, population variance, a per-channel fifth percentile
of wet-cell standard deviations, absolute floor `1e-6`, land mean/raw scale
zero, land scale one, and normalized land reset to zero. The increment divisor
is the per-channel RMS of normalized 10-day increments for starts 0--5,989:
5,990 pairs/regime and 17,970 total. Wind is pooled-standardized over
regime/wet cells; wet mask remains
0/1; Coriolis, DXF, and SST target use the exact parent wet-cell
standardizations.

Every recomputation must hash-match the existing parent artifact
`model_c_production_1in_1out_spectralnorm_v1_train_only_normalization.npz`,
SHA-256
`fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf`.
The point-mean, raw-scale, scale, floor, and increment component hashes are
also checked. A mismatch stops before training. Perturbed states and response
data never enter these quantities. The separate response RMS scales in
section 14 normalize only `L_response`; they do not modify any parent input,
target, static, increment, or forward-loss normalization.
The whole-NPZ byte check assumes the pinned writer/runtime (including fixed ZIP
metadata); component-array hashes remain the semantic cross-runtime gate.

### 6.4 Comparator caveat

**Proposed interpretation.** A versus B is now an exact-protocol reproduction,
not a different-split contextual comparison. B versus C is the prospective
causal comparison. The ft90 child remains contextual because it is a
continuation, but it shares the parent's nominal chronology. Investigators
have already seen A/ft90 final-inference metrics; the same starts remain sealed
from new-model training and selection, but are described honestly as held
nested inference rather than historically unseen data.

---

## 7. Exact new MITgcm response-data design

### 7.1 Existing anchors and the validation pickup bank

**Verified.** Complete, float64 MITgcm pickups already exist annually for every
regime. The trajectory-day/iteration relation is

$$
I(d)=2{,}592{,}000+72d,
$$

with a 1,200 s timestep and 360-day model year. Pilot, response-training, and
blind-response anchors use existing annual pickups. Three off-cycle
response-validation anchors are generated from one complete nominal pickup
chain per regime; no restart is ever reconstructed from the 46-channel Zarr
state.

At a segment boundary such as day 3,600, the source resolver enumerates every
candidate pickup against the canonical trajectory-v3 source-chain inventory.
If duplicate copies exist, all `.meta` and `.data` hashes must agree. The
resolver does **not** assume the canonical copy is in a downstream directory:
the current chains retain some unique boundary pickups in the upstream
segment. Missing or conflicting hashes fail the anchor. The manifest retains
every candidate path/hash and the canonical-choice reason.

For validation only, start from the verified annual day-5,760 pickup and run
one unperturbed, regime-correct chain through day 6,080, archiving complete
pickups every 10 days. This costs 320 model-days per regime, 960 total. Retain
days 6,010, 6,050, and 6,080 as sources. Their canonical P32 projections must
match trajectory-v3 before any perturbation is staged; otherwise Gate D0
stops. This bridge is response-data setup only. It is excluded from nominal
training, normalization, climatology, and checkpoint validation.

### 7.2 Amplitude pilot anchors

**Proposed.** Pilot anchors are days **720 and 3,600** in each regime, six
anchors total. These are inside training, have existing pickups, and are
separated by eight model years. Pilot centre/location IDs are reserved and are
not reused by production response directions, even though the anchor times are
also eligible for production.

At each pilot anchor and regime, run one U, one V, one Theta, and one SSH
direction. SSH is a wet-cell point at day 720 and a smooth 5 by 5 kernel at day
3,600. This gives 24 base directions: six each for U, V, Theta, and SSH.

The six `(regime,day)` cases are ordered
`[(S0,720),(S0,3600),(S1,720),(S1,3600),(S2,720),(S2,3600)]`. Their region
sequence is exactly
`[WBC,interior,eastern,northern,southern,WBC]`. U uses one-based levels
`[1,4,7,10,13,15]`; V uses `[7,10,13,15,1,4]`; and Theta uses
`[13,15,1,4,7,10]`. Thus every group samples upper, middle, and deep water,
all five regions occur, and WBC has two representatives. SSH inherits the
same region sequence and has no vertical level. Within each specified
region/level cell, the section-9 deterministic sampler chooses the centre;
the resulting exact `(j,i)` inventory is frozen before the pilot runs.

Every base direction is run at `alpha={0.025,0.05,0.10}` and both signs: 144
perturbed runs. Twelve base directions, exactly one per input group and regime,
continue to 90 days; the remainder stop at 10 days. The long assignment is:

| Regime | Day 720 long groups | Day 3,600 long groups |
| --- | --- | --- |
| S0 | U, Theta | V, SSH |
| S1 | V, SSH | U, Theta |
| S2 | U, SSH | V, Theta |

Thus the three long SSH pilot controls are the S0 smooth case and the S1/S2
point cases; their exact regions are interior/eastern/southern from the frozen
six-case sequence. The other three SSH cases still constrain day-10 amplitude.

All six paired nominal pilot branches run 90 days, and all six are duplicated
once to measure the deterministic/numerical floor. Pilot responses are used
only to freeze amplitudes and are excluded from train/validation/test losses.

### 7.3 Production response anchors

**Proposed exact days, per regime:**

| Role | Anchor days | Count/regime | Horizon guarantee |
| --- | --- | ---: | --- |
| response train | `0,360,720,1440,1800,2160,2520,3240,3600,3960,4320,5040,5400,5760` | 14 | every 60-day endpoint <=5,820, inside parent train |
| response validation | `6010,6050,6080` | 3 | every 60-day endpoint <=6,140, inside validation and before nested inference |
| blind response test | `7560,7920,8280` | 3 | every 90-day endpoint <=8,370, in model-unseen truth-only chronology |

The 14 training anchors are a count-preserving deterministic spread across the
17 eligible annual parent-training pickups. Response-validation starts are
distinct from all nominal selection starts `6000+6m`. The blind anchors avoid
the existing Phase-A source day 7,200, are frozen before model training, and
are never read by training, validation, hyperparameter selection, or
checkpoint selection.

### 7.4 Directions at each anchor

At each **training** anchor, create exactly 16 directions:

- four U patches;
- four V patches;
- four Theta patches;
- two SSH point directions;
- two SSH smooth directions.

This is 224 directions per regime: 56 U, 56 V, 56 Theta, and 56 SSH (28 point,
28 smooth).

At each **validation or blind-test** anchor, create exactly 24 directions:

- six U directions;
- six V directions;
- six Theta directions;
- three SSH point directions;
- three SSH smooth directions.

This is 72 directions per regime: 18 per input group, with nine point and nine
smooth SSH directions.

Every production direction has both signs. “Direction count” and “MITgcm run
count” are therefore kept separate in every report.

### 7.5 Sparse-long subsets

All directions provide a 10-day response. Sparse-long **training and
validation** directions continue only to the parent's six-call horizon and
write days 10,20,...,60. Blind long directions alone continue to day 90:

- **Training:** at days
  `{0,720,1800,2520,3600,4320,5040,5760}`, extend one preassigned direction
  from each input group. This is 8 long directions per group, 32 per regime,
  and 96 total. The other 576 training directions across all regimes are
  short-only.
- **Validation:** at each of the three anchors, extend one direction from each
  input group to 60 days. This is 12 per regime, 36 total.
- **Blind response test:** the same count and rule, with a disjoint centre and
  level inventory, integrated to 90 days: 12 per regime, 36 total.

The amplitude pilot and its numerical controls still run their preassigned
long cases to 90 days solely to prove that the final blind perturbations remain
in a measurable local regime. No FNO response loss, lambda decision,
checkpoint decision, or development response score reads a day-70/80/90
model response. Thus every gradient-carrying model unroll remains exactly six
calls/60 days, while day-90 response and adjoint skill remain genuinely blind
extrapolation tests.

Long membership is selected by a constrained deterministic inventory solve
before any response is run. For each training group/regime, its eight choices
must include two WBC cases, at least one case from each other region, and, for
U/V/Theta, at least two cases in each of levels 1-5, 6-10, and 11-15; SSH must
contain four point and four smooth cases. For each validation variable/regime,
the three long choices are exactly two single-level cases and one held-out
multi-level case, collectively covering upper, middle, and deep bands at three
distinct regions. The multi-level type is
`t=(r+o_h) mod 3` for upper/middle/deep type indices 0/1/2, so every type
appears for every variable across regimes. Validation SSH is
point/smooth/point. Blind U/V/Theta use the same constraints with multi-level
type `t=(r+o_h+1) mod 3`, disjoint centres, and the unseen combinations;
blind SSH is smooth/point/smooth. Thus single-versus-multi diagnostics are
populated at every long lead. Among
assignments satisfying those constraints, choose the one with maximum minimum
physical centre separation and break ties by the section-9 SHA order.
Validation and test assignments are solved jointly across regimes so each
input group has at least one long direction in all five regions. Failure to
find an assignment stops inventory construction. Membership cannot depend
on response magnitude or model performance.

---

## 8. Exact U/V/Theta/SSH perturbation definitions

### 8.1 Common horizontal kernel

For smooth directions, use a radius-2 isotropic Gaussian on the **native field
grid**:

$$
K(a,b)=\exp[-(a^2+b^2)/2],\qquad a,b\in\{-2,-1,0,1,2\},
$$

then normalize $K$ to unit discrete L2 norm. Thus `sigma_h=1 cell`, the native
support is exactly 5 by 5, and all 25 weights are positive. A centre is eligible
only if all 25 target-grid cells/faces are active at every perturbed level.
There is no clipping, land renormalization, or displaced centroid.

The direction is then scaled so its realized perturbation in the new
pointwise-normalized FNO input has unit RMS over its nonzero support. The pilot
coefficient `alpha_g` is the RMS standardized amplitude, not an arbitrary
physical-unit constant. Every manifest records the resulting physical peak,
RMS, and L2 magnitude.

### 8.2 U

- Native field edited: `Uvel` on the MITgcm W-face grid.
- Horizontal support: one 5 by 5 Gaussian, 25 active W faces per perturbed
  level.
- Primary training vertical support: exactly one of 15 levels.
- FNO input: apply the repository's exact centering
  `U_c(i)=0.5[U_face(i)+U_face(i+1)]` after the edit.
- Realized one-level FNO support: at most, and for the full-support eligible
  inventory exactly, 5 rows by 6 columns = 30 centred U cells.
- No V, Theta, SSH, salt, or history record is edited.

### 8.3 V

- Native field edited: `Vvel` on the MITgcm S-face grid.
- Horizontal support: one 5 by 5 Gaussian, 25 active S faces per perturbed
  level.
- Primary training vertical support: exactly one of 15 levels.
- FNO input: apply `V_c(j)=0.5[V_face(j)+V_face(j+1)]`.
- Realized one-level FNO support: exactly 6 rows by 5 columns = 30 centred V
  cells for eligible full-support patches.
- Every other pickup record remains unchanged.

### 8.4 Theta

- Native field edited: `Theta` at tracer centres.
- Horizontal support: one 5 by 5 Gaussian, exactly 25 wet tracer cells.
- Primary training vertical support: exactly one of 15 levels.
- Realized FNO support: exactly the same 25 cells in one Theta channel.
- Salt and `GtNm1` remain byte-identical; this is a partial derivative with
  unresolved restart histories held fixed.

### 8.5 SSH

Two equally represented training families are required:

1. **Point:** add the perturbation at one wet tracer cell; one affected pickup
   cell and one affected FNO SSH cell.
2. **Smooth:** the same unit-L2 5 by 5, sigma-1 Gaussian; exactly 25 wet tracer
   cells.

The pickup edit changes `EtaN` only. `EtaH` and `dEtaHdt` stay byte-identical.
This is consistent with the corrected adjoint contract: at startup with
`implicDiv2Dflow=1`, MITgcm recomputes `dEtaHdt` and sets `etaH=etaN`. The old
EtaN-plus-EtaH wording is not authoritative.

The pilot must select one common `alpha_SSH` that passes for both point and
smooth families. The maximum physical SSH magnitude is capped at 0.01 m.

### 8.6 Held-out vertical combinations

All training U/V/Theta directions are single-level, and every level appears at
least three times **per variable and regime** among the 56 directions. Order
the four direction slots within each of the 14 anchors and define
`q=4*anchor_slot+direction_slot`, `q=0,...,55`. With regime index
`r={S0:0,S1:1,S2:2}` and variable offsets `{U:0,V:5,Theta:10}`, assign

$$
\operatorname{level}(q,r,h)=1+[(q+5r+o_h)\bmod 15].
$$

This gives every level three occurrences and exactly 11 levels a fourth,
rotated across variable and regime. Direction-slot order is part of the frozen
inventory table, not filesystem enumeration order.

Each validation variable/regime has 18 directions:

- 15 single-level directions, exactly one at every level;
- one upper two-level direction on levels 1-2 with weights
  `[1,1]/sqrt(2)`;
- one mid-depth three-level Gaussian on levels 7-9 with weights
  `[exp(-1/2),1,exp(-1/2)]/sqrt(1+2 exp(-1))`;
- one deep two-level direction on levels 14-15 with weights
  `[1,1]/sqrt(2)`.

The blind test again has 15 single-level directions plus three unseen
combinations: levels 2-3, a Gaussian on levels 6-8, and levels 13-14. Horizontal
support is repeated on each involved level. Its two-level and Gaussian weights
are the same exact vectors above. A two-level U/V patch therefore
edits 50 native faces and maps to 60 centred cells; a three-level patch edits
75 faces and maps to 90 centred cells. Theta uses 50 or 75 tracer cells.

These combinations test linear superposition and vertical generalization.
Vertical weights multiply the native physical horizontal patch before the one
global standardized-RMS scaling in section 10; they are not reweighted level
by level after normalization. In validation, single level $l$ is assigned to
anchor index `[(l-1)+2r+o_h] mod 3`, and the upper/middle/deep combinations to
index `(t+r+o_h) mod 3` for type index `t={upper:0,middle:1,deep:2}`.
Blind-test assignments add one modulo three
to both formulas and use the stated unseen level combinations. Thus each
anchor has five singles and one combination per variable. The combinations
are never used to tune amplitude; their validation response must still pass
the forward numerical/SNR checks before model selection.

---

## 9. Spatial and vertical sampling plan

### 9.1 Exact region masks

The region of a direction is the region of its kernel centre. Masks are built
from the immutable wet/native masks in this precedence order:

1. **WBC:** the first four wet tracer cells east of the western wall in each
   wet row, reusing `src/oceanfno/dataset.py::western_boundary_mask`.
2. **Eastern:** the last four wet tracer cells west of the eastern wall,
   excluding WBC.
3. **Southern:** the ten southernmost wet tracer rows, excluding WBC/eastern.
4. **Northern:** the ten northernmost wet tracer rows, excluding WBC/eastern.
5. **Interior:** remaining eligible centres.

For Theta/SSH the centre is tracer index `(j,i)`. Under MITgcm's C-grid
convention, U face `(j,i)` is assigned to its carrier tracer cell immediately
east, `(j,i)`, and V face `(j,i)` to its carrier tracer cell immediately north,
also `(j,i)`; a face whose carrier is inactive at any requested level is not
eligible. The carrier tracer mask supplies the region label, while full patch
eligibility is checked on `hFacW`/`hFacS`. For smooth Theta/SSH, eligibility is
the erosion of the wet tracer mask by two cells. Inventory tests must verify
this face/carrier convention against the repository's centering equations.

### 9.2 Exact regional counts

Per **training** regime and per U/V/Theta family (56 directions):

| Region | Count | Fraction |
| --- | ---: | ---: |
| WBC | 20 | 35.7% |
| interior | 9 | 16.1% |
| eastern | 9 | 16.1% |
| northern | 9 | 16.1% |
| southern | 9 | 16.1% |

For each training SSH kernel separately (28 directions): WBC 8 and each of the
other four regions 5. The combined SSH family therefore has WBC 16 and 10 in
each other region.

Per **validation/test** regime and U/V/Theta family (18 directions): WBC 6 and
three in every other region. For nine point SSH directions use
`[WBC,interior,east,north,south]=[3,2,2,1,1]`; for nine smooth directions use
`[3,1,1,2,2]`. Combined SSH counts again equal `[6,3,3,3,3]`.

WBC is therefore sampled at roughly one third of directions even though the
four-column band occupies only about 6.7% of wet tracer cells. It is explicitly
oversampled, while every other boundary and the interior retain coverage.

### 9.3 Deterministic centre selection

**Proposed algorithm.** Before any response job is submitted:

1. Enumerate eligible native-grid centres for every family, level support, and
   region.
2. Divide WBC/eastern candidates into four latitude quartiles and
   northern/southern candidates into four longitude quartiles; divide interior
   into four quadrants.
3. Allocate pilot/train/validation/blind-test centre sets **jointly** within
   every `(regime,family,region)` stratum, including all role quotas before
   exposing any response. Centre IDs are constrained to be distinct across
   roles. Lexicographically maximize (i) minimum cross-role great-circle
   separation, (ii) minimum within-role separation, and (iii) summed WBC
   training-only mean surface speed or, outside WBC, proximity to the four
   subregion centroids. Physical distance uses `R=6371 km`, `(XC,YC)` for
   tracer centres, `(XG,YC)` for U faces, and `(XC,YG)` for V faces. Resolve
   every remaining tie by SHA-256 of
   `response-v1|split|regime|family|level-support|region|j|i`.
4. In role order `pilot < train < validation < blind`, require non-WBC
   validation and blind centres to have
   native-index Chebyshev distance at least three from every centre assigned
   to an earlier role; a joint solution that cannot meet that constraint
   fails. WBC is a predeclared capacity-limited exception: its centre IDs must
   be distinct and its joint maximin optimum is reported, but it has no
   hard distance-three threshold.
5. Assign levels by the explicit section-8 formulas, regions by the exact
   section-9.2 quotas, and long membership by the constrained section-7.5
   solve. A centre ID is
   `(regime,family,native_grid,j,i)`; coordinates may repeat across regimes but
   never across pilot/train/validation/test for the same regime/family.
6. Reserve disjoint centre inventories for validation and blind test before
   either is run. The blind list is written to an evaluator-only manifest and
   is not exposed to the training loader.

The five tracer cells used by the existing Phase-A point/kernel target
(`i=1`, zero-based `j=14..18`) are excluded as perturbation centres and as SSH
supports in all response splits; projected U/V/Theta footprints that intersect
those tracer cells are excluded as well. This uses the already frozen objective
location, not any adjoint output, and prevents direct training on the final
target stencil.

**Verified capacity warning.** A direct audit of the current masks finds only
56 full-support U-WBC carrier centres per regime and 47 after the Phase-A
footprint exclusion. The requested production roles consume 20 training + 6
validation + 6 blind centres, plus one pilot centre in the affected S0/S2
strata. This is enough for split-disjoint IDs but not for a global hard
distance-three rule, which is why the WBC exception above is explicit rather
than silently relaxed after seeing data.

**Unresolved until inventory materialization.** No trusted non-WBC sampler or
joint allocator currently exists. The concrete `(j,i)` list and achieved WBC
separations must be generated from this rule, reviewed for counts/full support,
frozen, and hashed before runs. Failure to meet the exact counts, distinct-ID
rule, or non-WBC distance rule is a stop, not permission to clip a kernel.

---

## 10. Perturbation-amplitude calibration protocol

No localized amplitude has been validated in the repository. The global S0
twin amplitudes do not establish a local U/V/Theta/SSH amplitude and will not
be reused as justification.

### 10.1 Standardized direction and candidate magnitudes

For direction $q$, form its native physical kernel and exact centred FNO
projection $v_q$. Scale it so

$$
\left[
\frac{1}{|\operatorname{supp}v_q|}
\sum_{c,j,i\in\operatorname{supp}v_q}
\left(\frac{v_{q,cji}}{\sigma_{c,ji}^{\rm train}}\right)^2
\right]^{1/2}=1.
$$

Test `alpha` in exactly `{0.025, 0.05, 0.10}` separately for U, V, Theta,
and SSH. The physical edit is `delta_x=alpha v_q`. SSH candidates must also
satisfy `max|delta Eta| <= 0.01 m`; a cap-triggered direction is recorded as a
failure of that alpha rather than silently assigned a smaller amplitude.

### 10.2 Precision and noise floor

- Source and endpoint pickups are big-endian float64.
- Nominal and perturbed branches write full pickups every 10 days.
- MITgcm responses are differenced after float64 native-grid extraction and
  face-to-centre projection.
- The FNO initial state uses the repository-canonical P32 projection: cast
  native fields to float32, then centre U/V. Exact nominal, plus, and minus
  P32 initial states are stored. Each realized signed P32 perturbation must be
  within 1% of the intended standardized RMS, and the relative antisymmetry
  error `||delta_P32^+ + delta_P32^-|| / mean(||delta_P32^+||,
  ||delta_P32^-||)` must be <=1%.
- Duplicate nominal pilot branches measure run-to-run noise at all nine leads.
  It is never assumed to be zero.

Define the group-balanced normalized norm

$$
\|z\|_{GB}^2=\frac14\sum_{g\in\{U,V,\Theta,SSH\}}
\operatorname{mean}_{c\in g,\,wet} z_c^2.
$$

Keep raw-response and differentiated-response units explicit. For each pilot
direction and lead, define the **oriented raw normalized responses**

$$
R_{q,k}^{s}=\frac{\mathcal N[M(x+s\alpha v)]-
\mathcal N[M(x)]}{s},\qquad s\in\{-1,+1\}.
$$

Thus $R^+$ is plus minus nominal, $R^-$ is nominal minus minus, both have raw
normalized-state units, and the derivative-like response used in section 4 is
$r^s=R^s/\alpha$. For any output slice $A$ (an individual physical group and
region, or the GB aggregate), the preliminary raw floor is

$$
n^{\rm raw}_{\rm nom,A,k}=\max_a\left\{
\|\mathcal N(M_{a,k}^{(1)})-\mathcal N(M_{a,k}^{(2)})\|_A,
8\epsilon_{64}\max[1,\|\mathcal N(M_{a,k}^{(1)})\|_A,
\|\mathcal N(M_{a,k}^{(2)})\|_A]
\right\},
$$

where `(1),(2)` are duplicate nominal branches and the maximum is over the six
pilot anchors. Candidate screening initially uses this raw floor; section 10.3
replaces it with the combined raw floor.

The sign-symmetry/nonlinearity score is

$$
Q_{\rm lin}=\frac{\|R^+-R^-\|_{GB}}
{0.5(\|R^+\|_{GB}+\|R^-\|_{GB})},
$$

and signal-to-noise is

$$
Q_{\rm SNR}=\frac{0.5(\|R^+\|_{GB}+\|R^-\|_{GB})}
{n^{\rm raw}_{h(q),GB,k}},
$$

where the denominator is initially $n^{\rm raw}_{\rm nom,GB,k}$ and is replaced
by the final input-family-specific combined raw floor after section-10.3
controls.

For adjacent-amplitude convergence use the centred estimate
$\widehat J_\alpha=[\mathcal N M(x+\alpha v)-\mathcal N
M(x-\alpha v)]/(2\alpha)$ and score
$\|\widehat J_\alpha-\widehat J_{\alpha/2}\|_{GB}/
\max(\|\widehat J_{\alpha/2}\|_{GB},
n^{\rm diff}_{h(q),GB,k}(\alpha/2))$, where
$n^{\rm diff}(\alpha)=n^{\rm raw}/\alpha$. Raw and differentiated floors are
never pooled in one maximum. Before the section-10.3 controls,
$n^{\rm diff}(\alpha)$ uses $n^{\rm raw}_{\rm nom}/\alpha$; the final check is
recomputed with the combined floor.

### 10.3 Solver-tolerance and perturbed-repeat controls

The main pilot first identifies a **provisional** group amplitude using the
candidate grid and nominal-repeat floor. Then use the already preassigned 12
long pilot directions—exactly one per input group and regime—at that
provisional amplitude for two additional forward-only controls:

1. duplicate both signed perturbed branches for 90 days at the production
   `cg2dTargetResidual=1e-7` (24 runs, 2,160 model-days);
2. rerun both signs at `cg2dTargetResidual=1e-10` (24 runs, 2,160 days) and
   rerun the six corresponding nominal anchors at `1e-10` (540 days).

For control condition $c\in\{\text{production},\text{duplicate},
\text{tight}\}$, construct $R_c^s$ from that condition's perturbed endpoint
and its own matched nominal endpoint. At every lead and sign define, with the
same normalized group-balanced norm,

$$
Q_{\rm repeat}=\frac{\|R_{\rm prod}^s-R_{\rm duplicate}^s\|_{GB}}
{\max[0.5(\|R_{\rm prod}^s\|_{GB}+\|R_{\rm duplicate}^s\|_{GB}),
n^{\rm raw}_{\rm nom,GB,k}]},
$$

and define $Q_{\rm CG}$ identically with $R_{\rm tight}^s$ in place of
$R_{\rm duplicate}^s$. Both must be <=0.01 for every controlled input group,
regime, sign, and lead. For each input family $h$, output slice $A$, and lead,
freeze

$$
n^{\rm raw}_{h,A,k}=\max\left\{
n^{\rm raw}_{\rm nom,A,k},
\|R_{\rm prod}^s-R_{\rm duplicate}^s\|_A,
\|R_{\rm prod}^s-R_{\rm tight}^s\|_A
\right\},
\qquad
n^{\rm diff}_{h,A,k}(\alpha_h)=
\frac{n^{\rm raw}_{h,A,k}}{\alpha_h},
$$

taking the maximum over the controlled directions, regimes, and signs; the
nominal term already includes the float64 rounding bound. `Q_lin` and
`Q_SNR` use raw $R$ and $n^{\rm raw}$; JVP targets, response loss, and response
validation use $r$ and $n^{\rm diff}$. Recompute final SNR with the combined raw
floor and require it to remain >=20. A failure stops v1; it cannot be fixed by
dropping a case or consulting an adjoint.

### 10.4 Frozen selection rule

For each input group independently, choose the **largest** candidate alpha for
which all of the following hold across the group's pilot cases:

1. `Q_lin <= 0.05` at day 10 for every direction;
2. `Q_lin <= 0.05` at every 10-day lead through day 90 for every long pilot
   direction;
3. `Q_SNR >= 20` for every scored direction/lead;
4. both realized P32 input perturbations pass the 1% magnitude and 1%
   antisymmetry checks and have a nonzero centre weight;
5. the SSH 1 cm cap holds for every point and smooth SSH pilot/planned
   production direction;
6. the centred JVP estimate at the selected alpha differs by <=5% from the
   next smaller passing alpha wherever such an alpha exists.
7. all selected-alpha solver-tolerance and perturbed-repeat controls in
   section 10.3 pass.

If no candidate passes for a group, production response generation stops. A
new, separately versioned pilot may add a smaller amplitude, but the failed
pilot and revised rule must be retained. No adjoint output may inform that
revision.

The frozen amplitude manifest records one selected `alpha_U`, `alpha_V`,
`alpha_Theta`, and `alpha_SSH`, plus every direction's actual physical peak and
RMS. There is no one-size physical amplitude invented in advance.

---

## 11. Short-versus-long response counts and integration cost

### 11.1 Direction and run inventory

| Role | Directions/regime | Directions total | Long directions total | Signed perturbed runs | Long signed runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 224 | 672 | 96 | 1,344 | 192 |
| validation | 72 | 216 | 36 | 432 | 72 |
| blind response test | 72 | 216 | 36 | 432 | 72 |
| pilot, base | 8 | 24 | 12 base | 144 across 3 alphas | 72 |

Every long run also supplies its 10-day response. The train set therefore has
672 distinct 10-day directions and 96 sparse long trajectories; validation and
blind test each have 216/36. Training/validation long trajectories contain six
leads; pilot/blind long trajectories contain nine.

### 11.2 Exact model-day cost

For a direction pair, a short direction costs `2 signs x 10 = 20` model-days.
A training/validation long pair costs `2 x 60 = 120`; a pilot/blind long pair
costs `2 x 90 = 180`.

| Component | Short signed runs | Long signed runs | Model-days | Model-years |
| --- | ---: | ---: | ---: | ---: |
| production train perturbations | 1,152 | 192 at 60 d | 23,040 | 64.0 |
| response validation perturbations | 360 | 72 at 60 d | 7,920 | 22.0 |
| blind response-test perturbations | 360 | 72 | 10,080 | 28.0 |
| 3-amplitude pilot perturbations | 72 | 72 | 7,200 | 20.0 |
| shared nominal branches | 18 controls | 42 controls, mixed 60/90 d | 3,150 | 8.75 |
| validation pickup-bank setup | -- | 3 x 320 d | 960 | 2.667 |
| duplicate 90-day pilot controls | -- | 6 controls | 540 | 1.5 |
| selected-alpha perturbed duplicates | -- | 24 | 2,160 | 6.0 |
| tight-CG selected-alpha perturbations | -- | 24 | 2,160 | 6.0 |
| tight-CG nominal controls | -- | 6 controls | 540 | 1.5 |
| **Total** | **1,944 perturbed + 18 controls** | **456 perturbed + 54 controls + 3 setup chains** | **57,750** | **160.417** |

Nominal costs are one shared control per anchor, not one per perturbation:

- training: six pilot-overlap controls at 90 days, 18 other long controls at
  60 days, and 18 short controls at 10 days = 1,800 days;
- validation: nine 60-day controls = 540 days;
- blind test: nine 90-day controls = 810 days.

The pilot reuses the training-anchor nominal branches. Annual pilot/train/blind
source pickups already exist. The 960-day off-cycle validation pickup bank is
included explicitly. If any annual pickup, generated validation pickup, or
forcing hash is missing, that anchor stops; it is not reconstructed from Zarr.

The complete plan launches 2,400 perturbed branches, 72 nominal/control
branches, and three validation pickup-bank chains: **2,475 logical branches**.
At 10-day segment equivalence this is 5,775 integrations and 4,158,000 MITgcm
timesteps. The 5,400 model-days from duplicate nominal,
selected-amplitude perturbed-repeat, and tight-CG branches are calibration
controls, not new training directions.

---

## 12. MITgcm restart/pickup implementation plan

### 12.1 Verified pickup layout

The actual pickup is 62 by 62, big-endian float64, 108 records, about 3.17 MiB:

| Field | Records |
| --- | ---: |
| `Uvel` | 15 |
| `Vvel` | 15 |
| `Theta` | 15 |
| `Salt` | 15 |
| `GuNm1` | 15 |
| `GvNm1` | 15 |
| `GtNm1` | 15 |
| `EtaN` | 1 |
| `dEtaHdt` | 1 |
| `EtaH` | 1 |

The trusted layout parser/editor/verification logic is in
`archive/src/bire_repro/af_s0_twin.py`. It already checks metadata, endian,
field order, record count, finite values, byte-identical untouched records,
and input/output hashes. It will be generalized, not copied into a parallel
implementation.

### 12.2 Exactly what changes

| Perturbation | Modified pickup records | Byte-identical records |
| --- | --- | --- |
| U | selected cells in selected `Uvel` level(s) | all other U; all V, Theta, Salt, Gu/Gv/Gt, EtaN, dEtaHdt, EtaH |
| V | selected cells in selected `Vvel` level(s) | all other V; all U, Theta, Salt, Gu/Gv/Gt, EtaN, dEtaHdt, EtaH |
| Theta | selected cells in selected `Theta` level(s) | all other Theta; all U, V, Salt, Gu/Gv/Gt, EtaN, dEtaHdt, EtaH |
| SSH | selected cells in `EtaN` | U, V, Theta, Salt, Gu/Gv/Gt, dEtaHdt, EtaH |

Holding Adams-Bashforth/history and unrepresented restart variables fixed is a
declared conditional derivative. It is the forward analogue of perturbing one
resolved state component while holding the others fixed. No restart is ever
reconstructed from the FNO state alone.

### 12.3 Regime forcing

Each run stages the exact source pickup plus the source regime's own
`windx_cosy.bin`, bathymetry, SST relaxation target, `data.pkg`, and physics
namelist. S1/S2 never borrow S0 forcing. The MITgcm commit, forward executable,
MPI decomposition (2 by 2, four ranks), convection, and all physical settings
remain identical to the nominal chain. Production, ordinary duplicate, and
perturbed-repeat branches use `cg2dTargetResidual=1e-7`. The sole exception is
the hashed section-10.3 solver-sensitivity condition, whose matched perturbed
and nominal branches use `1e-10`; no production response target uses that
tighter condition.

### 12.4 Output and restart gates

- Run training/validation long branches (including their nominal controls) as
  six validated 10-day segments, and pilot, blind, and calibration-only
  duplicate/tight-control long branches as nine, using
  `pChkptFreq=864000 s`, and archive/hash the endpoint before launching the
  next segment. MITgcm's rotating `pickup.ckptA/B` names are not assumed to
  retain all endpoints. A one-job wrapper may chain the segments, but
  each segment must restart from the just-verified preceding pickup.
  Apply the perturbation only to the original anchor pickup; later segments
  restart from the complete evolved pickup, including its evolved history
  records, with no repeated edit.
- Build each off-cycle validation source chain from the untouched day-5,760
  pickup as 32 validated 10-day segments through day 6,080. It uses production
  tolerances, no edit, and the same full-pickup/P32/hash gates. Archive all
  endpoints but designate only days 6,010/6,050/6,080 as response anchors.
- Set `dumpFreq=0`, `taveFreq=0`, and disable all gridded diagnostics output in
  the response-run `data.diagnostics`; retain stdout/scalar monitors and the
  declared 10-day pickups only. Diagnostics are passive, but their daily
  float32 files are neither response truth nor part of the storage budget.
- Extract U/V/Theta/EtaN from these pickups; do not subtract daily float32
  diagnostics or Zarr states.
- Canonical P32 gate: casting pickup faces to float32 before the trusted
  centering operator must reproduce the Zarr anchor state bit-for-bit.
- P64 response path: centre float64 faces first, subtract the paired nominal
  endpoint, and store float64 response arrays.
- For every edited pickup, assert metadata byte identity, exact target support,
  exact equality of all untouched records, sign reversal, amplitude, and SHA.
- A plus and minus case must cite the same source/nominal branch and differ
  only in the sign of the declared edit.

### 12.5 Per-run manifest

Every run manifest must contain at least:

- study/contract version and role
  (`pilot/train/validation/blind_test/validation_pickup_bank`);
- source pickup `.meta`/`.data` paths and SHA-256;
- regime, trajectory day, iteration, source-chain segment;
- perturbed variable, native grid, one-based level list;
- centre `(j,i)`, geographic coordinates, region;
- kernel name, sigma/radius, normalization, native and centred support counts;
- sign, candidate/selected alpha, physical peak/RMS/L2;
- horizon and output leads;
- control condition (`production`, `nominal_duplicate`,
  `perturbed_duplicate`, or `tight_cg`) and the exact
  `cg2dTargetResidual`;
- hashes of the edited pickup, nominal branch, endpoint pickups, executable,
  MITgcm commit, forcing files, data namelist, code/config, and Slurm script;
- job ID, ranks, host, start/end times, exit status, and validation gates.

Completed run directories are immutable. A directory without a complete
manifest is quarantined rather than reused.

---

## 13. Proposed response dataset schema

**Proposed development root:**
`/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/forward_response_v1.zarr`,
containing only `pilot`, `train`, and `validation`. The blind test is a separate
evaluator-only store,
`/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/forward_response_blind_v1.zarr`,
created after model freeze. Project-side digest manifests live under
`outputs/af_fno/response/forward_response_v1/` and
`outputs/af_fno/response/forward_response_blind_v1/`.
`trajectories_v3.zarr` is never modified.

The production-role dimensions are frozen as:

| Role | `A` | `A_short` | `A_long` | `Q_short` | `Q_long` |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 42 | 18 | 24 | 576 | 96 |
| validation | 9 | 0 | 9 | 180 | 36 |
| blind test | 9 | 0 | 9 | 180 | 36 |

Long directions are not duplicated in `short`; their day-10 response is the
first element of `long/response_p64`. Let `L=6` for train/validation and `L=9`
for blind test. Each role contains:

| Array/table | Shape/content | Dtype |
| --- | --- | --- |
| `anchors/state_p32` | `(A,46,62,62)` exact FNO input projection | float32 |
| `anchors/state_p64` | `(A,46,62,62)` float64 pickup projection | float64 |
| `anchors/nominal_short` | `(A_short,1,46,62,62)` | float64 |
| `anchors/nominal_short_anchor_row` | `(A_short,)` rows into `anchor_table` | int32 |
| `anchors/nominal_long` | `(A_long,L,46,62,62)` | float64 |
| `anchors/nominal_long_anchor_row` | `(A_long,)` rows into `anchor_table` | int32 |
| `short/anchor_row`, `short/direction_row` | each `(Q_short,)` | int32 |
| `short/input_state_p32` | `(Q_short,2,46,62,62)` exact plus/minus pickup projections | float32 |
| `short/response_p64` | `(Q_short,2,1,46,62,62)` signs x day 10 | float64 |
| `long/anchor_row`, `long/direction_row` | each `(Q_long,)` | int32 |
| `long/input_state_p32` | `(Q_long,2,46,62,62)` exact plus/minus pickup projections | float32 |
| `long/response_p64` | `(Q_long,2,L,46,62,62)` | float64 |
| `lead_days` | train/validation `[10,...,60]`; blind `[10,...,90]` | int16 |
| `anchor_table.jsonl` | one authoritative row per anchor | canonical JSON Lines |
| `direction_table.jsonl` | one row per direction; array row mappings above point here | canonical JSON Lines |

`anchor_table` columns are `anchor_id`, role, regime, day, iteration, horizon,
canonical source segment/path/hash, all duplicate-source hashes, forcing/static
hashes, and nominal endpoint hashes. `direction_table` columns are
`direction_id`, array group/row, anchor row, input family/native field,
one-based level list, exact vertical weights, native `(j,i)`, lon/lat, region,
kernel/sigma/radius, native/centred support counts, alpha, unit-direction
normalization, physical peak/RMS/L2, both P32 realized magnitudes,
antisymmetry, long flag, sparse edit lists `(record,j,i,value)`, and all input/
response hashes. IDs are UTF-8 strings of the frozen manifest grammar; levels,
weights, hashes, and sparse edits are typed JSON arrays validated against
`config/forward_response_schema_v1.json`. Each line uses sorted keys and compact
separators, and both line-order and file SHA-256 are frozen. This avoids adding
a Parquet engine that the current environment does not provide.

The sign axis is fixed as index 0 = `s=-1`, index 1 = `s=+1`. Stored response
arrays are raw float64 differences

$$
\Delta_{q,s,k}^{64}=P_{64}[M_k(x_q^s)]-P_{64}[M_k(x)],
$$

so the minus record is normally negative. The loader alone forms the oriented
target
$r_{M,q,k}^s=(\Delta_{q,s,k}^{64}/\sigma_{parent})/(s\alpha_q)$.
It never guesses sign from a filename. The scale is the exact production
parent pointwise scale from section 6, never a response-derived input
normalizer.

The pilot uses the same split short/long layout with 36 short and 36 long
`(base_direction,alpha)` rows, an alpha column, and a separate
`solver_control_table.jsonl`. It additionally stores
`controls/perturbed_response_p64` with shape
`(12,2_conditions,2_signs,9,46,62,62)` for production duplicates and tight-CG
branches, `controls/duplicate_nominal_p64` with shape
`(6,9,46,62,62)`, and `controls/tight_nominal_p64` with the same shape. The
solver-control table maps every array row to its source base direction/anchor,
condition, matched nominal row, exact CG tolerance, and run/endpoint hashes.
All curated numeric arrays use canonical
little-endian `<f4`/`<f8`. Response chunks are one
`(direction,sign,lead,46,62,62)` field, input chunks one
`(direction,sign,46,62,62)` field, and Blosc-Zstd level 3 with bitshuffle is
frozen for v1. Array-level and chunk-manifest SHA-256 digests are mandatory.

Loader reconstruction must hash-match both signed input states and all sparse
edits. Development loaders have no filesystem permission or config route to
the blind store; tests assert both. Every model's external response metrics
use the exact parent normalization. A, ft90, B, and C therefore share one
physical-to-normalized coordinate system.

**Estimated combined development-plus-blind curated size:** about 9-11 GB
uncompressed numeric arrays plus metadata; expected compressed size is
unresolved until pilot compression is measured.

---

## 14. Full objective function and response-weight selection

### 14.1 Parent objective stays intact

For every optimizer update, B and C evaluate the unchanged eight-term parent
objective on an effective batch of eight nominal 60-day sequences:

$$
L_{\rm total}=L_{\rm parent,nominal}
+\lambda_{\rm resp}I_{\rm joint}L_{\rm response}.
$$

No parent coefficient changes. No adjoint channel, contraction penalty, tanh
stabilizer, spectral-cap change, or extra state variable is introduced.

### 14.2 Response normalization and group balance

Let input family $h(q)\in\{U,V,\Theta,SSH\}$ and output group
$g\in\{U,V,\Theta,SSH\}$. Compute, from **response-training data only**, an RMS
scale

$$
d_{h,g,k}^2=
\operatorname{mean}_{q:h(q)=h,\,s}
\operatorname{mean}_{c\in g,wet}
\left(r_{M,q,k}^{s}\right)^2.
$$

Floor it at ten times the corresponding differentiated combined
rounding/repeat/CG-disagreement scale from section 10. The pilot SNR gate
should make the floor inactive for the directly excited signal; the floor
prevents a nearly zero cross-group response from producing an infinite loss.

For one response direction,

$$
\ell_{q,k}=\frac{1}{8}
\sum_{s\in\{-1,+1\}}
\sum_{g\in\{U,V,\Theta,SSH\}}
\frac{
\operatorname{mean}_{c\in g,wet}
\left(r_{F,q,k}^{s}-r_{M,q,k}^{s}\right)^2
}{d_{h(q),g,k}^2}.
$$

This gives each output physical group equal weight regardless of units or its
15-versus-1 channel count. The response sampler gives each input family equal
probability, so all 16 input-output blocks receive equal status in expectation.

Short samples use

$$
L_{\rm response}^{\rm short}=\ell_{q,10},
$$

and long samples use an unweighted lead mean

$$
L_{\rm response}^{\rm long}
=\frac{1}{6}\sum_{k\in\{10,20,\ldots,60\}}\ell_{q,k}.
$$

The signed formulation compares perturbation **responses** to responses. It is
not equivalent to fitting the perturbed state alone. No response target beyond
day 60 enters a gradient.

### 14.3 No ordinary state loss on perturbed trajectories

**Frozen choice: no.** C does not add ordinary state loss on the plus or minus
trajectory. The unchanged nominal parent loss anchors the common forward map,
while the signed response-difference term is the single scientific
modification. Perturbed absolute targets may be reconstructed for integrity
audits but are never passed to a state-loss function.

A perturbed-state-only arm, short-only arm, continuation, or additional
physics/Jacobian regularizer would answer a different multi-change question.
None is part of v1.


### 14.4 Response-loss-weight selection

**Proposed candidates:**
`lambda_resp in {0.03, 0.10, 0.30, 1.00}`.

Use primary seed `20260724` and identical initialization/batch order to run a 1,920-
step forward-only screen. For each candidate:

1. evaluate exact parent nominal validation and held response validation at
   leads 10-60 only;
2. reject it if any 10-90-day primary-field AUC is >1.05 times the matched
   lambda-zero control at the same step;
3. reject it if growth is >0.005 per call worse than that control or any
   rollout is nonfinite;
4. among remaining candidates, minimize the group/region/lead-balanced
   response relative-L2 composite;
5. candidates within 2% of the minimum tie in favour of smaller lambda.

Freeze the chosen lambda before full three-seed training. If no candidate is
forward-feasible, stop. No MITgcm adjoint, TAF file, FNO adjoint map, blind
response case, or test metric may be read during this screen.
Screen checkpoints and optimizer states are discarded. The full primary-seed
C run restarts from its original random initialization and step zero; it does
not receive 1,920 extra updates from the screen.

---

## 15. Training protocol

### 15.1 Shared B/C protocol

| Setting | Value |
| --- | --- |
| Initialization | neuraloperator default random initialization; local branch zero |
| Parent/ft90 state load | none |
| Normalizer | exact parent recomputation over days 0-5,999; required SHA `fe424b37...96bf` |
| Nominal rollout | 6 calls / 60 days |
| Teacher forcing | none after initial state |
| Adam | cold start, no optimizer-state load; betas `(0.9,0.95)`, weight decay 0 |
| Learning rate | `5e-4` steps 1-5,760; `1e-4` steps 5,761-7,680 |
| Updates | 7,680 |
| Effective nominal batch | 8 = microbatch 4 x accumulation 2 |
| Gradient clipping | none |
| Checkpoints | steps 1,920, 3,840, 5,760, 7,680 |
| Spectral normalization | exact parent machinery, `rho=1`, materialized checkpoints |
| Seeds | primary `20260724`; paired replications `20260911`, `20260912` |

Both arms use the same parameterized response-study runner. Its disabled path
must pass the section-23 primary-seed equivalence harness against the immutable
parent trainer before B is launched.

B and C see exactly the same nominal batch at every update. C does not replace
nominal samples with response samples.

### 15.2 Auxiliary response mixing

On exactly every fourth optimizer update (`I_joint=1`), C adds one response
direction pair. At every autoregressive lead, concatenate the response
anchor's unperturbed, minus, and plus states on the batch dimension and make
**one** model invocation, then split and advance the three branches.
Sequential sign forwards are forbidden:
the live spectral-normalization layer updates its power vectors on every
training forward, so sequential evaluation would quotient three slightly
different maps. Batched evaluation gives all signs identical normalized
weights at that lead. Thus the **response-update fraction is 25%**, while
nominal exposure remains identical to B.

Within either arm's own weight trajectory, the auxiliary path must leave
persistent spectral-normalization state as if that auxiliary chain had not
occurred. Immediately before an auxiliary chain,
snapshot every spectral left/right power vector; allow the parent's usual two
power iterations on each batched lead; backpropagate the auxiliary loss; then
restore every vector bit-for-bit before the optimizer step, in a `finally`
guard. Weight gradients are retained, estimator-buffer mutations are not. B
needs no dummy data pass; C alone uses this context manager. Tests
must establish branch-order invariance, zero response for zero perturbation,
and bit-identical within-arm pre/post auxiliary buffer hashes. A same-weight,
same-buffer counterfactual clone that enters a zero-auxiliary context must
match a clone that skips it after their next identical nominal pass. Across
real B/C runs, require equal buffer-mutation counts and schedule only: their
buffer values are not expected to match after response gradients make their
weights differ. A failure of the schedule-isolation test is a model-contract
failure.

Within the auxiliary stream:

- repeat the exact pattern short/short/short/long, so 75% of samples are
  short-only and 25% are long;
- input families are exactly balanced in blocks;
- regimes are exactly balanced in blocks;
- levels and regions follow the frozen inventory, with deterministic shuffling.

Over 7,680 updates this gives 1,920 response updates: 1,440 short and 480 long.
The 96 long directions therefore make exactly five complete hash-permuted
passes. The 576 short directions make two complete passes plus one frozen
288-direction balanced half-pass: exactly 72 cases per input family and 24 per
family/regime, with region/depth quotas chosen as close as integer arithmetic
allows to the full short inventory and tie-broken by the direction hash. There
is no replacement *within* a pass; reuse occurs only through these declared
complete/partial passes. The optimizer-step-to-direction-ID schedule is hashed
before training.
Because a long loss averages six leads, exactly 19/24 = 79.17% of total
response lead weight falls at day 10 and 1/24 = 4.17% at each of days 20-60.
The study therefore remains
primarily a direct constraint on the fundamental 10-day map.

A short pair adds one batch-three invocation/three state transitions; a long
pair adds six invocations/18 state transitions. The expected auxiliary load is
6.75 state transitions on a joint update, or 1.6875 averaged over all updates.
The nominal batch performs 48 state transitions/update, so raw transition
count rises by about 3.5%; batched-triplet and snapshot overhead suggests a
practical wall-time increase of roughly 8-20%, to be measured.

### 15.3 Maximum target horizon

The primary nominal and response-gradient target horizons are both exactly 60
days. The forward-only amplitude pilot may integrate to 90 days to certify the
blind perturbations' local regime, and the blind response/adjoint tests evaluate
day 90, but no FNO training loss, lambda choice, checkpoint choice, or
development response score reads a model response beyond day 60. Rollouts to
360/2,000 days are the unchanged parent forward diagnostics/evaluation only;
they provide no pointwise training target.

---

## 16. Validation and checkpoint-selection protocol

The study retains the parent's checkpoint selector exactly. Response
validation is an independent development view used to choose
\(\lambda_{\rm resp}\), to test the mechanism, and to report the selected
model; it is **not** spliced into the production checkpoint selector.

### 16.1 View 1: exact parent nominal forward validation

For every saved checkpoint of B and C, call the unchanged
`validate_checkpoint` implementation on the exact 102 production records:
34 starts per regime at 6000 + 6m, m = 0,...,33. Reuse the exact
training-days-0--5,999 climatology and all parent numerics. That function
produces:

- RMSE for surface speed, SST, and surface PHIHYD against MITgcm,
  persistence, and climatology;
- ACC for surface U, surface V, SST, and surface PHIHYD;
- 10--90-day RMSE-AUC;
- 90--360-day RMSE-AUC and its ratio to climatology;
- the implemented 330--360-day per-call gain;
- maximum normalized amplitude and day-360 slow-field biases.

The parent training report separately appends the exact perturbation-growth
diagnostic used by the selector: S0 validation starts 6,000 and 6,198;
0.01-relative random twins with direction seeds 0 and 1; 200 autoregressive
calls; a log-separation fit beginning at zero-based call index 50 (calls
51--200); and the worse of the two fitted per-call growth rates.

Finiteness is an explicit gate. Spatial variance, WBC/interior structure,
long-inference magnitude, and the other established stability/anomaly views
are generated after selection by the unchanged figure/anomaly numerics in
section 19; they are not falsely attributed to `validate_checkpoint` or added
to the selector.

No response anchor replaces or augments the 102 nominal records. No threshold,
metric definition, climatology, growth protocol, or production-validation
start changes.

### 16.2 View 2: held-out forward-response validation

Use only the 216 directions at response-validation anchors 6,010, 6,050, and
6,080. They are separate from response training and the nested final-inference
starts, and their centre IDs are disjoint from pilot, training, and blind-test
centres. Every U/V/Theta level is represented, and the predeclared multi-level
combinations are absent from response training. Only leads
\(k\in\{10,20,\ldots,60\}\) are available to development code. Because
the parent chronology is immutable, these cases and the nominal checkpoint
rollouts occupy the same validation block and may overlap in calendar time;
“independent views” means separate source starts, perturbation locations,
targets, metrics, and decision roles, not a newly invented temporal subsplit.
All 216 cases are scored at day 10; only the predeclared 36-direction long
subset is scored at days 20--60.

At each lead, report by input family, output group, regime, input-centre
region, output region, kernel, and vertical-support class:

- oriented signed-response relative L2;
- wet-cell spatial pattern correlation;
- amplitude ratio \(\|r_F\|/\|r_M\|\) and absolute log amplitude ratio;
- sign agreement;
- MITgcm and FNO plus/minus central-response consistency;
- WBC, interior, eastern, northern, and southern scores;
- point versus smooth SSH;
- single-level versus held-out multi-level directions;
- response norm, calibrated SNR, and lead dependence.

For output region \(\Omega\), define the conventional per-case relative L2

\[
E_{q,k,g,s}(\Omega)=
\left[
\frac{\sum_{c\in g,\Omega}(r_F-r_M)^2}
{\max\{\sum_{c\in g,\Omega}r_M^2,\,
N_{g,\Omega}n_{h(q),g,k}^2\}}
\right]^{1/2},
\]

where \(n_{h,g,k}\) is the differentiated training-only combined numerical
floor and \(N_{g,\Omega}\) is the number of scored wet values. This is distinct
from the training loss's RMS-normalized squared error.

For the development composite, first average \(E\) over signs, cases, and
regimes within each \((h,g,R,k)\) cell, where \(R\) is the input-centre region
and \(\Omega\) is all wet cells. Then define

\[
S_{\rm resp}^{10:60}=
\frac{1}{4\cdot4\cdot5}\sum_{h,g,R}
\left[
\frac{19}{24}\bar E_{h,g,R,10}
+\frac{1}{24}\sum_{k\in\{20,30,40,50,60\}}
\bar E_{h,g,R,k}
\right].
\]

These weights exactly match section 15's expected response-lead exposure.
Every cell must be populated. Output-region scores, pattern correlation, and
amplitude ratio are reported separately rather than hidden inside the
composite. A truth response at or below its calibrated numerical floor has
undefined correlation/amplitude, accompanied by its norm and SNR; it is never
coerced to zero or one. All models are scored in the exact parent external
normalization.

### 16.3 Exact checkpoint selection and forward-preservation gate

For each B and C run independently, pass its four checkpoint summaries to the
unchanged `select_by_validation` function with the parent's default
`tolerance=1.05` and `growth_ceiling=1.0`:

1. retain checkpoints within 5% of that run's best 10--90-day AUC in every
   primary field;
2. among those, retain checkpoints with measured perturbation growth at or
   below 1.0;
3. if any remain, minimize the worst 90--360-day AUC/climatology ratio and
   break ties by earlier optimizer step;
4. if none meets the growth ceiling but the short-feasible pool is nonempty,
   select its least-growing checkpoint, then earlier step;
5. if the short-feasible pool is empty, minimize the worst primary-field
   short-AUC ratio to that run's fieldwise best, then earlier step.

Publish exactly one materialized selected checkpoint per run. Response metrics
do not determine early stopping, eligibility, tie-breaking, or checkpoint
selection. For attribution, also report B/C comparisons at every matched saved
step (1,920, 3,840, 5,760, and 7,680), but do not substitute a favourable fixed
step for the production-selected model.

The section-14 lambda screen is the only development decision that reads
response validation. After lambda is frozen, response validation is a
mechanistic outcome. The primary selected C is considered forward-preserving
only if, relative to paired selected B:

- each primary 10--90-day AUC ratio is at most 1.05;
- the worst 90--360-day AUC/climatology ratio is at most 1.05 times B's;
- perturbation growth is no more than 0.005 per call worse;
- maximum normalized amplitude through 360 days is at most 1.05 times B's;
- all rollouts are finite.

It demonstrates the intended development response effect only if
\(S_{\rm resp}^{10:60}\) is at least 20% lower overall and at least 10% lower
within each input family than paired B, with no day-10 input-family/region
aggregate more than 1.10 times B. These are success/stop-go gates applied
*after* the exact production selector, not a replacement selector.

Failure freezes a negative v1 development result. There is no continuation,
curriculum, checkpoint reselection, new lambda, or relaxed gate. Provided the
technical adjoint gate later passes, the already-frozen blind evaluations still
run so that a negative forward/response tradeoff is measured rather than
hidden.

## 17. Blind forward-response test

The geometry/direction manifest for 216 directions at days 7,560, 7,920, and
8,280 is frozen before training and sealed under an evaluator-only path. No
blind numerical response or numeric store is generated then. Those forward
runs are generated/read only after every B/C checkpoint, lambda, development
report, and ordinary forward report is frozen and hashed. The directions have
disjoint centre IDs and unseen vertical combinations. They are never used for
amplitude calibration, lambda selection, early stopping, checkpoint selection,
architecture choice, or any retry decision.

Evaluate the frozen parent A, ft90 child as context, and all paired B/C seeds on
exactly the same cases. Report every section-16.2 diagnostic for all 216
directions at day 10 and for the predeclared 36-direction long subset at
days 20,30,...,90, with each available time/regime and every WBC/interior,
field, kernel, and vertical-support breakdown retained.

For leads 10--60, report \(S_{\rm resp}^{10:60}\) exactly as defined for
development. Using only the predeclared long cases, also define

\[
S_{\rm resp}^{90}=
\frac{1}{4\cdot4\cdot5}\sum_{h,g,R}\bar E_{h,g,R,90}.
\]

The primary seed is **20260724**. Its C checkpoint must reduce
`S_resp_10:60` by at least 15% and `S_resp_90` by at least 10% versus its
paired B; both scores must also be below frozen parent A. It must improve at
least three of four input families and worsen no input-family/region day-10
aggregate by more than 10% versus B. Across the three paired seeds, the median
10--60-day reduction must be at least 15%, and at least two seeds must improve
both the 10--60 and day-90 scores versus paired B.

This package is opened once. A failure is a negative result; it cannot change
the model, checkpoint, response weight, amplitudes, inventory, or evaluation
rule.

## 18. Blind MITgcm-adjoint test

### 18.1 Seal and timing

**Verified limitation.** Existing Phase-A numbers are readable in the
repository and summarized earlier in this document because the requested audit
required inspecting them. “Blind” can therefore mean a locked external test
not used for any new decision; it cannot honestly mean the investigators have
never seen the historical result.

Before response work begins, freeze hashes and deny all development code access
to:

- `outputs/af_fno/adjoint/mitgcm_s0_adjoint_v1/**`;
- `outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/**`;
- `outputs/af_fno/adjoint/comparison_phase_a_v1/**`;
- `outputs/af_fno/adjoint/fno_*adjoint*/**`;
- `/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1/**`;
- `/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v2/**`;
- every `ADJ*`, `adxx_*`, TAF log, gradient-check result, and derived copy.

Training configs retain `read_contract.adjoint_state=false`. Enforcement is a
lightweight, code-level convention rather than OS-level account/mount
separation: a path/glob scan (`**/ADJ*`, `**/adxx_*`, `**/*TAF*.log`,
`**/*grdchk*`, and the explicit forbidden roots above) runs over every
training/selection script, config, and log actually used for a B/C run, and
the run's provenance report records that the scan passed. Any evaluator-only
artifact (the blind-response store, its geometry manifest) lives under its
own path, is generated only after the freeze conditions below are met, and is
treated as write-once; nothing beyond the scan and the write-once convention
gates access to it. This is a single-researcher project, so the check is
mechanical but not adversarial-proof -- it catches accidental leakage, not a
determined attempt to defeat it.

### 18.2 Existing primary suite

Reuse the scalar-gate-validated MITgcm/TAF products; do not rerun them merely
to change provenance, but resolve the G1 plateau caveat at Gate A0. Run the
trusted FNO-side derivative machinery for:

1. frozen parent A;
2. all paired B and C replicates, with `20260724` primary;
3. the existing ft90 result as context.

Use source day 7,200 and leads 10, 20, 30, and 90 days for:

- point SSH anomaly objective;
- five-point meridional Gaussian SSH anomaly objective;
- analytic area-mean conservation probe;
- truth-forced and free FNO chains;
- existing fixed-source and backward-sweep views.

Report, without reselection:

- relative L2, pattern correlation, amplitude ratio, sign agreement;
- WBC/interior structure, radial decay, and lead dependence;
- conservation-mode error;
- absolute 12-bin spectra with the 4.625-cell FNO cutoff;
- the established unweighted metrics for continuity with Phase A, plus clearly
  labeled area-weighted companions if added;
- FNO finite-difference, forward/reverse-mode, masking, dtype, and hash gates.

The FNO spectral path must use the validated complex128 work-buffer correction;
casting weights to float64 without replacing neuralop's hard-coded complex64
buffer is not an acceptable “double precision” adjoint.

Truth-forced relative L2 is the primary local-Jacobian endpoint. The free-chain
metric is secondary because it includes trajectory-linearization drift.

### 18.3 Primary blind scientific endpoint

For model $m$ and seed $s$, define the truth-forced primary score

$$
S_{m,s}=\frac18\sum_{o\in\{point,kernel\}}
\sum_{k\in\{10,20,30,90\}}
\log[\max(E^{\rm relL2}_{m,s,o,k},10^{-12})].
$$

The two primary effects are
$\Delta_{A,s}=S_{C,s}-S_A$ and
$\Delta_{B,s}=S_{C,s}-S_{B,s}$; negative values improve on the historical
parent and prospective paired nominal control. The primary seed succeeds only
if $\Delta_{B,20260724}\le\log(0.8)$, $\Delta_{A,20260724}<0$, at least six of
the eight objective/lead relative-L2 cells improve versus paired B, and none is
>1.10 times B. Across replication, the median $\Delta_B$ must be
<=`log(0.9)` and at least two of three seeds must have $\Delta_B<0$.

Secondary endpoints are the free-chain score, higher pattern correlation,
amplitude ratio closer to one, and improved mean-
mode preservation. Results are reported per seed; no best seed is chosen after
opening TAF data.

The scientific answer is positive only if the frozen forward gates/tests and
the quantitative adjoint criteria above all pass. A failure is publishable and
closes v1. Any later v2 must use a
new preregistered development cycle and, preferably, new sealed adjoint targets.

### 18.4 Additional final-only exploratory adjoint tests

These may be preregistered before training but generated/opened only after the
primary freeze:

- new runtime-weight SSH objectives at one interior and one eastern target;
- directional projections of raw MITgcm U/V/Theta adjoints onto the exact
  native-grid response kernels, comparing the valid scalars
  $\langle\nabla_{face}J,v_{face}\rangle$ for MITgcm and
  $\langle\nabla_{centre}J,Pv_{face}\rangle$ for the FNO, avoiding an
  unvalidated direct C-grid map;
- balanced/geostrophically consistent U/V/SSH perturbation projections;
- S1/S2 copies of the 10/30/90-day SSH suite with regime-correct forcing.

The existing tape is sized to 200 days. New objective weights do not require a
TAF rebuild, but U/V packaging and face-to-centre adjoint conventions are
**unresolved** and require independent gates before use.

Unless exact target coordinates/direction IDs are materialized and hashed in
the pretraining freeze manifest, these analyses are explicitly exploratory and
cannot rescue or overturn the section-18.3 confirmatory result. A genuinely
new confirmatory S1/S2/interior/eastern suite should be a separately frozen
contract, not chosen after inspecting v1 maps.

---

## 19. Forward figure and anomaly evaluation after training

After the production selector has frozen every B/C checkpoint and before the
adjoint evaluator is enabled:

1. Run the unchanged 360-day nominal validation package on the exact 102
   production-validation records.
2. Run the established final S0 figure package on the exact 15 starts listed in
   section 6.2; do not invent a per-regime or new nominal-test package.
3. Run the standard streamfunction figures at days 0--40, 60, and 2,000;
   RMSE/ACC through day 200; and RMSE through day 2,000.
4. Through the contract-parameterized `anomaly_response.py` adapter, reuse
   the existing anomaly numerics on each sealed 15-start figure array. The
   adapter verifies the complete start vector; the mapped day-60/day-2,000
   anomaly panels use member 0, start 6,263, exactly as the parent package.
   The reference is MITgcm's derived S0 time-mean barotropic streamfunction
   over parent training days 0--5,999. Subtract that identical field from truth
   and every model; never subtract a model's own mean.
5. Report anomaly RMS, spatial variance, streamfunction extrema,
   WBC/interior ratio, zonal/meridional spectra, perturbation growth, maximum
   normalized magnitude, persistence, and climatology.
6. Compare frozen parent A, ft90 child as context, and every B/C seed on
   identical starts and physical metrics. Preserve the existing A/ft90 reports
   rather than regenerating a different contract for them.
7. Freeze all ordinary forward outputs; then run and freeze the blind
   forward-response package from section 17.
8. Only after both packages and their hashes are in the freeze manifest may
   the MITgcm/TAF adjoint evaluator be enabled.

The 90--360- and 2,000-day products are stability/statistical evaluations, not
pointwise targets. Nominal and response-gradient training remain exactly
six calls/60 days; only the sealed forward-response and adjoint evaluations
probe day 90.

## 20. Required controls and attribution checks

### 20.1 Required for the main paper

1. **A versus B:** verifies that the production parent can be replayed under
   its exact implemented contract and quantifies run-to-run/seed effects.
2. **B versus C:** the primary causal contrast. The arms share seed,
   initialization, every nominal batch, optimizer step count, learning-rate
   schedule, six-call rollout, parent loss, selector, and nominal evaluation;
   C alone receives the auxiliary response term.
3. **A versus C:** tests whether the new random-initialized model preserves the
   actual deployed parent's forward skill, not merely its paired rerun.
4. **Matched saved-step tables:** compare paired B/C at 1,920, 3,840, 5,760,
   and 7,680 updates. This makes any response effect separable from checkpoint
   timing while leaving the production selector untouched.
5. **Three paired B/C seeds:** primary seed 20260724 and replications
   20260911/20260912; report all seeds and the paired distribution, never a
   post hoc best seed.
6. **ft90 context:** report its already-established forward and adjoint
   products, but never use it as an architecture definition, initializer,
   normalizer source, hyperparameter control, or causal baseline.

B and C have identical optimizer budgets and nominal state exposure. C's
additional compute is solely the declared forward-response information; it
does not receive extra optimizer updates or a later training stage.

### 20.2 Explicitly outside v1

There is no perturbed-state-only arm, short-only arm, parent continuation,
90-day continuation, curriculum, alternate spectral cap, architecture
ablation, or loss-coefficient ablation in this contract. Those would violate
the single-change question posed here. If a later study tests point-only versus
smooth-only SSH, WBC sampling, or continuation, it must be separately approved
and use a newly sealed blind-response inventory; no v1 blind result may trigger
its design.

## 21. Compute and storage estimate

### 21.1 MITgcm forward response work

The exact pilot/production/control budget in section 11 is 57,750 model-days
= 160.417 model-years = 4,158,000 forward timesteps. A current S0 production
manifest records 680.78 s for 3,600 model-days on four MPI ranks. At that rate,
pure integration is about 3.0 four-rank wall-hours; current segment manifests
support a conservative 3--6 h integration range.

Operationally, the design contains 2,400 signed perturbed branches, 72
nominal/control branches, and three 320-day validation pickup-bank chains:
2,475 logical branches and 5,775 validated 10-day segment equivalents. Job
startup, staging, hashing, and filesystem latency will dominate. Reserve
**45--65 four-rank node-hours**, use arrays/bundles, and report measured
integration separately from the operational allocation. No TAF license,
adjoint tape, or adjoint output is involved.

### 21.2 FNO

The parent measured 3.215 V100-hours per complete seed. Budget:

- B, three seeds: about 9.7 GPU-hours;
- C, three seeds at the estimated 8--20% response overhead: 10.4--11.6
  GPU-hours;
- four-candidate, 1,920-step lambda screen: about 3.5--4.5 GPU-hours;
- forward/response/adjoint evaluation: less than 3 GPU-hours, with much of the
  validated double-precision adjoint work on CPU.

The total is approximately **27--31 V100-equivalent GPU-hours**. There is no
continuation or additional training arm in this budget. Actual auxiliary-path
wall-time is unresolved until the response loader and spectral-buffer context
are benchmarked.

### 21.3 Storage

- curated development-plus-blind response arrays and metadata: about 9--11 GB
  uncompressed; compressed size is unresolved until the pilot;
- existing annual source pickups are reused; edited inputs may be represented
  by complete provenance plus sparse byte edits only if the approved retention
  policy proves exact recoverability;
- transient endpoint pickups/logs: approximately 50--100 GB;
- six B/C runs, screen artifacts, reports, and figures: approximately 15--25
  GB.

Reserve **150 GB scratch** and **40 GB durable project storage**. Freeze raw
retention and compression policy after the pilot integrity test but before
production response generation.

## 22. Failure criteria and stop/go gates

### Gate D0 — production-contract, source, and inventory audit

- the B/C nominal split, records, normalizer recipe/hash, architecture,
  optimizer schedule, rollout, forward objective, checkpoint cadence, and
  production selector match sections 5--6 exactly;
- every source pickup, metadata, forcing file, executable, and grid hash
  resolves; duplicate boundary copies agree byte-for-byte;
- the regime-specific day-5,760-to-6,080 pickup-bank chains complete, and
  their day-6,010/6,050/6,080 P32 projections match trajectory-v3;
- every response anchor lies in its declared role and its complete rollout
  remains within that role's chronology;
- family, region, level, kernel, sign, and long-subset counts are exact;
- face/carrier labels, full support, coordinates, WBC capacity exception, and
  all cross-role centre-disjointness/separation rules pass;
- the blind store and all adjoint paths are excluded from the training/
  selection code path and its configs, verified by the section-18.1 path/glob
  scan rather than an OS-level account or mount boundary.

Failure: stop before production MITgcm response generation or FNO training.

### Gate D1 — pickup surgery

- only declared records/cells change;
- every untouched record is byte-identical;
- plus/minus edits are exact sign reversals;
- native support and centred P32 state/delta match the manifest and pass the
  1% realization/antisymmetry gates;
- the unedited pickup's P32 projection matches trajectory-v3.

Failure: fix and retest the generic editor before any batch submission.

### Gate D2 — forward-only amplitude pilot

Each separately selected U, V, Theta, and SSH amplitude must satisfy every
section-10 linearity, SNR, P32 precision, adjacent-amplitude, SSH-cap,
perturbed-repeat, and tight-CG criterion. If any family has no passing
candidate, stop production generation. A smaller-amplitude follow-up requires
a separately versioned pilot contract; no adjoint result may inform it.

### Gate D3 — curated response dataset

- every declared nominal and signed branch completes;
- response extraction reproduces direct float64 pickup differences;
- train/validation cases satisfy `Q_lin<=0.05` and `Q_SNR>=20` at every
  available lead through day 60 using the final combined numerical floor;
- no NaN/Inf is present; counts, signs, lead arrays, schemas, and hashes are
  exact;
- the training loader cannot read validation, blind, or adjoint paths, and
  development evaluators cannot read blind or adjoint paths.

A failed train/validation case invalidates the dataset version; it is never
silently dropped or rescaled, and v1 stops. If a successor changes amplitude,
inventory, or extraction after seeing that failure, every failed validation
case becomes development data and the successor must create new response-
validation and blind inventories. Blind linearity/SNR is computed only after
model freeze and reported for every case; it cannot trigger any change.

### Gate M0 — exact parent replay

The B code/config audit must prove scientific equality to the production
parent for every invariant in section 5.1, subject only to the explicit
section-23 infrastructure whitelist. The primary seed is frozen in advance;
its checkpoint is selected only by the unchanged production selector.
Relative to frozen parent A on the identical 102 validation records, selected
primary-seed B must have:

- each 10--90-day primary-field AUC at most 1.05 times A;
- worst 90--360-day AUC/climatology ratio at most 1.05 times A;
- perturbation growth no more than 0.005 per call above A;
- maximum normalized amplitude through day 360 at most 1.05 times A;
- no nonfinite rollout.

All A/B checkpoint-step tables and selector branches are reported, including
differences. Failure means the exact-parent reproduction has not recovered the
baseline adequately. Only a demonstrated contract/code-integrity defect may be
corrected under a newly reviewed version. If all equality/integrity tests pass
but these metric tolerances fail, freeze a negative M0 and close v1: no
seed, hardware, protocol, checkpoint, or threshold retry is allowed.

### Gate M1 — response-aware development result

Select C with the unchanged production selector, then apply section 16.3's
forward-preservation and held-response success criteria against paired B.
Failure labels the development result negative. It does not authorize another
lambda, seed, checkpoint, continuation, curriculum, or data edit. Once the
technically valid model identities are frozen, the preregistered blind
forward-response and adjoint evaluations still run so the mechanistic result
is not censored by a development outcome.

### Gate M2 — frozen ordinary-forward and response tests

For the primary selected C, ordinary-forward preservation must hold against
both paired B and frozen parent A:

- every primary 10--90-day AUC ratio is at most 1.05;
- the worst 90--360-day AUC/climatology ratio and maximum normalized amplitude
  through day 360 are each at most 1.05 times the comparator;
- perturbation growth is no more than 0.005 per call worse;
- on the exact 15 S0 final-inference starts, each day-2,000 primary RMSE and
  maximum normalized magnitude is at most 1.10 times each comparator;
- no rollout is nonfinite.

The blind response criteria are exactly those in section 17. These packages
are run once. Failure is a negative result and cannot alter any trained
artifact or decision. The adjoint evaluation still proceeds if technical Gate
A0 passes.

### Gate A0 — adjoint pipeline validity

Parent/B/C FNO finite differences, forward/reverse identity, dtype, masks,
checkpoint/normalizer hashes, and weight-field hashes must pass before
comparison; ft90 retains its validated contextual result. The existing MITgcm
G0--G5 scalar gates remain required. Extend the one offshore G1 curve whose
current minimum is at `epsilon=1e-5` with predeclared `1e-6` and `1e-7`
forward differences; obtain an interior minimum or retain and report a failed
plateau flag.

Because v2 G0 currently checks ETAN only, add a final-evaluation, forward-only
F90 extraction of U/V/Theta/ETAN at FNO 10-day nodes and require its canonical
P32 projection to match trajectory-v3. Reports must say either
“46-channel G0 at FNO 10-day nodes; ETAN daily” after that gate passes or
“ETAN-only daily G0” before it does. These technical checks occur only after
model freeze and cannot affect model decisions.

### Gate A1 — confirmatory scientific result

No threshold selects a model after TAF access. Report the predeclared primary
and secondary endpoints for every seed. V1 supports the full hypothesis only
if ordinary-forward preservation in M2, the section-17 blind response
criteria, and every section-18.3 primary adjoint criterion pass. Any failure,
including a forward/adjoint tradeoff, rejects “improved Jacobian/adjoint
without degrading forward skill” for v1 even if a mechanistic sub-result
improves.

## 23. Exact implementation files to create or modify after approval

No file below is implemented by this planning task.

### 23.1 Create

**Frozen contracts/configs**

- `config/forward_response_amplitude_pilot_v1.json`
- `config/forward_response_dataset_v1.json`
- `config/forward_response_schema_v1.json`
- `config/forward_response_lambda_screen_v1.json`
- `config/model_c_adjoint_faithful_nominal_control_v1.json`
- `config/model_c_adjoint_faithful_response_v1.json`
- `config/adjoint_faithful_forward_evaluation_v1.json`
- `config/adjoint_faithful_blind_adjoint_evaluation_v1.json`

The B config must match every **scientific field** in the production parent.
The equality checker permits only this explicit infrastructure whitelist:
study/version label, output/report roots, declared seed, new-runner source
hashes, and a response block fixed to `enabled=false`. The parent loss
contract SHA-256 `6a233883b3c9a6347f0d343f295bee2aa841b143b547acc9f71fea05e8e8d2e1`
and all model/data/training/validation fields must match. C is byte-identical
to B outside response enablement, response-store/schedule fields, and the
frozen nonzero lambda.

**MITgcm forward-response generation**

- `archive/src/bire_repro/af_response_pickup_bank.py` — implemented (see
  amendments above)
- `scripts/build_response_pickup_bank.py` — implemented, supersedes the
  originally planned `build_response_validation_pickup_bank.py` name
- `scripts/build_forward_response_inventory.py` — implemented
- `slurm/mitgcm/af_response_pickup_bank_segment.sbatch` and
  `scripts/submit_af_response_pickup_bank.sh` — implemented, supersede the
  originally planned `af_forward_response_pickup_bank.sbatch` name
- `scripts/stage_forward_response_run.py`
- `scripts/extract_forward_response_dataset.py`
- `scripts/verify_forward_response_dataset.py`
- `slurm/mitgcm/af_forward_response_array.sbatch`

**FNO response path and reports**

- `src/oceanfno/response_dataset.py`
- `src/oceanfno/response_objective.py`
- `src/oceanfno/response_spectral_context.py`
- `src/oceanfno/response_validation.py`
- `src/oceanfno/train_response.py`: the one common parameterized runner for
  both B (response disabled) and C (response enabled), importing the trusted
  parent dataset/model/objective/validation utilities;
- `src/oceanfno/figures_response.py`
- `src/oceanfno/anomaly_response.py`: contract adapter that reuses the
  numerical helpers in the frozen anomaly module but accepts B/C figure
  identities;
- `slurm/models/c/train_adjoint_faithful_nominal_control_v1.sbatch`
- `slurm/models/c/train_adjoint_faithful_response_v1.sbatch`
- `slurm/models/c/figures_adjoint_faithful_response_v1.sbatch`

**Blind adjoint adapters**

- `scripts/fno_adjoint_model.py`: contract-parameterized adapter of the
  trusted one-input ft90 runner, retaining its validated complex128 fix;
- `scripts/compare_adjoint_models_response_v1.py`.

**Tests**

- `tests/test_forward_response_inventory.py` — implemented
- `tests/test_forward_response_pickup.py` — implemented
- `tests/test_af_response_pickup_bank.py` — implemented (not in the original
  plan; covers the pickup-bank driver above)
- `tests/test_forward_response_dataset.py`
- `tests/test_response_objective.py`
- `tests/test_response_spectral_context.py`
- `tests/test_response_training.py`
- `tests/test_response_validation.py`
- `tests/test_fno_adjoint_model.py`

### 23.2 Modify minimally

- `archive/src/bire_repro/af_s0_twin.py`: factor its trusted byte-level
  pickup parser/editor into a generic declared-record/cell edit function while
  keeping the existing S0 twin wrapper byte-compatible.
- `pyproject.toml`: add entry points only after the modules exist and preserve
  the user's unrelated current edit.

The frozen `src/oceanfno/train.py` cannot be parameterized in place: it
hard-codes the parent version, primary seed, output suffix, and pinned source
hashes. Before any study training, the new common runner with response disabled
must pass a primary-seed equivalence harness against that original trainer:
same initialization tensors, nominal batches, per-term losses, gradients,
spectral buffers, optimizer states, and checkpoints step-for-step in the
pinned environment. A full primary-seed replay must also satisfy Gate M0.
Failure stops implementation; it is not permission to fork parent numerics or
maintain separate B/C loops.

### 23.3 Reuse unchanged

- `src/oceanfno/dataset.py`, including its production split, normalizers,
  `RolloutDataset`, and `ChunkAwareBatchSampler`;
- `src/oceanfno/model.py`, `objective.py`, `spectral_norm.py`,
  `pressure_gradient.py`, `continuity.py`, and
  `barotropic_transport.py`;
- `src/oceanfno/validation.py`, including `validate_checkpoint`,
  `train_only_climatology`, and `select_by_validation`;
- `src/oceanfno/train.py` as the immutable primary-seed equivalence
  reference, not the B/C study runner;
- numerical routines in `src/oceanfno/figures.py`,
  `figures_ft90.py`, and `anomaly.py`;
- `scripts/adjoint_metrics.py`, `fno_adjoint_ft90.py`,
  `compare_adjoint_maps_phase_a.py`, and
  `stage_adjoint_run.py`;
- `af_fno/mitgcm/code_ad/**`, `input_ad/**`, and `tamc.h`;
- every existing parent/ft90 config, output, checkpoint, report, and required
  document.

There is deliberately no chronological-split adapter, new nominal
normalizer, replacement selector, alternate model definition, or forked
forward objective.

## 24. Reproducibility and provenance requirements

Before each phase, write an immutable contract plus SHA-256. Every report must
include:

- repository commit, dirty-worktree diff hash, Python lock/environment, NumPy,
  Torch, neuralop, CUDA, compiler, and MPI versions;
- MITgcm commit and executable hash;
- source pickup, metadata, forcing, grid, namelist, and static-input hashes;
- trajectory-v3 metadata/manifest hash (`766cae893593...` currently);
- a field-by-field production-parent equality report for split, records,
  normalizers, architecture, loss, optimizer, rollout, checkpoint cadence,
  spectral normalization, validation, and selector;
- exact split arrays, anchor/centre inventory, level and region counts;
- pilot raw metrics, nominal/perturbed repeats, tight-CG comparisons, combined
  floor, selected amplitudes, and decision trace;
- response-store metadata/chunk/compressor hashes and per-array digests;
- model architecture/loss contract, seeds, batch order, optimizer, learning
  rates, response schedule, lambda screen, and checkpoint steps;
- pre/post auxiliary spectral power-vector hashes and triplet-order tests;
- normalizer/increment-scale hashes;
- materialized checkpoint and optimizer-step hashes;
- the unmodified nominal-selection trace, including any production fallback,
  plus a separately labeled response-validation report proving it did not
  select the full-run checkpoint;
- hardware, Slurm IDs, wall times, failures/retries, and quarantined cases;
- blind freeze timestamp, forbidden-path scan result, and hashes of every
  artifact existing before access;
- final evaluator version and all MITgcm/FNO gate results.

Model randomness is limited to the three declared paired seeds. The response
sampler uses a separately hashed counter stream and cannot advance the parent
initialization or nominal-batch RNGs. Spatial selection is hash/maximin
deterministic. Completed artifacts are write-once. Reports state inclusive and
half-open day conventions side by side to prevent off-by-one leakage.

The existing adjoint products stay under their own paths, excluded from every
training/selection run by the section-18.1 forbidden-path scan. Repository
readability in the present audit checkout is not itself a blind barrier --
the scan, not general filesystem access, is what the freeze report relies on.

---

## 25. Numbered execution order

1. Review and approve this document. Submit no MITgcm or FNO compute before
   approval.
2. Freeze a machine-readable equality contract against
   `model_c_production_1in_1out_spectralnorm_v1`: exact active split,
   nominal records, normalization recipes/hash, architecture, parent loss,
   optimizer schedule, six-call rollout, checkpoint cadence, and selector,
   with only the section-23 infrastructure whitelist permitted.
3. Freeze the three paired seeds, response counts/leads, masks, kernels,
   candidate alphas/lambdas, and the forbidden-path scan contract (section
   18.1) that keeps adjoint/blind artifacts out of training/selection.
4. Materialize the joint pilot/train/validation/blind centre, level, kernel,
   and long-subset inventory; verify all counts/full support and seal the blind
   inventory from development.
5. Implement the common B/C runner and equality harness; pass response-disabled
   initialization/batch/loss/gradient/optimizer dry-run equivalence against the
   frozen parent trainer. In parallel, generalize and unit-test the trusted
   pickup editor; prove the old twin path remains byte-compatible and only
   requested bytes change.
6. Build the three regime-specific day-5,760-to-6,080 nominal pickup-bank
   chains; hash every 10-day pickup and verify their P32 projections.
7. Run the six-anchor, forward-only amplitude pilot at both signs and all three
   candidate amplitudes, with paired nominal duplicates.
8. Choose provisional U/V/Theta/SSH amplitudes; run the selected-alpha
   perturbed duplicates and tight-CG signed/nominal controls; apply every
   section-10 criterion and freeze the four final amplitudes. Stop if any
   family fails.
9. Generate all shared nominal and signed response-training/validation
   branches, short cases first and then the frozen 60-day long subsets. Do not
   generate or expose blind response data.
10. Extract and verify the development response store; freeze training-only
    response scales, numerical floors, schemas, and hashes.
11. Pass the common runner's response-disabled primary-seed equivalence
    harness, then run B from random initialization for all three seeds through
    that runner. Apply the exact production selector and Gate M0.
12. Run the four-lambda, 1,920-step primary-seed screen using nominal and
    response validation through day 60 only. Freeze lambda and discard screen
    checkpoints/optimizer states.
13. Restart from step zero and train C for all three paired seeds; C alone
    enables the response auxiliary path.
14. Apply the unchanged production selector independently to every C run.
    Report response validation only after selection, apply Gate M1, and freeze
    model/checkpoint/normalizer/config/report hashes. Do not retrain on failure.
15. Run and freeze the complete ordinary forward validation, S0 figure,
    anomaly, streamfunction, and matched-step packages for A, ft90, B, and C.
16. Generate/extract the already-designed evaluator-only blind
    forward-response store; evaluate it once for A, ft90, B, and C and freeze
    Gate M2 results.
17. Enable the evaluator-only adjoint path. Run all FNO derivative checks and
    the MITgcm G0--G5/A0 technical extensions without changing a model.
18. Evaluate parent A and every B/C seed against the existing MITgcm/TAF
    point, kernel, and conservation objectives at 10/20/30/90 days; retain
    ft90 as context.
19. Run only preregistered exploratory adjoint objectives whose independent
    gates were frozen before training.
20. Produce paper tables/figures for nominal forward skill, anomalies, blind
    responses, JVP/adjoint metrics, lead dependence, spectra, conservation,
    paired controls, compute, and every failure.
21. Archive the provenance/access-log bundle and answer the sole confirmatory
    question: did forward-only response supervision improve the learned
    Jacobian/adjoint without degrading the production-parent forward emulator?

## Frozen proposed contract

Numerical outcomes that require the forward-only pilot or lambda screen are
frozen below as procedures, not invented values.

| Contract item | Frozen proposal |
| --- | --- |
| Baseline model | Frozen `model_c_production_1in_1out_spectralnorm_v1` (A); its current implementation is authoritative; no checkpoint is loaded by B or C |
| Context-only child | `model_c_production_1in_1out_spectralnorm_ft90_v1`; report only; never an initializer, baseline design, or decision source |
| New model | `model_c_adjoint_faithful_response_v1` (C), random initialization, paired with response-disabled B in the same common runner; C differs only by response data and `lambda_resp * L_response` |
| State channels | U15 + V15 + Theta15 + SSH1 = 46; unchanged; no adjoint outputs |
| Architecture/statics | Exact full parent architecture dictionary: 27,297,960 parameters, 32x32 modes, width 128, three blocks, six pointwise LayerNorms, padding 0.1, dense/full-precision spectral weights, local branch, five physical statics, position encoding, and identical spectral normalization |
| Nominal train/validation periods | Exact parent train `[0,6000)` and validation `[6000,7200)`; no buffer and no replacement split |
| Nominal validation/inference | Checkpoint starts `6000 + 6m`, `m=0,...,33`, 34/regime; nested final-inference block `[6200,7200)` and exact 15 S0 starts in section 6.2; `[7200,9000)` is truth-only |
| Normalization | Exact parent functions and days 0--5,999 over S0/S1/S2; recomputation must match SHA-256 `fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf`; response data never change parent normalization |
| Response anchors | Train 14/regime at 0,360,720,1440,1800,2160,2520,3240,3600,3960,4320,5040,5400,5760; validation 3/regime at 6010/6050/6080; blind 3/regime at 7560/7920/8280 |
| Response counts | Train 224 directions/regime = 672 total, 96 long; validation 72/regime = 216 total, 36 long; blind same; every direction has both signs |
| Perturbation families | Native-face 5x5 Gaussian U/V; tracer-centred 5x5 Gaussian Theta; SSH wet-cell point and 5x5 Gaussian; sigma 1 cell, radius 2, unit-L2 kernel, full active support only |
| Spatial/vertical design | WBC about one third and oversampled; interior/east/north/south retained; deterministic joint maximin/hash allocation; every U/V/Theta level represented; training single-level, held validation/blind multi-level combinations |
| Amplitude calibration | Forward-only +/- pilot at alpha 0.025/0.05/0.10 training sigma; choose the largest separately for U/V/Theta/SSH passing <=5% sign asymmetry, SNR>=20, P32/adjacent-alpha/repeat/tight-CG gates; SSH peak <=1 cm |
| Response horizons | All cases 10 d; training/validation sparse-long targets 10,20,...,60 d; pilot and blind evaluation may reach 90 d; no model-development response metric or gradient beyond 60 d |
| MITgcm cost | 57,750 model-days = 160.417 model-years, 4,158,000 steps, including pickup bank, pilot, nominal, duplicate, and tight-CG controls |
| Restart semantics | Edit only selected `Uvel`, `Vvel`, `Theta`, or `EtaN` cells; Salt, AB histories, `dEtaHdt`, `EtaH`, and every unselected byte remain identical; never reconstruct a pickup from Zarr |
| Forward loss | Exact eight-term parent objective and coefficients, unchanged; exact six-call/60-day nominal rollout |
| Response loss | Signed oriented response/JVP error, balanced equally across input/output physical groups; six-lead mean for long cases; no ordinary perturbed-state loss |
| Response mixing | Exact parent nominal minibatches and nominal-loss definition/weight every update; isolated response pair every fourth update; 75% short/25% long; batched nominal/-/+ branches and spectral power-vector snapshot/restore |
| Response weight | Choose from 0.03/0.10/0.30/1.00 using only a 1,920-step primary-seed forward/response-validation screen; discard screen states and restart C at step zero |
| Optimizer/training | Cold Adam with no state load, exact parent betas, no weight decay/clipping, batch 8, 7,680 updates, LR 5e-4 through 5,760 then 1e-4, checkpoints 1,920/3,840/5,760/7,680, six-call/60-day rollout |
| Seeds | Primary 20260724; paired replications 20260911 and 20260912 |
| Checkpoint selection | Exact unchanged parent `select_by_validation` on exact nominal validation, independently for B and C; response validation never selects a full-run checkpoint |
| Forward-preservation criteria | Selected C must remain within the predeclared 5% nominal validation envelope versus paired B and frozen A, with final-inference gates; response-development and blind improvements are separate success tests |
| Blind forward tests | Presealed geometry; evaluator-only 216-direction/432-signed numeric response set generated after freeze at 7560/7920/8280, with all cases at day 10 and 36 long through day 90; ordinary parent final-inference/figure package; run once |
| Blind adjoint tests | Existing scalar-gated S0 point/kernel/mean suite at 10/20/30/90 d, plus predeclared A0 technical extensions; compare parent, all B/C seeds, and ft90 context |
| Required controls | A frozen parent, B exact-scientific-contract response-disabled replay, C response-aware in the same runner; matched saved steps and three paired seeds; ft90 context only |
| Forbidden data | No MITgcm/TAF adjoint, `ADJ*`, `adxx_*`, adjoint-derived map/metric, blind numeric response, nested final-inference result, or new FNO adjoint may influence training, amplitude/lambda choice, architecture, checkpoint selection, or retry decisions |

**Approval boundary:** stop here. After approval, the first authorized work is
contract implementation plus no-compute inventory/pickup unit testing—not an
MITgcm submission and not FNO training.
