#!/bin/bash -l
#SBATCH --job-name=af-s0-adj-pa
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

# Phase A of docs/Adjoint_study_Phase_A.md: the ninety-day adjoint window.
#
#   AF_ADJ_RUN=F90   FORWARD, day 7200 -> 7290 plus one tail day.
#                    Run this first.  It produces (a) the day-7280 pickup V10
#                    starts from, which no archive segment holds, (b) daily
#                    snapshots for gate G0 across the whole window, and (c) the
#                    day-7290 snapshot -- no adjoint run dumps it, because
#                    data.diagnostics sets dumpAtLast=.FALSE. and the cost sits
#                    at the final iteration.
#
#   AF_ADJ_RUN=P10|P20|P30|P90    the point objective, source day 7200 fixed
#   AF_ADJ_RUN=K10|K20|K30|K90    the smooth-kernel objective, same windows
#   AF_ADJ_RUN=C90                the mean-only conservation probe (gate G3)
#   AF_ADJ_RUN=V10                day 7280 -> 7290 (gate G2b)
#   AF_ADJ_RUN=G90                grdchk at the ninety-day window (gate G1-90)
#
# Each run's window and weight field come from config/adjoint_phase_a_v1.json,
# so the runs and the plan cannot drift apart, and the weight field follows
# from the run name rather than from a flag that can be forgotten.  Two sides
# weighting eta differently is the single largest silent-failure risk here.
#
# NO TAF SUBMISSION.  code_ad/tamc.h is sized 72 x 200 x 1 = 14,400 >= 6,480
# steps, so ninety days fits the existing tape and mitgcmuv_ad is reused
# byte-for-byte.  Nothing in the differentiated source changes.
#
# ON NODE EXCLUSIVITY.  mitgcm-cg2d-needs-exclusive-nodes records that shared
# nodes made the 248^2 runs ~100x slower because the cg2d solve is
# reduction-bound, and a ninety-day adjoint is nine times runB and dominated by
# the same reduction.  So prefer an exclusive node:
#
#     sbatch --exclusive --export=ALL,AF_ADJ_RUN=P90 ...
#
# It is NOT a #SBATCH directive here, because it cannot be turned off from the
# command line and this cluster frequently has no whole node free -- on
# 2026-08-19, 50 of 62 batch nodes were allocated and an exclusive request sat
# in (Resources) indefinitely while nine nodes had free cores.  These runs are
# 62^2, an eighth the linear size of the case that motivated the finding, and
# the v1 runs completed on shared nodes.  Watch the wall time: a ninety-day
# adjoint that takes hours rather than tens of minutes is the reduction
# contention showing up at this grid size too, and is worth recording.
#
# cg2dTargetResidual stays at 1e-7 for every production run: it is what every
# S0 forward segment was integrated with and what the emulator was trained on.
# Only G90 tightens it, and only because grdchk-limited-by-cg2d established
# that at 1e-7 the FINITE DIFFERENCE, not the adjoint, is the noisy party.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v2}"
AF_ADJ_RUN="${AF_ADJ_RUN:?Set AF_ADJ_RUN to one of F90 P10 P20 P30 P90 K10 K20 K30 K90 C90 V10 G90}"

case "${AF_ADJ_RUN}" in
  F90|P10|P20|P30|P90|K10|K20|K30|K90|C90|V10|G90) ;;
  *) echo "unknown AF_ADJ_RUN '${AF_ADJ_RUN}'"; exit 1 ;;
esac

cd "${AF_PROJECT_ROOT}"

STAGE_ARGS=(
  scripts/stage_adjoint_run.py "${AF_ADJ_RUN}"
  --project-root "${AF_PROJECT_ROOT}"
  --scratch-root "${AF_SCRATCH_ROOT}"
  --force
)
if [ "${AF_ADJ_RUN}" = "G90" ]; then
  # Diagnostic only, and labelled as such in the run manifest.  The production
  # maps keep 1e-7; tightening it there would differentiate a model the
  # emulator never saw.
  STAGE_ARGS+=(--cg2d-target-residual "${AF_GRDCHK_CG2D:-1e-12}")
  if [ -n "${AF_GRDCHK_EPS:-}" ]; then
    STAGE_ARGS+=(--grdchk-eps "${AF_GRDCHK_EPS}")
  fi
fi

PYTHONPATH="${AF_PROJECT_ROOT}/scripts" \
  "${AF_PROJECT_ROOT}/.venv/bin/python" "${STAGE_ARGS[@]}"

RUN_DIR="${AF_SCRATCH_ROOT}/${AF_ADJ_RUN}"
cd "${RUN_DIR}"

if [ "${AF_ADJ_RUN}" = "F90" ]; then
  EXE=./mitgcmuv
else
  EXE=./mitgcmuv_ad
fi

# A bare srun fails at MPI_Init here ("PML add procs failed"); every S0 forward
# segment was launched as srun --mpi=pmix -n 4 and the adjoint must match.
srun --mpi=pmix -n 4 "${EXE}" > run.log 2>&1

echo "=== ${AF_ADJ_RUN} complete ==="

if [ "${AF_ADJ_RUN}" = "F90" ]; then
  echo "--- pickups (V10 needs 0003116160 = day 7280) ---"
  ls -1 pickup.*.data | sed 's/^/  /'
  echo "--- snapshot count (gate G0 wants days 7200-7290) ---"
  echo "  dynState  $(ls -1 dynState.*.data 2>/dev/null | wc -l)"
  echo "  surfState $(ls -1 surfState.*.data 2>/dev/null | wc -l)"
  exit 0
fi

test -f adxx_etan.0000000000.data || {
  echo "ERROR: no gradient written; adxx_etan.0000000000.data is missing"
  echo "A complete, valid build can return an identically zero gradient -- check"
  echo "STDOUT.0000 for 'the independent variables have no influence on'."
  tail -60 run.log
  exit 1
}

echo "--- gradient and dumps ---"
ls -l adxx_etan.0000000000.data | sed 's/^/  /'
echo "  ADJetan dumps: $(ls -1 ADJetan.*.data 2>/dev/null | wc -l)   (91 expected for a 90-day run)"
echo "  ADJuvel/ADJvvel/ADJtheta also written; Phase A archives them for Phase B"
echo "--- cost function value (gate G5 compares this against numpy) ---"
# fc goes to the model's own STDOUT.0000, not to the stdout srun captures.
grep -iE "global fc|early fc|objf_test" STDOUT.0000 | tail -5

if [ "${AF_ADJ_RUN}" = "G90" ]; then
  echo "--- gradient check (gate G1-90) ---"
  grep -A3 -iE "grad-res|precision_derivative" STDOUT.0000 | tail -40
fi
