#!/bin/bash -l
#SBATCH --job-name=af-pa-g90
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-20

# Gate G1-90: the gradient check at the ninety-day window.
#
# Seven test points x three epsilons, one array task each.  Each task is one
# MITgcm invocation: two forward runs of 6,480 steps and one adjoint.
#
# WHY cg2dTargetResidual = 1e-12 HERE AND ONLY HERE.  grdchk-limited-by-cg2d
# established that at the production 1e-7 the FINITE DIFFERENCE, not the
# adjoint, is the noisy party: the v1 gate-G1 error was flat in epsilon --- no
# plateau to find, because there was nothing converging.  At 1e-12 every point
# passed at every epsilon with the plateau visible.  The adjoint gradient itself
# moved by only ~5e-06 relative between the two settings, so this changes what
# the FD can resolve and not what is being certified.  The production maps keep
# 1e-7, because that is what every S0 forward segment was integrated with and
# what the emulator was trained on.
#
# WHY THIS EPSILON RANGE.  P90's own map sets it.  The weakest of the seven
# points carries |S_90| = 4.0e-07, so at eps = 1e-5 the predicted change in fc
# is 4.0e-12 --- about 1e4 times the fc round-off floor at this solver setting.
# Below that the finite difference stops resolving anything; above 1e-3 the
# ninety-day response may leave the linear range, which is itself the thing the
# sweep measures rather than assumes.
#
# nbeg MUST stay 0 or grdchk silently tests the first wet cell (2,2) while
# printing a perfectly plausible grad-res block.  The stager substitutes the
# TILE-LOCAL position, which is what grdchk_readparms.F actually reads.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v2}"

# The plan's seven points, global 1-based (i,j).  Land is not reachable from
# here --- GRDCHK_GET_POSITION only ever lands on wet cells --- so it is covered
# by gate G4 over all 244 land cells instead, which is the stronger statement.
POINTS=("2,17" "2,14" "2,11" "4,17" "31,17" "61,17" "31,55")
LABELS=("pstar" "wbc14" "wbc11" "offshore" "interior" "eastern" "northern")
# Overridable, because where the plateau sits depends on the magnitude of the
# sensitivity being checked and that is not known before P90 runs.  At ninety
# days |S| at p* is 5.3e-06 against the ten-day study's 9.3e-05 -- seventeen
# times smaller -- so the same epsilon buys seventeen times less signal against
# the same fc round-off floor, and the optimum moves up by about that factor.
# The first sweep {1e-3,1e-4,1e-5} put the minimum at its own top edge; the
# extension {1e-1,1e-2} brackets it from above.  Whether the error turns back up
# there is the linear-range measurement, not an assumption: s0-not-chaotic
# predicts it will not, at 90 days in this regime.
IFS=',' read -r -a EPSILONS <<< "${AF_GRDCHK_EPSILONS:-1e-3,1e-4,1e-5}"

N=${#EPSILONS[@]}
P=$(( SLURM_ARRAY_TASK_ID / N ))
E=$(( SLURM_ARRAY_TASK_ID % N ))
POINT="${POINTS[$P]}"
LABEL="${LABELS[$P]}"
EPS="${EPSILONS[$E]}"
NAME="G90_${LABEL}_${EPS}"

cd "${AF_PROJECT_ROOT}"
PYTHONPATH="${AF_PROJECT_ROOT}/scripts" "${AF_PROJECT_ROOT}/.venv/bin/python" \
  scripts/stage_adjoint_run.py G90 \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --run-name "${NAME}" \
  --grdchk-point "${POINT}" \
  --grdchk-eps "${EPS}" \
  --cg2d-target-residual 1e-12 \
  --force

cd "${AF_SCRATCH_ROOT}/${NAME}"
srun --mpi=pmix -n 4 ./mitgcmuv_ad > run.log 2>&1

echo "=== ${NAME} complete  (global point ${POINT}, eps ${EPS}) ==="
grep -iE "grad-res|precision_derivative|adjoint value|finite diff" STDOUT.0000 | tail -20
