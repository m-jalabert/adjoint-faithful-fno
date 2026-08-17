#!/bin/bash -l
#SBATCH --job-name=af-s0-adj-grdchk
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# Stage 7 of docs/mitgcm_adjoint_ground_truth_plan.md: acceptance gate G1.
#
# pkg/grdchk perturbs one control element by +/- grdchk_eps, runs the forward
# model twice, and compares the centred finite difference against the adjoint
# gradient.  This is the only thing standing between "a 62x62 array of
# plausible numbers" and ground truth.  Do not skip it.
#
# Sweep epsilon: a correct adjoint shows a PLATEAU, with FD/adjoint at 1 across
# the middle of the range, degrading only at 1e-6 where float round-off takes
# over.  A single epsilon agreeing is weak evidence.
#
# Test points are set in af_fno/mitgcm/input_ad/data.grdchk (iGloPos/jGloPos are
# TILE-LOCAL; tile (1,1) covers global i,j = 1..31 so local == global there).
# The plan lists eight; point 1 is p* = (i=2, j=17).
#
# Env:
#   AF_GRDCHK_EPS_LIST  epsilons to sweep (default "1.d-3 1.d-4 1.d-5 1.d-6")
#   AF_GRDCHK_I, AF_GRDCHK_J   override the tile-local test position
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1}"
AF_GRDCHK_EPS_LIST="${AF_GRDCHK_EPS_LIST:-1.d-3 1.d-4 1.d-5 1.d-6}"

cd "${AF_PROJECT_ROOT}"

"${AF_PROJECT_ROOT}/.venv/bin/python" scripts/stage_adjoint_run.py grdchk \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --force

RUN_DIR="${AF_SCRATCH_ROOT}/grdchk"
cd "${RUN_DIR}"

if [ -n "${AF_GRDCHK_I:-}" ]; then
  sed -i "s/^ iGloPos .*/ iGloPos        = ${AF_GRDCHK_I},/" data.grdchk
fi
if [ -n "${AF_GRDCHK_J:-}" ]; then
  sed -i "s/^ jGloPos .*/ jGloPos        = ${AF_GRDCHK_J},/" data.grdchk
fi

for EPS in ${AF_GRDCHK_EPS_LIST}; do
  TAG="${EPS//./p}"
  echo "=== grdchk_eps = ${EPS} ==="
  sed -i "s/^ grdchk_eps .*/ grdchk_eps     = ${EPS},/" data.grdchk
  srun --ntasks=4 ./mitgcmuv_ad > "run_eps_${TAG}.log" 2>&1 || {
    echo "run failed at eps=${EPS}"; tail -40 "run_eps_${TAG}.log"; exit 1; }
  cp data.grdchk "data.grdchk_eps_${TAG}"
  # grdchk_print writes the comparison to STDOUT; keep the block that matters
  grep -A4 -iE "grdchk|gradient check|adjoint gradient|finite diff" \
    "run_eps_${TAG}.log" > "grdchk_eps_${TAG}.txt" || true
  echo "--- eps=${EPS} summary ---"
  cat "grdchk_eps_${TAG}.txt"
done

echo
echo "Gate G1 needs |FD/adjoint - 1| < 1e-4 at eps = 1e-4 AND 1e-5, with the"
echo "plateau visible across the sweep.  Land cells must return exactly 0."
echo "Results in ${RUN_DIR}/grdchk_eps_*.txt"
