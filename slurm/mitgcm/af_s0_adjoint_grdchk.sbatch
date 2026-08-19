#!/bin/bash -l
#SBATCH --job-name=af-s0-adj-grdchk
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# Stage 7 of docs/mitgcm_adjoint_ground_truth_plan.md, acceptance gate G1.
#
# pkg/grdchk perturbs one control element by +/- grdchk_eps, runs the forward
# model twice, and compares the centred finite difference against the adjoint
# gradient.  This is the only thing standing between "a 62x62 array of
# plausible numbers" and ground truth.
#
# The sweep is points x epsilons.  A single epsilon agreeing is weak evidence;
# the plateau across epsilons is the strong evidence, so every point gets the
# full epsilon list.
#
# grdchk writes its result to STDOUT.0000 as "grad-res" lines and overwrites
# that file on every run, so each combination is archived immediately.  The
# per-rank run_eps_*.log holds only the Fortran STOP message -- it is NOT where
# the numbers are.
#
# Positions are TILE-LOCAL (grdchk_readparms.F:116 rejects iGloPos > sNx).
# With sNx = sNy = 31 on the 2x2 decomposition:
#     iGloTile=1, jGloTile=1  ->  global i=1..31,  j=1..31
#     iGloTile=2, jGloTile=1  ->  global i=32..62, j=1..31
#     iGloTile=1, jGloTile=2  ->  global i=1..31,  j=32..62
#     iGloTile=2, jGloTile=2  ->  global i=32..62, j=32..62
#
# Env:
#   AF_GRDCHK_EPS_LIST  epsilons to sweep (default "1.d-3 1.d-4 1.d-5 1.d-6")
#   AF_GRDCHK_POINTS    override the point list, same "label:i:j:iTile:jTile" form
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1}"
AF_GRDCHK_EPS_LIST="${AF_GRDCHK_EPS_LIST:-1.d-3 1.d-4 1.d-5 1.d-6}"
# DIAGNOSTIC ONLY (plan section 12.2).  1e-7 is what every S0 forward segment
# used and the only value a reportable result may use.  A tighter setting
# separates solver-tolerance error in the FINITE DIFFERENCE -- grdchk's two
# forward runs -- from a genuine error in the adjoint.  Results produced with
# anything else are labelled by AF_GRDCHK_SUFFIX and must not be reported as
# gate G1.
AF_CG2D_TARGET_RESIDUAL="${AF_CG2D_TARGET_RESIDUAL:-1e-7}"
AF_GRDCHK_SUFFIX="${AF_GRDCHK_SUFFIX:-}"

# label : tile-local i : tile-local j : iGloTile : jGloTile     (global i,j in the label)
AF_GRDCHK_POINTS="${AF_GRDCHK_POINTS:-\
pstar_i2_j17:2:17:1:1 \
wbc_upstream_i2_j14:2:14:1:1 \
wbc_upstream_i2_j11:2:11:1:1 \
offshore_i4_j17:4:17:1:1 \
interior_i31_j17:31:17:1:1 \
eastern_i61_j17:30:17:2:1 \
northern_i31_j55:31:24:1:2}"

cd "${AF_PROJECT_ROOT}"

"${AF_PROJECT_ROOT}/.venv/bin/python" scripts/stage_adjoint_run.py grdchk \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --cg2d-target-residual "${AF_CG2D_TARGET_RESIDUAL}" \
  --force

RUN_DIR="${AF_SCRATCH_ROOT}/grdchk"
cd "${RUN_DIR}"
RESULTS="results${AF_GRDCHK_SUFFIX}"
mkdir -p "${RESULTS}"

for POINT in ${AF_GRDCHK_POINTS}; do
  IFS=: read -r LABEL PI PJ PTI PTJ <<< "${POINT}"
  sed -i "s/^ iGloPos .*/ iGloPos        = ${PI},/"  data.grdchk
  sed -i "s/^ jGloPos .*/ jGloPos        = ${PJ},/"  data.grdchk
  sed -i "s/^ iGloTile .*/ iGloTile       = ${PTI},/" data.grdchk
  sed -i "s/^ jGloTile .*/ jGloTile       = ${PTJ},/" data.grdchk

  for EPS in ${AF_GRDCHK_EPS_LIST}; do
    TAG="${LABEL}_eps${EPS//./p}"
    echo "=== ${LABEL}  local(i=${PI},j=${PJ}) tile(${PTI},${PTJ})  eps=${EPS} ==="
    sed -i "s/^ grdchk_eps .*/ grdchk_eps     = ${EPS},/" data.grdchk
    srun --mpi=pmix -n 4 ./mitgcmuv_ad > "${RESULTS}/${TAG}.runlog" 2>&1 || {
      echo "run failed: ${TAG}"; tail -40 "${RESULTS}/${TAG}.runlog"; exit 1; }
    # STDOUT.0000 carries the grad-res block and is clobbered by the next run.
    cp STDOUT.0000 "${RESULTS}/${TAG}.stdout"
    cp data.grdchk "${RESULTS}/${TAG}.data.grdchk"
    grep -E "grad-res|ADM " "${RESULTS}/${TAG}.stdout" || echo "  (no grad-res block)"
  done
done

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" scripts/verify_gate_g1.py \
  --results "${RUN_DIR}/${RESULTS}" \
  --report  "${RUN_DIR}/gate_g1${AF_GRDCHK_SUFFIX}.json"
