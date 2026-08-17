#!/bin/bash -l
#SBATCH --job-name=af-s0-adj-pickup
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# Stage 6 of docs/mitgcm_adjoint_ground_truth_plan.md, plus acceptance gate G0.
#
# Restarts the FORWARD executable from the archived day-7200 pickup and
# integrates 1440 steps to day 7220.  This produces three things:
#
#   pickup.0003111120  day 7210 -- the initial condition Run A starts from
#   pickup.0003111840  day 7220
#   dynState/surfState daily snapshots for days 7200..7220 -- gate G0
#
# The partition, rank count and 2x2 decomposition match the ones the whole S0
# control was integrated on, so the restart and the archive share the same node
# and arithmetic.  The twin study established that this matters; G0 is
# bit-for-bit and will not forgive a mismatch.
#
# Needs no TAF licence: this is the forward model.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1}"

cd "${AF_PROJECT_ROOT}"

"${AF_PROJECT_ROOT}/.venv/bin/python" scripts/stage_adjoint_run.py pickup \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --force

RUN_DIR="${AF_SCRATCH_ROOT}/pickup"
cd "${RUN_DIR}"
srun --ntasks=4 ./mitgcmuv > run.log 2>&1

# The pickups Run A and Run B depend on must exist before anything downstream.
for ITER in 0003111120 0003111840; do
  test -f "pickup.${ITER}.data" || { echo "MISSING pickup.${ITER}.data"; tail -40 run.log; exit 1; }
done

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" scripts/verify_gate_g0.py \
  --run-dir "${RUN_DIR}" \
  --report  "${RUN_DIR}/gate_g0.json"
