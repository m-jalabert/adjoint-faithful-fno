#!/bin/bash -l
#SBATCH --job-name=af-turb
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=slurm/mitgcm/logs/af-turb-%x-%j.out

# One ten-model-year segment of a 0.25-degree turbulent wind regime.
#
# The 248 x 248 grid is decomposed 8 x 8, so every rank owns the same 31 x 31
# tile the validated 1-degree configuration used on four ranks.  With deltaT =
# 300 s a model year is 103,680 steps.
#
# Two settings here are performance-critical and were measured, not guessed:
#
#   --exclusive   The surface-pressure solve does a global reduction per
#                 iteration, so the run goes only as fast as its slowest rank.
#                 On shared nodes the same benchmark ran 11 ms/step one moment
#                 and 620 ms/step the next; exclusive nodes hold it at 7 ms.
#   PSM2          The Omni-Path fabric is reached through the psm2 MTL.  Left to
#                 its default transport the cross-node reduction costs about
#                 30 per cent more.  The fabric is probed below and the pinning
#                 is skipped if it is absent, because pinning to a transport
#                 that is not there fails the job outright.
#
# Measured cost is roughly 750 s per model-year of spin-up plus daily-output
# overhead in production, so a ten-year segment is 2-4 hours; the eight-hour
# limit is headroom, not an estimate.
#
# Segments chain through MITgcm pickups and are restart-safe: a completed
# segment is a no-op, a partially written one is refused rather than overwritten.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

if srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
     test -e /sys/class/infiniband/hfi1_0 2>/dev/null; then
  export OMPI_MCA_pml=cm
  export OMPI_MCA_mtl=psm2
  echo "using psm2 over Omni-Path"
else
  echo "Omni-Path not found on every node; leaving the MPI transport at its default"
fi

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_turb/mitgcmuv}"
: "${AF_REGIME:?Set AF_REGIME to S0_turb, S1_turb or S2_turb}"
: "${AF_PHASE:?Set AF_PHASE to spinup or production}"
: "${AF_START_YEAR:?Set AF_START_YEAR}"
: "${AF_YEARS:?Set AF_YEARS}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/af_turb_trajectories.py" run-segment \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --executable "${AF_EXECUTABLE}" \
  --regime "${AF_REGIME}" \
  --phase "${AF_PHASE}" \
  --start-year "${AF_START_YEAR}" \
  --years "${AF_YEARS}"
