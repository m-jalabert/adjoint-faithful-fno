#!/bin/bash -l
#SBATCH --job-name=af-indep
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# One ten-model-year segment of an independently equilibrated wind regime.
# Each regime starts from the original MITgcm tutorial initial condition and
# carries its own wind amplitude through the full 100-year spin-up, then
# produces 25 years of daily output.  Segments chain through MITgcm pickups and
# are restart-safe: a completed segment is a no-op, a partial one is refused.
#
# Measured cost is about 700 s per ten model-years on four MPI ranks.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_s0/mitgcmuv}"
: "${AF_REGIME:?Set AF_REGIME to S0, S1 or S2}"
: "${AF_PHASE:?Set AF_PHASE to spinup or production}"
: "${AF_START_YEAR:?Set AF_START_YEAR}"
: "${AF_YEARS:?Set AF_YEARS}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_independent_wind_trajectories \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --executable "${AF_EXECUTABLE}" \
  --regime "${AF_REGIME}" \
  --phase "${AF_PHASE}" \
  --start-year "${AF_START_YEAR}" \
  --years "${AF_YEARS}"
