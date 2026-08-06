#!/bin/bash -l
#SBATCH --job-name=af-wind
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_s0/mitgcmuv}"
: "${AF_EXPERIMENT:?Set AF_EXPERIMENT to S1 or S2}"
: "${AF_PHASE:?Set AF_PHASE to adjust or production}"
: "${AF_START_YEAR:?Set AF_START_YEAR}"
: "${AF_YEARS:?Set AF_YEARS}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_wind_trajectories \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --executable "${AF_EXECUTABLE}" \
  --experiment "${AF_EXPERIMENT}" \
  --phase "${AF_PHASE}" \
  --start-year "${AF_START_YEAR}" \
  --years "${AF_YEARS}"
