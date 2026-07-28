#!/bin/bash -l
#SBATCH --job-name=af-s0
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
: "${AF_S0_PHASE:?Set AF_S0_PHASE to spinup or production}"
: "${AF_S0_START_YEAR:?Set AF_S0_START_YEAR}"
: "${AF_S0_YEARS:?Set AF_S0_YEARS}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_s0 run-segment \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --executable "${AF_EXECUTABLE}" \
  --phase "${AF_S0_PHASE}" \
  --start-year "${AF_S0_START_YEAR}" \
  --years "${AF_S0_YEARS}"
