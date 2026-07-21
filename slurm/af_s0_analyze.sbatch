#!/bin/bash -l
#SBATCH --job-name=af-s0-check
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_ANALYSIS_OUTPUT="${AF_ANALYSIS_OUTPUT:-${AF_PROJECT_ROOT}/outputs/af_fno/S0/tutorial_validation}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_tutorial_analysis \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --output-dir "${AF_ANALYSIS_OUTPUT}"
