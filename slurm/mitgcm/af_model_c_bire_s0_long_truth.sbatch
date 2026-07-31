#!/bin/bash -l
#SBATCH --job-name=af-c-s0-long
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# Six continuous S0 years used only as unseen MITgcm truth for the frozen
# 15-member, dt=10-day, day-2000 Bire-style Model C evaluation.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_s0_long_truth_v1.json}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_bire_s0_long_truth \
  run \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --contract "${AF_CONTRACT}"
