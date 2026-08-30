#!/bin/bash -l
#SBATCH --job-name=af-c-lambda-screen
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00

# Execution step 12 of docs/Adjoint_faithful_response_training_plan.md: one
# candidate of the four-lambda, 1,920-step primary-seed forward-only screen
# (plan section 14.4, config/forward_response_lambda_screen_v1.json). Submit
# four times, once per AF_LAMBDA_RESP in {0.03, 0.10, 0.30, 1.00}. Loads arm
# B's already-published, hash-verified normalizer rather than recomputing it
# (identical initialization/batch order already pins the recipe and result),
# so this is faster than a from-scratch run of the same step count.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_LAMBDA_RESP="${AF_LAMBDA_RESP:?Set AF_LAMBDA_RESP to one of 0.03, 0.10, 0.30, 1.00}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" scripts/run_lambda_screen.py --lambda-resp "${AF_LAMBDA_RESP}" --device cuda
