#!/bin/bash -l
#SBATCH --job-name=af-c-lambda-screen-v2
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Execution step 12 of docs/Adjoint_faithful_response_training_plan.md, second
# pass: the four-lambda, full-length 7,680-step primary-seed forward-only
# screen (config/forward_response_lambda_screen_v2.json). Submit four times,
# once per AF_LAMBDA_RESP in {0.0003, 0.001, 0.003, 0.01}.
#
# Differs from lambda_screen_v1.sbatch in two frozen constants and one
# correctness fix, all set out in the plan's "Implementation status and
# amendments (2026-08-27, step 12 re-screen)": the grid moved down (v1's
# smallest candidate already carried ~26% of the raw training signal), the
# budget and matched arm B step moved to 7,680 (at fixed lambda the auxiliary
# share grows 3.3x across the run, so step 1,920 gates the weight at its
# mildest), and the script now applies section 15.1's step-5,761 decay, which
# a 1,920-step screen never reached.
#
# Four candidates are independent; submit them together. Each is a full C-run
# budget, so allow ~4-10 h depending on shared-cluster load.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_LAMBDA_RESP="${AF_LAMBDA_RESP:?Set AF_LAMBDA_RESP to one of 0.0003, 0.001, 0.003, 0.01}"
AF_SCREEN_CONTRACT="${AF_SCREEN_CONTRACT:-config/forward_response_lambda_screen_v2.json}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" scripts/run_lambda_screen.py \
  --lambda-resp "${AF_LAMBDA_RESP}" \
  --contract "${AF_SCREEN_CONTRACT}" \
  --device cuda
