#!/bin/bash -l
#SBATCH --job-name=af-c-3l-eval
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Evaluation-only recovery of the seven checkpoints fully trained by job
# 304526. No retraining and no validation, held, long-term, response, or
# adjoint archive access.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_anomaly_direct_three_layer_no_padding_recovery_v2.json}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_anomaly_direct_three_layer_no_padding_recovery \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
