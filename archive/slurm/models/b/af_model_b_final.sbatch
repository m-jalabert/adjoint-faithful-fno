#!/bin/bash -l
#SBATCH --job-name=af-model-b-final
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Frozen from full-loss development job 284306: exactly 12 epochs. The code
# verifies this value and the complete loss hash against that report.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_B_OUTPUT="${AF_MODEL_B_OUTPUT:-${AF_SCRATCH_ROOT}/models/B/final_v1}"
AF_MODEL_B_DEVELOPMENT_REPORT="${AF_MODEL_B_DEVELOPMENT_REPORT:-${AF_SCRATCH_ROOT}/models/B/development_full_v1/model_b_development_report.json}"
AF_MODEL_B_FINAL_EPOCHS="${AF_MODEL_B_FINAL_EPOCHS:-12}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_b final \
  --dataset "${AF_DATASET}" \
  --output-dir "${AF_MODEL_B_OUTPUT}" \
  --development-report "${AF_MODEL_B_DEVELOPMENT_REPORT}" \
  --epochs "${AF_MODEL_B_FINAL_EPOCHS}" \
  --device cuda
