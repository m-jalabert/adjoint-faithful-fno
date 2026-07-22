#!/bin/bash -l
#SBATCH --job-name=af-model-a-evaluate
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Project-facing numerical and figure outputs for frozen Model A, including a
# direct 10-day comparison to the frozen A0 metrics.  No model is retrained.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_A_CHECKPOINT="${AF_MODEL_A_CHECKPOINT:-${AF_SCRATCH_ROOT}/models/A/final_v1/model_a_final.pt}"
AF_A0_METRICS="${AF_A0_METRICS:-${AF_PROJECT_ROOT}/outputs/af_fno/A0/final_v1/a0_metrics.json}"
AF_MODEL_A_EVALUATION_OUTPUT="${AF_MODEL_A_EVALUATION_OUTPUT:-${AF_PROJECT_ROOT}/outputs/af_fno/A/final_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_a_evaluate \
  --dataset "${AF_DATASET}" \
  --checkpoint "${AF_MODEL_A_CHECKPOINT}" \
  --a0-metrics "${AF_A0_METRICS}" \
  --output-dir "${AF_MODEL_A_EVALUATION_OUTPUT}" \
  --device cuda
