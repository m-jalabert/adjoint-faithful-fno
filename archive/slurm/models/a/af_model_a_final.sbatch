#!/bin/bash -l
#SBATCH --job-name=af-model-a-final
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Frozen M6 realization selected from development job 284152: exactly ten
# epochs, batch eight, lr 0.001, AdamW beta=(0.9,0.95), wd=1e-5.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_A_OUTPUT="${AF_MODEL_A_OUTPUT:-${AF_SCRATCH_ROOT}/models/A/final_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_a final \
  --dataset "${AF_DATASET}" \
  --output-dir "${AF_MODEL_A_OUTPUT}" \
  --device cuda
