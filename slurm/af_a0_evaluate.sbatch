#!/bin/bash -l
#SBATCH --job-name=af-a0-evaluate
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Project-facing figures and numerical metrics for the frozen A0 checkpoint.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_A0_CHECKPOINT="${AF_A0_CHECKPOINT:-${AF_SCRATCH_ROOT}/models/A0/final_v1/a0_final.pt}"
AF_A0_EVALUATION_OUTPUT="${AF_A0_EVALUATION_OUTPUT:-${AF_PROJECT_ROOT}/outputs/af_fno/A0/final_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_a0_evaluate \
  --dataset "${AF_DATASET}" \
  --checkpoint "${AF_A0_CHECKPOINT}" \
  --output-dir "${AF_A0_EVALUATION_OUTPUT}" \
  --device cuda
