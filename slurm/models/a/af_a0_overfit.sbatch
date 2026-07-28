#!/bin/bash -l
#SBATCH --job-name=af-a0-overfit
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# M5 entry gate: exact adapted-Bire A0 architecture, 96 stratified training
# pairs, 10-day normalized next-state target, and bitwise save/reload check.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_A0_OUTPUT="${AF_A0_OUTPUT:-${AF_SCRATCH_ROOT}/models/A0/overfit_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_a0 overfit \
  --dataset "${AF_DATASET}" \
  --output-dir "${AF_A0_OUTPUT}" \
  --samples 96 \
  --epochs 160 \
  --batch-size 4 \
  --device cuda
