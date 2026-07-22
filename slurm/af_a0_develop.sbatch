#!/bin/bash -l
#SBATCH --job-name=af-a0-develop
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# M5 development gate: the declared A0 architecture is trained only on the
# 7,530 sealed chronological train pairs and assessed on 780 validation pairs.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_A0_OUTPUT="${AF_A0_OUTPUT:-${AF_SCRATCH_ROOT}/models/A0/development_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_a0 develop \
  --dataset "${AF_DATASET}" \
  --output-dir "${AF_A0_OUTPUT}" \
  --epochs 12 \
  --batch-size 8 \
  --device cuda
