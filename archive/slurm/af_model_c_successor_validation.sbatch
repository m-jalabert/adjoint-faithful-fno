#!/bin/bash -l
#SBATCH --job-name=af-c-v2-valid
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00

# Apply the prospectively frozen, Bire-aligned 10--90-day gate to all three
# Model C seeds on the fresh trajectory-v2 validation block.  This job may read
# split codes 1 and 2 only.  Inference, intermediate winds, response data, and
# adjoints remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_QUALITY_REPORT="${AF_QUALITY_REPORT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.quality.json}"
AF_SUCCESSOR_VALIDATION_CONTRACT="${AF_SUCCESSOR_VALIDATION_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_successor_validation_v2.json}"
AF_SUCCESSOR_VALIDATION_OUTPUT="${AF_SUCCESSOR_VALIDATION_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/successor_validation_v1/fresh_validation}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_successor_validation evaluate \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY_REPORT}" \
  --contract "${AF_SUCCESSOR_VALIDATION_CONTRACT}" \
  --output-dir "${AF_SUCCESSOR_VALIDATION_OUTPUT}" \
  --device cuda
