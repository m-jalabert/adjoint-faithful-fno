#!/bin/bash -l
#SBATCH --job-name=af-c-bias-proj
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Diagnose teacher-forced slow-field bias and score post-hoc conservation,
# static-bias, and damped-persistence alternatives on the frozen 540-record
# split-1 rollout set. This job does not train and opens no held state archive.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_BIAS_PROJECTION_CONTRACT="${AF_BIAS_PROJECTION_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_slow_field_bias_projection_v1.json}"
AF_BIAS_PROJECTION_OUTPUT="${AF_BIAS_PROJECTION_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/slow_field_bias_projection_v1}"
AF_BIAS_PROJECTION_FIGURES="${AF_BIAS_PROJECTION_FIGURES:-${AF_PROJECT_ROOT}/outputs/af_fno/C/slow_field_bias_projection_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_slow_field_bias_projection \
  --dataset "${AF_DATASET}" \
  --contract "${AF_BIAS_PROJECTION_CONTRACT}" \
  --output-dir "${AF_BIAS_PROJECTION_OUTPUT}" \
  --figure-dir "${AF_BIAS_PROJECTION_FIGURES}" \
  --device cuda
