#!/bin/bash -l
#SBATCH --job-name=af-c-arm-r
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Arm R changes only the selected Model C dynamical channel task: 46 outputs
# become the ten recovered Bire-facing outputs.  Phase 1 trains and selects
# using split-1 state only.  Phase 2 then generates the prospectively fixed
# six S0 Figure 3--8 analogues through day 2000.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_reduced_channel_control_v1.json}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/model_c_reduced_bire_channels_v1.zarr}"
AF_QUALITY="${AF_QUALITY:-${AF_SCRATCH_ROOT}/datasets/model_c_reduced_bire_channels_v1.quality.json}"
AF_TRAINING_OUTPUT="${AF_TRAINING_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/anomaly_direct_reduced_channel_control_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_reduced_channel_control \
  preflight \
  --contract "${AF_CONTRACT}" \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY}" \
  --output-dir "${AF_TRAINING_OUTPUT}"

"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_reduced_channel_control \
  run \
  --contract "${AF_CONTRACT}" \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY}" \
  --output-dir "${AF_TRAINING_OUTPUT}" \
  --device cuda

"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_reduced_channel_s0_figures \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_reduced_channel_s0_figures \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
