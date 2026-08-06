#!/bin/bash -l
#SBATCH --job-name=af-c-rc-lossv3
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Fine-tune the diagnostic-best Model C checkpoint with the frozen loss-v3:
# projected loss-v1, spatial signed-bias penalty, and one differentiable call
# from forecast ages spanning the complete day-10--90 training curve.
# Validation, inference, response, and adjoint states remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_LOSS_V3_CONTRACT="${AF_LOSS_V3_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_rollout_conditioned_loss_v3.json}"
AF_LOSS_V3_OUTPUT="${AF_LOSS_V3_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/rollout_conditioned_loss_v3}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_rollout_conditioned_loss_v3 run \
  --dataset "${AF_DATASET}" \
  --contract "${AF_LOSS_V3_CONTRACT}" \
  --output-dir "${AF_LOSS_V3_OUTPUT}" \
  --device cuda
