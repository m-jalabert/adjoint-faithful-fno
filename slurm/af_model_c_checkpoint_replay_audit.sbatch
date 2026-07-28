#!/bin/bash -l
#SBATCH --job-name=af-c-ckpt-replay
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Replay the accepted width-128 reference training exactly, save its six
# predeclared late checkpoints, and score them only on split-1 10--90-day
# rollouts. Validation, inference, later wind, response, and adjoint data stay
# sealed. The job decides checkpoint-selection-only versus objective correction.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_CHECKPOINT_AUDIT_CONTRACT="${AF_CHECKPOINT_AUDIT_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_checkpoint_replay_audit_v1.json}"
AF_CHECKPOINT_AUDIT_OUTPUT="${AF_CHECKPOINT_AUDIT_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/checkpoint_replay_audit_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_checkpoint_replay_audit run \
  --dataset "${AF_DATASET}" \
  --contract "${AF_CHECKPOINT_AUDIT_CONTRACT}" \
  --output-dir "${AF_CHECKPOINT_AUDIT_OUTPUT}" \
  --device cuda
