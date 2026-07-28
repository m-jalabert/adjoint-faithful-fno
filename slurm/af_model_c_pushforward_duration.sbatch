#!/bin/bash -l
#SBATCH --job-name=af-c-push-duration
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Replay pushforward v1 exactly through step 1,920, then continue the identical
# objective at its already-decayed learning rate.  The first 1,920-step state
# must reproduce bitwise before extension metrics are trusted.  Validation,
# inference, response, and adjoint states remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_DURATION_CONTRACT="${AF_DURATION_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_pushforward_duration_v1.json}"
AF_DURATION_OUTPUT="${AF_DURATION_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/pushforward_duration_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_pushforward_duration run \
  --dataset "${AF_DATASET}" \
  --contract "${AF_DURATION_CONTRACT}" \
  --output-dir "${AF_DURATION_OUTPUT}" \
  --device cuda
