#!/bin/bash -l
#SBATCH --job-name=af-c-pushforward
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# One bounded objective correction after the exact checkpoint replay.  The
# width-128 architecture and loss-v1 core stay fixed.  Only a detached SST and
# surface-PHIHYD endpoint term at alternating days 60/90 is added.  All
# validation, inference, response, and adjoint data remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_PUSHFORWARD_CONTRACT="${AF_PUSHFORWARD_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_pushforward_objective_v1.json}"
AF_PUSHFORWARD_OUTPUT="${AF_PUSHFORWARD_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/pushforward_objective_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_pushforward_objective run \
  --dataset "${AF_DATASET}" \
  --contract "${AF_PUSHFORWARD_CONTRACT}" \
  --output-dir "${AF_PUSHFORWARD_OUTPUT}" \
  --device cuda
