#!/bin/bash -l
#SBATCH --job-name=af-c-trunc3
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Continue from the diagnostic-best duration checkpoint with one frozen
# objective change: supervise three consecutive endpoints while
# backpropagating through three model calls. Validation, inference, response,
# and adjoint states remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_TRUNCATED_CONTRACT="${AF_TRUNCATED_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_truncated_unroll_objective_v1.json}"
AF_TRUNCATED_OUTPUT="${AF_TRUNCATED_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/truncated_unroll_objective_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_truncated_unroll_objective run \
  --dataset "${AF_DATASET}" \
  --contract "${AF_TRUNCATED_CONTRACT}" \
  --output-dir "${AF_TRUNCATED_OUTPUT}" \
  --device cuda
