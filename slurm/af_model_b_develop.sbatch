#!/bin/bash -l
#SBATCH --job-name=af-model-b-develop
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# M6 incremental development profile. Set AF_MODEL_B_PROFILE to rollout,
# rollout_spectral, or full and give each profile a distinct output directory.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_B_PROFILE="${AF_MODEL_B_PROFILE:-full}"
AF_MODEL_B_OUTPUT="${AF_MODEL_B_OUTPUT:-${AF_SCRATCH_ROOT}/models/B/development_${AF_MODEL_B_PROFILE}_v1}"

case "${AF_MODEL_B_PROFILE}" in
  rollout|rollout_spectral|full) ;;
  *) echo "invalid AF_MODEL_B_PROFILE=${AF_MODEL_B_PROFILE}" >&2; exit 2 ;;
esac

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_b develop \
  --dataset "${AF_DATASET}" \
  --output-dir "${AF_MODEL_B_OUTPUT}" \
  --profile "${AF_MODEL_B_PROFILE}" \
  --device cuda
