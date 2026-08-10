#!/bin/bash -l
#SBATCH --job-name=af-c-y32
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Retained successor to the local24 arm: Fourier modes (Y,X) 24x24 -> 32x24.
# The trained bias-free 49-to-46 local 3x3 branch is copied unchanged. The old
# spectral tensors are centered in indices 4:28 of the enlarged Y axis, with
# four zero modes on either side, so the migrated map is initially exact.
# Data, normalization, six-step loss, optimizer schedule, seed, validation
# starts, and checkpoint selection remain inherited from local24.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_rollout_ft_local24_y32_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  echo "The /sw software mount was repointed and anaconda-3.12 is gone;" >&2
  echo "restore a Python 3.12 and repoint or rebuild .venv before submitting." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" \
  -m oceanfno.train \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m oceanfno.train \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
