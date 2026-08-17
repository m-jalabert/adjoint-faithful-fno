#!/bin/bash -l
#SBATCH --job-name=af-c-2in1out
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00

# Successor to the 32x32 one-input arm: temporal context only.
#
#   one-in / one-out:   x_t             -> x_(t+10)
#   two-in / one-out:  (x_(t-10), x_t)  -> x_(t+10)
#
# and autoregression slides the pair forward, so the operator sees an empirical
# tendency (x_t - x_(t-10)) / 10 days at every call. The external input block
# grows 49 -> 95 and lifting 51 -> 97; the two input-facing tensors gain 46
# leading channels that start at exact zero, so the warm start ignores the
# history state and reproduces the parent's map. Fourier modes stay at 32x32,
# and data, normalization, six-step loss, optimizer schedule, seed, validation
# starts and checkpoint selection remain inherited from the 32x32 arm.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_2in_1out_v1.json}"
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
