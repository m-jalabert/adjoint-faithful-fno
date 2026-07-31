#!/bin/bash -l
#SBATCH --job-name=af-c-bire-aligned
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Bire-aligned full-state FNO over the retained 46-channel closed state:
# three FNO blocks, six pointwise LayerNorms, the exact oceanfourcast
# sine/cosine position fields appended immediately before lifting, no external
# 3x3 raw-input branch, MSE + 0.01 MAE on wet cells, and Bire's one-step
# pretraining followed by two-step autoregressive fine-tuning under
# Adam(1e-2, betas 0.9/0.95, no weight decay) at batch size 8.
#
# Training split only.  The held S0 figure suite opens through its own
# contract.  Inference, intermediate-wind, response, and adjoint archives
# remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_aligned_full_state_v1.json}"
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
  -m bire_repro.af_model_c_bire_aligned_full_state \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_full_state \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
