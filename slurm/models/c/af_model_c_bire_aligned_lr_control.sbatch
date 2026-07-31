#!/bin/bash -l
#SBATCH --job-name=af-c-bire-lr5e4
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# One-factor learning-rate control for the Bire-aligned full-state FNO.
# The single declared change from model_c_bire_aligned_full_state_v1 is the
# optimizer learning rate, 1e-2 -> 5e-4.  Architecture, inputs, positional
# encoder, objective, two-stage protocol, seed, batch size, betas, weight
# decay, absent gradient clipping, and decay schedule are all frozen against
# the parent, and the contract loader asserts that field by field.
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
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_aligned_full_state_lr5e4_v1.json}"
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
  -m bire_repro.af_model_c_bire_aligned_lr_control \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_lr_control \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
