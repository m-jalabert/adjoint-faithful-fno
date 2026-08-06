#!/bin/bash -l
#SBATCH --job-name=af-c-bire-faith
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Bire-faithful protocol arm for the Bire-aligned full-state FNO.
# Corrects, as one bundle on the working 5e-4 base, the three unintended
# divergences from the public oceanfourcast implementation:
#   MAE weight            0.01 -> 0.05
#   LR schedule           step decay -> CosineAnnealingLR(T_max 3, eta_min 1e-5)
#   checkpoint selection  fixed steps -> lowest validation loss within each stage
# ChannelMLP dropout stays at zero; the repository default of 0.5 is a
# regulariser the paper does not specify and belongs to its own arm.
#
# Validation is a seeded random 10% of split-1 training records, so no sealed
# archive is opened.  Training split only; the held S0 figure suite opens
# through its own contract.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_aligned_faithful_v1.json}"
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
  -m bire_repro.af_model_c_bire_aligned_faithful \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_faithful \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
