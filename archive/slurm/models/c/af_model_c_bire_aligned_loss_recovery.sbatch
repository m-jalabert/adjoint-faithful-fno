#!/bin/bash -l
#SBATCH --job-name=af-c-bire-lossrec
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Architecture-fixed loss-recovery control.  Keeps every architectural and
# optimizer choice of the working Bire-aligned 5e-4 arm -- three FNO blocks,
# 24x16 modes, width 128, six pointwise LayerNorms, Bire PosEmbed, no external
# 3x3 branch, 10% padding, Adam(5e-4, betas 0.9/0.95, wd 0), batch 8, 7,680
# steps -- and changes only the objective and the rollout exposure back to the
# incumbent three-step group-balanced Model C loss v1.
#
# It answers whether the bounded behaviour comes from the Bire-aligned
# architecture while the loss of skill comes from the Bire objective.
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
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_aligned_loss_recovery_v1.json}"
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
  -m bire_repro.af_model_c_bire_aligned_loss_recovery \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_loss_recovery \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
