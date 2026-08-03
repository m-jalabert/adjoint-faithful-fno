#!/bin/bash -l
#SBATCH --job-name=af-c-chrono
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

# Loss-recovery model retrained from scratch under a strictly chronological
# split:  train 0-5039  >  buffer  >  validation 5130-5759  >  buffer  >
# test 5850-7199.  The model is identical to
# model_c_bire_aligned_loss_recovery_v1 (same seed, three FNO blocks, six
# pointwise LayerNorms, 24x16 modes, width 128, Bire PosEmbed, 10% padding, no
# local branch, Model C loss v1, three-step rollout, Adam 5e-4 -> 1e-4 at 75%,
# batch 8, 7,680 steps).
#
# Every train-derived statistic is recomputed from indices 0-5039 only, and
# checkpoints are selected on 90 held 360-day validation rollouts.
#
# Training and validation only.  The test block and the held S0 archive stay
# sealed; the figure suite opens through its own contract.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_aligned_loss_recovery_chronological_v1.json}"
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
  -m bire_repro.af_model_c_bire_aligned_chronological \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_chronological \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
