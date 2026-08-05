#!/bin/bash -l
#SBATCH --job-name=af-c-bpdur
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

# Matched-budget duration control for the Bire Section 3.2 protocol arm.
#
# Exactly one quantity moves against model_c_bire_protocol_pooled_v1:
#   maximum_steps  7,680 -> 15,360
# with checkpoints at the quarters of the new budget (3,840 / 7,680 / 11,520 /
# 15,360) and decay_fraction held at 0.75, so the decay falls at step 11,520
# rather than 5,760 and this run's step-7,680 checkpoint is NOT comparable to
# the parent's final checkpoint.  Everything else -- architecture, Model C loss
# v1 over a three-step rollout, the Bire split, the train-only normalizers,
# seed, batch, learning rate, betas, weight decay, clipping, decay factor, and
# the selection rule -- is asserted field by field against the parent contract.
#
# The parent trained 7,680 steps in 40m24s, so budget on roughly double that
# plus the unchanged validation pass.
#
# Training and validation only; the inference set opens through the figure
# contract, S0 only.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_duration_v1.json}"
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
  -m bire_repro.af_model_c_bire_protocol_duration \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_protocol_duration \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
