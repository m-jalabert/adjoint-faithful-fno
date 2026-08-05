#!/bin/bash -l
#SBATCH --job-name=af-c-bproft
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Six-step rollout fine-tune of the Bire-protocol step-15,360 checkpoint.
#
# Warm start from
#   .../models/C/bire_protocol_duration_v1/selected.pt
# loading model_state_dict only; the optimizer is reset.  The train-only
# pointwise normalizers are REUSED from that package rather than recomputed, so
# the fine-tuned weights stay in exactly the normalized coordinates they start
# in.
#
# What moves, and it is one change:
#   rollout_steps  3 -> 6   (t -> t+10 -> ... -> t+60, prediction fed back from
#                            the second call; no teacher forcing after the
#                            initial state)
#   rollout_weight 0.15 -> 0.50
#   learning rate  5e-4 -> 2e-5, decaying to 4e-6 at step 2,880
#   batch          8 -> 4   (4 x 6 = 8 x 3, so activation memory is unchanged)
#   budget         3,840 steps, checkpoints at 960 / 1,920 / 2,880 / 3,840
#
# Architecture, Fourier modes, static inputs, positional encoding, 46-channel
# output, the Bire Section 3.2 split and the selection rule are asserted field
# by field against the parent contract at load time.
#
# Training draws 5,940 starts per regime (17,820 pooled); the latest start is
# 5,939 so every six-step target sequence stays inside training 0-5999.
#
# Budget: the parent ran 15,360 steps at batch 8 x 3 calls in about 1h35m
# including validation.  This arm runs a quarter of the steps at the same
# per-step activation load, plus the same four-checkpoint validation pass.
#
# Training and validation only; the inference set opens through the figure
# contract, S0 only.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_rollout_ft_v1.json}"
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
  -m bire_repro.af_model_c_bire_protocol_rollout_ft \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_protocol_rollout_ft \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
