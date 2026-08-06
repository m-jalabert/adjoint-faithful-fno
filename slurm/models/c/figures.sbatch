#!/bin/bash -l
#SBATCH --job-name=af-c-bproftfigs
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Publish the six Bire-style S0 Figure 3-8 plots for the six-step rollout
# fine-tune on S0's inference set (6200-6999), with lead-matched truth through
# day 2000 supplied by days 7200-8999 -- which the model never saw in any
# capacity -- then evaluate the 2,000-day half of the final acceptance gate.
#
# Figure 6 is a literal pre-train / fine-tune pair: black is the step-15,360
# checkpoint the fine-tune started from, red is the selected fine-tuned
# checkpoint.  The two carry different objective hashes (v1 over three steps
# against the six-step contract) and each is verified against its own.
#
# `finalize` fills the four contract fields that cannot be known until training
# has run -- the selected optimizer step and the three artifact hashes -- from
# the training run's own report.  It is idempotent and refuses to overwrite a
# field that is already filled with a different value, so it is safe to run
# before every figure job.
#
# Held-evaluation only: no training, no checkpoint selection, no promotion.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_rollout_ft_s0_figures_v2.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" \
  -m oceanfno.figures \
  finalize \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m oceanfno.figures \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m oceanfno.figures \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
