#!/bin/bash -l
#SBATCH --job-name=af-c-response-train
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Execution step 13 of docs/Adjoint_faithful_response_training_plan.md: arm C,
# the response-aware model, trained from random initialization for all three
# paired seeds. Submit three times, once per AF_SEED in {20260724, 20260911,
# 20260912}; 20260724 is the primary seed.
#
# C is arm B plus exactly one term: lambda_resp * L_response, with
# lambda_resp = 0.001 frozen by the v2 screen
# (config/forward_response_lambda_screen_v2.json, selected 2026-08-28). The
# nominal path -- initialization, batch order, normalizer, rollout, optimizer
# schedule, parent loss, checkpoint cadence and selector -- is identical to
# B's by construction, which is what makes the paired B run a valid matched
# control. The section-15.2 auxiliary stream mixes in one response direction
# on every 4th optimizer step (1,920 updates: 1,440 short, 480 long) from a
# schedule keyed by an isolated counter/hash stream, so it cannot perturb the
# parent RNG.
#
# Response metrics take no part in checkpoint selection: the run applies the
# unchanged production selector on nominal validation only (plan section
# 16.3). Gate M1 is applied afterwards, at step 14, not here.
#
# Expect ~3.7 h per seed on a V100, from the lambda=1e-3 v2 screen run, which
# is the computationally identical job.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_adjoint_faithful_response_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_SEED="${AF_SEED:?Set AF_SEED to one of 20260724 (primary), 20260911, or 20260912}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" -m oceanfno.train_response preflight --contract "${AF_CONTRACT}" --seed "${AF_SEED}"
"${AF_PYTHON}" -m oceanfno.train_response run --contract "${AF_CONTRACT}" --device cuda --seed "${AF_SEED}"
