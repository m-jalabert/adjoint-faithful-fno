#!/bin/bash -l
#SBATCH --job-name=af-c-v3-figs
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Publish the six Bire-style Figure 3-8 plots for the pooled trajectory-v3
# model on each regime's held test block: S0 as the primary comparison, S1 and
# S2 as wind-regime robustness.  Truth comes entirely from the v3 store; the
# 2,520-day test block covers every lead so no evaluation-only continuation is
# needed.  Starts are a new fixed draw from 6480-6999.
#
# Held-evaluation only: no training, no checkpoint selection, no promotion.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_aligned_v3_figures_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_v3_figures \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_v3_figures \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
