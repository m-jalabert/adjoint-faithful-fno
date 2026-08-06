#!/bin/bash -l
#SBATCH --job-name=af-c-chrono-figs
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Publish the same six Bire-style S0 Figure 3-8 plots for the
# validation-selected checkpoint of the chronological-split arm, on the same 15
# starts in 6660-7199 that are held out under BOTH the stored and the
# chronological split.  The climatology baseline is rebuilt from the
# chronological training interval 0-5039, because the frozen runner would
# otherwise derive it from stored split-1 codes that include test days.
#
# Held-evaluation only: no training, no checkpoint selection, no promotion.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_aligned_chronological_s0_figures_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_STAGE="${AF_STAGE:-selected}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_chronological_s0_figures \
  preflight \
  --contract "${AF_CONTRACT}" \
  --stage "${AF_STAGE}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_aligned_chronological_s0_figures \
  run \
  --contract "${AF_CONTRACT}" \
  --stage "${AF_STAGE}" \
  --device cuda
