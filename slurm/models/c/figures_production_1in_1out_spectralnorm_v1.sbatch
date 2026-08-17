#!/bin/bash -l
#SBATCH --job-name=af-c-prodsn-fig
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# S0 Figures 3-8 for the selected production checkpoint, plus the 2,000-day
# half of the acceptance gate. Held evaluation only: 15 members from the
# inference block, autoregressed to day 2,000 at ten-day steps. There is no
# comparator model, so every plate carries a single model curve against
# persistence and climatology.
#
# `finalize` stamps the selected optimizer step and the checkpoint /
# normalization / report digests into the contract from the training report;
# run it only after the training job has completed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" -m oceanfno.figures finalize --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures run --contract "${AF_CONTRACT}" --device cuda
