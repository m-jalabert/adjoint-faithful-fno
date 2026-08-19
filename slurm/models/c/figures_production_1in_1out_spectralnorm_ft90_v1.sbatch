#!/bin/bash -l
#SBATCH --job-name=af-c-ft90-fig
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# S0 Figures 3-8 for the ninety-day fine-tune's selected checkpoint, plus the
# 2,000-day half of its acceptance gate. Held evaluation only: the same 15
# members drawn from the inference block by the same seed the parent used,
# autoregressed to day 2,000 at ten-day steps.
#
# The protocol is the parent's to the member, deliberately: the point of this
# package is a like-for-like comparison with
# model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1, and a fine-tune
# scored on different starts would not be comparable to the thing it fine-tuned.
#
# What is being asked. At day 2,000 the parent reaches a maximum normalized
# magnitude of 10.33 against a declared ceiling of 8, and its day-2,000 RMSE
# divided by climatology's is 0.48 / 1.43 / 1.76 in pressure / SST / speed --
# worse than predicting the training mean in two of three fields. A model that
# has settled onto the right attractor should sit near or below one there. The
# parent's numbers are read from its sealed figure summary, pinned by digest,
# and written beside this arm's own in the acceptance gate, so the comparison is
# never made from memory.
#
# The plates themselves carry a single model curve, exactly as the parent's do:
# adding a comparator series would mean editing the frozen plot definitions in
# plots.py, which the parent's contract also pins.
#
# `finalize` stamps the selected optimizer step and the fine-tuning digests into
# the contract from the fine-tuning report; run it only after that job has
# completed. It is idempotent and refuses to overwrite a field that already
# disagrees, so it is safe to leave in the script.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_production_1in_1out_spectralnorm_ft90_v1_s0_figures_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" -m oceanfno.figures_ft90 finalize  --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures_ft90 preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures_ft90 run       --contract "${AF_CONTRACT}" --device cuda
