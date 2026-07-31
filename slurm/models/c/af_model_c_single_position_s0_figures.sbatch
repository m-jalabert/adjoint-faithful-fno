#!/bin/bash -l
#SBATCH --job-name=af-c-sp-figs
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Publish the same six Bire-style S0 Figure 3-8 plots as
# anomaly_direct_bire_s0_inference_v1, for the single-position LayerNorm arm
# selected at optimizer step 14,880.  Held-evaluation only: no training, no
# checkpoint selection, and no promotion.  Intermediate-wind, response, and
# adjoint archives remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_single_position_layernorm_s0_figures_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_single_position_s0_figures \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_single_position_s0_figures \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
