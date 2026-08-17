#!/bin/bash -l
#SBATCH --job-name=af-c-sgate
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Inference-only ablation of the local 3x3 branch on the selected 32x32
# checkpoint: z = G(x) + gamma L(x) for gamma = 1, 0.5, 0. No training, no
# selection, no promotion, and nothing on disk is rewritten -- gamma is applied
# by scaling the bias-free local.weight in memory on a freshly loaded model.
# gamma = 1 must reproduce the published 32x32 report or the script aborts.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_rollout_ft_y32_x32_s0_figures_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_OUTPUT="${AF_OUTPUT:-${AF_PROJECT_ROOT}/outputs/af_fno/C/local_branch_spatial_gate_ablation_v1}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" scripts/ablate_local_branch_spatial.py \
  --contract "${AF_CONTRACT}" \
  --output-dir "${AF_OUTPUT}" \
  --device cuda
