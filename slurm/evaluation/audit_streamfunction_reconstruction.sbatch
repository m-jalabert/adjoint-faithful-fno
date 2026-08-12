#!/bin/bash -l
#SBATCH --job-name=af-c-psiaudit
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Inference-only three-way streamfunction reconstruction audit of the retained
# Model C. Retrains nothing, promotes nothing, and rewrites no
# published figure package.
set -euo pipefail
module purge
module load gnu12/12.2.0 cuda/12.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

AF_GROUP="${AF_PROJECT_ROOT}/outputs/af_fno/C"
AF_FIGURES="${AF_GROUP}/model_c_2in_1out_new_channels_p_cont_BT_loss_s0_figures_v1/S0"
AF_OUTPUT="${AF_GROUP}/model_c_2in_1out_new_channels_p_cont_BT_loss_s0_streamfunction_audit_v1/S0"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${AF_PYTHON}" scripts/audit_streamfunction_reconstruction.py \
  --output "${AF_OUTPUT}" \
  --published-arrays "${AF_FIGURES}/model_c_bire_s0_figures_arrays.npz" \
  --device cuda
