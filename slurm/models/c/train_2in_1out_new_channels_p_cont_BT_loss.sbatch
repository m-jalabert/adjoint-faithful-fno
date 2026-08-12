#!/bin/bash -l
#SBATCH --job-name=af-c-bt
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00

# Loss-only fine-tune of model_c_2in_1out_new_channels_pressure_gradient_continuity_v1.
# Architecture/data/normalization/rollout remain unchanged; Adam is reset, the
# pressure-gradient and continuity terms are retained, and a depth-integrated
# barotropic transport-tendency loss is added at weight 0.05.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_2in_1out_new_channels_p_cont_BT_loss_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" -m oceanfno.train preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.train run --contract "${AF_CONTRACT}" --device cuda
