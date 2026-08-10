#!/bin/bash -l
#SBATCH --job-name=af-c-y32figs
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Held S0 evaluation of the retained Y32 checkpoint. This reproduces the exact
# six Figure 3--8 definitions, members, starts, truth, climatology, and
# persistence baselines used for local24. Figure 6 compares local24 (black)
# with its 32x24 meridional-mode fine-tune (red). No training or selection.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_rollout_ft_local24_y32_s0_figures_v1.json}"
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
