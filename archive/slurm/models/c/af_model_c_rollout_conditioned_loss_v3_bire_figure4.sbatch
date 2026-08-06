#!/bin/bash -l
#SBATCH --job-name=af-c-v3-fig4
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Apply the selected loss-v3 checkpoint and its exact output projection to the
# prospectively fixed 15-member S2 Figure-4 protocol. No inference or later
# archive is opened.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_FIGURE_CONTRACT="${AF_FIGURE_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_rollout_conditioned_loss_v3_bire_figure4_v1.json}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.analysis.af_model_c_rollout_conditioned_loss_v3_bire_figure4 \
  run \
  --contract "${AF_FIGURE_CONTRACT}" \
  --device cuda
