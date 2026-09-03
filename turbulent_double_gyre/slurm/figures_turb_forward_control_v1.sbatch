#!/bin/bash -l
#SBATCH --job-name=turb-figures
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

# The frozen S0_turb evaluation suite: figures 3-8 on the 15-start protocol out
# to day 2,000.  One GPU; the 15-member rollout was measured at 9.6 GiB peak on
# a 31.7 GiB card, so it needs no chunking.
set -euo pipefail
module purge; module load gnu12/12.2.0 cuda/12.4
R="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
STUDY="${R}/turbulent_double_gyre"
PY="${AF_PYTHON:-${R}/.venv/bin/python}"
CONTRACT="${STUDY}/config/turb_forward_control_v1_s0_turb_figures_v1.json"
export PYTHONPATH="${STUDY}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MPLBACKEND=Agg
cd "$R"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"${PY}" -m turbfno.figures preflight --contract "${CONTRACT}"
"${PY}" -m turbfno.figures run --contract "${CONTRACT}" --device cuda
