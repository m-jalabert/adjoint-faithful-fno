#!/bin/bash -l
#SBATCH --job-name=turb-figmem
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:20:00
set -euo pipefail
module purge; module load gnu12/12.2.0 cuda/12.4
R=/home/mjalabert314/bire_james25_repro
export PYTHONPATH="${R}/turbulent_double_gyre/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "$R"
"${R}/.venv/bin/python" turbulent_double_gyre/scripts/probe_figure_rollout_memory.py
"${R}/.venv/bin/python" -m turbfno.figures preflight --contract turbulent_double_gyre/config/turb_forward_control_v1_s0_turb_figures_v1.json
