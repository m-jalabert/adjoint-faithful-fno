#!/bin/bash -l
#SBATCH --job-name=turb-dryrun
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
set -euo pipefail
module purge
module load gnu12/12.2.0 cuda/12.4
R=/home/mjalabert314/bire_james25_repro
export PYTHONPATH="${R}/turbulent_double_gyre/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export AF_DATALOADER_WORKERS=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "$R"
"${R}/.venv/bin/python" turbulent_double_gyre/scripts/verify_rank_write_guards.py
"${R}/.venv/bin/python" turbulent_double_gyre/scripts/dryrun_resume_validation.py
