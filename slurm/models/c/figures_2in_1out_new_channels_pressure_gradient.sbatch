#!/bin/bash -l
#SBATCH --job-name=af-c-pgfig
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail
module purge
module load gnu12/12.2.0 cuda/12.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_2in_1out_new_channels_pressure_gradient_s0_figures_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
cd "${AF_PROJECT_ROOT}"

"${AF_PYTHON}" -m oceanfno.figures_pressure_gradient finalize --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures_pressure_gradient preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures_pressure_gradient run --contract "${AF_CONTRACT}" --device cuda
