#!/bin/bash -l
#SBATCH --job-name=af-c-figures-resp
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Execution step 15 of docs/Adjoint_faithful_response_training_plan.md, plan
# section 19 step 2: the established final S0 figure package on the exact 15
# starts of section 6.2, for one B/C arm and seed. Submit six times, once per
# AF_FIGURE_CONTRACT.
#
# Runs through src/oceanfno/figures_response.py, a contract adapter: the
# numerics (evaluate_regime, the climatology, the static block, every plot,
# the summary and the acceptance gate) are imported from the frozen
# src/oceanfno/figures.py and executed unchanged, so each arm's figures are
# produced by exactly the code that produced the parent's. Only the identity
# binding -- which is hard-coded to the parent in the frozen module, and whose
# file is pinned by the parent's own contract and so must not be edited -- is
# re-expressed here.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_FIGURE_CONTRACT="${AF_FIGURE_CONTRACT:?Set AF_FIGURE_CONTRACT to one of the six B/C figure contracts}"

cd "${AF_PROJECT_ROOT}"
export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" -m oceanfno.figures_response finalize  --contract "${AF_FIGURE_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures_response preflight --contract "${AF_FIGURE_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.figures_response run       --contract "${AF_FIGURE_CONTRACT}" --device cuda
