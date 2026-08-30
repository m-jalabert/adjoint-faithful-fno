#!/bin/bash -l
#SBATCH --job-name=af-c-gate-m1
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Execution step 14 of docs/Adjoint_faithful_response_training_plan.md:
# Gate M1. Scores all six published selected checkpoints (arms B and C, three
# paired seeds) on held-out forward-response validation, then applies section
# 16.3's forward-preservation and response-effect criteria to each B/C pair.
#
# Strictly post-selection: the production selector already ran inside each
# training job on nominal validation only, and every number this produces
# comes afterwards, so none of it can have influenced a checkpoint choice.
# Reads only the validation role of the curated response store and the six
# published checkpoints -- no adjoint, TAF, FNO adjoint map, blind response
# case, or nested final-inference start.
#
# Write-once: refuses to overwrite an existing gate_m1_result.json.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"
export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" scripts/apply_gate_m1.py --device cuda
