#!/bin/bash -l
#SBATCH --job-name=af-c-gate-m2
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Execution step 16 (docs/Adjoint_faithful_response_training_plan.md section
# 17): the blind forward-response test. Scores frozen parent A, the ft90 child
# as context, and all six paired B/C selected checkpoints on the same 216
# blind directions -- day 10 for all, days 20..90 for the 36-direction long
# subset -- then applies section 17's seven conditions and freezes the verdict.
#
# Write-once: refuses to overwrite an existing gate_m2_result.json, because
# section 17 opens this package once.
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

"${AF_PYTHON}" scripts/apply_gate_m2.py --device cuda
