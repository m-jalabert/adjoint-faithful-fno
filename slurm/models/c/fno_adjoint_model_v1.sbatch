#!/bin/bash -l
#SBATCH --job-name=af-fno-adjoint
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Execution step 17 (docs/Adjoint_faithful_response_training_plan.md sections
# 18.2 and 22/Gate A0): the trusted FNO derivative machinery for one model of
# {A, B_<seed>, C_<seed>, ft90}, via scripts/fno_adjoint_model.py.
#
# Every gate and objective, and in particular the validated complex128
# spectral-buffer promotion, is the ft90 runner's own code executed unchanged;
# only the model identity differs. Runs on CPU in float64 by design -- the
# adjoint identities are precision-sensitive and this project has already lost
# one signal to a float32 quantisation floor.
#
# These checks occur only after model freeze and cannot affect model decisions.
set -euo pipefail

module purge
module load gnu12/12.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_ADJOINT_MODEL="${AF_ADJOINT_MODEL:?Set AF_ADJOINT_MODEL to A, B_<seed>, C_<seed> or ft90}"

cd "${AF_PROJECT_ROOT}"
export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

"${AF_PYTHON}" scripts/fno_adjoint_model.py \
  --model "${AF_ADJOINT_MODEL}" --threads "${SLURM_CPUS_PER_TASK}"
