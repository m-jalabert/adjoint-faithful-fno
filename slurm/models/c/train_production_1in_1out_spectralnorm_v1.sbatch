#!/bin/bash -l
#SBATCH --job-name=af-c-prodsn-train
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Production emulator v3: F([x_t, S]) -> x_(t+10), trained once from random
# initialization. Architecture, data, split, schedule, optimizer, batch, seed
# and the eight original loss terms have not moved since v1. Only the stability
# term changes:
#
#   v1  none.                          growth rate 1.0168, x28 over 200 calls
#   v2  [max(0, g-1)]^2 at 0.01.       growth rate 1.01126, day-2000 anomaly
#                                      RMS ratio 8.93 -> 4.25, no accuracy cost,
#                                      but the term was ~0.003 % of the objective
#   v3  max(0, log g) at 0.1.          ~350x the weighted derivative wrt g
#
# Linear in log because growth over N calls is exp(sum_k log g_k), so the
# penalty attacks the growth exponent and its derivative 1/g does not vanish
# for small violations the way the squared hinge did. Weight 0.1 rather than
# 0.5 deliberately: 0.5 would be ~1760x in one jump and risks making stability
# the dominant objective. The success criterion is the growth rate reaching
# 1.0 or below with 0-360 day skill approximately unchanged.
#
# The run recomputes its own train-only normalization over days 0-5999 of
# S0/S1/S2 before the first optimizer step; that streaming pass reads roughly
# 13 GB from the trajectory store and takes some minutes. The contraction term
# adds one forward pass per microbatch on top of the six-call rollout, and the
# growth-rate diagnostic adds two 200-call rollouts per checkpoint. The
# gradient-share probe adds two backward passes every 120 steps.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_production_1in_1out_spectralnorm_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" -m oceanfno.train preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.train run --contract "${AF_CONTRACT}" --device cuda
