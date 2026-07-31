#!/bin/bash -l
#SBATCH --job-name=af-c-pos-ln
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-0

# Full 46-channel pointwise-anomaly direct-state Model C with the duplicated
# positional encoding removed and the Bire pointwise LayerNorm enabled.
# Training split only; the held S0 figure suite opens only if the training
# gate passes.  Inference, intermediate-wind, response, and adjoint archives
# remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_anomaly_direct_single_position_layernorm_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  echo "The /sw software mount was repointed and anaconda-3.12 is gone;" >&2
  echo "restore a Python 3.12 and repoint or rebuild .venv before submitting." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PYTHON}" \
  -m bire_repro.af_model_c_anomaly_direct_single_position_layernorm \
  preflight \
  --contract "${AF_CONTRACT}" \
  --arm-index "${SLURM_ARRAY_TASK_ID}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_anomaly_direct_single_position_layernorm \
  run \
  --contract "${AF_CONTRACT}" \
  --arm-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda
