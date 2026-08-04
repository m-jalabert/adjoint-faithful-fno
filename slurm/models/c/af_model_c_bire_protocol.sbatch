#!/bin/bash -l
#SBATCH --job-name=af-c-bireproto
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

# Pooled loss-recovery model under the Bire et al. Section 3.2 arrangement:
# 6,000 training days, 1,200 validation days, and the final 1,000 validation
# days as the inference set, with no buffers (the paper has none).  Days
# 7200-8999 lie beyond the paper's record and serve only as rollout truth.
#
# Training and validation only; the inference set opens through the figure
# contract, S0 only.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_pooled_v1.json}"
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
  -m bire_repro.af_model_c_bire_protocol \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m bire_repro.af_model_c_bire_protocol \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
