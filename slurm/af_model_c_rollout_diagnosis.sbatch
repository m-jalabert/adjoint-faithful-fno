#!/bin/bash -l
#SBATCH --job-name=af-c-rollout-audit
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Diagnose the rejected successor's 10--90-day behavior on a fixed, balanced
# subset of split-1 starts. This job does not train and does not read split 2
# states, inference, intermediate winds, response data, or adjoints.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_VALIDATION_REPORT="${AF_VALIDATION_REPORT:-${AF_SCRATCH_ROOT}/models/C/successor_validation_v1/fresh_validation/model_c_successor_validation_report.json}"
AF_ROLLOUT_DIAGNOSIS_CONTRACT="${AF_ROLLOUT_DIAGNOSIS_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_rollout_diagnosis_v1.json}"
AF_ROLLOUT_DIAGNOSIS_OUTPUT="${AF_ROLLOUT_DIAGNOSIS_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/rollout_diagnosis_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_rollout_diagnosis \
  --dataset "${AF_DATASET}" \
  --validation-report "${AF_VALIDATION_REPORT}" \
  --contract "${AF_ROLLOUT_DIAGNOSIS_CONTRACT}" \
  --output-dir "${AF_ROLLOUT_DIAGNOSIS_OUTPUT}" \
  --device cuda
