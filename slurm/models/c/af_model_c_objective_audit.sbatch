#!/bin/bash -l
#SBATCH --job-name=af-c-obj-audit
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-1

# Audit the late width-32 and width-64 diagnostic checkpoints independently on
# their exact 96 training rollouts.  Each array task writes a distinct immutable
# result. Validation, inference, intermediate-wind, response, and adjoint data
# remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    AF_MODEL_C_REPORT="${AF_SCRATCH_ROOT}/models/C/overfit_duration_v2/model_c_overfit_rejected_report.json"
    AF_MODEL_C_AUDIT="${AF_SCRATCH_ROOT}/models/C/late_checkpoint_objective_audit_width32_v1"
    ;;
  1)
    AF_MODEL_C_REPORT="${AF_SCRATCH_ROOT}/models/C/overfit_width64_gpu_v1/model_c_overfit_rejected_report.json"
    AF_MODEL_C_AUDIT="${AF_SCRATCH_ROOT}/models/C/late_checkpoint_objective_audit_width64_v1"
    ;;
  *)
    echo "Unsupported Model C objective-audit array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.diagnostics.af_model_c_objective_audit \
  --overfit-report "${AF_MODEL_C_REPORT}" \
  --output-dir "${AF_MODEL_C_AUDIT}" \
  --batch-size 4 \
  --device cuda
