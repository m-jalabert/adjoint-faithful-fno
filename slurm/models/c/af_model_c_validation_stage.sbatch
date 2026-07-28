#!/bin/bash -l
#SBATCH --job-name=af-c-val
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-9%5

# Generic immutable Model C successive-halving stage. Stage 1 uses the ten
# contract candidates. Later stages must receive the preceding immutable
# selection manifest and should override the array extent at submission.
# Pair codes 1 and 2 are permitted; inference and all later datasets remain
# sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_C_DIAGNOSTICS="${AF_MODEL_C_DIAGNOSTICS:-${AF_SCRATCH_ROOT}/models/C/training_diagnostics_v2/model_c_training_diagnostics.json}"
AF_MODEL_C_CALIBRATION="${AF_MODEL_C_CALIBRATION:-${AF_SCRATCH_ROOT}/models/C/loss_calibration_cpu_v1/model_c_loss_calibration.json}"
AF_MODEL_C_OPTIMIZATION="${AF_MODEL_C_OPTIMIZATION:-${AF_PROJECT_ROOT}/config/model_c_optimization_v1.json}"
AF_MODEL_C_LOSS_MANIFEST="${AF_MODEL_C_LOSS_MANIFEST:-${AF_PROJECT_ROOT}/config/model_c_loss_v1.json}"
AF_MODEL_C_VALIDATION_CONTRACT="${AF_MODEL_C_VALIDATION_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_validation_search_v1.json}"
AF_MODEL_C_VALIDATION_ROOT="${AF_MODEL_C_VALIDATION_ROOT:-${AF_SCRATCH_ROOT}/models/C/validation_search_v1}"
AF_MODEL_C_VALIDATION_STAGE="${AF_MODEL_C_VALIDATION_STAGE:-1}"
AF_MODEL_C_SELECTION_MANIFEST="${AF_MODEL_C_SELECTION_MANIFEST:-}"

selection_arguments=()
if [[ "${AF_MODEL_C_VALIDATION_STAGE}" -gt 1 ]]; then
  if [[ -z "${AF_MODEL_C_SELECTION_MANIFEST}" ]]; then
    echo "validation stage ${AF_MODEL_C_VALIDATION_STAGE} requires a selection manifest" >&2
    exit 2
  fi
  selection_arguments+=(--selection-manifest "${AF_MODEL_C_SELECTION_MANIFEST}")
fi

cd "${AF_PROJECT_ROOT}"
candidate_id="$(
  "${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_validation resolve \
    --contract "${AF_MODEL_C_VALIDATION_CONTRACT}" \
    --stage "${AF_MODEL_C_VALIDATION_STAGE}" \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    "${selection_arguments[@]}"
)"
output="${AF_MODEL_C_VALIDATION_ROOT}/stage_${AF_MODEL_C_VALIDATION_STAGE}/${candidate_id}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_validation run \
  --dataset "${AF_DATASET}" \
  --diagnostics "${AF_MODEL_C_DIAGNOSTICS}" \
  --calibration "${AF_MODEL_C_CALIBRATION}" \
  --optimization-manifest "${AF_MODEL_C_OPTIMIZATION}" \
  --loss-manifest "${AF_MODEL_C_LOSS_MANIFEST}" \
  --contract "${AF_MODEL_C_VALIDATION_CONTRACT}" \
  --stage "${AF_MODEL_C_VALIDATION_STAGE}" \
  --candidate-id "${candidate_id}" \
  "${selection_arguments[@]}" \
  --output-dir "${output}" \
  --device cuda
