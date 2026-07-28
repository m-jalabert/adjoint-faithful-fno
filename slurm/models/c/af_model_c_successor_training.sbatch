#!/bin/bash -l
#SBATCH --job-name=af-c-v2-train
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2

# Training-only trajectory-v2 successor diagnosis.  Phase 1 runs the unchanged
# width-64 data control and the isolated 4C Channel-MLP ablation.  Submit phase
# 2 only after both phase-1 reports complete, overriding --array=0 and setting
# AF_MODEL_C_SUCCESSOR_PHASE=2.  Fresh validation and all later data stay sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_QUALITY_REPORT="${AF_QUALITY_REPORT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.quality.json}"
AF_COVERAGE_REPORT="${AF_COVERAGE_REPORT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2_coverage_v1/trajectories_v2_coverage_report.json}"
AF_V1_DECISION="${AF_V1_DECISION:-${AF_SCRATCH_ROOT}/models/C/validation_search_v1/final_seed_gate/model_c_three_seed_freeze_decision.json}"
AF_MODEL_C_SUCCESSOR_CONTRACT="${AF_MODEL_C_SUCCESSOR_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_successor_training_v1.json}"
AF_MODEL_C_SUCCESSOR_ROOT="${AF_MODEL_C_SUCCESSOR_ROOT:-${AF_SCRATCH_ROOT}/models/C/successor_training_v1}"
AF_MODEL_C_SUCCESSOR_PHASE="${AF_MODEL_C_SUCCESSOR_PHASE:-1}"

cd "${AF_PROJECT_ROOT}"
candidate_id="$(
  "${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_successor resolve \
    --contract "${AF_MODEL_C_SUCCESSOR_CONTRACT}" \
    --phase "${AF_MODEL_C_SUCCESSOR_PHASE}" \
    --array-index "${SLURM_ARRAY_TASK_ID}"
)"
output="${AF_MODEL_C_SUCCESSOR_ROOT}/phase_${AF_MODEL_C_SUCCESSOR_PHASE}/${candidate_id}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_successor run \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY_REPORT}" \
  --coverage-report "${AF_COVERAGE_REPORT}" \
  --v1-decision "${AF_V1_DECISION}" \
  --contract "${AF_MODEL_C_SUCCESSOR_CONTRACT}" \
  --candidate-id "${candidate_id}" \
  --output-dir "${output}" \
  --device cuda
