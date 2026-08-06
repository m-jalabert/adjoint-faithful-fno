#!/bin/bash -l
#SBATCH --job-name=af-c-v2-seeds
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --array=0-1%2

# Train the two predeclared independent replicas of the already selected
# width-128 Model C successor.  This job reads split 1 only; fresh validation
# remains sealed until all seed artifacts are complete and independently
# checked by the separately frozen validation runner.
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
AF_SUCCESSOR_TRAINING_CONTRACT="${AF_SUCCESSOR_TRAINING_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_successor_training_v1.json}"
AF_SUCCESSOR_VALIDATION_CONTRACT="${AF_SUCCESSOR_VALIDATION_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_successor_validation_v1.json}"
AF_SUCCESSOR_VALIDATION_ROOT="${AF_SUCCESSOR_VALIDATION_ROOT:-${AF_SCRATCH_ROOT}/models/C/successor_validation_v1}"

cd "${AF_PROJECT_ROOT}"
seed="$(
  "${AF_PROJECT_ROOT}/.venv/bin/python" \
    -m bire_repro.af_model_c_successor_validation resolve-seed \
    --contract "${AF_SUCCESSOR_VALIDATION_CONTRACT}" \
    --array-index "${SLURM_ARRAY_TASK_ID}"
)"
output="${AF_SUCCESSOR_VALIDATION_ROOT}/training/seed_${seed}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_successor run \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY_REPORT}" \
  --coverage-report "${AF_COVERAGE_REPORT}" \
  --v1-decision "${AF_V1_DECISION}" \
  --contract "${AF_SUCCESSOR_TRAINING_CONTRACT}" \
  --candidate-id v2_bireprop_w128_mlp4 \
  --seed "${seed}" \
  --output-dir "${output}" \
  --device cuda
