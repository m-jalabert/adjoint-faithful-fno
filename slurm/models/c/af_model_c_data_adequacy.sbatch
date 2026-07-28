#!/bin/bash -l
#SBATCH --job-name=af-c-data-audit
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Evaluate only the already-open training and validation splits for the frozen
# three-seed Model C rejection. Inference, I1/I2, response, and adjoint data
# remain sealed.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_TRAINING_DIAGNOSTICS="${AF_TRAINING_DIAGNOSTICS:-${AF_SCRATCH_ROOT}/models/C/training_diagnostics_v2/model_c_training_diagnostics.json}"
AF_FINAL_DECISION="${AF_FINAL_DECISION:-${AF_SCRATCH_ROOT}/models/C/validation_search_v1/final_seed_gate/model_c_three_seed_freeze_decision.json}"
AF_DATA_ADEQUACY_CONTRACT="${AF_DATA_ADEQUACY_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_data_adequacy_v1.json}"
AF_DATA_ADEQUACY_OUTPUT="${AF_DATA_ADEQUACY_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/data_adequacy_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.diagnostics.af_model_c_data_adequacy \
  --dataset "${AF_DATASET}" \
  --training-diagnostics "${AF_TRAINING_DIAGNOSTICS}" \
  --final-decision "${AF_FINAL_DECISION}" \
  --contract "${AF_DATA_ADEQUACY_CONTRACT}" \
  --output-dir "${AF_DATA_ADEQUACY_OUTPUT}" \
  --batch-size 16 \
  --device cuda
