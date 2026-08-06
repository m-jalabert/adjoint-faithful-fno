#!/bin/bash -l
#SBATCH --job-name=af-c-anom-seeds
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2

# Train the two independent replicas of the pointwise-anomaly direct-state
# Model C.  Only the initialization/batch-order seed changes.  Checkpoints are
# selected on split 1 before the already frozen 15-member S2 characterization.
# Inference, intermediate-wind, response, and adjoint archives remain sealed.
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
AF_REPLICATION_CONTRACT="${AF_REPLICATION_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_anomaly_direct_replication_v1.json}"
AF_REPLICATION_ROOT="${AF_REPLICATION_ROOT:-${AF_SCRATCH_ROOT}/models/C/anomaly_direct_replication_v1}"
AF_REPLICATION_PLOTS="${AF_REPLICATION_PLOTS:-${AF_PROJECT_ROOT}/outputs/af_fno/C/anomaly_direct_replication_v1}"

cd "${AF_PROJECT_ROOT}"
seed="$(
  "${AF_PROJECT_ROOT}/.venv/bin/python" \
    -m bire_repro.af_model_c_anomaly_direct_replication resolve-seed \
    --contract "${AF_REPLICATION_CONTRACT}" \
    --array-index "${SLURM_ARRAY_TASK_ID}"
)"
output="${AF_REPLICATION_ROOT}/seeds/seed_${seed}"
plots="${AF_REPLICATION_PLOTS}/seed_${seed}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_anomaly_direct_replication run \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY_REPORT}" \
  --contract "${AF_REPLICATION_CONTRACT}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda

"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.analysis.af_model_c_anomaly_direct_plots \
  --report "${output}/model_c_anomaly_direct_v1_report.json" \
  --arrays "${output}/model_c_anomaly_direct_v1_arrays.npz" \
  --output-dir "${plots}"
