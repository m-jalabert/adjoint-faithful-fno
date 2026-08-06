#!/bin/bash -l
#SBATCH --job-name=af-c-anom-direct
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Train Model C from scratch in pooled pointwise-anomaly coordinates and
# predict each ten-day future anomaly state directly.  Checkpoint selection is
# split-1 only.  After selection, publish the fixed S2 0--200-day Figure-4
# characterization and related plots.  Inference/response/adjoint states stay
# sealed.
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
AF_ANOMALY_DIRECT_CONTRACT="${AF_ANOMALY_DIRECT_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_anomaly_direct_v1.json}"
AF_ANOMALY_DIRECT_OUTPUT="${AF_ANOMALY_DIRECT_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/anomaly_direct_v1}"
AF_ANOMALY_DIRECT_PLOTS="${AF_ANOMALY_DIRECT_PLOTS:-${AF_PROJECT_ROOT}/outputs/af_fno/C/anomaly_direct_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_anomaly_direct run \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY_REPORT}" \
  --contract "${AF_ANOMALY_DIRECT_CONTRACT}" \
  --output-dir "${AF_ANOMALY_DIRECT_OUTPUT}" \
  --device cuda

"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.analysis.af_model_c_anomaly_direct_plots \
  --report "${AF_ANOMALY_DIRECT_OUTPUT}/model_c_anomaly_direct_v1_report.json" \
  --arrays "${AF_ANOMALY_DIRECT_OUTPUT}/model_c_anomaly_direct_v1_arrays.npz" \
  --output-dir "${AF_ANOMALY_DIRECT_PLOTS}"
