#!/bin/bash -l
#SBATCH --job-name=af-c-train-spec2
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Corrected split-1 attribution: every model call advances ten daily dataset
# snapshots. The invalid stride-one job 304125 remains preserved as provenance.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_anomaly_direct_training_spectral_attribution_v2.json}"
AF_OUTPUT="${AF_OUTPUT:-${AF_PROJECT_ROOT}/outputs/af_fno/C/anomaly_direct_training_spectral_attribution_v2}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_anomaly_direct_training_spectral_attribution_v2 \
  run \
  --contract "${AF_CONTRACT}" \
  --output-dir "${AF_OUTPUT}" \
  --device cuda
