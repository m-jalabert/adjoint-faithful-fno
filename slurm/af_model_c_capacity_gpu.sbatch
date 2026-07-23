#!/bin/bash -l
#SBATCH --job-name=af-c-width64-gpu
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Model C stage C1d GPU replica: isolate capacity after width-32 C1c failed
# only the SSH increment/persistence gate. Keep the exact 96 records, loss,
# modes, layers, padding, optimizer, seed, batch order, and 320 epochs; change
# only hidden width 32 -> 64. Validation and inference remain sealed.
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
AF_MODEL_C_CAPACITY="${AF_MODEL_C_CAPACITY:-${AF_SCRATCH_ROOT}/models/C/overfit_width64_gpu_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_overfit \
  --dataset "${AF_DATASET}" \
  --diagnostics "${AF_MODEL_C_DIAGNOSTICS}" \
  --calibration "${AF_MODEL_C_CALIBRATION}" \
  --output-dir "${AF_MODEL_C_CAPACITY}" \
  --samples 96 \
  --epochs 320 \
  --batch-size 4 \
  --learning-rate 0.0005 \
  --hidden-channels 64 \
  --device cuda
