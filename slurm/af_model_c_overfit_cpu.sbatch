#!/bin/bash -l
#SBATCH --job-name=af-model-c-overfit
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Model C stage C1b: exercise the frozen balanced objective on the same 96
# training-only rollouts.  Acceptance requires every physical group to pass,
# substantial total/spectral reduction, bounded rollout/boundary error, and
# bitwise-exact three-step inference after checkpoint reload.
set -euo pipefail

module purge
module load gnu12/12.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_C_DIAGNOSTICS="${AF_MODEL_C_DIAGNOSTICS:-${AF_SCRATCH_ROOT}/models/C/training_diagnostics_v2/model_c_training_diagnostics.json}"
AF_MODEL_C_CALIBRATION="${AF_MODEL_C_CALIBRATION:-${AF_SCRATCH_ROOT}/models/C/loss_calibration_cpu_v1/model_c_loss_calibration.json}"
AF_MODEL_C_OVERFIT="${AF_MODEL_C_OVERFIT:-${AF_SCRATCH_ROOT}/models/C/overfit_v1}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_overfit \
  --dataset "${AF_DATASET}" \
  --diagnostics "${AF_MODEL_C_DIAGNOSTICS}" \
  --calibration "${AF_MODEL_C_CALIBRATION}" \
  --output-dir "${AF_MODEL_C_OVERFIT}" \
  --samples 96 \
  --epochs 160 \
  --batch-size 4 \
  --device cpu
