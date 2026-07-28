#!/bin/bash -l
#SBATCH --job-name=af-model-c-cal-cpu
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# CPU replica of Model C stage C1a.  CUDA is not required for the calibration:
# this job reads the same 96 balanced training rollouts and writes to an
# independent directory.  If both CPU and GPU jobs complete, compare their raw
# loss/gradient scales before manually freezing weights; neither job validates
# a candidate or reads validation/inference records.
set -euo pipefail

module purge
module load gnu12/12.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_C_DIAGNOSTICS="${AF_MODEL_C_DIAGNOSTICS:-${AF_SCRATCH_ROOT}/models/C/training_diagnostics_v2/model_c_training_diagnostics.json}"
AF_MODEL_C_CALIBRATION="${AF_MODEL_C_CALIBRATION:-${AF_SCRATCH_ROOT}/models/C/loss_calibration_cpu_v1}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.diagnostics.af_model_c_calibrate \
  --dataset "${AF_DATASET}" \
  --diagnostics "${AF_MODEL_C_DIAGNOSTICS}" \
  --output-dir "${AF_MODEL_C_CALIBRATION}" \
  --warmup-epochs 20 \
  --device cpu
