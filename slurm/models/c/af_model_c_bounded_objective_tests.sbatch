#!/bin/bash -l
#SBATCH --job-name=af-c-bounded
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-1

# Two isolated, training-only width-32 tests follow the completed late-checkpoint
# audit. Task 0 keeps loss v1 and adds one late LR decay. Task 1 keeps the
# constant optimizer and changes only increment weight through loss v2.
# A combined variant is not authorized by this job.
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

extra_arguments=()
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    AF_MODEL_C_OUTPUT="${AF_SCRATCH_ROOT}/models/C/overfit_v1_lr_decay_v1"
    extra_arguments+=(--loss-version v1 --lr-decay-epoch 240 --lr-decay-factor 0.2)
    ;;
  1)
    AF_MODEL_C_OUTPUT="${AF_SCRATCH_ROOT}/models/C/overfit_increment_loss_v2"
    extra_arguments+=(--loss-version v2)
    ;;
  *)
    echo "Unsupported bounded-test array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
set +e
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_overfit \
  --dataset "${AF_DATASET}" \
  --diagnostics "${AF_MODEL_C_DIAGNOSTICS}" \
  --calibration "${AF_MODEL_C_CALIBRATION}" \
  --output-dir "${AF_MODEL_C_OUTPUT}" \
  --samples 96 \
  --epochs 320 \
  --batch-size 4 \
  --learning-rate 0.0005 \
  --hidden-channels 32 \
  --device cuda \
  "${extra_arguments[@]}"
status=$?
set -e

if [[ ${status} -eq 0 ]]; then
  test -f "${AF_MODEL_C_OUTPUT}/model_c_overfit_report.json"
  exit 0
fi
if [[ -f "${AF_MODEL_C_OUTPUT}/model_c_overfit_rejected_report.json" ]]; then
  echo "Model C bounded task ${SLURM_ARRAY_TASK_ID} completed as a scientific rejection"
  exit 0
fi
echo "Model C bounded task ${SLURM_ARRAY_TASK_ID} failed operationally" >&2
exit "${status}"
