#!/bin/bash -l
#SBATCH --job-name=af-c-cpu-scale
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

# Benchmark the real width-64, full-loss Model C workload at 4/8/16 threads
# before choosing the CPU count for the full C1d CPU replica. Each trial uses
# the same 96 training-only records, seed, batch order, and 20 epochs. Its
# scientific gate result is ignored; only elapsed time and resource use select
# the CPU count.
set -euo pipefail

module purge
module load gnu12/12.2.0

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_C_DIAGNOSTICS="${AF_MODEL_C_DIAGNOSTICS:-${AF_SCRATCH_ROOT}/models/C/training_diagnostics_v2/model_c_training_diagnostics.json}"
AF_MODEL_C_CALIBRATION="${AF_MODEL_C_CALIBRATION:-${AF_SCRATCH_ROOT}/models/C/loss_calibration_cpu_v1/model_c_loss_calibration.json}"
AF_BENCHMARK_ROOT="${AF_BENCHMARK_ROOT:-${AF_SCRATCH_ROOT}/benchmarks/C/cpu_scaling_width64_v2}"

cd "${AF_PROJECT_ROOT}"
for threads in 4 8 16; do
  output="${AF_BENCHMARK_ROOT}/threads_${threads}"
  timing="${AF_BENCHMARK_ROOT}/threads_${threads}.time"
  mkdir -p "${AF_BENCHMARK_ROOT}"
  SECONDS=0
  set +e
  OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
    "${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_overfit \
      --dataset "${AF_DATASET}" \
      --diagnostics "${AF_MODEL_C_DIAGNOSTICS}" \
      --calibration "${AF_MODEL_C_CALIBRATION}" \
      --output-dir "${output}" \
      --samples 96 \
      --epochs 20 \
      --batch-size 4 \
      --learning-rate 0.0005 \
      --hidden-channels 64 \
      --device cpu
  status=$?
  elapsed_seconds="${SECONDS}"
  set -e
  printf 'real %s\n' "${elapsed_seconds}" > "${timing}"
  if [[ ${status} -ne 0 && ! -f "${output}/model_c_overfit_rejected_report.json" ]]; then
    echo "CPU scaling trial ${threads} failed without a scientific rejection report" >&2
    exit "${status}"
  fi
  echo "threads=${threads}"
  cat "${timing}"
done
