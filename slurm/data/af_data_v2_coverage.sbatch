#!/bin/bash -l
#SBATCH --job-name=af-data-v2-ess
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Training-only effective-coverage audit. Validation, inference, I1/I2,
# response, and adjoint data remain sealed.
set -euo pipefail

module purge
module load gnu14/14.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET_V2="${AF_DATASET_V2:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_DATASET_V2_REPORT="${AF_DATASET_V2_REPORT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.quality.json}"
AF_COVERAGE_CONTRACT="${AF_COVERAGE_CONTRACT:-${AF_PROJECT_ROOT}/config/trajectories_v2_coverage_audit.json}"
AF_COVERAGE_OUTPUT="${AF_COVERAGE_OUTPUT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2_coverage_v1}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.diagnostics.af_data_v2_coverage \
  --dataset "${AF_DATASET_V2}" \
  --quality-report "${AF_DATASET_V2_REPORT}" \
  --contract "${AF_COVERAGE_CONTRACT}" \
  --output-dir "${AF_COVERAGE_OUTPUT}" \
  --chunk-days 32
