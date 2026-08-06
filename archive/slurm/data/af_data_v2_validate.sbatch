#!/bin/bash -l
#SBATCH --job-name=af-data-v2-qc
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Independently check the immutable v2 store against v1 and selected raw MDS
# records. No model, inference score, or later archive is opened.
set -euo pipefail

module purge
module load gnu14/14.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET_V1="${AF_DATASET_V1:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_DATASET_V2="${AF_DATASET_V2:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_DATASET_V2_REPORT="${AF_DATASET_V2_REPORT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.quality.json}"
AF_EXPANSION_CONTRACT="${AF_EXPANSION_CONTRACT:-${AF_PROJECT_ROOT}/config/trajectories_v2_expansion.json}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_data_v2 \
  validate \
  --dataset "${AF_DATASET_V2}" \
  --v1 "${AF_DATASET_V1}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --expansion-contract "${AF_EXPANSION_CONTRACT}" \
  --report "${AF_DATASET_V2_REPORT}" \
  --chunk-days 16
