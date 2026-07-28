#!/bin/bash -l
#SBATCH --job-name=af-data-v2
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Convert only the already-authorized S0--S2 continuations into a new Zarr
# store. This job computes no model or inference metric.
set -euo pipefail

module purge
module load gnu14/14.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET_V1="${AF_DATASET_V1:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_DATASET_V2="${AF_DATASET_V2:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_EXPANSION_CONTRACT="${AF_EXPANSION_CONTRACT:-${AF_PROJECT_ROOT}/config/trajectories_v2_expansion.json}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_data_v2 \
  build \
  --v1 "${AF_DATASET_V1}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --expansion-contract "${AF_EXPANSION_CONTRACT}" \
  --output "${AF_DATASET_V2}" \
  --chunk-days 8
