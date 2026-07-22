#!/bin/bash -l
#SBATCH --job-name=af-data-s0-s2
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Build the immutable shared dataset for the A0/A--C comparison ladder.
# The raw MDS archives remain read-only; conversion first writes a temporary
# Zarr directory and atomically renames it only after all checks succeed.
set -euo pipefail

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_data convert \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --output "${AF_DATASET}" \
  --chunk-days 8
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_data validate \
  --output "${AF_DATASET}"
