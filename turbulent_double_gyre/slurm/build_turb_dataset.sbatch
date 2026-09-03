#!/bin/bash -l
#SBATCH --job-name=turb-build-dataset
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Build trajectories_turb_v1.zarr from the three completed 0.25-degree MITgcm
# production chains.  chunk_days=1: each training sample reads seven
# non-adjacent day slices, so a multi-day chunk decompresses far more than it
# uses, and unlike the 13 GB 1-degree store this one will not sit in page cache.
set -euo pipefail

module purge
module load gnu12/12.2.0

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
STUDY="${AF_PROJECT_ROOT}/turbulent_double_gyre"
SCRATCH=/bigscratch/mjalabert314/bire_james25_repro/af_fno
OUTPUT="${SCRATCH}/datasets/trajectories_turb_v1.zarr"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export BLOSC_NTHREADS="${SLURM_CPUS_PER_TASK}"

cd "${AF_PROJECT_ROOT}"
"${AF_PYTHON}" "${STUDY}/scripts/build_turb_dataset.py" build \
  --scratch-root "${SCRATCH}/mitgcm_turb_v1" \
  --output "${OUTPUT}" \
  --chunk-days 1

"${AF_PYTHON}" "${STUDY}/scripts/build_turb_dataset.py" validate --dataset "${OUTPUT}"
