#!/bin/bash -l
#SBATCH --job-name=af-v3-build
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

# Build the immutable trajectory-v3 store: 3 regimes x 9,000 daily records on
# independently equilibrated 100-year spin-ups, with the strictly chronological
# 5040 / 90 / 1260 / 90 / 2520 split.  Then re-validate it independently.
set -euo pipefail

module purge
module load gnu12/12.2.0

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_OUTPUT="${AF_OUTPUT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v3.zarr}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_data_v3 build \
  --scratch-root "${AF_SCRATCH_ROOT}" --output "${AF_OUTPUT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_data_v3 validate \
  --dataset "${AF_OUTPUT}"
