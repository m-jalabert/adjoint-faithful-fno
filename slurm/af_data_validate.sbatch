#!/bin/bash -l
#SBATCH --job-name=af-data-validate
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Independent M4 acceptance check.  It recomputes the training-only moments
# directly from the published store and writes a small tracked-quality report.
set -euo pipefail

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_QUALITY_REPORT="${AF_QUALITY_REPORT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.quality.json}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_data validate \
  --output "${AF_DATASET}" \
  --report "${AF_QUALITY_REPORT}"
