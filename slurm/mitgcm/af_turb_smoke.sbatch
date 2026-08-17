#!/bin/bash -l
#SBATCH --job-name=af-turb-smoke
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/mitgcm/logs/af-turb-smoke-%j.out

# Thirty cold-start model days of the turbulent configuration: proves the time
# step is stable and the elliptic solver converges, and measures the per-step
# cost that sizes the real segments.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/af_turb_smoke.py" \
  --project-root "${AF_PROJECT_ROOT}" \
  --run-dir "${AF_SCRATCH_ROOT}/mitgcm_turb_smoke/cold_start" \
  --executable "${AF_PROJECT_ROOT}/build/af_turb/mitgcmuv" \
  --days "${AF_TURB_SMOKE_DAYS:-30}"
