#!/bin/bash -l
#SBATCH --job-name=af-s0-twin2
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# One segment of the second MITgcm twin of the S0 control trajectory.  The
# first segment perturbs only Uvel and Vvel in the year-100 S0 pickup by a
# factor 1+1e-3; every later segment is an unmodified restart.  Nothing here
# reads or writes the FNO tree.  The partition matches the one the S0 control
# and the 1e-6 twin were integrated on, so all three share the same node and
# arithmetic -- the epsilon comparison is only clean if nothing else differs.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_s0/mitgcmuv}"
: "${AF_TWIN_START_YEAR:?Set AF_TWIN_START_YEAR}"
: "${AF_TWIN_YEARS:?Set AF_TWIN_YEARS}"

# bire_repro lives in the archived tree and is not installed into the venv.
export PYTHONPATH="${AF_PROJECT_ROOT}/archive/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_s0_twin2 run-segment \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --executable "${AF_EXECUTABLE}" \
  --start-year "${AF_TWIN_START_YEAR}" \
  --years "${AF_TWIN_YEARS}"
