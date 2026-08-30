#!/bin/bash -l
#SBATCH --job-name=af-resp-bank
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/mitgcm/logs/af-resp-bank-%x-%j.out

# Plan step 6 (docs/Adjoint_faithful_response_training_plan.md section 7.1):
# one regime's 320-day unperturbed continuation from the existing day-5,760
# trajectory-v3 pickup through day 6,080, archiving a complete pickup every
# 10 days.  This is response-data setup only -- it is excluded from nominal
# training, normalization, climatology, and checkpoint validation, and it
# touches no adjoint/TAF/blind data, so it does not require the study's
# development/evaluator identity separation (that machinery only matters
# once FNO training/selection on response data begins).
#
# 320 days at the ~700s-per-10-model-years-ish throughput measured for the
# independently-equilibrated 1-degree chains (archive/slurm/mitgcm/
# af_independent_wind_segment.sbatch) is on the order of an hour; the
# four-hour limit is headroom, not an estimate -- this has not been timed yet.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_s0/mitgcmuv}"
: "${AF_REGIME:?Set AF_REGIME to S0, S1 or S2}"

# bire_repro lives in the archived tree and is not installed into the venv.
export PYTHONPATH="${AF_PROJECT_ROOT}/archive/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/build_response_pickup_bank.py" run \
  --regime "${AF_REGIME}" \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --executable "${AF_EXECUTABLE}"
