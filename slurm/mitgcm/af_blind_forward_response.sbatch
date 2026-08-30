#!/bin/bash -l
#SBATCH --job-name=af-blind-resp-run
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/mitgcm/logs/af-blind-resp-%x-%j.out
#SBATCH --error=slurm/mitgcm/logs/af-blind-resp-%x-%j.err

# Execution step 16 (docs/Adjoint_faithful_response_training_plan.md section
# 17): one evaluator-only blind forward-response branch -- a signed perturbed
# direction (AF_RESPONSE_KIND=signed, both signs together) or a shared nominal
# branch (AF_RESPONSE_KIND=nominal, 90 days for every blind anchor, since the
# blind long subset is evaluated at days 20..90).
#
# The sibling of slurm/mitgcm/af_forward_response_array.sbatch, which
# deliberately never reads the blind manifest. This one does, through
# scripts/stage_blind_forward_response_run.py, which asserts section 17's
# precondition -- lambda frozen, all six training reports complete, Gate M1
# frozen, the step-15 ordinary-forward freeze manifest written -- before it
# will stage anything. The MITgcm numerics, the pickup editor and the frozen
# amplitudes are the step-9 machinery unchanged, so a blind case is generated
# by exactly the code that generated the development cases.
#
# Section 17: this package is opened once.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_s0/mitgcmuv}"
: "${AF_RESPONSE_KIND:?Set AF_RESPONSE_KIND to signed or nominal}"

export PYTHONPATH="${AF_PROJECT_ROOT}/archive/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${AF_PROJECT_ROOT}"

if [[ "${AF_RESPONSE_KIND}" == "signed" ]]; then
  # Addressed by (regime, day, family, slot), never by direction_id: that
  # string embeds a JSON fragment with commas, which sbatch --export truncates.
  : "${AF_REGIME:?Set AF_REGIME to S0, S1 or S2}"
  : "${AF_DAY:?Set AF_DAY to 7560, 7920 or 8280}"
  : "${AF_FAMILY:?Set AF_FAMILY to U, V, Theta or SSH}"
  : "${AF_SLOT:?Set AF_SLOT}"
  "${AF_PROJECT_ROOT}/.venv/bin/python" \
    "${AF_PROJECT_ROOT}/scripts/stage_blind_forward_response_run.py" run-signed \
    --regime "${AF_REGIME}" --day "${AF_DAY}" --family "${AF_FAMILY}" --slot "${AF_SLOT}"
else
  : "${AF_REGIME:?Set AF_REGIME to S0, S1 or S2}"
  : "${AF_DAY:?Set AF_DAY to 7560, 7920 or 8280}"
  "${AF_PROJECT_ROOT}/.venv/bin/python" \
    "${AF_PROJECT_ROOT}/scripts/stage_blind_forward_response_run.py" run-nominal \
    --regime "${AF_REGIME}" --day "${AF_DAY}"
fi
