#!/bin/bash -l
#SBATCH --job-name=af-resp-run
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/mitgcm/logs/af-resp-run-%x-%j.out

# Plan step 9 (docs/Adjoint_faithful_response_training_plan.md section 25):
# one production train/validation response branch -- a signed perturbed
# direction (AF_RESPONSE_KIND=signed, both signs run together) or a shared
# nominal branch (AF_RESPONSE_KIND=nominal). Never touches blind or pilot
# data (pilot already ran in step 7/8; blind is a separate, evaluator-only
# manifest step 9 never reads), so it needs none of the OS-identity
# machinery retired in the roadmap amendments.
#
# Individual segments are short (10 or 60 model-days) -- comparable to or
# smaller than the amplitude pilot's proven 90-day nominal (37s wall time)
# and 10-day signed branches (<7s), so the "short" partition used for that
# same reason is a comfortable fit here too.
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
  # Addressed by (regime, day, family, slot), not the row's own direction_id:
  # that string embeds a JSON fragment with commas, which sbatch --export
  # silently truncates on (verified directly against a real submission).
  : "${AF_REGIME:?Set AF_REGIME to S0, S1 or S2}"
  : "${AF_DAY:?Set AF_DAY}"
  : "${AF_FAMILY:?Set AF_FAMILY to U, V, Theta or SSH}"
  : "${AF_SLOT:?Set AF_SLOT}"
  "${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/stage_forward_response_run.py" run-signed \
    --regime "${AF_REGIME}" --day "${AF_DAY}" --family "${AF_FAMILY}" --slot "${AF_SLOT}" \
    --project-root "${AF_PROJECT_ROOT}" --scratch-root "${AF_SCRATCH_ROOT}" --executable "${AF_EXECUTABLE}"
else
  : "${AF_ROLE:?Set AF_ROLE to train or validation}"
  : "${AF_REGIME:?Set AF_REGIME to S0, S1 or S2}"
  : "${AF_DAY:?Set AF_DAY}"
  "${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/stage_forward_response_run.py" run-nominal \
    --role "${AF_ROLE}" --regime "${AF_REGIME}" --day "${AF_DAY}" \
    --project-root "${AF_PROJECT_ROOT}" --scratch-root "${AF_SCRATCH_ROOT}" --executable "${AF_EXECUTABLE}"
fi
