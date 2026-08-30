#!/bin/bash -l
#SBATCH --job-name=af-pilot
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/mitgcm/logs/af-pilot-%x-%j.out

# Plan step 7/8 (docs/Adjoint_faithful_response_training_plan.md sections
# 7.2/8/10): one branch of the forward-only amplitude pilot -- a signed
# perturbed direction (AF_PILOT_KIND=signed), a paired nominal branch
# (AF_PILOT_KIND=nominal), or a section-10.3 duplicate/tight-CG confirmatory
# control (AF_PILOT_KIND=control). Touches no adjoint/TAF/blind data, so it
# needs none of the OS-identity machinery retired in the roadmap amendments.
#
# Individual segments are short (10 or 90 model-days); a live canary run
# (S0/day720/V, 10 days) completed in under 7 seconds, so the "short"
# partition (used by the S0 twin study for the same reason) is a comfortable
# fit -- this is not the 320-day step-6 bridge, which used "batch".
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_s0/mitgcmuv}"
: "${AF_PILOT_KIND:?Set AF_PILOT_KIND to signed or nominal}"
: "${AF_REGIME:?Set AF_REGIME to S0, S1 or S2}"
: "${AF_DAY:?Set AF_DAY to 720 or 3600}"

export PYTHONPATH="${AF_PROJECT_ROOT}/archive/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${AF_PROJECT_ROOT}"

if [[ "${AF_PILOT_KIND}" == "signed" ]]; then
  : "${AF_FAMILY:?Set AF_FAMILY to U, V, Theta or SSH}"
  : "${AF_ALPHA:?Set AF_ALPHA}"
  : "${AF_SIGN:?Set AF_SIGN to 1 or -1}"
  "${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/build_amplitude_pilot.py" run-signed \
    --regime "${AF_REGIME}" --day "${AF_DAY}" --family "${AF_FAMILY}" \
    --alpha "${AF_ALPHA}" --sign "${AF_SIGN}" \
    --project-root "${AF_PROJECT_ROOT}" --scratch-root "${AF_SCRATCH_ROOT}" --executable "${AF_EXECUTABLE}"
elif [[ "${AF_PILOT_KIND}" == "control" ]]; then
  : "${AF_FAMILY:?Set AF_FAMILY to U, V, Theta or SSH}"
  : "${AF_ALPHA:?Set AF_ALPHA}"
  : "${AF_SIGN:?Set AF_SIGN to 1 or -1}"
  : "${AF_CONDITION:?Set AF_CONDITION to duplicate or tight}"
  "${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/build_amplitude_pilot.py" run-control \
    --regime "${AF_REGIME}" --day "${AF_DAY}" --family "${AF_FAMILY}" \
    --alpha "${AF_ALPHA}" --sign "${AF_SIGN}" --condition "${AF_CONDITION}" \
    --project-root "${AF_PROJECT_ROOT}" --scratch-root "${AF_SCRATCH_ROOT}" --executable "${AF_EXECUTABLE}"
else
  duplicate_flag=()
  [[ "${AF_PILOT_DUPLICATE:-0}" == "1" ]] && duplicate_flag=(--duplicate)
  tight_flag=()
  [[ "${AF_PILOT_TIGHT:-0}" == "1" ]] && tight_flag=(--tight)
  "${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/build_amplitude_pilot.py" run-nominal \
    --regime "${AF_REGIME}" --day "${AF_DAY}" "${duplicate_flag[@]}" "${tight_flag[@]}" \
    --project-root "${AF_PROJECT_ROOT}" --scratch-root "${AF_SCRATCH_ROOT}" --executable "${AF_EXECUTABLE}"
fi
