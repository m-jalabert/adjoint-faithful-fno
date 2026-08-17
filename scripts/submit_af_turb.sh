#!/usr/bin/env bash
# Submit the three 0.25-degree turbulent ground-truth campaigns.
#
# Each regime is an independent chain of thirteen restart-safe segments: ten
# ten-year spin-up legs (years 0-100) followed by 10 + 10 + 5 production years
# (years 100-125), giving 9,000 daily states per regime.  The three chains are
# independent of one another and run concurrently; within a chain each segment
# depends on its predecessor's pickup, so they are linked with afterok.
#
# Re-running this script is safe.  A segment whose segment_result.json already
# exists returns immediately, so a chain that died halfway can simply be
# resubmitted.
#
# Usage:
#   scripts/submit_af_turb.sh                 # all three regimes
#   scripts/submit_af_turb.sh S1_turb S2_turb # a subset
#   AF_TURB_AFTER=<jobid> scripts/submit_af_turb.sh   # start after a build job
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SCRIPT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
cd "${AF_PROJECT_ROOT}"
mkdir -p slurm/mitgcm/logs

regimes=("$@")
if [[ ${#regimes[@]} -eq 0 ]]; then
  regimes=(S0_turb S1_turb S2_turb)
fi

# phase:start_year:years, in chain order.
segments=(
  spinup:0:10 spinup:10:10 spinup:20:10 spinup:30:10 spinup:40:10
  spinup:50:10 spinup:60:10 spinup:70:10 spinup:80:10 spinup:90:10
  production:100:10 production:110:10 production:120:5
)

for regime in "${regimes[@]}"; do
  parent_id="${AF_TURB_AFTER:-}"
  for segment in "${segments[@]}"; do
    IFS=: read -r phase start_year years <<<"${segment}"
    exports="ALL,AF_PROJECT_ROOT=${AF_PROJECT_ROOT},AF_SCRATCH_ROOT=${AF_SCRATCH_ROOT}"
    exports="${exports},AF_REGIME=${regime},AF_PHASE=${phase}"
    exports="${exports},AF_START_YEAR=${start_year},AF_YEARS=${years}"
    args=(--parsable --job-name "${regime}-${phase}-${start_year}" --export="${exports}")
    if [[ -n "${parent_id}" ]]; then
      args+=(--dependency="afterok:${parent_id}")
    fi
    job_id="$(sbatch "${args[@]}" slurm/mitgcm/af_turb_segment.sbatch)"
    printf '%-8s %-10s years %03d-%03d  job %s\n' \
      "${regime}" "${phase}" "${start_year}" "$((start_year + years))" "${job_id}"
    parent_id="${job_id}"
  done
done
