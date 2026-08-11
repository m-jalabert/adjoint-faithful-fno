#!/usr/bin/env bash
# Submit the three restart-chained segments of the MITgcm S0 twin2 experiment
# (epsilon = 1e-3): years 100-110 (perturbed start), 110-120, and 120-125.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SCRIPT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
cd "${AF_PROJECT_ROOT}"

parent_id=""
for segment in "100:10" "110:10" "120:5"; do
  start_year="${segment%%:*}"
  years="${segment##*:}"
  exports="ALL,AF_PROJECT_ROOT=${AF_PROJECT_ROOT},AF_SCRATCH_ROOT=${AF_SCRATCH_ROOT}"
  exports="${exports},AF_TWIN_START_YEAR=${start_year},AF_TWIN_YEARS=${years}"
  if [[ -z "${parent_id}" ]]; then
    job_id="$(sbatch --parsable --export="${exports}" slurm/mitgcm/af_s0_twin2_segment.sbatch)"
  else
    job_id="$(sbatch --parsable --dependency="afterok:${parent_id}" --export="${exports}" \
      slurm/mitgcm/af_s0_twin2_segment.sbatch)"
  fi
  printf 'S0 twin2 years %03d-%03d: %s\n' "${start_year}" "$((start_year + years))" "${job_id}"
  parent_id="${job_id}"
done
