#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SCRIPT_ROOT}}"
cd "${AF_PROJECT_ROOT}"

for experiment in S1 S2; do
  adjust_id="$(sbatch --parsable \
    --export=ALL,AF_PROJECT_ROOT="${AF_PROJECT_ROOT}",AF_EXPERIMENT="${experiment}",AF_PHASE=adjust,AF_START_YEAR=0,AF_YEARS=5 \
    slurm/mitgcm/af_wind_segment.sbatch)"
  printf '%s adjustment years 000-005: %s\n' "${experiment}" "${adjust_id}"
  production_id="$(sbatch --parsable --dependency="afterok:${adjust_id}" \
    --export=ALL,AF_PROJECT_ROOT="${AF_PROJECT_ROOT}",AF_EXPERIMENT="${experiment}",AF_PHASE=production,AF_START_YEAR=5,AF_YEARS=10 \
    slurm/mitgcm/af_wind_segment.sbatch)"
  printf '%s production years 005-015: %s\n' "${experiment}" "${production_id}"
done
