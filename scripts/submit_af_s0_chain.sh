#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SCRIPT_ROOT}}"
cd "${AF_PROJECT_ROOT}"

build_id="$(sbatch --parsable --export=ALL,AF_PROJECT_ROOT="${AF_PROJECT_ROOT}" slurm/af_s0_build.sbatch)"
printf 'AF--FNO S0 build job: %s\n' "${build_id}"
parent_id="${build_id}"

for start_year in 0 10 20 30 40 50 60 70 80 90; do
  job_id="$(sbatch --parsable --dependency="afterok:${parent_id}" \
    --export=ALL,AF_PROJECT_ROOT="${AF_PROJECT_ROOT}",AF_S0_PHASE=spinup,AF_S0_START_YEAR="${start_year}",AF_S0_YEARS=10 \
    slurm/af_s0_segment.sbatch)"
  printf 'S0 spin-up years %03d-%03d: %s\n' "${start_year}" "$((start_year + 10))" "${job_id}"
  parent_id="${job_id}"
done

production_id="$(sbatch --parsable --dependency="afterok:${parent_id}" \
  --export=ALL,AF_PROJECT_ROOT="${AF_PROJECT_ROOT}",AF_S0_PHASE=production,AF_S0_START_YEAR=100,AF_S0_YEARS=10 \
  slurm/af_s0_segment.sbatch)"
printf 'S0 production years 100-110: %s\n' "${production_id}"
