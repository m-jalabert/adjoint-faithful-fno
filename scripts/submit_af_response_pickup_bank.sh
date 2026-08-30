#!/usr/bin/env bash
# Submit the three regime-independent validation pickup-bank bridge segments
# (plan step 6, docs/Adjoint_faithful_response_training_plan.md section 7.1).
#
# S0/S1/S2 are independent of one another -- unlike the spin-up chains, there
# is no afterok dependency chaining here.
#
# Usage:
#   scripts/submit_af_response_pickup_bank.sh                 # all three regimes
#   scripts/submit_af_response_pickup_bank.sh S1 S2           # a subset
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SCRIPT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
cd "${AF_PROJECT_ROOT}"
mkdir -p slurm/mitgcm/logs

regimes=("$@")
if [[ ${#regimes[@]} -eq 0 ]]; then
  regimes=(S0 S1 S2)
fi

for regime in "${regimes[@]}"; do
  job_id="$(sbatch --parsable \
    --job-name "af-resp-bank-${regime}" \
    --export=ALL,AF_PROJECT_ROOT="${AF_PROJECT_ROOT}",AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT}",AF_REGIME="${regime}" \
    slurm/mitgcm/af_response_pickup_bank_segment.sbatch)"
  printf '%-4s validation pickup bank (day 5760-6080): job %s\n' "${regime}" "${job_id}"
done
