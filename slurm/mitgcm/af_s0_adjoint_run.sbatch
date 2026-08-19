#!/bin/bash -l
#SBATCH --job-name=af-s0-adj-run
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# Stage 8 of docs/mitgcm_adjoint_ground_truth_plan.md: the production adjoint.
#
#   AF_ADJ_MODE=runA   day 7210 -> 7220, the primary 10-day map
#                      adxx_etan.0000000000.data == S^10_ij exactly
#   AF_ADJ_MODE=runB   day 7200 -> 7220, the 20-day map and the lead sweep
#                      adxx_etan is the history-slot analogue; ADJetan.* gives
#                      every lead from 1 to 20 days in one run
#
# The adjoint state at time t inside a run whose cost sits at T is dJ/deta(.,t)
# for every t in [start, T], independent of when the run started.  So Run B's
# ADJetan at day 7210 must equal Run A's adxx_etan -- that is gate G2, and it
# is free.
#
#   AF_ADJ_WEIGHT   which quantity of interest to use.  Default is the SSH
#                   anomaly at p*.  Use costWeight_mean_only.bin for gate G3:
#                   by conservation of the area integral of eta, the returned
#                   map must equal the weight field itself at every dump time.
#
# A new p*, a new mean convention or any other linear-in-eta QoI is a different
# weight file and nothing else.  No rebuild, and so no TAF submission.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1}"
AF_ADJ_MODE="${AF_ADJ_MODE:-runA}"
AF_ADJ_WEIGHT="${AF_ADJ_WEIGHT:-${AF_PROJECT_ROOT}/work/costWeight_ssh_anomaly.bin}"

case "${AF_ADJ_MODE}" in
  runA|runB|runG3) ;;
  *) echo "AF_ADJ_MODE must be runA, runB or runG3, got '${AF_ADJ_MODE}'"; exit 1 ;;
esac

cd "${AF_PROJECT_ROOT}"

"${AF_PROJECT_ROOT}/.venv/bin/python" scripts/stage_adjoint_run.py "${AF_ADJ_MODE}" \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --weight-file  "${AF_ADJ_WEIGHT}" \
  --force

RUN_DIR="${AF_SCRATCH_ROOT}/${AF_ADJ_MODE}"
cd "${RUN_DIR}"
srun --mpi=pmix -n 4 ./mitgcmuv_ad > run.log 2>&1

test -f adxx_etan.0000000000.data || {
  echo "ERROR: no gradient written; adxx_etan.0000000000.data is missing"
  tail -60 run.log
  exit 1
}

echo "=== ${AF_ADJ_MODE} complete ==="
ls -l adxx_etan.0000000000.* ADJetan.*.data 2>/dev/null | head -30
echo "--- cost function value (gate G5 compares this against numpy) ---"
# fc is printed by cost_final.F to the model's own STDOUT.0000, not to the
# stdout srun captures; run.log holds only the Fortran STOP messages.
grep -iE "global fc|early fc|objf_test" STDOUT.0000 | tail -5
