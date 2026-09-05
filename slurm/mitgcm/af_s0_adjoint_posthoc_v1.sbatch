#!/bin/bash -l
#SBATCH --job-name=af-s0-adj-ph
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

# Post-hoc interior/eastern adjoint diagnostic -- NOT part of the frozen
# adjoint-faithful contract.  Gate A1 already closed v1 negative on
# 2026-08-29 (outputs/af_fno/adjoint/comparison_response_v1/gate_a1_result.json)
# and the interior/eastern exploratory extension in
# config/adjoint_faithful_blind_adjoint_evaluation_v1.json was never unlocked
# before B/C training, so nothing produced by this script counts as
# confirmatory or exploratory evidence under that contract. See
# scripts/make_cost_weight_posthoc_v1.py for target provenance.
#
# Reuses the validated Phase A machinery unmodified: same mitgcmuv_ad binary
# (NO TAF SUBMISSION -- see af_s0_adjoint_phase_a.sbatch), same window/cost-day
# timing as the frozen P10/P30/P90/K10/K30/K90 runs in
# config/adjoint_phase_a_v1.json, same day-7200 archived pickup. Only the
# weight field (which grid cell the cost function reads) and the output
# run-name change, via stage_adjoint_run.py's own --weight-file/--run-name
# overrides -- no edits to any frozen script or config.
#
#   AF_ADJ_BASE_MODE   one of P10 P30 P90 K10 K30 K90 (window/cost-day source)
#   AF_TARGET_NAME     interior | eastern
#   AF_WEIGHT_FILE     work/costWeight_<target>_<ssh_anomaly|ssh_anomaly_kernel>.bin
#   AF_RUN_NAME        defaults to ${AF_ADJ_BASE_MODE}_${AF_TARGET_NAME}
#
# cg2dTargetResidual stays at the production 1e-7 default (no override here):
# this is a science run, not a grdchk diagnostic, and both new points already
# passed G1-90 at 1e-7 in the frozen v2 table.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_posthoc_v1}"
AF_ADJ_BASE_MODE="${AF_ADJ_BASE_MODE:?Set AF_ADJ_BASE_MODE to one of P10 P30 P90 K10 K30 K90}"
AF_TARGET_NAME="${AF_TARGET_NAME:?Set AF_TARGET_NAME to western, interior or eastern}"
AF_WEIGHT_FILE="${AF_WEIGHT_FILE:?Set AF_WEIGHT_FILE to the target-specific costWeight .bin}"
AF_RUN_NAME="${AF_RUN_NAME:-${AF_ADJ_BASE_MODE}_${AF_TARGET_NAME}}"

case "${AF_ADJ_BASE_MODE}" in
  P10|P30|P90|K10|K30|K90) ;;
  *) echo "unknown AF_ADJ_BASE_MODE '${AF_ADJ_BASE_MODE}'"; exit 1 ;;
esac
case "${AF_TARGET_NAME}" in
  western|interior|eastern) ;;
  *) echo "unknown AF_TARGET_NAME '${AF_TARGET_NAME}'"; exit 1 ;;
esac

cd "${AF_PROJECT_ROOT}"

PYTHONPATH="${AF_PROJECT_ROOT}/scripts" \
  "${AF_PROJECT_ROOT}/.venv/bin/python" scripts/stage_adjoint_run.py "${AF_ADJ_BASE_MODE}" \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --run-name "${AF_RUN_NAME}" \
  --weight-file "${AF_WEIGHT_FILE}" \
  --force

RUN_DIR="${AF_SCRATCH_ROOT}/${AF_RUN_NAME}"

# stage_adjoint_run.py's manifest always records the FROZEN western contract's
# target_i_global/target_j_global, regardless of --weight-file -- that field is
# provenance-only and not read back by anything downstream, but leaving it
# pointed at (2,17) inside an interior/eastern run directory is exactly the
# "two sides disagreeing about where eta is measured" class of silent-wrong-
# answer bug this study keeps flagging. Patch it from whichever
# costWeight_*_manifest.json a make_cost_weight_*.py script wrote -- there may
# be more than one as new weight families are added, so merge all of them
# rather than hard-coding one filename.
"${AF_PROJECT_ROOT}/.venv/bin/python" - "${RUN_DIR}/run_manifest.json" "${AF_WEIGHT_FILE}" \
    "${AF_PROJECT_ROOT}"/work/costWeight_*_manifest.json <<'PYEOF'
import json, sys
run_manifest_path, weight_file, *weight_manifest_paths = sys.argv[1:]
run_manifest = json.load(open(run_manifest_path))
weight_manifest = {}
for path in weight_manifest_paths:
    weight_manifest.update(json.load(open(path)))
entry = weight_manifest[weight_file.rsplit("/", 1)[-1]]
run_manifest["target_i_global"] = entry["i_global"]
run_manifest["target_j_global"] = entry["j_global"]
run_manifest["target_name"] = entry["target_name"]
run_manifest["target_patched_from"] = weight_manifest_paths
json.dump(run_manifest, open(run_manifest_path, "w"), indent=2, sort_keys=True)
open(run_manifest_path, "a").write("\n")
print(f"patched target_i_global={entry['i_global']} target_j_global={entry['j_global']} into {run_manifest_path}")
PYEOF

cd "${RUN_DIR}"

# A bare srun fails at MPI_Init here ("PML add procs failed"); every S0
# adjoint run so far has been launched as srun --mpi=pmix -n 4.
srun --mpi=pmix -n 4 ./mitgcmuv_ad > run.log 2>&1

echo "=== ${AF_RUN_NAME} complete ==="

test -f adxx_etan.0000000000.data || {
  echo "ERROR: no gradient written; adxx_etan.0000000000.data is missing"
  tail -60 run.log
  exit 1
}

echo "--- gradient and dumps ---"
ls -l adxx_etan.0000000000.data | sed 's/^/  /'
echo "  ADJetan dumps: $(ls -1 ADJetan.*.data 2>/dev/null | wc -l)"
echo "--- cost function value ---"
grep -iE "global fc|early fc|objf_test" STDOUT.0000 | tail -5
