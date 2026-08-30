#!/bin/bash -l
#SBATCH --job-name=af-resp-inv-v2
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=slurm/response_inventory/logs/af-resp-inv-v2-%x-%j.out

# Step 4 of docs/Adjoint_faithful_response_training_plan.md: materialize the
# public (pilot+train+validation) and write-once blind geometry inventories
# under config/forward_response_dataset_v2.json.
#
# Runs with --max-workers 1 (fully sequential over the twelve
# (regime,family) groups), not the default one-worker-per-group parallelism.
# Verified directly on this shared, multi-user machine: running all twelve
# groups concurrently stalled for 90+ minutes with zero progress even after
# pinning each worker to one BLAS thread (ruling out thread oversubscription
# specifically -- confirmed via /proc/PID/status showing exactly one OS
# thread per worker), while an identical single case run in true isolation
# (no other worker processes competing) solved in 503s, matching the
# original small-scale benchmark. The parallel path's contention has a real
# but not yet root-caused cause; running sequentially avoids it entirely by
# construction, in exchange for wall time -- estimated 15-20+ hours across
# all twelve groups (very approximate: extrapolated from a single confirmed
# 503s WBC-region timing and partial multi-region timing on one family,
# not a full dry run), hence the generous 48-hour budget rather than a
# tight one.
#
# This writes real output (config/forward_response_dataset_v2.json's public
# manifest at 0444 and the blind manifest at 0400, both write-once/O_EXCL --
# see materialize_inventory's docstring). No response/blind numeric data is
# created; only geometry.
set -euo pipefail

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

echo "materialize_forward_response_inventory_v2: starting at $(date)"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"

"${AF_PYTHON}" -u scripts/build_forward_response_inventory.py materialize \
  --max-workers 1

echo "materialize_forward_response_inventory_v2: finished at $(date)"
