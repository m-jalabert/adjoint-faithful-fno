#!/bin/bash -l
#SBATCH --job-name=af-traj-v2
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3

# Exact ten-year S0--S2 production extensions authorized by the immutable
# Model C data-adequacy decision. Each task writes a separate versioned root.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_EXECUTABLE="${AF_EXECUTABLE:-${AF_PROJECT_ROOT}/build/af_s0/mitgcmuv}"
AF_EXPANSION_CONTRACT="${AF_EXPANSION_CONTRACT:-${AF_PROJECT_ROOT}/config/trajectories_v2_expansion.json}"

cd "${AF_PROJECT_ROOT}"
experiment="$(
  "${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_trajectory_expansion \
    resolve-experiment \
    --array-index "${SLURM_ARRAY_TASK_ID}"
)"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_trajectory_expansion \
  run \
  --project-root "${AF_PROJECT_ROOT}" \
  --scratch-root "${AF_SCRATCH_ROOT}" \
  --executable "${AF_EXECUTABLE}" \
  --contract "${AF_EXPANSION_CONTRACT}" \
  --experiment "${experiment}"
