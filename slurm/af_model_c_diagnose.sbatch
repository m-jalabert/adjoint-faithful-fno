#!/bin/bash -l
#SBATCH --job-name=af-model-c-diagnose
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Model C stage C0: read only the frozen S0--S2 training split.  Measure
# ten-day increment scales, temporal dependence, and tapered state/increment
# energy captured by the predeclared 12/16/24-mode candidates.  No model is
# trained and validation/inference records are not opened.
set -euo pipefail

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_MODEL_C_DIAGNOSTICS="${AF_MODEL_C_DIAGNOSTICS:-${AF_SCRATCH_ROOT}/models/C/training_diagnostics_v2}"

cd "${AF_PROJECT_ROOT}"
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_model_c_diagnostics \
  --dataset "${AF_DATASET}" \
  --output-dir "${AF_MODEL_C_DIAGNOSTICS}" \
  --samples 96 \
  --seed 20260723
