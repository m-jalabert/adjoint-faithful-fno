#!/bin/bash -l
#SBATCH --job-name=af-c-nomctl-train
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Execution step 11 of docs/Adjoint_faithful_response_training_plan.md: arm B,
# the exact parent-protocol replay through the common response-study runner
# with response disabled. Same architecture/data/split/schedule/optimizer/
# batch/loss as model_c_production_1in_1out_spectralnorm_v1; the only
# permitted delta from that parent (plan section 5.1) is the study identity
# (version string, output paths) and, for the two secondary seeds, the seed
# itself. Submit three times, once per AF_SEED in {20260724, 20260911,
# 20260912} -- 20260724 is the primary seed and Gate M0's comparator.
#
# Must not be submitted before scripts/verify_response_training_equivalence.py
# passes (plan section 23.2's equivalence harness) -- that script recomputes
# every deterministic pre-training artifact (record sets, per-seed nominal
# batch schedule, static block, normalization) through this same runner's
# code path and hash-matches it against the values already frozen in the
# contract's study_contract.equality_artifact_hashes.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_adjoint_faithful_nominal_control_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_SEED="${AF_SEED:?Set AF_SEED to one of 20260724 (primary), 20260911, or 20260912}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" -m oceanfno.train_response preflight --contract "${AF_CONTRACT}" --seed "${AF_SEED}"
"${AF_PYTHON}" -m oceanfno.train_response run --contract "${AF_CONTRACT}" --device cuda --seed "${AF_SEED}"
