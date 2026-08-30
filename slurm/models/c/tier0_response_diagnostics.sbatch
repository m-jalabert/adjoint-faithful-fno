#!/bin/bash -l
#SBATCH --job-name=af-c-tier0-resp-diag
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Tier-0 response diagnostics: the two baselines step 12 never measured --
# arm B's own S_resp_10:60 at every saved step (the matched lambda-zero
# value that makes the four candidates' 12.13/13.96/15.01/14.82
# interpretable), and the reference scale of section 14.2's training loss
# l_{q,k} (the zero-response predictor, ~1.0 by construction of d).
#
# Inference only: trains nothing, writes no decision artifact, and does not
# touch lambda_selection_result.json. Reads only the train/validation roles
# of the curated forward-response store and arm B's published,
# hash-verified checkpoints and normalizer -- no adjoint, TAF, FNO adjoint
# map, blind response case, or nested final-inference start.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" scripts/tier0_response_diagnostics.py --device cuda
