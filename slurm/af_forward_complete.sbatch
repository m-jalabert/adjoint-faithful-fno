#!/bin/bash -l
#SBATCH --job-name=af-forward-complete
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Complete, non-training forward evaluation against the predeclared Bire-trend
# protocol. Required model/checkpoint/output variables make accidental reuse of
# an old output directory impossible.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
: "${AF_FORWARD_MODEL:?Set AF_FORWARD_MODEL to a0, a, or b}"
: "${AF_FORWARD_CHECKPOINT:?Set AF_FORWARD_CHECKPOINT to the frozen checkpoint}"
: "${AF_FORWARD_OUTPUT:?Set AF_FORWARD_OUTPUT to a new versioned output directory}"

arguments=(
  --model "${AF_FORWARD_MODEL}"
  --dataset "${AF_DATASET}"
  --checkpoint "${AF_FORWARD_CHECKPOINT}"
  --output-dir "${AF_FORWARD_OUTPUT}"
  --device cuda
)
if [[ -n "${AF_A0_COMPLETE_REFERENCE:-}" ]]; then
  arguments+=(--a0-reference "${AF_A0_COMPLETE_REFERENCE}")
fi

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.af_forward_complete "${arguments[@]}"
