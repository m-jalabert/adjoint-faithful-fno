#!/bin/bash -l
#SBATCH --job-name=af-a0-paper-plots
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Re-evaluate the frozen A0 checkpoint only.  This produces paper-style
# diagnostics in a new project-facing output directory; it never changes A0.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v1.zarr}"
AF_A0_CHECKPOINT="${AF_A0_CHECKPOINT:-${AF_SCRATCH_ROOT}/models/A0/final_v1/a0_final.pt}"
AF_A0_PAPER_OUTPUT="${AF_A0_PAPER_OUTPUT:-${AF_PROJECT_ROOT}/outputs/af_fno/A0/paper_style_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" -m bire_repro.analysis.af_a0_paper_plots \
  --dataset "${AF_DATASET}" \
  --checkpoint "${AF_A0_CHECKPOINT}" \
  --output-dir "${AF_A0_PAPER_OUTPUT}" \
  --device cuda
