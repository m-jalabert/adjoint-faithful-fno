#!/bin/bash -l
#SBATCH --job-name=af-c-nchan
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00

# Successor to the two-in / one-out arm: static channels only.
#
#   parent    [tau_x, wet mask, distance to wall]
#   this arm  [tau_x, wet mask, f(phi), dx(phi), theta_clim(x, y)]
#
# The map, the two-time-level input and the 32x32 Fourier modes are unchanged.
# f is derived from the store's latitude (== MITgcm YC), dx is MITgcm's own DXF
# at tracer points, and theta_clim is the thetaClimFile the run restores towards
# on a 30-day timescale; all three are pinned by SHA-256 and the MITgcm `data`
# declaration is parsed before they are trusted. The external block grows 95 ->
# 97 and lifting 97 -> 99. Unlike earlier arms the warm start is NOT
# function-preserving -- the parent has trained weights on the removed
# distance-to-wall channel -- so preflight measures and reports the size of the
# step before any optimization.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_2in_1out_new_channels_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  echo "The /sw software mount was repointed and anaconda-3.12 is gone;" >&2
  echo "restore a Python 3.12 and repoint or rebuild .venv before submitting." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${AF_PYTHON}" \
  -m oceanfno.train \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m oceanfno.train \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
