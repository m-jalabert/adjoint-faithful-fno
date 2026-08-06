#!/bin/bash -l
#SBATCH --job-name=af-c-bproftanom
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Streamfunction-anomaly companions to figures 3 and 7 of the rollout fine-tune.
#
#   psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)
#
# with psi_bar_S0 the MITgcm time-mean barotropic streamfunction over the S0
# TRAINING block, days 0-5999.  The same two-dimensional field is subtracted
# from truth and from prediction; the model's own time mean is deliberately not
# used, because that would absorb any bias in the mean circulation it carries.
#
# This package ADDS two plates.  It modifies no published figure, reads no model
# weights, and rolls nothing out -- it consumes the sealed figure package's
# published arrays, whose SHA-256 the contract pins.  Amplitude diagnostics,
# including the day-2000 streamfunction minimum, stay defined on the total
# field and are unaffected.
#
# No GPU: the only real work is one pass over 6,000 training days to build the
# reference field, which takes seconds against the consolidated Zarr store.
set -euo pipefail

module purge
module load gnu12/12.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_protocol_rollout_ft_s0_anomaly_v2.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

"${AF_PYTHON}" \
  -m oceanfno.anomaly \
  preflight \
  --contract "${AF_CONTRACT}"

"${AF_PYTHON}" \
  -m oceanfno.anomaly \
  run \
  --contract "${AF_CONTRACT}"
