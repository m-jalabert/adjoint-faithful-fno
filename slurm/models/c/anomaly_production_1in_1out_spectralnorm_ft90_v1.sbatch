#!/bin/bash -l
#SBATCH --job-name=af-c-ft90-anom
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

# Streamfunction-anomaly companions to the fine-tune's figures 3 and 7.
#
#   psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)
#
# psi_bar_S0 is MITgcm's own time-mean barotropic streamfunction over training
# days 0-5999 -- the *same* field the parent's anomaly package removed. That
# identity is what makes the two arms' transient amplitudes comparable, and it
# is asserted by the contract rather than assumed: the reference depends on
# neither model, so no bias in either one's stationary circulation can hide
# inside its own anomaly.
#
# With the standing gyre removed, a day-2,000 anomaly RMS ratio below one means
# the transients were damped and above one that they were manufactured. The
# parent's published ratio and western-boundary/interior structure are read from
# its sealed anomaly report, pinned by digest, and tabulated beside this arm's.
#
# Reads the sealed figure arrays and model-visible MITgcm training state only:
# no weights are loaded and nothing is rolled out, so this needs no GPU.
#
# `finalize` pins the fine-tuned figure package's contract / report / arrays /
# manifest digests; run it only after that figure job has completed.
set -euo pipefail

module purge
module load gnu12/12.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_production_1in_1out_spectralnorm_ft90_v1_s0_anomaly_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

"${AF_PYTHON}" -m oceanfno.anomaly_ft90 finalize  --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.anomaly_ft90 preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.anomaly_ft90 run       --contract "${AF_CONTRACT}"
