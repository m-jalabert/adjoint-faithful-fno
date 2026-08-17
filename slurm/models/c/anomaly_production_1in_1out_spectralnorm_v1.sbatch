#!/bin/bash -l
#SBATCH --job-name=af-c-prodsn-anom
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

# Streamfunction-anomaly companions to figures 3 and 7. Reads the sealed figure
# arrays and model-visible MITgcm training state only: no weights are loaded and
# nothing is rolled out, so this needs no GPU.
#
# `finalize` pins the published figure report / arrays / manifest digests; run
# it only after the figure job has completed.
set -euo pipefail

module purge
module load gnu12/12.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

"${AF_PYTHON}" -m oceanfno.anomaly finalize --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.anomaly preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.anomaly run --contract "${AF_CONTRACT}"
