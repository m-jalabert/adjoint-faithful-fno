#!/bin/bash -l
#SBATCH --job-name=turb-anomaly
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Streamfunction-anomaly companions to figures 3 and 7.  This package reads no
# model weights and rolls nothing out -- it subtracts the MITgcm S0_turb
# training-mean streamfunction from the fields the figure package already
# published -- so it needs no GPU.
set -euo pipefail
module purge; module load gnu12/12.2.0
R="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
STUDY="${R}/turbulent_double_gyre"
PY="${AF_PYTHON:-${R}/.venv/bin/python}"
CONTRACT="${STUDY}/config/turb_forward_control_v1_s0_turb_anomaly_v1.json"
export PYTHONPATH="${STUDY}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
cd "$R"
"${PY}" -m turbfno.anomaly preflight --contract "${CONTRACT}"
"${PY}" -m turbfno.anomaly run --contract "${CONTRACT}"
