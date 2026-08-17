#!/bin/bash -l
#SBATCH --job-name=af-turb-build
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=slurm/mitgcm/logs/af-turb-build-%j.out

# Compile the 0.25-degree turbulent MITgcm executable from af_fno/mitgcm/code_turb.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SOURCE="${AF_PROJECT_ROOT}/external/MITgcm"
AF_BUILD="${AF_PROJECT_ROOT}/build/af_turb"
AF_CODE="${AF_PROJECT_ROOT}/af_fno/mitgcm/code_turb"
mkdir -p "${AF_BUILD}"
cd "${AF_BUILD}"

if [[ ! -f Makefile ]]; then
  "${AF_SOURCE}/tools/genmake2" \
    -rootdir "${AF_SOURCE}" \
    -mods "${AF_CODE}" \
    -mpi \
    -of "${AF_SOURCE}/tools/build_options/linux_amd64_gfortran"
fi
make depend
make -j "${SLURM_CPUS_PER_TASK}"

"${AF_PROJECT_ROOT}/.venv/bin/python" "${AF_PROJECT_ROOT}/scripts/af_turb_trajectories.py" build-manifest \
  --project-root "${AF_PROJECT_ROOT}" \
  --executable "${AF_BUILD}/mitgcmuv"
