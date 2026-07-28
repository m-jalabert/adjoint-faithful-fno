#!/bin/bash -l
#SBATCH --job-name=af-c-psi-leads
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# Save eight per-lead 1-degree streamfunction comparisons for the same frozen
# Model C checkpoint and S2 validation member used by the Bire Figure-3 plot.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_SCRATCH_ROOT="${AF_SCRATCH_ROOT:-/bigscratch/mjalabert314/bire_james25_repro/af_fno}"
AF_DATASET="${AF_DATASET:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.zarr}"
AF_QUALITY_REPORT="${AF_QUALITY_REPORT:-${AF_SCRATCH_ROOT}/datasets/trajectories_v2.quality.json}"
AF_VALIDATION_REPORT="${AF_VALIDATION_REPORT:-${AF_SCRATCH_ROOT}/models/C/successor_validation_v1/fresh_validation/model_c_successor_validation_report.json}"
AF_STREAMFUNCTION_CONTRACT="${AF_STREAMFUNCTION_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_bire_streamfunction_leads_v1.json}"
AF_STREAMFUNCTION_OUTPUT="${AF_STREAMFUNCTION_OUTPUT:-${AF_SCRATCH_ROOT}/models/C/bire_streamfunction_leads_v1}"
AF_STREAMFUNCTION_PROJECT_OUTPUT="${AF_STREAMFUNCTION_PROJECT_OUTPUT:-${AF_PROJECT_ROOT}/outputs/af_fno/C/bire_streamfunction_leads_v1}"

cd "${AF_PROJECT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${AF_PROJECT_ROOT}/.venv/bin/python" \
  -m bire_repro.af_model_c_bire_streamfunction_leads \
  --dataset "${AF_DATASET}" \
  --quality-report "${AF_QUALITY_REPORT}" \
  --validation-report "${AF_VALIDATION_REPORT}" \
  --contract "${AF_STREAMFUNCTION_CONTRACT}" \
  --output-dir "${AF_STREAMFUNCTION_OUTPUT}" \
  --figure-dir "${AF_STREAMFUNCTION_PROJECT_OUTPUT}" \
  --device cuda
