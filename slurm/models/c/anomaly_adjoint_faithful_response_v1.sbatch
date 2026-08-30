#!/bin/bash -l
#SBATCH --job-name=af-c-anomaly-resp
#SBATCH --account=pod-users
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Execution step 15 of docs/Adjoint_faithful_response_training_plan.md, plan
# section 19 step 4: the streamfunction-anomaly companions for one B/C arm and
# seed, taken about MITgcm's derived S0 time-mean barotropic streamfunction
# over parent training days 0-5,999 and subtracted identically from truth and
# from the model. A model's own mean is never subtracted.
#
# Reads no model weights and rolls nothing out -- it operates on the sealed
# figure package's arrays -- so this runs on CPU, not the GPU partition.
#
# Runs through src/oceanfno/anomaly_response.py, a contract adapter: every
# numerical helper is imported from the frozen src/oceanfno/anomaly.py and
# executed unchanged; only the identity binding, which that module hard-wires
# to the parent and whose file the parent's own contract pins, is re-expressed.
set -euo pipefail

module purge
module load gnu12/12.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"
AF_ANOMALY_CONTRACT="${AF_ANOMALY_CONTRACT:?Set AF_ANOMALY_CONTRACT to one of the six B/C anomaly contracts}"

cd "${AF_PROJECT_ROOT}"
export PYTHONPATH="${AF_PROJECT_ROOT}/src:${PYTHONPATH:-}"

"${AF_PYTHON}" -m oceanfno.anomaly_response finalize  --contract "${AF_ANOMALY_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.anomaly_response preflight --contract "${AF_ANOMALY_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.anomaly_response run       --contract "${AF_ANOMALY_CONTRACT}"
