#!/bin/bash -l
#SBATCH --job-name=turb-fwd-resume
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00

# Validate the four checkpoints an interrupted run already wrote, select one and
# publish.  The training loop is skipped and the normalization artifact is read
# back rather than recomputed, so this picks up where job 446822 stopped: it
# finished all 7,680 optimizer steps and then exhausted the card in validation,
# which was rolling all 102 members as a single batch.
#
# One process, one GPU: validation and selection are rank 0's alone, so there is
# nothing for a second rank to do.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

R="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
STUDY="${R}/turbulent_double_gyre"
PY="${AF_PYTHON:-${R}/.venv/bin/python}"
CONTRACT="${AF_CONTRACT:-${STUDY}/config/turb_forward_control_v1.json}"

export PYTHONPATH="${STUDY}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export AF_DATALOADER_WORKERS=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Validation members carried through the 360-day rollout together. Every
# reduction over members is taken after all chunks have contributed, so this
# changes memory and nothing else.
export AF_VALIDATION_MEMBER_CHUNK="${AF_VALIDATION_MEMBER_CHUNK:-4}"

cd "${R}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

echo "=== rank write guards ==="
"${PY}" "${STUDY}/scripts/verify_rank_write_guards.py"

echo "=== resume validation ==="
"${PY}" -m turbfno.train run --contract "${CONTRACT}" --device cuda --resume-validation
