#!/bin/bash -l
#SBATCH --job-name=turb-fwd-train
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=48:00:00

# Train the 0.25-degree turbulent forward emulator, seed 20260911, 64x64 modes.
#
# Data-parallel across the GPUs of one node.  Submit with the world size on the
# command line.  It must divide both the contract's eight microbatches per
# optimizer step and the 17,820 in an epoch, which leaves 1, 2 or 4 -- never 3
# (it does not divide the step) and never 8 (it does not divide the epoch, so
# the ranks would cross the epoch boundary at different steps):
#
#   sbatch --gres=gpu:v100:2 --ntasks=2 --cpus-per-task=10 <this>
#   sbatch --gres=gpu:v100:4 --ntasks=4 --cpus-per-task=5  <this>
#
# Measured single-GPU cost is 19.2 s per optimizer step, so 7,680 steps is
# ~41 h on one card, ~21 h on two and ~10 h on four, plus a ~1.5 h tail in
# which rank 0 validates the four checkpoints and selects one, alone.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

R="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
STUDY="${R}/turbulent_double_gyre"
PY="${AF_PYTHON:-${R}/.venv/bin/python}"
CONTRACT="${AF_CONTRACT:-${STUDY}/config/turb_forward_control_v1.json}"

if [ ! -x "${PY}" ]; then echo "missing interpreter ${PY}" >&2; exit 1; fi
if [ ! -f "${CONTRACT}" ]; then echo "missing contract ${CONTRACT}" >&2; exit 1; fi

export PYTHONPATH="${STUDY}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Loader processes per rank. Each sample pulls seven 11.3 MB day slices from a
# 207 GB store that cannot sit in page cache, so this read has to overlap the step.
export AF_DATALOADER_WORKERS="${AF_DATALOADER_WORKERS:-4}"
# seed_everything asks for deterministic algorithms; without this cuBLAS quietly
# declines and the run is only reproducible up to GEMM workspace scheduling.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# 20.8 GiB peak on a 32 GB card leaves little room for allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Rendezvous for the NCCL process group; rank and size come from SLURM_PROCID
# and SLURM_NTASKS, which turbfno.distributed reads directly.
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

cd "${R}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "world size ${SLURM_NTASKS}, ${SLURM_CPUS_PER_TASK} cpus per rank"

# Gate the run on the contract audit: every pinned source and MITgcm input is
# re-hashed, the record counts are recomputed and each physics term is proved
# identically zero when truth is scored against itself.  One process, not under
# srun; a non-zero exit here means training never starts.
# Static gate: under data parallelism the output tree has one owner, and a
# single unguarded write only shows itself when a rank reaches it -- an hour in,
# past the normalizer scans.  Checked before anything is allocated.
echo "=== rank write guards ==="
"${PY}" "${STUDY}/scripts/verify_rank_write_guards.py"

echo "=== preflight ==="
"${PY}" -m turbfno.train preflight --contract "${CONTRACT}"

echo "=== train ==="
srun --kill-on-bad-exit=1 "${PY}" -m turbfno.train run \
  --contract "${CONTRACT}" --device cuda
