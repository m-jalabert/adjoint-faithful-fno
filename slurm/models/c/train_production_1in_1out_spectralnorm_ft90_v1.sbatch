#!/bin/bash -l
#SBATCH --job-name=af-c-ft90-train
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

# Ninety-day staged fine-tune of the published production operator.
#
#   parent  model_c_production_1in_1out_spectralnorm_v1/selected.pt  (step 7,680)
#     |
#     v
#   child   model_c_production_1in_1out_spectralnorm_ft90_v1
#
# Zero architecture changes. Four changes in total:
#
#   initialization   random          ->  the parent's selected.pt
#   rollout          6 (60 days)     ->  9 (90 days)
#   learning rate    5e-4 -> 1e-4    ->  5e-5, constant
#   optimizer steps  7,680           ->  1,920
#
# Everything else -- the eight loss terms and their weights, the per-mode
# spectral cap at rho = 1, the dataset, the split, the normalization, batch 8,
# Adam betas, the validation protocol and the selection rule -- is the parent's.
#
# What this targets. The parent's forecast skill is fine; its long-horizon
# behaviour is not. Its 90-360 day RMSE curve steepens rather than flattening
# (late/early secant slope 1.65 in speed, 1.73 in SST) and the day-2000 anomaly
# amplitude is order 10. It was asked to survive sixty autonomous days and was
# trained from scratch with everything active at once. Bire et al. instead learn
# the propagator first and then fine-tune on self-generated states; this run
# applies that more strongly to a model already known to work.
#
# Ninety days and not further: Bire estimate this double gyre decorrelates in
# roughly 90 days, past which pointwise truth stops being a defensible training
# target and minimizing it rewards suppressing variability.
#
# Cost. No normalization pass -- the parent's is reused byte-for-byte, which
# saves the ~13 GB streaming read the parent needed. The train-only climatology
# is still recomputed for validation. Training is 1,920 steps x 8 samples x 9
# calls = 138,240 state transitions, 37.5 % of the parent's exposure at 1.5x the
# per-step cost, so roughly 55 % of the parent's ~3.2 h. Validation is four
# 360-day rollouts over 102 pooled starts plus two 200-call growth diagnostics
# per checkpoint, as before. Eight hours is a generous ceiling.
#
# GPU memory pulls in two directions and is worth stating plainly rather than
# waving at. The batch-dependent unroll activations go DOWN: they scale as
# microbatch x calls, 2 x 9 = 18 against the parent's 4 x 6 = 24, so 0.75x. The
# batch-INDEPENDENT part goes UP: the per-mode cap materializes a normalized
# (128,128,32,17) complex weight per block on every call and each is retained by
# the graph -- 71 MB x 3 blocks x 9 calls = 1.93 GB against the parent's 1.28 GB,
# so +0.65 GB. Net it should land near the parent's footprint, and the parent
# fit. If it does not, the failure is an OOM inside the first few optimizer
# steps -- fast, obvious, and fixed by halving the microbatch to 1 and doubling
# accumulation to 8, which leaves the effective batch of 8 unchanged.
set -euo pipefail

module purge
module load gnu12/12.2.0 cuda/12.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_production_1in_1out_spectralnorm_ft90_v1.json}"
AF_PYTHON="${AF_PYTHON:-${AF_PROJECT_ROOT}/.venv/bin/python}"

cd "${AF_PROJECT_ROOT}"

if [ ! -x "${AF_PYTHON}" ]; then
  echo "Project interpreter ${AF_PYTHON} is missing or not executable." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# preflight verifies the parent's digest, loads its weights strictly into the
# declared architecture, and refuses to continue if reinstalling the spectral
# cap moved the forward map by more than 1e-2 of its own rms -- i.e. if this run
# would not actually be starting from the published operator.
"${AF_PYTHON}" -m oceanfno.finetune preflight --contract "${AF_CONTRACT}"
"${AF_PYTHON}" -m oceanfno.finetune run --contract "${AF_CONTRACT}" --device cuda
