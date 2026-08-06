#!/bin/bash
#SBATCH --job-name=af-c-s0-spgate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-/home/mjalabert314/bire_james25_repro}"
AF_CONTRACT="${AF_CONTRACT:-${AF_PROJECT_ROOT}/config/model_c_s0_reflected_spectral_gate_control_v1.json}"

cd "${AF_PROJECT_ROOT}"
source .venv/bin/activate

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -m bire_repro.af_model_c_s0_reflected_spectral_gate_control \
  preflight \
  --contract "${AF_CONTRACT}"
python -m bire_repro.af_model_c_s0_reflected_spectral_gate_control \
  run \
  --contract "${AF_CONTRACT}" \
  --device cuda
