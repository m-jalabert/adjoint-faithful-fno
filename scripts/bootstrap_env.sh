#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${BIRE_PYTHON:-python3}"

module purge
module load gnu14/14.2.0
module load cuda/12.4
module load anaconda/3.12

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install \
  torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
"${ENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}[fno,dev]"
"${ENV_DIR}/bin/python" -m pip freeze > "${ROOT_DIR}/env/requirements-resolved.txt"

echo "Environment ready: ${ENV_DIR}"
