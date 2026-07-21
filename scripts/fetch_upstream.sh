#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

mkdir -p "${EXTERNAL_DIR}"

if [[ ! -d "${EXTERNAL_DIR}/MITgcm/.git" ]]; then
  git clone --depth 1 --branch checkpoint68j \
    https://github.com/MITgcm/MITgcm.git "${EXTERNAL_DIR}/MITgcm"
fi
if [[ ! -d "${EXTERNAL_DIR}/oceanfourcast/.git" ]]; then
  git clone --depth 1 \
    https://github.com/suyashbire1/oceanfourcast.git "${EXTERNAL_DIR}/oceanfourcast"
fi

MIT_SHA="$(git -C "${EXTERNAL_DIR}/MITgcm" rev-parse HEAD)"
PAPER_CODE_SHA="$(git -C "${EXTERNAL_DIR}/oceanfourcast" rev-parse HEAD)"
[[ "${MIT_SHA}" == "f03a2f5e214bc57b8393f6201a6a1266dd1f53d6" ]]
[[ "${PAPER_CODE_SHA}" == "96b814af00e36665878ef295f94f766b2420b18a" ]]

echo "Pinned upstream repositories verified."

