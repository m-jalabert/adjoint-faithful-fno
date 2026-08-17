#!/usr/bin/env bash
# Build one 0.25-degree turbulent MITgcm executable for a given rank layout.
#
#   scripts/build_af_turb.sh <nPx> <nPy> [build_suffix]
#
# 248 = 8 x 31, so nPx and nPy must each divide 248; the tile is 248/nPx by
# 248/nPy.  The executable lands in build/af_turb<suffix>/mitgcmuv and the code
# directory it was compiled from is kept beside it for provenance.
set -euo pipefail

npx="${1:?usage: build_af_turb.sh <nPx> <nPy> [suffix]}"
npy="${2:?usage: build_af_turb.sh <nPx> <nPy> [suffix]}"
suffix="${3:-_${npx}x${npy}}"

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SCRIPT_ROOT}}"
source_dir="${AF_PROJECT_ROOT}/external/MITgcm"
code_dir="${AF_PROJECT_ROOT}/af_fno/mitgcm/code_turb${suffix}"
build_dir="${AF_PROJECT_ROOT}/build/af_turb${suffix}"

snx=$((248 / npx))
sny=$((248 / npy))
if [[ $((snx * npx)) -ne 248 || $((sny * npy)) -ne 248 ]]; then
  echo "nPx=${npx} nPy=${npy} does not tile 248 x 248" >&2
  exit 1
fi

mkdir -p "${code_dir}"
cp "${AF_PROJECT_ROOT}/af_fno/mitgcm/code_turb/DIAGNOSTICS_SIZE.h" "${code_dir}/"
cp "${AF_PROJECT_ROOT}/af_fno/mitgcm/code_turb/packages.conf" "${code_dir}/"
cat >"${code_dir}/SIZE.h" <<EOF
CBOP
C    !ROUTINE: SIZE.h
C    !DESCRIPTION: AF--FNO 0.25-degree turbulent grid on $((npx * npy)) MPI ranks
C                  (${npx} x ${npy} decomposition, ${snx} x ${sny} tiles).
CEOP
      INTEGER sNx, sNy, OLx, OLy
      INTEGER nSx, nSy, nPx, nPy
      INTEGER Nx, Ny, Nr
      PARAMETER (
     &           sNx = ${snx},
     &           sNy = ${sny},
     &           OLx =  2,
     &           OLy =  2,
     &           nSx =  1,
     &           nSy =  1,
     &           nPx =  ${npx},
     &           nPy =  ${npy},
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = 15)

      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )
EOF

mkdir -p "${build_dir}"
cd "${build_dir}"
if [[ ! -f Makefile ]]; then
  "${source_dir}/tools/genmake2" \
    -rootdir "${source_dir}" \
    -mods "${code_dir}" \
    -mpi \
    -of "${source_dir}/tools/build_options/linux_amd64_gfortran" >genmake.stdout
fi
make depend >make_depend.stdout 2>&1
make -j "${MAKE_JOBS:-8}" >make.stdout 2>&1
echo "built ${build_dir}/mitgcmuv  (${npx} x ${npy} ranks, ${snx} x ${sny} tiles)"
