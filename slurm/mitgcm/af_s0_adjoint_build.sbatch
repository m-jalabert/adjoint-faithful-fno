#!/bin/bash -l
#SBATCH --job-name=af-s0-adj-build
#SBATCH --account=pod-users
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Stage 5 of docs/mitgcm_adjoint_ground_truth_plan.md: build mitgcmuv_ad.
#
# THIS IS THE ONLY STEP THAT NEEDS THE TAF LICENCE.  The rule at
# build/af_s0_ad/Makefile:1979 invokes `staf`, which uploads ad_input_code.f to
# fastopt.net and receives ad_input_code_ad.f.  Everything before it is CPP and
# everything after it is gfortran.
#
#   - staf must be on PATH.  Smoke-test it first (plan section 2).
#   - Do NOT pass -j to `make adall`; the TAF step is not parallel-safe.
#   - gfortran needs >=16 GB and 20-90 min on ad_taf_output.f, hence --mem=32G.
#
# EGRESS: fastopt.net is reachable from the login node on :80 and :443.  If
# compute nodes lack that egress this job fails at the staf line.  The fix is
# to split the build -- run `make adtaf` on the login node, then submit this
# job, which will find ad_taf_output.f already present and go straight to the
# compile.
set -euo pipefail

module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7

AF_PROJECT_ROOT="${AF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set AF_PROJECT_ROOT}}"
AF_BUILD_DIR="${AF_BUILD_DIR:-${AF_PROJECT_ROOT}/build/af_s0_ad}"
MITGCM_ROOT="${AF_PROJECT_ROOT}/external/MITgcm"

command -v staf >/dev/null 2>&1 || {
  echo "ERROR: staf is not on PATH."
  echo "  TAF is licensed from FastOpt; see docs/mitgcm_adjoint_ground_truth_plan.md section 2."
  echo "  Everything up to and including 'make ad_input_code.f' works without it."
  exit 1
}

# Regenerate the Makefile only if it is absent, so a rerun does not discard a
# completed TAF round trip.
if [ ! -f "${AF_BUILD_DIR}/Makefile" ]; then
  mkdir -p "${AF_BUILD_DIR}"
  cd "${AF_BUILD_DIR}"
  "${MITGCM_ROOT}/tools/genmake2" \
    -rootdir "${MITGCM_ROOT}" \
    -mods    "${AF_PROJECT_ROOT}/af_fno/mitgcm/code_ad" \
    -of      "${MITGCM_ROOT}/tools/build_options/linux_amd64_gfortran" \
    -adof    "${MITGCM_ROOT}/tools/adjoint_options/adjoint_default" \
    -mpi
  make -j 8 depend
fi

cd "${AF_BUILD_DIR}"
make adall

test -x mitgcmuv_ad || { echo "ERROR: mitgcmuv_ad was not produced"; exit 1; }

# taf_ad.log and the generated source are the provenance of every gradient this
# executable will produce.  A different TAF version is a different executable.
echo "=== build complete ==="
ls -l mitgcmuv_ad ad_taf_output.f
sha256sum mitgcmuv_ad
grep -m5 -i "version" taf_ad.log 2>/dev/null || true
