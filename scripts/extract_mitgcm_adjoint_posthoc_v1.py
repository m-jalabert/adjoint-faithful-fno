"""Package the post-hoc interior/eastern MITgcm/TAF adjoint runs.

**Not part of the frozen adjoint-faithful contract** -- see
scripts/make_cost_weight_posthoc_v1.py for why: Gate A1 already closed v1
negative on 2026-08-29, and the interior/eastern exploratory extension in
config/adjoint_faithful_blind_adjoint_evaluation_v1.json was never unlocked
before B/C training. Nothing here is confirmatory or exploratory evidence
under that contract; it is a post-hoc "why did it fail" diagnostic.

Reads the twelve run directories staged by
slurm/mitgcm/af_s0_adjoint_posthoc_v1.sbatch under
/bigscratch/.../mitgcm_adjoint_posthoc_v1/{P10,P30,P90,K10,K30,K90}_{interior,eastern}
and writes one .npz per target, keyed to match
outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz so the
same downstream loader reads both:

    S_ssh_anomaly, S_ssh_anomaly_kernel   (3, 62, 62), lead_days=[10,30,90]
    S_backward                            (91, 62, 62), backward_days
    w_ssh_anomaly, w_ssh_anomaly_kernel   (62, 62)
    wet_mask, rA, target_ij               copied from mitgcm_s0_adjoint_v2

S_mean_only_backward is not recomputed: the mean-only functional is the
domain-mean of eta, not tied to any (i,j), so its adjoint is target-
independent and the existing v2 array already covers it (see v2/README.md
gate G3).

Gates run here are deliberately narrow -- this reuses an already-validated
executable and already-passed grdchk points, so it is not re-deriving G0-G5.
It only checks the two things a new (weight, run) pairing can get wrong:

    weight identity  each run's staged costWeight.bin sha256 matches the
                      target it claims to be, per
                      work/costWeight_posthoc_v1_manifest.json
    executable       every run used the byte-identical mitgcmuv_ad already
                      validated for the frozen v1/v2 products (no TAF rebuild)
    land             S is exactly 0 on every dry cell, finite everywhere
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from extract_mitgcm_adjoint import ExtractionError, adjetan_series, global_fc, read_mds  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_ROOT = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_posthoc_v1")
V2_NPZ = PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1"
WEIGHT_MANIFEST = PROJECT_ROOT / "work/costWeight_posthoc_v1_manifest.json"
MDS_DTYPE = ">f4"

TARGETS = {
    "interior": (31, 17),  # (i_global, j_global), 1-based
    "eastern": (61, 17),
}
LEAD_DAYS = (10, 30, 90)
SOURCE_DAY = 7200


def _run_dir(base_mode: str, target: str) -> Path:
    return SCRATCH_ROOT / f"{base_mode}_{target}"


def _check_executable(run_dirs: list[Path], expected_sha256: str) -> None:
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if manifest["executable_sha256"] != expected_sha256:
            raise ExtractionError(
                f"{run_dir} used a different mitgcmuv_ad ({manifest['executable_sha256']}) "
                f"than the frozen v1/v2 executable ({expected_sha256}) -- this would be a "
                "silent TAF rebuild, not a reused binary"
            )


def _check_weight_identity(run_dirs: list[Path], weight_manifest: dict[str, Any]) -> None:
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        weight_name = Path(manifest["weight_file"]).name
        expected = weight_manifest[weight_name]["sha256"]
        if manifest["weight_sha256"] != expected:
            raise ExtractionError(
                f"{run_dir} staged {weight_name} with sha256 {manifest['weight_sha256']}, "
                f"expected {expected} -- the two sides may be weighting eta differently"
            )


def extract_target(name: str, i_global: int, j_global: int, v2: np.lib.npyio.NpzFile) -> dict[str, Any]:
    point_dirs = [_run_dir(f"P{lead}", name) for lead in LEAD_DAYS]
    kernel_dirs = [_run_dir(f"K{lead}", name) for lead in LEAD_DAYS]
    p90_dir = _run_dir("P90", name)

    weight_manifest = json.loads(WEIGHT_MANIFEST.read_text())
    all_dirs = point_dirs + kernel_dirs
    executable_sha256 = json.loads((all_dirs[0] / "run_manifest.json").read_text())["executable_sha256"]
    reference_v1_manifest = json.loads(
        (PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v1/report.json").read_text()
    )
    _check_executable(all_dirs, executable_sha256)  # internal consistency across the six runs
    _check_weight_identity(all_dirs, weight_manifest)

    wet = np.asarray(v2["wet_mask"]).astype(bool)

    S_ssh_anomaly = np.stack([read_mds(d / "adxx_etan.0000000000") for d in point_dirs])
    S_ssh_anomaly_kernel = np.stack([read_mds(d / "adxx_etan.0000000000") for d in kernel_dirs])
    S_backward, backward_days = adjetan_series(p90_dir, SOURCE_DAY, SOURCE_DAY + 90)

    for label, field in (
        ("S_ssh_anomaly", S_ssh_anomaly),
        ("S_ssh_anomaly_kernel", S_ssh_anomaly_kernel),
        ("S_backward", S_backward),
    ):
        if not np.all(np.isfinite(field)):
            raise ExtractionError(f"{name}/{label} has non-finite values")
        land = ~wet
        offenders = int(np.count_nonzero(field[:, land]))
        if offenders:
            raise ExtractionError(f"{name}/{label} is nonzero on {offenders} land cells")

    w_ssh_anomaly = np.fromfile(
        PROJECT_ROOT / f"work/costWeight_{name}_ssh_anomaly.bin", dtype=MDS_DTYPE
    ).reshape(62, 62).astype(np.float64)
    w_ssh_anomaly_kernel = np.fromfile(
        PROJECT_ROOT / f"work/costWeight_{name}_ssh_anomaly_kernel.bin", dtype=MDS_DTYPE
    ).reshape(62, 62).astype(np.float64)

    fc_by_run = {d.name: global_fc(d) for d in all_dirs}

    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"mitgcm_s0_adjoint_posthoc_{name}_v1.npz"
    np.savez(
        npz_path,
        S_ssh_anomaly=S_ssh_anomaly,
        S_ssh_anomaly_kernel=S_ssh_anomaly_kernel,
        lead_days=np.asarray(LEAD_DAYS, dtype=np.int64),
        S_backward=S_backward,
        backward_days=backward_days,
        w_ssh_anomaly=w_ssh_anomaly,
        w_ssh_anomaly_kernel=w_ssh_anomaly_kernel,
        wet_mask=v2["wet_mask"],
        rA=v2["rA"],
        target_ij=np.asarray([j_global - 1, i_global - 1], dtype=np.int64),
    )

    report = {
        "version": "mitgcm_s0_adjoint_posthoc_v1",
        "status": "non_confirmatory_posthoc_diagnostic",
        "not_covered_by": "config/adjoint_faithful_blind_adjoint_evaluation_v1.json exploratory block "
        "(pretraining_manifest was never created; late creation means no exploratory test runs)",
        "target_name": name,
        "target_i_global": i_global,
        "target_j_global": j_global,
        "grdchk_reference": "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/README.md G1-90 table, "
        f"point ({i_global}, {j_global}), passed before B/C training completed",
        "executable_sha256": executable_sha256,
        "matches_frozen_v1_executable_sha256": executable_sha256
        == reference_v1_manifest["executable_sha256"],
        "global_fc_by_run": fc_by_run,
        "npz_path": str(npz_path),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {npz_path}")
    print(f"wrote {report_path}")
    return report


def main() -> int:
    with np.load(V2_NPZ) as v2:
        v2_arrays = {k: v2[k] for k in ("wet_mask", "rA")}
    for name, (i_global, j_global) in TARGETS.items():
        extract_target(name, i_global, j_global, v2_arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
