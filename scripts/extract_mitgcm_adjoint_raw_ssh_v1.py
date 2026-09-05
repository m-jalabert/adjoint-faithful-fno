"""Package the raw-SSH (point_only) MITgcm/TAF runs, at all three targets.

Non-confirmatory, same status as extract_mitgcm_adjoint_posthoc_v1.py.

Reads the nine run directories staged by
slurm/mitgcm/af_s0_adjoint_posthoc_v1.sbatch under
mitgcm_adjoint_posthoc_v1/{P10,P30,P90}_{western,interior,eastern}_raw and
writes one .npz per target:

    S_point_only   (3, 62, 62), lead_days=[10,30,90]
    w_point_only   (62, 62)
    wet_mask, rA, target_ij

Because MITgcm's adjoint is exactly linear in the cost weight (see
make_cost_weight_raw_ssh_v1.py), this also checks

    S_point_only == S_ssh_anomaly - w_mean_only

against each target's existing ssh_anomaly product (the frozen
mitgcm_s0_adjoint_v2.npz for western, the round-1 posthoc npz for
interior/eastern) -- an independent verification that the new weight family
is wired correctly, at the same precision Gate G3 already established
(worst 3.57e-8 over 91 dumps in the frozen record).
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

from extract_mitgcm_adjoint import ExtractionError, read_mds  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_ROOT = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_posthoc_v1")
V2_NPZ = PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz"
WEIGHT_MANIFEST = PROJECT_ROOT / "work/costWeight_raw_ssh_v1_manifest.json"
V1_REPORT = PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v1/report.json"
MDS_DTYPE = ">f4"
LEAD_DAYS = (10, 30, 90)

TARGETS = {
    "western": (2, 17),
    "interior": (31, 17),
    "eastern": (61, 17),
}


def _run_dir(base_mode: str, target: str) -> Path:
    return SCRATCH_ROOT / f"{base_mode}_{target}_raw"


def _ssh_anomaly_reference(target: str, wet: np.ndarray) -> np.ndarray:
    """S_ssh_anomaly at leads (10, 30, 90) for ``target``, from whichever
    already-extracted product owns it -- never recomputed."""

    if target == "western":
        with np.load(V2_NPZ) as v2:
            leads = list(int(v) for v in v2["lead_days"])
            return np.stack([v2["S_ssh_anomaly"][leads.index(lead)] for lead in LEAD_DAYS])
    path = (
        PROJECT_ROOT
        / f"outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/{target}/mitgcm_s0_adjoint_posthoc_{target}_v1.npz"
    )
    with np.load(path) as store:
        leads = list(int(v) for v in store["lead_days"])
        return np.stack([store["S_ssh_anomaly"][leads.index(lead)] for lead in LEAD_DAYS])


def extract_target(name: str, i_global: int, j_global: int) -> dict[str, Any]:
    run_dirs = [_run_dir(f"P{lead}", name) for lead in LEAD_DAYS]
    weight_manifest = json.loads(WEIGHT_MANIFEST.read_text())
    reference_v1_manifest = json.loads(V1_REPORT.read_text())

    executable_sha256 = json.loads((run_dirs[0] / "run_manifest.json").read_text())["executable_sha256"]
    for run_dir in run_dirs:
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if run_manifest["executable_sha256"] != executable_sha256:
            raise ExtractionError(f"{run_dir}: executable differs across the target's own three runs")
        weight_name = Path(run_manifest["weight_file"]).name
        expected = weight_manifest[weight_name]["sha256"]
        if run_manifest["weight_sha256"] != expected:
            raise ExtractionError(f"{run_dir}: staged weight sha256 does not match {weight_name}'s manifest entry")
    matches_frozen_executable = executable_sha256 == reference_v1_manifest["executable_sha256"]

    with np.load(V2_NPZ) as v2:
        wet = np.asarray(v2["wet_mask"]).astype(bool)
        rA = np.asarray(v2["rA"])
        w_mean_only = np.asarray(v2["w_mean_only"], dtype=np.float64)

    S_point_only = np.stack([read_mds(d / "adxx_etan.0000000000") for d in run_dirs])
    if not np.all(np.isfinite(S_point_only)):
        raise ExtractionError(f"{name}/S_point_only has non-finite values")
    offenders = int(np.count_nonzero(S_point_only[:, ~wet]))
    if offenders:
        raise ExtractionError(f"{name}/S_point_only is nonzero on {offenders} land cells")

    w_point_only = np.fromfile(
        PROJECT_ROOT / f"work/costWeight_{name}_point_only.bin", dtype=MDS_DTYPE
    ).reshape(62, 62).astype(np.float64)

    # linearity check: S_point_only should equal S_ssh_anomaly - w_mean_only exactly
    # (to Gate G3 precision), since MITgcm's adjoint is linear in the cost weight.
    ssh_anomaly_reference = _ssh_anomaly_reference(name, wet)
    predicted_point_only = ssh_anomaly_reference - w_mean_only[None, :, :]
    linearity_relative_l2 = [
        float(np.linalg.norm((S_point_only[k] - predicted_point_only[k])[wet]))
        / max(float(np.linalg.norm(predicted_point_only[k][wet])), 1e-30)
        for k in range(len(LEAD_DAYS))
    ]
    worst_linearity_error = max(linearity_relative_l2)
    if worst_linearity_error > 1e-4:
        raise ExtractionError(
            f"{name}: S_point_only disagrees with S_ssh_anomaly - w_mean_only by relative L2 "
            f"{worst_linearity_error:.2e} (expected Gate-G3-level agreement, <1e-4) -- the new "
            "weight family may be wired incorrectly"
        )

    out_dir = PROJECT_ROOT / f"outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"mitgcm_s0_adjoint_raw_ssh_{name}_v1.npz"
    np.savez(
        npz_path,
        S_point_only=S_point_only,
        lead_days=np.asarray(LEAD_DAYS, dtype=np.int64),
        w_point_only=w_point_only,
        wet_mask=wet,
        rA=rA,
        target_ij=np.asarray([j_global - 1, i_global - 1], dtype=np.int64),
    )

    report = {
        "version": "mitgcm_s0_adjoint_raw_ssh_v1",
        "status": "non_confirmatory_posthoc_diagnostic",
        "objective": "point_only (raw SSH, no basin-mean subtraction)",
        "target_name": name,
        "target_i_global": i_global,
        "target_j_global": j_global,
        "executable_sha256": executable_sha256,
        "matches_frozen_v1_executable_sha256": matches_frozen_executable,
        "linearity_check": {
            "identity": "S_point_only == S_ssh_anomaly - w_mean_only",
            "relative_l2_by_lead": dict(zip(LEAD_DAYS, linearity_relative_l2)),
            "worst": worst_linearity_error,
            "reference_gate": "Gate G3, mitgcm_s0_adjoint_v2 (worst 3.57e-8 over 91 dumps)",
        },
        "npz_path": str(npz_path),
    }
    (out_dir / "raw_ssh_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {npz_path}")
    print(f"  linearity check worst relative L2 = {worst_linearity_error:.3e}")
    return report


def main() -> int:
    for name, (i_global, j_global) in TARGETS.items():
        extract_target(name, i_global, j_global)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
