"""Emit the figure and anomaly contracts for the turbulent forward study.

Both are generated from the module constants and the published training report
rather than hand-written, so no field can disagree with the code that audits it.
Run after the training run has published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "turbulent_double_gyre"
sys.path.insert(0, str(STUDY / "src"))

from turbfno import anomaly as A  # noqa: E402
from turbfno import figures as F  # noqa: E402
from turbfno import plots as P  # noqa: E402
from turbfno.dataset import STATIC_FEATURES  # noqa: E402
from turbfno.model import ProductionArchitecture  # noqa: E402
from turbfno.objective import LOSS_CONTRACT_SHA256  # noqa: E402
from turbfno.train import ROLLOUT_STEPS, SEED, VERSION as TRAIN_VERSION  # noqa: E402

SCRATCH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno")
TRAIN_PROJECT = STUDY / "outputs" / TRAIN_VERSION
TRAIN_SCRATCH = SCRATCH / "models" / "turb" / TRAIN_VERSION
TURB_SPINUP = SCRATCH / "mitgcm_turb_v1" / "S0_turb" / "spinup" / "years_000_010"
DATASET = SCRATCH / "datasets" / "trajectories_turb_v1.zarr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def figure_contract() -> dict:
    report_path = TRAIN_PROJECT / f"{TRAIN_VERSION}_report.json"
    report = json.loads(report_path.read_text())
    step = int(report["published_checkpoint"]["optimizer_step"])
    return {
        "version": F.VERSION,
        "contract_status": F.CONTRACT_STATUS,
        "purpose": (
            "the frozen S0_turb evaluation suite for turb_forward_control_v1, on "
            "the identical 15-start protocol, leads, fields and baselines the "
            "1-degree package used, so the two resolutions are directly comparable"
        ),
        "dataset": {
            "path": str(DATASET),
            "version": F.DATASET_VERSION,
            "train": list(F.TRAIN_RANGE),
            "validation": list(F.VALIDATION_RANGE),
            "inference": list(F.INFERENCE_RANGE),
            "tau0_n_m2": dict(F.TAU0_N_M2),
        },
        "baselines": dict(F._EXPECTED_BASELINES),
        "truth": dict(F._EXPECTED_TRUTH),
        "protocol": {
            "member_count": F.MEMBER_COUNT,
            "start_seed": F.START_SEED,
            "start_draw_order": [int(v) for v in F.declared_inference_starts()],
            "start_window": list(F.INFERENCE_START_RANGE),
            "inference_set": list(F.INFERENCE_RANGE),
            "regimes": list(F.REGIMES),
            "primary_regime": "S0_turb",
            "figure_names": list(P.FIGURE_NAMES),
            "figure3_lead_days": list(P.FIGURE_3_LEADS),
            "figure7_lead_days": list(P.FIGURE_7_LEADS),
            "rmse_fields": list(P.RMSE_FIELDS),
            "acc_fields": list(P.ACC_FIELDS),
            "maximum_lead_days": F.MAXIMUM_INFERENCE_ROLLOUT_DAYS,
            "prediction_interval_days": 10,
            "short_lead_days": "0_to_200_inclusive_by_10",
            "long_lead_days": "0_to_2000_inclusive_by_10",
            "comparator_model": None,
            "nesting": (
                "nested_validation_inference_protocol_no_independent_third_test_split"
            ),
            "static_channels": list(STATIC_FEATURES),
        },
        "selected_model": {
            "version": TRAIN_VERSION,
            "optimizer_step": step,
            "rollout_steps": ROLLOUT_STEPS,
            "loss_contract_sha256": LOSS_CONTRACT_SHA256,
            "architecture": ProductionArchitecture().to_dict(),
            "from_scratch": True,
            "seed": SEED,
            "training_contract": str(STUDY / "config" / f"{TRAIN_VERSION}.json"),
        },
        "artifacts": {
            "dataset_metadata": _artifact(DATASET / ".zmetadata"),
            "selected_checkpoint": _artifact(TRAIN_SCRATCH / "selected.pt"),
            "selected_normalization": _artifact(
                TRAIN_SCRATCH / f"{TRAIN_VERSION}_train_only_normalization.npz"
            ),
            "selected_report": _artifact(report_path),
            "mitgcm_declaration": _artifact(TURB_SPINUP / "data"),
            "mitgcm_sst_relaxation": _artifact(TURB_SPINUP / "SST_relax.bin"),
            "mitgcm_zonal_spacing": _artifact(TURB_SPINUP / "DXF.data"),
        },
        "output": {
            "project_root": str(STUDY / "outputs" / F.VERSION),
            "scratch_root": str(SCRATCH / "models" / "turb" / F.VERSION),
            "overwrite": False,
            "one_folder_per_regime": True,
            "required": list(F._EXPECTED_OUTPUTS),
        },
        "source_hashes": {
            name: sha256(STUDY / name) for name in sorted(F._REQUIRED_SOURCE_HASHES)
        },
    }


def anomaly_contract(figure_path: Path) -> dict:
    figure_project = STUDY / "outputs" / F.VERSION / "S0_turb"
    return {
        "version": A.VERSION,
        "contract_status": A.CONTRACT_STATUS,
        "purpose": (
            "streamfunction-anomaly companions to figures 3 and 7 for "
            f"{TRAIN_VERSION}, about the same MITgcm S0_turb training-mean field "
            "the parent package used"
        ),
        "adds_only": True,
        "modifies_published_figures": False,
        "dataset": {
            "version": F.DATASET_VERSION,
            "train": list(F.TRAIN_RANGE),
            "tau0_n_m2": dict(F.TAU0_N_M2),
        },
        "reference": {
            "source": "mitgcm",
            "regime": "S0_turb",
            "days": list(F.TRAIN_RANGE),
            "is_two_dimensional_field": True,
            "not_a_scalar_spatial_mean": True,
            "model_own_mean_used": False,
            "subtracted_from": "both_truth_and_prediction",
        },
        "protocol": {
            "primary_regime": "S0_turb",
            "member": 0,
            "reads_model_weights": False,
            "rolls_nothing_out": True,
            "figure_names": list(A.FIGURE_NAMES),
            "figure3_lead_days": list(P.FIGURE_3_LEADS),
            "figure7_lead_days": list(P.FIGURE_7_LEADS),
            "day2000_structure_diagnostics": list(A._EXPECTED_DIAGNOSTICS),
        },
        "artifacts": {
            "figure_package_contract": _artifact(figure_path),
            "figure_package_report": _artifact(figure_project / P.REPORT_NAME),
            "figure_package_arrays": _artifact(figure_project / P.ARRAYS_NAME),
            "figure_package_manifest": _artifact(figure_project / "manifest.json"),
            "dataset_metadata": _artifact(DATASET / ".zmetadata"),
        },
        "output": {
            "project_root": str(STUDY / "outputs" / A.VERSION),
            "scratch_root": str(SCRATCH / "models" / "turb" / A.VERSION),
            "overwrite": False,
            "one_folder_per_regime": True,
            "required": list(A._EXPECTED_REQUIRED),
        },
        "source_hashes": {
            name: sha256(STUDY / name) for name in sorted(A._REQUIRED_SOURCE_HASHES)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=("figures", "anomaly", "both"), default="figures")
    args = parser.parse_args()
    config = STUDY / "config"
    figure_path = config / f"{F.VERSION}.json"
    if args.which in ("figures", "both"):
        figure_path.write_text(json.dumps(figure_contract(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {figure_path}")
    if args.which in ("anomaly", "both"):
        path = config / f"{A.VERSION}.json"
        path.write_text(json.dumps(anomaly_contract(figure_path), indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
