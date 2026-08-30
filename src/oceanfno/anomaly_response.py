"""Contract adapter: the frozen S0 anomaly package for the B/C study arms.

Plan section 19 step 4 and section 23.1: "contract adapter that reuses the
numerical helpers in the frozen anomaly module but accepts B/C figure
identities."

The numerics are imported and executed unchanged -- the MITgcm training-mean
reference field, the anomaly subtraction itself, ``variability_summary``,
``day2000_structure_summary``, ``wet_rms`` and both plates all come from
``anomaly.py``. What this module re-expresses is only the identity binding,
which in the frozen module is hard-wired to the parent through ``VERSION``,
``FIGURE_PACKAGE_VERSION`` (= ``figures.VERSION``) and
``figures.TRAINING_VERSION``.

As with ``figures_response.py``, the frozen module is not edited: the parent's
anomaly contract pins ``src/oceanfno/anomaly.py`` in its ``source_hashes``, so
changing that file would retire the A package's ability to re-verify itself,
and section 19 step 6 requires the existing A/ft90 reports be preserved.

The reference field is the same one the parent package used -- MITgcm's derived
S0 time-mean barotropic streamfunction over training days 0--5,999 -- and it is
subtracted identically from truth and from every model. A model's own mean is
never subtracted, so no model bias in the mean circulation can hide.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import plots
from . import anomaly as frozen
from . import figures_response
from .anomaly import (
    ARRAYS_NAME,
    AnomalyContractError,
    DATASET_VERSION,
    FIGURE_3_LEADS,
    FIGURE_7_LEADS,
    FIGURE_NAMES,
    MANIFEST_NAME,
    PENDING,
    PENDING_PATHS,
    README_NAME,
    REFERENCE_FIGURE,
    REGIME,
    REPORT_NAME,
    SEALED_KEYS,
    TAU0,
    TRAIN_RANGE,
    _EXPECTED_DIAGNOSTICS,
    _EXPECTED_REQUIRED,
    _REQUIRED_ARRAYS,
    _integer,
    _plot_anomaly_grid,
    _plot_anomaly_long,
    _style,
    day2000_structure_summary,
    training_mean_streamfunction,
    variability_summary,
    wet_rms,
)
from .runtime import _file_sha256, _json_sha256, json_safe

CONTRACT_STATUS = frozen.CONTRACT_STATUS

#: anomaly package version -> the figure package it seals over.
IDENTITIES: dict[str, str] = {
    figure_version.replace("_s0_figures_v1", "_s0_anomaly_v1"): figure_version
    for figure_version in figures_response.IDENTITIES
}

_REQUIRED_SOURCE_HASHES = frozenset(
    frozen._REQUIRED_SOURCE_HASHES
    | {"src/oceanfno/anomaly_response.py", "src/oceanfno/figures_response.py"}
)


def identity_of(contract: Mapping[str, Any]) -> tuple[str, str, str, int]:
    """(anomaly version, figure version, training version, seed)."""

    version = str(contract.get("version", ""))
    figure_version = IDENTITIES.get(version)
    if figure_version is None:
        raise AnomalyContractError(f"{version!r} is not a declared B/C anomaly identity")
    training_version = figures_response.IDENTITIES[figure_version]
    seed = int(figure_version.split("_seed_")[1].split("_")[0])
    return version, figure_version, training_version, seed


def unfilled_fields(contract: Mapping[str, Any]) -> list[str]:
    return frozen.unfilled_fields(contract)


def _sealed_figure_provenance(contract: Mapping[str, Any]) -> dict[str, str]:
    """``anomaly._sealed_figure_provenance`` with the arm's identity supplied.

    Every condition is the frozen module's; only the two version strings it
    compares against are resolved from ``IDENTITIES`` rather than the parent
    constants.
    """

    _, figure_version, training_version, _ = identity_of(contract)
    artifacts = contract.get("artifacts", {})
    paths = {key: Path(str(artifacts.get(key, {}).get("path", ""))).resolve() for key in SEALED_KEYS}
    expected_names = (f"{figure_version}.json", plots.REPORT_NAME, plots.ARRAYS_NAME, plots.MANIFEST_NAME)
    if tuple(paths[key].name for key in SEALED_KEYS) != expected_names or not all(
        path.is_file() for path in paths.values()
    ):
        raise AnomalyContractError("the sealed figure package is incomplete")

    digests = {key: _file_sha256(path) for key, path in paths.items()}
    figure_contract = json.loads(paths["figure_package_contract"].read_text())
    report = json.loads(paths["figure_package_report"].read_text())
    manifest = json.loads(paths["figure_package_manifest"].read_text())
    expected_arrays = Path(figure_contract["output"]["project_root"]).resolve() / REGIME / plots.ARRAYS_NAME
    expected_report = expected_arrays.with_name(plots.REPORT_NAME)
    manifest_artifacts = manifest.get("artifacts", {})
    if (
        figure_contract.get("version") != figure_version
        or figure_contract.get("selected_model", {}).get("version") != training_version
        or report.get("status") != "complete"
        or report.get("version") != figure_version
        or report.get("regime") != REGIME
        or tuple(report.get("start_draw_order", ()))
        != tuple(int(v) for v in figures_response.declared_inference_starts())
        or tuple(report.get("figures", ())) != tuple(plots.FIGURE_NAMES)
        or Path(str(report.get("contract", ""))).resolve() != paths["figure_package_contract"]
        or report.get("contract_sha256") != digests["figure_package_contract"]
        or report.get("arrays_sha256") != digests["figure_package_arrays"]
        or expected_arrays != paths["figure_package_arrays"]
        or expected_report != paths["figure_package_report"]
        or paths["figure_package_manifest"] != paths["figure_package_report"].parent / plots.MANIFEST_NAME
        or manifest.get("version") != figure_version
        or manifest.get("regime") != REGIME
        or manifest.get("contract_sha256") != digests["figure_package_contract"]
        or manifest.get("report_content_sha256") != report.get("report_content_sha256")
        or manifest_artifacts.get(plots.ARRAYS_NAME, {}).get("sha256") != digests["figure_package_arrays"]
        or manifest_artifacts.get(plots.REPORT_NAME, {}).get("sha256") != digests["figure_package_report"]
        or artifacts["dataset_metadata"].get("path") != figure_contract["artifacts"]["dataset_metadata"].get("path")
        or artifacts["dataset_metadata"].get("sha256")
        != figure_contract["artifacts"]["dataset_metadata"].get("sha256")
    ):
        raise AnomalyContractError("the sealed figure provenance changed")

    with np.load(paths["figure_package_arrays"]) as stored:
        if not _REQUIRED_ARRAYS.issubset(stored.files):
            raise AnomalyContractError("the figure arrays are incomplete")
        if (
            np.asarray(stored["figure3_truth_streamfunction"]).shape != (len(FIGURE_3_LEADS), 62, 62)
            or np.asarray(stored["figure3_model_streamfunction"]).shape != (len(FIGURE_3_LEADS), 62, 62)
            or np.asarray(stored["figure7_truth_streamfunction"]).shape != (len(FIGURE_7_LEADS), 62, 62)
            or np.asarray(stored["figure7_model_streamfunction"]).shape != (len(FIGURE_7_LEADS), 62, 62)
            or tuple(np.asarray(stored["start_draw_order"]).astype(int))
            != tuple(int(v) for v in figures_response.declared_inference_starts())
        ):
            raise AnomalyContractError("the figure array protocol changed")
    return digests


def finalize(contract_path: str | Path) -> dict[str, Any]:
    """Fill the deferred sealed-package hashes, idempotently."""

    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    version, figure_version, _, seed = identity_of(contract)
    digests = _sealed_figure_provenance(contract)
    changed = []
    for path in PENDING_PATHS:
        node: Any = contract
        for key in path[:-1]:
            node = node[key]
        current = node[path[-1]]
        wanted = digests[path[-2]]
        if current == wanted:
            continue
        if current != PENDING:
            raise AnomalyContractError(
                f"{'.'.join(path)} is already {current!r}, not {PENDING!r}; refusing to overwrite"
            )
        node[path[-1]] = wanted
        changed.append(".".join(path))
    if changed:
        resolved.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "status": "finalized",
        "version": version,
        "figure_version": figure_version,
        "seed": seed,
        "filled": changed,
        "contract": str(resolved),
        "contract_sha256": _file_sha256(resolved),
    }


def load_contract(path: str | Path, *, verify_sources: bool = True) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    pending = unfilled_fields(contract)
    if pending:
        raise AnomalyContractError(
            "the anomaly contract still carries figure fields: "
            + ", ".join(pending)
            + " -- run `python -m oceanfno.anomaly_response finalize` first"
        )
    version, _, _, _ = identity_of(contract)
    reference = contract.get("reference", {})
    protocol = contract.get("protocol", {})
    output = contract.get("output", {})
    dataset = contract.get("dataset", {})
    if (
        contract.get("contract_status") != CONTRACT_STATUS
        or dataset.get("version") != DATASET_VERSION
        or tuple(dataset.get("train", ())) != TRAIN_RANGE
        or dataset.get("tau0_n_m2") != {REGIME: TAU0}
        or protocol.get("primary_regime") != REGIME
        or _integer(protocol.get("member")) != 0
        or protocol.get("reads_model_weights") is not False
        or protocol.get("rolls_nothing_out") is not True
        or tuple(protocol.get("figure3_lead_days", ())) != tuple(FIGURE_3_LEADS)
        or tuple(protocol.get("figure7_lead_days", ())) != tuple(FIGURE_7_LEADS)
        or tuple(protocol.get("figure_names", ())) != FIGURE_NAMES
        or tuple(protocol.get("day2000_structure_diagnostics", ())) != _EXPECTED_DIAGNOSTICS
        or reference.get("source") != "mitgcm"
        or tuple(reference.get("days", ())) != TRAIN_RANGE
        or reference.get("regime") != REGIME
        or reference.get("subtracted_from") != "both_truth_and_prediction"
        or reference.get("model_own_mean_used") is not False
        or reference.get("is_two_dimensional_field") is not True
        or reference.get("not_a_scalar_spatial_mean") is not True
        or contract.get("modifies_published_figures") is not False
        or contract.get("adds_only") is not True
        or not str(output.get("project_root", "")).endswith(version)
        or not str(output.get("scratch_root", "")).endswith(version)
        or output.get("overwrite") is not False
        or output.get("one_folder_per_regime") is not True
        or tuple(output.get("required", ())) != _EXPECTED_REQUIRED
    ):
        raise AnomalyContractError("the B/C anomaly contract changed")

    digests = _sealed_figure_provenance(contract)
    for key, digest in digests.items():
        if contract["artifacts"][key].get("sha256") != digest:
            raise AnomalyContractError(f"{key} changed after finalization")
    if verify_sources:
        for label, specification in contract.get("artifacts", {}).items():
            artifact = Path(str(specification.get("path", ""))).resolve()
            target = artifact / ".zmetadata" if artifact.is_dir() else artifact
            if not target.is_file() or _file_sha256(target) != specification.get("sha256"):
                raise AnomalyContractError(f"{label} changed on disk")
        hashes = contract.get("source_hashes", {})
        if not _REQUIRED_SOURCE_HASHES.issubset(hashes):
            raise AnomalyContractError("the anomaly source declaration is incomplete")
        root = resolved.parents[1]
        for relative, expected in hashes.items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise AnomalyContractError(f"an anomaly source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def _readme(report: Mapping[str, Any]) -> str:
    return (
        f"# {report['version']}\n\n"
        f"Barotropic streamfunction anomalies for `{report['training_version']}` seed "
        f"{report['seed']}, produced by the unchanged numerics of `src/oceanfno/anomaly.py` "
        "through the study adapter `src/oceanfno/anomaly_response.py`.\n\n"
        "The reference is MITgcm's derived S0 time-mean barotropic streamfunction over "
        f"training days {report['reference']['days'][0]}--{report['reference']['days'][1] - 1} "
        f"({report['reference']['days_averaged']} days), subtracted identically from truth and "
        "from the model. A model's own mean is never subtracted.\n\n"
        "This package adds plates; it modifies no published figure.\n"
    )


def run(contract_path: str | Path) -> dict[str, Any]:
    """Publish the anomaly plates and variability summary for one arm and seed."""

    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    version, figure_version, training_version, seed = identity_of(contract)
    dataset = Path(contract["artifacts"]["dataset_metadata"]["path"]).parent
    suite_arrays = Path(contract["artifacts"]["figure_package_arrays"]["path"])

    group = zarr.open_consolidated(str(dataset), mode="r")
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)

    mean_field, days = training_mean_streamfunction(group, wet)

    with np.load(suite_arrays) as stored:
        figure3_truth = np.asarray(stored["figure3_truth_streamfunction"], dtype=np.float64)
        figure3_model = np.asarray(stored["figure3_model_streamfunction"], dtype=np.float64)
        figure7_truth = np.asarray(stored["figure7_truth_streamfunction"], dtype=np.float64)
        figure7_model = np.asarray(stored["figure7_model_streamfunction"], dtype=np.float64)

    anomalies = {
        "figure3_truth": figure3_truth - mean_field,
        "figure3_model": figure3_model - mean_field,
        "figure7_truth": figure7_truth - mean_field,
        "figure7_model": figure7_model - mean_field,
    }

    project = Path(contract["output"]["project_root"]).resolve() / REGIME
    scratch = Path(contract["output"]["scratch_root"]).resolve() / REGIME
    for path in (project, scratch):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    project_tmp = project.with_name(project.name + ".tmp")
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.mkdir()
    scratch_tmp.mkdir()

    try:
        _style()
        _plot_anomaly_grid(
            project_tmp, anomalies["figure3_truth"], anomalies["figure3_model"], longitude, latitude, wet
        )
        _plot_anomaly_long(
            project_tmp, anomalies["figure7_truth"], anomalies["figure7_model"], mean_field,
            longitude, latitude, wet,
        )

        arrays_path = scratch_tmp / ARRAYS_NAME
        np.savez_compressed(
            arrays_path,
            reference_time_mean_streamfunction=mean_field,
            figure3_lead_days=np.asarray(FIGURE_3_LEADS, dtype=np.int16),
            figure7_lead_days=np.asarray(FIGURE_7_LEADS, dtype=np.int16),
            wet_mask=wet.astype(np.uint8),
            longitude_deg=longitude,
            latitude_deg=latitude,
            **{name: value.astype(np.float32) for name, value in anomalies.items()},
        )

        report = {
            "status": "complete",
            "version": version,
            "figure_version": figure_version,
            "training_version": training_version,
            "seed": seed,
            "regime": REGIME,
            "tau0_n_m2": TAU0,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "dataset_version": DATASET_VERSION,
            "member": "member_0_of_15_the_same_member_figures_3_and_7_plot",
            "reference": {
                "definition": "mitgcm_time_mean_barotropic_streamfunction",
                "days": list(TRAIN_RANGE),
                "days_averaged": days,
                "regime": REGIME,
                "subtracted_from": "both_truth_and_prediction",
                "model_own_mean_used": False,
                "is_two_dimensional_field": True,
                "rms_sv": wet_rms(mean_field, wet),
                "range_sv": [float(mean_field[wet].min()), float(mean_field[wet].max())],
            },
            "variability": {
                "figure3": variability_summary(
                    anomalies["figure3_truth"], anomalies["figure3_model"], FIGURE_3_LEADS, wet
                ),
                "figure7": variability_summary(
                    anomalies["figure7_truth"], anomalies["figure7_model"], FIGURE_7_LEADS, wet
                ),
            },
            "day2000_structure": day2000_structure_summary(
                anomalies["figure7_truth"][-1], anomalies["figure7_model"][-1], wet
            ),
            "total_field_note": (
                "amplitude diagnostics including the day-2000 streamfunction minimum "
                "remain defined on the total field and are unaffected by this package"
            ),
            "modifies_published_figures": False,
            "figures": list(FIGURE_NAMES) + [REFERENCE_FIGURE],
            "arrays": str(scratch / ARRAYS_NAME),
            "arrays_sha256": _file_sha256(arrays_path),
            "numerics_source": "src/oceanfno/anomaly.py (unchanged); adapter src/oceanfno/anomaly_response.py",
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        report = json_safe(report)
        report["content_sha256"] = _json_sha256(report)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        (scratch_tmp / REPORT_NAME).write_text(rendered)
        (project_tmp / REPORT_NAME).write_text(rendered)
        shutil.copy2(arrays_path, project_tmp / ARRAYS_NAME)
        (project_tmp / README_NAME).write_text(_readme(report))
        manifest = {
            "version": version,
            "regime": REGIME,
            "contract_sha256": contract_sha,
            "report_content_sha256": report["content_sha256"],
            "artifacts": {
                path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
                for path in sorted(project_tmp.iterdir())
                if path.is_file()
            },
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        (project_tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(scratch_tmp, scratch)
        os.replace(project_tmp, project)
    except Exception:
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        shutil.rmtree(project_tmp, ignore_errors=True)
        raise
    return report


def preflight(contract_path: str | Path) -> dict[str, Any]:
    contract, resolved, digest = load_contract(contract_path)
    version, figure_version, training_version, seed = identity_of(contract)
    return {
        "status": "pass",
        "version": version,
        "figure_version": figure_version,
        "training_version": training_version,
        "seed": seed,
        "contract": str(resolved),
        "contract_sha256": digest,
        "regime": REGIME,
        "reference": "mitgcm_time_mean_barotropic_streamfunction_train_days_0_5999",
        "reads_model_weights": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("finalize", "preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        result: Any = finalize(args.contract)
    elif args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
