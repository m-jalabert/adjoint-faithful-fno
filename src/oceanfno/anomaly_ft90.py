"""Streamfunction-anomaly companions to the fine-tune's figures 3 and 7.

The same single operation the production anomaly package performs, applied to
the fine-tuned model's sealed figure arrays:

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

``psi_bar_S0`` is MITgcm's own two-dimensional time-mean barotropic
streamfunction over training days 0--5999 --- the *same* reference field the
parent package removed, because a comparison between two models is only
meaningful if the standing gyre subtracted from both is identical. The model's
own mean is never used, so a bias in the stationary circulation cannot hide
inside the anomaly.

**This module adds a lineage, not a method.** The reference mean, the anomaly
diagnostics, the variability summary, the day-2,000 structure metrics and all
three plates come from :mod:`oceanfno.anomaly` unchanged. What is reimplemented
is only which sealed figure package is read and which arm's provenance chain is
verified: the production package binds to ``figures.VERSION`` and
``train.VERSION``, and this one binds to the fine-tune's equivalents.

Keeping ``anomaly.py`` byte-identical preserves the parent's anomaly package as
a re-runnable comparison, which is the point --- the parent's day-2,000 anomaly
RMS ratio and western-boundary structure are what the fine-tune is measured
against.

Reads the sealed figure arrays and model-visible MITgcm training state only: no
weights are loaded, nothing is rolled out, and no published figure is modified.

Entry points::

    python -m oceanfno.anomaly_ft90 finalize  --contract config/...json
    python -m oceanfno.anomaly_ft90 preflight --contract config/...json
    python -m oceanfno.anomaly_ft90 run       --contract config/...json
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
from .runtime import _file_sha256, _json_sha256, json_safe
from .dataset import DATASET_VERSION, TRAIN_RANGE
from .model import MANIFEST_NAME, README_NAME
from .plots import FIGURE_3_LEADS, FIGURE_7_LEADS, _style
from . import figures_ft90
from .figures_ft90 import PARENT_VERSION, declared_inference_starts

# The production anomaly package's own machinery, reused rather than rewritten.
from .anomaly import (
    ARRAYS_NAME,
    AnomalyContractError,
    FIGURE_NAMES,
    PENDING,
    PENDING_PATHS,
    REFERENCE_FIGURE,
    REGIME,
    REPORT_NAME,
    SEALED_KEYS,
    TAU0,
    _EXPECTED_DIAGNOSTICS,
    _EXPECTED_REQUIRED,
    _REQUIRED_ARRAYS,
    _integer,
    _plot_anomaly_grid,
    _plot_anomaly_long,
    _read,
    day2000_structure_summary,
    training_mean_streamfunction,
    unfilled_fields,
    variability_summary,
    wet_rms,
)

VERSION = "model_c_production_1in_1out_spectralnorm_ft90_v1_s0_anomaly_v1"

CONTRACT_STATUS = (
    "frozen_after_the_fine_tuned_figure_package_and_before_any_anomaly_metric"
)

#: The sealed figure package this reads: the fine-tune's, not the parent's.
FIGURE_PACKAGE_VERSION = figures_ft90.VERSION

TRAINING_VERSION = figures_ft90.TRAINING_VERSION

#: The parent's published anomaly report, pinned by digest and read only so the
#: two arms' day-2,000 anomaly numbers land in one place. It is a *reference*,
#: never an input to any computation here.
PARENT_ANOMALY_ARTIFACT = "parent_anomaly_report"

PARENT_ANOMALY_VERSION = "model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1"

#: ``variability_summary`` keys its per-lead record by ``str(int(lead))``, and
#: the day-2,000 structure block names the western band by its width rather than
#: generically. Both are spelled out here so a rename upstream fails loudly
#: instead of silently dropping the comparison.
DAY2000_KEY = str(int(FIGURE_7_LEADS[-1]))

WESTERN_KEY = "western_first_4_wet_cells"

_REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/oceanfno/anomaly.py",
        "src/oceanfno/anomaly_ft90.py",
        "src/oceanfno/dataset.py",
        "src/oceanfno/diagnostics.py",
        "src/oceanfno/figures.py",
        "src/oceanfno/figures_ft90.py",
        "src/oceanfno/finetune.py",
        "src/oceanfno/plots.py",
        "src/oceanfno/runtime.py",
    }
)


def _sealed_figure_provenance(contract: Mapping[str, Any]) -> dict[str, str]:
    """Verify the fine-tune's contract/report/arrays chain and sealed manifest."""

    artifacts = contract.get("artifacts", {})
    paths = {
        key: Path(str(artifacts.get(key, {}).get("path", ""))).resolve()
        for key in SEALED_KEYS
    }
    expected_names = (
        f"{FIGURE_PACKAGE_VERSION}.json",
        plots.REPORT_NAME,
        plots.ARRAYS_NAME,
        plots.MANIFEST_NAME,
    )
    if tuple(paths[key].name for key in SEALED_KEYS) != expected_names or not all(
        path.is_file() for path in paths.values()
    ):
        raise AnomalyContractError("the sealed figure package is incomplete")

    digests = {key: _file_sha256(path) for key, path in paths.items()}
    figure_contract = json.loads(paths["figure_package_contract"].read_text())
    report = json.loads(paths["figure_package_report"].read_text())
    manifest = json.loads(paths["figure_package_manifest"].read_text())
    expected_arrays = (
        Path(figure_contract["output"]["project_root"]).resolve()
        / REGIME
        / plots.ARRAYS_NAME
    )
    expected_report = expected_arrays.with_name(plots.REPORT_NAME)
    manifest_artifacts = manifest.get("artifacts", {})
    if (
        figure_contract.get("version") != FIGURE_PACKAGE_VERSION
        or figure_contract.get("selected_model", {}).get("version") != TRAINING_VERSION
        # The chain must reach all the way back to the parent, not merely to a
        # fine-tune: a package produced from some other lineage would otherwise
        # satisfy every check above.
        or figure_contract.get("selected_model", {}).get("parent_version")
        != PARENT_VERSION
        or report.get("status") != "complete"
        or report.get("version") != FIGURE_PACKAGE_VERSION
        or report.get("regime") != REGIME
        or report.get("training_version") != TRAINING_VERSION
        or report.get("parent_version") != PARENT_VERSION
        or tuple(report.get("start_draw_order", ()))
        != tuple(int(value) for value in declared_inference_starts())
        or tuple(report.get("figures", ())) != tuple(plots.FIGURE_NAMES)
        or Path(str(report.get("contract", ""))).resolve()
        != paths["figure_package_contract"]
        or report.get("contract_sha256") != digests["figure_package_contract"]
        or report.get("arrays_sha256") != digests["figure_package_arrays"]
        or expected_arrays != paths["figure_package_arrays"]
        or expected_report != paths["figure_package_report"]
        or paths["figure_package_manifest"]
        != paths["figure_package_report"].parent / plots.MANIFEST_NAME
        or manifest.get("version") != FIGURE_PACKAGE_VERSION
        or manifest.get("regime") != REGIME
        or manifest.get("contract_sha256") != digests["figure_package_contract"]
        or manifest.get("report_content_sha256") != report.get("report_content_sha256")
        or manifest_artifacts.get(plots.ARRAYS_NAME, {}).get("sha256")
        != digests["figure_package_arrays"]
        or manifest_artifacts.get(plots.REPORT_NAME, {}).get("sha256")
        != digests["figure_package_report"]
        or artifacts["dataset_metadata"].get("path")
        != figure_contract["artifacts"]["dataset_metadata"].get("path")
        or artifacts["dataset_metadata"].get("sha256")
        != figure_contract["artifacts"]["dataset_metadata"].get("sha256")
    ):
        raise AnomalyContractError("the sealed figure provenance changed")

    with np.load(paths["figure_package_arrays"]) as stored:
        if not _REQUIRED_ARRAYS.issubset(stored.files):
            raise AnomalyContractError("the figure arrays are incomplete")
        if (
            np.asarray(stored["figure3_truth_streamfunction"]).shape
            != (len(plots.FIGURE_3_LEADS), 62, 62)
            or np.asarray(stored["figure3_model_streamfunction"]).shape
            != (len(plots.FIGURE_3_LEADS), 62, 62)
            or np.asarray(stored["figure7_truth_streamfunction"]).shape
            != (len(plots.FIGURE_7_LEADS), 62, 62)
            or np.asarray(stored["figure7_model_streamfunction"]).shape
            != (len(plots.FIGURE_7_LEADS), 62, 62)
            or tuple(np.asarray(stored["start_draw_order"]).astype(int))
            != tuple(int(value) for value in declared_inference_starts())
        ):
            raise AnomalyContractError("the figure array protocol changed")
    return digests


def parent_anomaly(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The parent's published day-2,000 anomaly numbers, for the comparison."""

    specification = contract["artifacts"][PARENT_ANOMALY_ARTIFACT]
    path = Path(str(specification["path"])).resolve()
    if not path.is_file() or _file_sha256(path) != specification.get("sha256"):
        raise AnomalyContractError("the parent anomaly report is missing or changed")
    report = json.loads(path.read_text())
    if report.get("version") != PARENT_ANOMALY_VERSION or report.get("status") != "complete":
        raise AnomalyContractError("the parent anomaly report is not the published one")
    structure = report["day2000_structure"]
    return {
        "version": PARENT_ANOMALY_VERSION,
        "report": str(path),
        "report_sha256": _file_sha256(path),
        "day2000_structure": structure,
        "day2000_anomaly_rms_ratio": float(
            report["variability"]["figure7"][DAY2000_KEY]["anomaly_rms_ratio"]
        ),
        "day2000_western_boundary": structure[WESTERN_KEY],
        "reference_rms_sv": float(report["reference"]["rms_sv"]),
        "note": (
            "the same MITgcm training-mean field is removed in both packages, so "
            "these anomaly diagnostics are directly comparable"
        ),
    }


def finalize(contract_path: str | Path) -> dict[str, Any]:
    """Fill the deferred sealed-package digests, idempotently."""

    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    digests = _sealed_figure_provenance(contract)
    applied: dict[str, str] = {}
    for path in PENDING_PATHS:
        key = path[-2]
        value = digests[key]
        current = _read(contract, path)
        if current not in (None, PENDING) and current != value:
            raise AnomalyContractError(
                f"{'.'.join(path)} is already {current!r}, not {value!r}; "
                "refusing to overwrite a filled field"
            )
        if current != value:
            contract["artifacts"][key]["sha256"] = value
            applied[".".join(path)] = value
    if applied:
        resolved.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "status": "filled" if applied else "already_complete",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": _file_sha256(resolved),
        "applied": applied,
    }


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the frozen anomaly declaration and its sealed figure provenance."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    pending = unfilled_fields(contract)
    if pending:
        raise AnomalyContractError(
            "the anomaly contract still carries figure fields: "
            + ", ".join(pending)
            + " -- run `python -m oceanfno.anomaly_ft90 finalize` first"
        )
    reference = contract.get("reference", {})
    protocol = contract.get("protocol", {})
    output = contract.get("output", {})
    dataset = contract.get("dataset", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or dataset.get("version") != DATASET_VERSION
        or tuple(dataset.get("train", ())) != TRAIN_RANGE
        or dataset.get("tau0_n_m2") != {REGIME: TAU0}
        or protocol.get("primary_regime") != REGIME
        or _integer(protocol.get("member")) != 0
        or protocol.get("reads_model_weights") is not False
        or protocol.get("rolls_nothing_out") is not True
        or tuple(protocol.get("figure3_lead_days", ())) != tuple(plots.FIGURE_3_LEADS)
        or tuple(protocol.get("figure7_lead_days", ())) != tuple(plots.FIGURE_7_LEADS)
        or tuple(protocol.get("figure_names", ())) != FIGURE_NAMES
        or tuple(protocol.get("day2000_structure_diagnostics", ()))
        != _EXPECTED_DIAGNOSTICS
        or protocol.get("training_version") != TRAINING_VERSION
        or protocol.get("parent_version") != PARENT_VERSION
        # The reference field must be identical to the parent's, or the two
        # arms' anomalies are not the same quantity.
        or reference.get("source") != "mitgcm"
        or tuple(reference.get("days", ())) != TRAIN_RANGE
        or reference.get("regime") != REGIME
        or reference.get("subtracted_from") != "both_truth_and_prediction"
        or reference.get("model_own_mean_used") is not False
        or reference.get("is_two_dimensional_field") is not True
        or reference.get("not_a_scalar_spatial_mean") is not True
        or reference.get("identical_to_the_parent_packages_reference") is not True
        or contract.get("modifies_published_figures") is not False
        or contract.get("adds_only") is not True
        or not str(output.get("project_root", "")).endswith(VERSION)
        or not str(output.get("scratch_root", "")).endswith(VERSION)
        or output.get("overwrite") is not False
        or output.get("one_folder_per_regime") is not True
        or tuple(output.get("required", ())) != _EXPECTED_REQUIRED
    ):
        raise AnomalyContractError("the fine-tuned anomaly contract changed")

    digests = _sealed_figure_provenance(contract)
    for key, digest in digests.items():
        if contract["artifacts"][key].get("sha256") != digest:
            raise AnomalyContractError(f"{key} changed after finalization")
    parent_anomaly(contract)
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
    figure7 = report["variability"]["figure7"]
    structure = report["day2000_structure"]
    boundary = structure[WESTERN_KEY]
    comparison = report["comparison_to_the_parent"]
    rms = comparison["day2000_anomaly_rms_ratio"]
    parent_boundary = report["parent"]["day2000_western_boundary"]
    return f"""# Ninety-day fine-tune, {REGIME}: streamfunction anomalies

One operation, applied identically to truth and to the prediction:

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

`psi_bar_S0` is MITgcm's own time-mean barotropic streamfunction over training
days {TRAIN_RANGE[0]}--{TRAIN_RANGE[1] - 1}, RMS
{report['reference']['rms_sv']:.3f} Sv. It is the **same reference field the
parent's anomaly package removed**, which is what makes the two arms' numbers
comparable; the model's own mean is never subtracted, so a bias in the
stationary circulation cannot hide inside the anomaly.

With the standing gyre removed, an anomaly RMS ratio below one means the
transients were damped and above one that they were manufactured.

## Day 2,000, against the parent

| | fine-tuned | parent | target |
| --- | --- | --- | --- |
| anomaly RMS ratio | {rms['fine_tuned']:.3f} | {rms['parent']:.3f} | 1.000 |
| western boundary / interior, model | {boundary['model_boundary_to_interior_rms_ratio']:.3f} | {parent_boundary['model_boundary_to_interior_rms_ratio']:.3f} | {boundary['truth_boundary_to_interior_rms_ratio']:.3f} |

An anomaly RMS ratio above one means the transients were manufactured, below one
that they were damped. The western-boundary target column is MITgcm's own value,
which both packages must reproduce identically for truth --- that they do is a
check on the comparison, not a result.

Anomaly RMS ratio at day 60:
{figure7[str(FIGURE_7_LEADS[0])]['anomaly_rms_ratio']:.3f}, at day 2,000:
{figure7[str(FIGURE_7_LEADS[-1])]['anomaly_rms_ratio']:.3f}.

This package reads the sealed figure arrays and model-visible MITgcm training
state only. It rolls out no model, promotes nothing, and does not modify the
total-field figures or the acceptance gate.

Report content SHA-256: `{report['content_sha256']}`.
"""


def run(contract_path: str | Path) -> dict[str, Any]:
    """Publish the anomaly plates and the variability summary."""

    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(contract["artifacts"]["dataset_metadata"]["path"]).parent
    suite_arrays = Path(contract["artifacts"]["figure_package_arrays"]["path"])
    parent = parent_anomaly(contract)

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
            project_tmp,
            anomalies["figure3_truth"],
            anomalies["figure3_model"],
            longitude,
            latitude,
            wet,
        )
        _plot_anomaly_long(
            project_tmp,
            anomalies["figure7_truth"],
            anomalies["figure7_model"],
            mean_field,
            longitude,
            latitude,
            wet,
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

        structure = day2000_structure_summary(
            anomalies["figure7_truth"][-1], anomalies["figure7_model"][-1], wet
        )
        variability = {
            "figure3": variability_summary(
                anomalies["figure3_truth"],
                anomalies["figure3_model"],
                FIGURE_3_LEADS,
                wet,
            ),
            "figure7": variability_summary(
                anomalies["figure7_truth"],
                anomalies["figure7_model"],
                FIGURE_7_LEADS,
                wet,
            ),
        }
        day2000_ratio = float(
            variability["figure7"][DAY2000_KEY]["anomaly_rms_ratio"]
        )
        report = {
            "status": "complete",
            "version": VERSION,
            "regime": REGIME,
            "tau0_n_m2": TAU0,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "dataset_version": DATASET_VERSION,
            "training_version": TRAINING_VERSION,
            "parent_version": PARENT_VERSION,
            "figure_package_version": FIGURE_PACKAGE_VERSION,
            "member": "member_0_of_15_the_same_member_figures_3_and_7_plot",
            "reference": {
                "definition": "mitgcm_time_mean_barotropic_streamfunction",
                "days": list(TRAIN_RANGE),
                "days_averaged": days,
                "regime": REGIME,
                "subtracted_from": "both_truth_and_prediction",
                "model_own_mean_used": False,
                "is_two_dimensional_field": True,
                "identical_to_the_parent_packages_reference": True,
                "rms_sv": wet_rms(mean_field, wet),
                "range_sv": [float(mean_field[wet].min()), float(mean_field[wet].max())],
            },
            "variability": variability,
            "day2000_structure": structure,
            "parent": parent,
            "comparison_to_the_parent": {
                "day2000_anomaly_rms_ratio": {
                    "fine_tuned": day2000_ratio,
                    "parent": parent["day2000_anomaly_rms_ratio"],
                    "target": (
                        "one; below one means transients were damped, above one "
                        "that they were manufactured"
                    ),
                },
                "day2000_western_boundary_to_interior_rms_ratio": {
                    "fine_tuned": structure[WESTERN_KEY][
                        "model_boundary_to_interior_rms_ratio"
                    ],
                    "parent": parent["day2000_western_boundary"][
                        "model_boundary_to_interior_rms_ratio"
                    ],
                    "truth": structure[WESTERN_KEY][
                        "truth_boundary_to_interior_rms_ratio"
                    ],
                    "note": (
                        "the truth value is a property of MITgcm and must agree "
                        "between the two packages; that it does is a check on the "
                        "comparison, not a result"
                    ),
                },
                "same_reference_field_removed": True,
            },
            "total_field_note": (
                "amplitude diagnostics including the day-2000 streamfunction minimum "
                "remain defined on the total field and are unaffected by this package"
            ),
            "modifies_published_figures": False,
            "figures": list(FIGURE_NAMES) + [REFERENCE_FIGURE],
            "arrays": str(scratch / ARRAYS_NAME),
            "arrays_sha256": _file_sha256(arrays_path),
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
            "version": VERSION,
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
        (project_tmp / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(scratch_tmp, scratch)
        os.replace(project_tmp, project)
    except Exception:
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        shutil.rmtree(project_tmp, ignore_errors=True)
        raise
    return report


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the contract and the sealed figure package without plotting."""

    contract, resolved, digest = load_contract(contract_path)
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "regime": REGIME,
        "training_version": TRAINING_VERSION,
        "parent_version": PARENT_VERSION,
        "figure_package_version": FIGURE_PACKAGE_VERSION,
        "reference_days": list(TRAIN_RANGE),
        "reference_source": "mitgcm_training_block_only",
        "same_reference_for_truth_and_prediction": True,
        "same_reference_as_the_parent_package": True,
        "figure_names": list(FIGURE_NAMES),
        "modifies_published_figures": False,
        "reads_model_weights": False,
        "rolls_nothing_out": True,
        "figure_package_arrays": contract["artifacts"]["figure_package_arrays"]["path"],
        "parent_anomaly": parent_anomaly(contract),
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
    print(json.dumps(json_safe(result), indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
