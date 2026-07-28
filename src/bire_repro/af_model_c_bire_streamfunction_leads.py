"""Per-lead Bire-style streamfunction maps for the rejected Model C successor.

This descriptive evaluator extends the fixed Figure-3 member from days 20
through 90.  Each output compares MITgcm truth, Model C prediction, and their
difference.  Truth and prediction share one scale across all leads; every
difference panel uses a labeled lead-specific scale so small errors are visible.
Inference and all later archives remain sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from .af_a0_evaluate import _normalizers
from .af_data import STATE_CHANNELS
from .af_forward_complete import _state_fields
from .af_model_c_bire_figures import _verify_external_sources
from .af_model_c_overfit import _file_sha256
from .af_model_c_successor_validation import (
    _load_successor_stepper,
    load_validation_contract,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_bire_streamfunction_leads_v1"
REPORT_NAME = "model_c_bire_streamfunction_leads_report.json"
ARRAYS_NAME = "model_c_bire_streamfunction_leads_arrays.npz"
SUMMARY_NAME = "streamfunction_lead_summary.json"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"
EVALUATION_LEADS = tuple(range(0, 91, 10))
FIGURE_LEADS = tuple(range(20, 91, 10))
FIGURE_NAMES = tuple(
    f"model_c_bire_figure3_streamfunction_1deg_day{lead:03d}.png"
    for lead in FIGURE_LEADS
)


class ModelCStreamfunctionLeadError(RuntimeError):
    """Raised when the frozen per-lead characterization contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def figure_name(lead_day: int) -> str:
    """Return the immutable filename for one requested lead."""

    if lead_day not in FIGURE_LEADS:
        raise ValueError(f"lead day is not one of {FIGURE_LEADS}")
    return f"model_c_bire_figure3_streamfunction_1deg_day{lead_day:03d}.png"


def streamfunction_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    wet: np.ndarray,
) -> dict[str, float]:
    """Return wet-cell error and amplitude metrics in Sverdrups."""

    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(wet, dtype=bool)
    if (
        truth_array.shape != prediction_array.shape
        or truth_array.shape != mask.shape
        or not np.any(mask)
    ):
        raise ValueError("streamfunction arrays and wet mask are incompatible")
    error = truth_array[mask] - prediction_array[mask]
    truth_values = truth_array[mask]
    rmse = float(np.sqrt(np.mean(error**2)))
    truth_rms = float(np.sqrt(np.mean(truth_values**2)))
    return {
        "rmse_sv": rmse,
        "maximum_absolute_error_sv": float(np.max(np.abs(error))),
        "truth_rms_sv": truth_rms,
        "relative_rmse": rmse / max(truth_rms, np.finfo(float).eps),
    }


def load_streamfunction_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and validate the frozen per-lead visualization contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_before_20_to_90_day_streamfunction_maps"
    ):
        raise ValueError("per-lead streamfunction contract is not frozen")
    evaluation = contract.get("evaluation", {})
    if (
        int(evaluation.get("regime_index", -1)) != 2
        or int(evaluation.get("validation_snapshot_code", -1)) != 2
        or int(evaluation.get("start", -1)) != 6335
        or tuple(evaluation.get("evaluation_lead_days", ()))
        != EVALUATION_LEADS
        or tuple(evaluation.get("figure_lead_days", ())) != FIGURE_LEADS
        or tuple(evaluation.get("figure_names", ())) != FIGURE_NAMES
    ):
        raise ValueError("per-lead streamfunction evaluation changed")
    plot = contract.get("plot_contract", {})
    if (
        plot.get("panels")
        != [
            "MITgcm_ground_truth",
            "Model_C_prediction",
            "ground_truth_minus_prediction",
        ]
        or plot.get("truth_prediction_scale")
        != "single_shared_symmetric_scale_across_all_leads"
        or plot.get("difference_scale")
        != "lead_specific_symmetric_maximum_absolute_error"
        or plot.get("annotate_rmse_and_maximum_error") is not True
    ):
        raise ValueError("per-lead streamfunction plot contract changed")
    read = contract.get("read_contract", {})
    if (
        read.get("fresh_validation_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "training_state",
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ValueError("per-lead streamfunction read contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(
                    f"per-lead streamfunction source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _plot_lead(
    output: Path,
    lead: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
    state_bound: float,
) -> dict[str, float]:
    metrics = streamfunction_metrics(truth, prediction, wet)
    difference = np.asarray(truth) - np.asarray(prediction)
    difference_bound = max(
        metrics["maximum_absolute_error_sv"],
        np.finfo(float).eps,
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.75),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    state_image = axes[0].pcolormesh(
        longitude,
        latitude,
        np.ma.masked_where(~wet, truth),
        cmap="RdBu_r",
        vmin=-state_bound,
        vmax=state_bound,
        shading="auto",
    )
    axes[1].pcolormesh(
        longitude,
        latitude,
        np.ma.masked_where(~wet, prediction),
        cmap="RdBu_r",
        vmin=-state_bound,
        vmax=state_bound,
        shading="auto",
    )
    difference_image = axes[2].pcolormesh(
        longitude,
        latitude,
        np.ma.masked_where(~wet, difference),
        cmap="RdBu_r",
        vmin=-difference_bound,
        vmax=difference_bound,
        shading="auto",
    )
    axes[0].set_title("MITgcm ground truth")
    axes[1].set_title("Model C prediction")
    axes[2].set_title(
        f"GT − prediction\nerror scale ±{difference_bound:.3f} Sv"
    )
    axes[0].set_ylabel("Latitude (°)")
    for axis in axes:
        axis.set_xlabel("Longitude (°)")
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    figure.colorbar(
        state_image,
        ax=axes[:2].tolist(),
        label="Barotropic streamfunction (Sv)",
        shrink=0.84,
    )
    figure.colorbar(
        difference_image,
        ax=axes[2],
        label="GT − prediction (Sv)",
        shrink=0.84,
    )
    figure.suptitle(
        "Model C barotropic streamfunction at "
        f"1° — day {lead}\n"
        f"RMSE {metrics['rmse_sv']:.4f} Sv; "
        f"max |error| {metrics['maximum_absolute_error_sv']:.4f} Sv; "
        f"relative RMSE {100.0 * metrics['relative_rmse']:.3f}%"
    )
    figure.savefig(output / figure_name(lead), bbox_inches="tight", dpi=180)
    plt.close(figure)
    return {
        **metrics,
        "truth_prediction_symmetric_bound_sv": state_bound,
        "difference_symmetric_bound_sv": difference_bound,
    }


def _project_readme(report: Mapping[str, Any], manifest_hash: str) -> str:
    metrics = report["lead_metrics"]
    lines = [
        "# Model C Bire-style per-lead streamfunction maps",
        "",
        "Status: complete descriptive characterization of the scientifically",
        "rejected Model C successor. This package cannot authorize tuning or",
        "inference.",
        "",
        "The eight figures show S2 validation start 6335 separately at days",
        "20, 30, ..., 90. Each contains MITgcm truth, Model C prediction, and",
        "truth-minus-prediction at the native 1-degree resolution. Truth and",
        "prediction share one scale across every lead. Each error map uses its",
        "own labeled symmetric scale so the spatial error remains visible.",
        "",
        "| Lead (days) | RMSE (Sv) | Max absolute error (Sv) | Relative RMSE |",
        "|---:|---:|---:|---:|",
    ]
    for lead in FIGURE_LEADS:
        values = metrics[str(lead)]
        lines.append(
            f"| {lead} | {values['rmse_sv']:.6f} | "
            f"{values['maximum_absolute_error_sv']:.6f} | "
            f"{100.0 * values['relative_rmse']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "The complete report and arrays remain immutable in scratch.",
            f"Figure-manifest content SHA-256: `{manifest_hash}`.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_streamfunction_leads(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    validation_report_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate and save the frozen eight-figure per-lead package."""

    if torch is None:
        raise RuntimeError("per-lead streamfunction figures require PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_streamfunction_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    quality_path = Path(quality_report_path).resolve()
    validation_path = Path(validation_report_path).resolve()
    output = Path(output_dir).resolve()
    figures = Path(figure_dir).resolve()
    expected_output = contract["output_contract"]
    if (
        output != Path(expected_output["scratch_output"]).resolve()
        or figures != Path(expected_output["project_output"]).resolve()
    ):
        raise ModelCStreamfunctionLeadError("output path changed")
    temporary_output = output.with_name(output.name + ".tmp")
    temporary_figures = figures.with_name(figures.name + ".tmp")
    if any(
        path.exists()
        for path in (output, figures, temporary_output, temporary_figures)
    ):
        raise FileExistsError(
            "refusing to overwrite per-lead streamfunction output"
        )
    validation_report = _verify_external_sources(
        contract,
        dataset,
        quality_path,
        validation_path,
    )
    source = contract["source_artifacts"]
    validation_contract, _, validation_contract_sha = (
        load_validation_contract(source["successor_validation_contract"])
    )
    if (
        validation_contract_sha
        != source["successor_validation_contract_sha256"]
    ):
        raise ModelCStreamfunctionLeadError(
            "successor validation contract changed"
        )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA characterization requested without a GPU")
    device = torch.device(device_name)

    group = zarr.open_consolidated(str(dataset), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise ModelCStreamfunctionLeadError("trajectory-v2 channels changed")
    state = group["state"]
    static = group["static_features"]
    evaluation = contract["evaluation"]
    start = int(evaluation["start"])
    regime = int(evaluation["regime_index"])
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    offsets = np.asarray(EVALUATION_LEADS, dtype=np.int64)
    if (
        not np.all(snapshot_codes[start + offsets] == 2)
        or not np.all(pair_codes[start + offsets[:-1]] == 2)
    ):
        raise ModelCStreamfunctionLeadError(
            "fixed member no longer has complete validation truth"
        )

    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    stepper, payload = _load_successor_stepper(
        Path(source["checkpoint"]),
        device,
        wet,
        mean,
        scale,
        wind_mean,
        wind_scale,
        validation_contract["architecture"],
    )
    if int(payload.get("seed", -1)) != int(source["checkpoint_seed"]):
        raise ModelCStreamfunctionLeadError("checkpoint seed changed")
    initial = np.asarray(state[regime, start], dtype=np.float32)[None]
    current = stepper.normalized_state(initial)
    forcing = stepper.normalized_static(
        static,
        np.asarray([regime], dtype=np.int64),
    )
    truth_streamfunction = np.empty(
        (len(EVALUATION_LEADS), *wet.shape),
        dtype=np.float32,
    )
    prediction_streamfunction = np.empty_like(truth_streamfunction)
    finite = np.empty(len(EVALUATION_LEADS), dtype=np.uint8)
    normalized_max_abs = np.empty(len(EVALUATION_LEADS), dtype=np.float32)
    wet_t = torch.from_numpy(wet).to(device)
    with torch.no_grad():
        for index, lead in enumerate(EVALUATION_LEADS):
            if lead:
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
            else:
                prediction = initial.copy()
            truth = np.asarray(
                state[regime, start + lead],
                dtype=np.float32,
            )[None]
            truth_streamfunction[index] = _state_fields(
                truth,
                wet,
            )["streamfunction"][0, 0]
            prediction_streamfunction[index] = _state_fields(
                prediction,
                wet,
            )["streamfunction"][0, 0]
            finite[index] = np.isfinite(prediction).all()
            normalized_max_abs[index] = (
                torch.amax(torch.abs(current[:, :, wet_t]))
                .detach()
                .cpu()
                .item()
            )
    if not np.all(finite):
        raise ModelCStreamfunctionLeadError(
            "non-finite prediction in the requested lead range"
        )

    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    state_bound = max(
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        (
                            truth_streamfunction[:, wet],
                            prediction_streamfunction[:, wet],
                        ),
                        axis=0,
                    )
                )
            )
        ),
        np.finfo(float).eps,
    )
    arrays = {
        "lead_days": np.asarray(EVALUATION_LEADS, dtype=np.int16),
        "truth_streamfunction": truth_streamfunction,
        "prediction_streamfunction": prediction_streamfunction,
        "longitude_deg": longitude,
        "latitude_deg": latitude,
        "wet_mask": wet.astype(np.uint8),
        "finite": finite,
        "normalized_max_abs": normalized_max_abs,
    }

    temporary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_figures.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.mkdir(exist_ok=False)
    temporary_figures.mkdir(exist_ok=False)
    try:
        arrays_path = temporary_output / ARRAYS_NAME
        np.savez_compressed(arrays_path, **arrays)
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "figure.dpi": 120,
                "savefig.dpi": 180,
            }
        )
        lead_metrics: dict[str, dict[str, float]] = {}
        for lead in FIGURE_LEADS:
            index = EVALUATION_LEADS.index(lead)
            lead_metrics[str(lead)] = _plot_lead(
                temporary_figures,
                lead,
                truth_streamfunction[index],
                prediction_streamfunction[index],
                longitude,
                latitude,
                wet,
                state_bound,
            )
        report = {
            "version": VERSION,
            "status": "complete",
            "purpose": "descriptive_per_lead_streamfunction_visualization",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "dataset": str(dataset),
            "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
            "fresh_validation_report": str(validation_path),
            "fresh_validation_report_sha256": _file_sha256(validation_path),
            "fresh_validation_report_content_sha256": validation_report[
                "report_content_sha256"
            ],
            "checkpoint": source["checkpoint"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "checkpoint_seed": int(source["checkpoint_seed"]),
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
            "regime": "S2",
            "start": start,
            "evaluation_lead_days": list(EVALUATION_LEADS),
            "figure_lead_days": list(FIGURE_LEADS),
            "truth_prediction_symmetric_bound_sv": state_bound,
            "lead_metrics": lead_metrics,
            "all_predictions_finite": True,
            "maximum_normalized_abs": float(np.max(normalized_max_abs)),
            "arrays": str(output / ARRAYS_NAME),
            "arrays_sha256": _file_sha256(arrays_path),
            "figures": {
                name: {
                    "path": str(figures / name),
                    "sha256": _file_sha256(temporary_figures / name),
                }
                for name in FIGURE_NAMES
            },
            "read_contract": {
                "training_state": False,
                "fresh_validation_state": True,
                "inference_state": False,
                "intermediate_wind_state": False,
                "response_state": False,
                "adjoint_state": False,
            },
            "inference_opened": False,
            "tuning_authorized": False,
        }
        report["report_content_sha256"] = _json_sha256(report)
        report_path = temporary_output / REPORT_NAME
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "version": VERSION,
            "status": "complete",
            "report": str(output / REPORT_NAME),
            "report_sha256": _file_sha256(report_path),
            "report_content_sha256": report["report_content_sha256"],
            "start": start,
            "lead_metrics": lead_metrics,
            "all_predictions_finite": True,
            "maximum_normalized_abs": report["maximum_normalized_abs"],
            "inference_opened": False,
            "tuning_authorized": False,
        }
        summary["summary_content_sha256"] = _json_sha256(summary)
        summary_path = temporary_figures / SUMMARY_NAME
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "version": VERSION,
            "status": "complete",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "report": str(output / REPORT_NAME),
            "report_sha256": _file_sha256(report_path),
            "report_content_sha256": report["report_content_sha256"],
            "arrays": str(output / ARRAYS_NAME),
            "arrays_sha256": report["arrays_sha256"],
            "summary": str(figures / SUMMARY_NAME),
            "summary_sha256": _file_sha256(summary_path),
            "figures": report["figures"],
            "inference_opened": False,
            "tuning_authorized": False,
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        (temporary_figures / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary_figures / README_NAME).write_text(
            _project_readme(report, manifest["manifest_content_sha256"])
        )
        os.replace(temporary_output, output)
        os.replace(temporary_figures, figures)
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        shutil.rmtree(temporary_figures, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_streamfunction_leads(
        args.dataset,
        args.quality_report,
        args.validation_report,
        args.contract,
        args.output_dir,
        args.figure_dir,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
