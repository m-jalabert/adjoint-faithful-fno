"""Corrected ten-day-stride training-only spectral attribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import af_model_c_anomaly_direct_training_spectral_attribution as base
from .af_a0_evaluate import _normalizers
from .af_forward_complete import derived_fields, radial_spectrum
from .af_model_c_anomaly_direct import PointwiseDirectStepper
from .af_model_c_overfit import _device, _file_sha256
from .af_model_c_successor import ModelCSuccessorArchitecture, build_successor
from .af_pressure import phihyd_from_theta_eta

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_anomaly_direct_training_spectral_attribution_v2"
HORIZON_DAYS = 10
ROLLOUT_STEPS = base.ROLLOUT_STEPS
PRESSURE_LEVELS = base.PRESSURE_LEVELS
OUTPUT_NAMES = base.OUTPUT_NAMES


class CorrectedTrainingSpectralAttributionError(RuntimeError):
    """Raised when the corrected attribution contract changes."""


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_after_v1_stride_rejection_before_corrected_state_open"
        or protocol.get("split_code") != 1
        or protocol.get("horizon_days") != HORIZON_DAYS
        or protocol.get("snapshot_stride_per_model_call") != HORIZON_DAYS
        or protocol.get("rollout_steps") != ROLLOUT_STEPS
        or protocol.get("tail_start_radial_mode") != base.TAIL_START_MODE
        or len(contract["sources"].get("seeds", ())) != 3
        or read.get("inference_state") is not False
        or read.get("response_or_adjoint_state") is not False
    ):
        raise ValueError("corrected training spectral-attribution contract changed")
    return contract, resolved, _file_sha256(resolved)


def _verify_sources(contract: Mapping[str, Any], dataset: Path) -> None:
    sources = contract["sources"]
    if (
        dataset != Path(sources["dataset"]["path"]).resolve()
        or _file_sha256(dataset / ".zmetadata")
        != sources["dataset"]["metadata_sha256"]
    ):
        raise CorrectedTrainingSpectralAttributionError("dataset source changed")
    for name in ("replication_summary", "invalid_v1_report", "invalid_v1_manifest"):
        record = sources[name]
        path = Path(record["path"])
        if not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise CorrectedTrainingSpectralAttributionError(f"{name} changed")
    runner = sources["runner"]
    if (
        Path(runner["path"]).resolve() != Path(__file__).resolve()
        or _file_sha256(Path(__file__)) != runner["sha256"]
    ):
        raise CorrectedTrainingSpectralAttributionError("corrected runner changed")
    parent = sources["rejected_v1_runner"]
    if (
        Path(parent["path"]).resolve() != Path(base.__file__).resolve()
        or _file_sha256(Path(base.__file__)) != parent["sha256"]
    ):
        raise CorrectedTrainingSpectralAttributionError("rejected v1 runner changed")
    for record in sources["seeds"]:
        for key in ("checkpoint", "normalization"):
            path = Path(record[key]["path"])
            if not path.is_file() or _file_sha256(path) != record[key]["sha256"]:
                raise CorrectedTrainingSpectralAttributionError(
                    f"seed {record['seed']} {key} changed"
                )


def training_records(
    contract: Mapping[str, Any],
    split: np.ndarray,
) -> np.ndarray:
    """Return records whose complete 360-day daily window remains split 1."""

    times = np.asarray(contract["protocol"]["start_times"], dtype=np.int32)
    if base._array_sha256(times) != contract["protocol"]["start_times_sha256"]:
        raise CorrectedTrainingSpectralAttributionError("start times changed")
    records = np.asarray(
        [(experiment, int(time)) for experiment in range(3) for time in times],
        dtype=np.int32,
    )
    if base._array_sha256(records) != contract["protocol"]["records_sha256"]:
        raise CorrectedTrainingSpectralAttributionError("training records changed")
    final_offset = HORIZON_DAYS * ROLLOUT_STEPS
    for _, time in records:
        if not np.all(split[time : time + final_offset + 1] == 1):
            raise CorrectedTrainingSpectralAttributionError(
                "a corrected attribution trajectory leaves split 1"
            )
    return records


def target_snapshot_offset(model_step: int) -> int:
    """Convert a one-based model call to its daily dataset offset."""

    if model_step < 1 or model_step > ROLLOUT_STEPS:
        raise ValueError("model step is outside the corrected rollout")
    return model_step * HORIZON_DAYS


def _gather_target(state: Any, records: np.ndarray, model_step: int) -> np.ndarray:
    return base._gather_states(
        state,
        records,
        target_snapshot_offset(model_step),
    )


def _evaluate_seed(
    seed_record: Mapping[str, Any],
    *,
    device: Any,
    initial: np.ndarray,
    raw_static: np.ndarray,
    experiments: np.ndarray,
    state: Any,
    records: np.ndarray,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    modes: np.ndarray,
) -> dict[str, Any]:
    checkpoint_path = Path(seed_record["checkpoint"]["path"])
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(payload["optimizer_step"]) != int(seed_record["optimizer_step"]):
        raise CorrectedTrainingSpectralAttributionError(
            f"seed {seed_record['seed']} optimizer step changed"
        )
    architecture = ModelCSuccessorArchitecture(**payload["architecture"])
    model = build_successor(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with np.load(seed_record["normalization"]["path"], allow_pickle=False) as z:
        mean = np.asarray(z["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(z["pointwise_scale"], dtype=np.float32)
    stepper = PointwiseDirectStepper(
        model=model,
        device=device,
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
    )
    current = stepper.normalized_state(initial)
    forcing = stepper.normalized_static(raw_static, experiments)
    ratio = np.empty((ROLLOUT_STEPS, PRESSURE_LEVELS), dtype=np.float64)
    integrated = np.empty_like(ratio)
    tail_model = np.empty_like(ratio)
    tail_truth = np.empty_like(ratio)
    primary_fields = ("surface_speed", "sst", "phihyd_surface")
    model_rmse = np.empty((ROLLOUT_STEPS, len(primary_fields)), dtype=np.float64)
    persistence_rmse = np.empty_like(model_rmse)
    truth_day360 = np.empty((PRESSURE_LEVELS, modes.size), dtype=np.float64)
    model_day360 = np.empty_like(truth_day360)
    persistence_fields = derived_fields(initial, wet)

    with torch.no_grad():
        for lead_index in range(ROLLOUT_STEPS):
            current = stepper.step(current, forcing)
            prediction = stepper.physical(current)
            truth = _gather_target(state, records, lead_index + 1)
            predicted_pressure = phihyd_from_theta_eta(
                prediction[:, 30:45],
                prediction[:, 45],
                wet,
            )
            truth_pressure = phihyd_from_theta_eta(
                truth[:, 30:45],
                truth[:, 45],
                wet,
            )
            for level in range(PRESSURE_LEVELS):
                observed_modes, truth_spectrum = radial_spectrum(
                    truth_pressure[:, level],
                    wet,
                )
                _, predicted_spectrum = radial_spectrum(
                    predicted_pressure[:, level],
                    wet,
                )
                if not np.array_equal(observed_modes, modes):
                    raise CorrectedTrainingSpectralAttributionError(
                        "radial-mode grid changed"
                    )
                summary = base.spectral_summary(
                    modes,
                    truth_spectrum,
                    predicted_spectrum,
                )
                ratio[lead_index, level] = summary[
                    "frozen_median_modewise_ratio"
                ]
                integrated[lead_index, level] = summary[
                    "integrated_energy_ratio"
                ]
                tail_model[lead_index, level] = summary[
                    "tail_model_fraction"
                ]
                tail_truth[lead_index, level] = summary[
                    "tail_truth_fraction"
                ]
                if lead_index == ROLLOUT_STEPS - 1:
                    truth_day360[level] = truth_spectrum.mean(axis=0)
                    model_day360[level] = predicted_spectrum.mean(axis=0)

            predicted_fields = derived_fields(prediction, wet)
            truth_fields = derived_fields(truth, wet)
            for field_index, field in enumerate(primary_fields):
                model_rmse[lead_index, field_index] = base._masked_rmse(
                    predicted_fields[field],
                    truth_fields[field],
                    wet,
                )
                persistence_rmse[lead_index, field_index] = base._masked_rmse(
                    persistence_fields[field],
                    truth_fields[field],
                    wet,
                )

    del model, stepper, current, forcing
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "optimizer_step": int(payload["optimizer_step"]),
        "ratio": ratio,
        "integrated": integrated,
        "tail_model": tail_model,
        "tail_truth": tail_truth,
        "model_rmse": model_rmse,
        "persistence_rmse": persistence_rmse,
        "truth_day360": truth_day360,
        "model_day360": model_day360,
    }


def run(
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("corrected spectral attribution requires PyTorch")
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    output = Path(output_dir).resolve()
    if output != Path(contract["output"]["directory"]).resolve():
        raise ValueError("output directory differs from corrected contract")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite corrected output: {output}")
    _verify_sources(contract, dataset)

    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = training_records(contract, split)
    experiments = records[:, 0]
    initial = base._gather_states(state, records, 0)
    raw_static = base._gather_static(static, records)
    _, _, wet, _, wind_mean, wind_scale = _normalizers(group)
    wet = np.asarray(wet, dtype=bool)
    device = _device(device_name)
    seeds = np.asarray(
        [record["seed"] for record in contract["sources"]["seeds"]],
        dtype=np.int32,
    )
    lead_days = np.arange(1, ROLLOUT_STEPS + 1, dtype=np.int32) * HORIZON_DAYS
    modes = np.arange(1, 31, dtype=np.float32)
    evaluated = [
        _evaluate_seed(
            record,
            device=device,
            initial=initial,
            raw_static=raw_static,
            experiments=experiments,
            state=state,
            records=records,
            wet=wet,
            wind_mean=float(wind_mean),
            wind_scale=float(wind_scale),
            modes=modes,
        )
        for record in contract["sources"]["seeds"]
    ]
    ratio = np.stack([value["ratio"] for value in evaluated])
    integrated = np.stack([value["integrated"] for value in evaluated])
    tail_model = np.stack([value["tail_model"] for value in evaluated])
    tail_truth = np.stack([value["tail_truth"] for value in evaluated])
    model_rmse = np.stack([value["model_rmse"] for value in evaluated])
    persistence_rmse = np.stack(
        [value["persistence_rmse"] for value in evaluated]
    )
    model_day360 = np.stack([value["model_day360"] for value in evaluated])
    truth_day360 = evaluated[0]["truth_day360"]

    attribution: dict[str, Any] = {}
    reproducible = True
    primary_fields = ("surface_speed", "sst", "phihyd_surface")
    for seed_index, seed in enumerate(seeds):
        result: dict[str, Any] = {}
        for level, name in ((7, "phihyd_mid"), (14, "phihyd_bottom")):
            failed = np.flatnonzero(ratio[seed_index, :, level] > 4.0)
            detail = {
                "first_factor_four_failure_day": (
                    int(lead_days[failed[0]]) if failed.size else None
                ),
                "day360_frozen_median_modewise_ratio": float(
                    ratio[seed_index, -1, level]
                ),
                "day360_integrated_energy_ratio": float(
                    integrated[seed_index, -1, level]
                ),
                "day360_tail_model_fraction": float(
                    tail_model[seed_index, -1, level]
                ),
                "day360_tail_truth_fraction": float(
                    tail_truth[seed_index, -1, level]
                ),
            }
            reproducible = (
                reproducible
                and detail["day360_frozen_median_modewise_ratio"] > 4.0
            )
            result[name] = detail
        result["primary_10_to_90_rmse_ratio_to_persistence"] = {
            field: float(
                np.mean(model_rmse[seed_index, :9, field_index])
                / np.mean(persistence_rmse[seed_index, :9, field_index])
            )
            for field_index, field in enumerate(primary_fields)
        }
        attribution[str(int(seed))] = result

    classification = (
        "seed_consistent_training_split_tail"
        if reproducible
        else "not_seed_consistent_on_training_split"
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "purpose": "corrected_ten_day_stride_training_only_spectral_attribution",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "device": str(device),
        "snapshot_stride_per_model_call": HORIZON_DAYS,
        "training_records": records.tolist(),
        "training_records_sha256": base._array_sha256(records),
        "checkpoint_optimizer_steps": [
            value["optimizer_step"] for value in evaluated
        ],
        "seed_attribution": attribution,
        "classification": classification,
        "next_decision": contract["next_decision"][classification],
        "rejected_v1": contract["rejected_v1"],
        "read_contract": contract["read_contract"],
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    report["content_sha256"] = base._json_sha256(report)

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary corrected output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    np.savez_compressed(
        temporary / OUTPUT_NAMES[1],
        seeds=seeds,
        lead_days=lead_days,
        pressure_levels=np.arange(PRESSURE_LEVELS, dtype=np.int16),
        spectral_modes=modes,
        frozen_median_modewise_ratio=ratio.astype(np.float32),
        integrated_energy_ratio=integrated.astype(np.float32),
        tail_model_fraction=tail_model.astype(np.float32),
        tail_truth_fraction=tail_truth.astype(np.float32),
        primary_model_rmse=model_rmse.astype(np.float32),
        primary_persistence_rmse=persistence_rmse.astype(np.float32),
        truth_spectrum_day360=truth_day360.astype(np.float32),
        model_spectrum_day360=model_day360.astype(np.float32),
        training_records=records,
    )
    (temporary / OUTPUT_NAMES[0]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    base._plot_attribution(
        temporary / OUTPUT_NAMES[2],
        lead_days,
        seeds,
        ratio,
        tail_model,
    )
    base._plot_day360_spectra(
        temporary / OUTPUT_NAMES[3],
        modes,
        seeds,
        truth_day360,
        model_day360,
    )
    (temporary / OUTPUT_NAMES[5]).write_text(
        "# Corrected Model C training-only pressure spectral attribution\n\n"
        "This v2 package uses a ten-daily dataset offset per autoregressive "
        "model call. The v1 package is retained as an invalid temporal-alignment "
        "diagnostic and is not scientific evidence.\n"
    )
    manifest = {
        "version": VERSION,
        "status": "complete",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "artifacts": {
            name: _file_sha256(temporary / name)
            for name in OUTPUT_NAMES
            if name != "manifest.json"
        },
        "rejected_v1_report_sha256": contract["sources"]["invalid_v1_report"][
            "sha256"
        ],
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    manifest["content_sha256"] = base._json_sha256(manifest)
    (temporary / OUTPUT_NAMES[4]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def preflight(contract_path: str | Path) -> dict[str, Any]:
    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    _verify_sources(contract, dataset)
    split = np.asarray(
        zarr.open_consolidated(str(dataset), mode="r")["snapshot_split"][:],
        dtype=np.uint8,
    )
    records = training_records(contract, split)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "snapshot_stride_per_model_call": HORIZON_DAYS,
        "final_target_offset_days": target_snapshot_offset(ROLLOUT_STEPS),
        "training_records": int(records.shape[0]),
        "seeds": [record["seed"] for record in contract["sources"]["seeds"]],
        "read_contract": contract["read_contract"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--output-dir", type=Path, required=True)
            child.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        preflight(args.contract)
        if args.command == "preflight"
        else run(args.contract, args.output_dir, device_name=args.device)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
