"""Training-only spectral attribution for the three accepted Model C seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

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


VERSION = "model_c_anomaly_direct_training_spectral_attribution_v1"
ROLLOUT_STEPS = 36
STATE_CHANNELS = 46
PRESSURE_LEVELS = 15
TAIL_START_MODE = 10
OUTPUT_NAMES = (
    "training_spectral_attribution.json",
    "training_spectral_attribution_arrays.npz",
    "training_only_pressure_spectral_attribution.png",
    "training_only_day360_pressure_spectra.png",
    "manifest.json",
    "README.md",
)


class TrainingSpectralAttributionError(RuntimeError):
    """Raised when the frozen training-only attribution contract changes."""


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def spectral_summary(
    modes: np.ndarray,
    truth_members: np.ndarray,
    model_members: np.ndarray,
) -> dict[str, float | int | bool]:
    """Apply the frozen modewise gate and scale-aware tail statistics."""

    modes = np.asarray(modes, dtype=np.float64)
    truth = np.asarray(truth_members, dtype=np.float64).mean(axis=0)
    model = np.asarray(model_members, dtype=np.float64).mean(axis=0)
    if truth.shape != modes.shape or model.shape != modes.shape:
        raise ValueError("spectra do not match the radial-mode vector")
    valid = truth > max(float(np.max(truth)) * 1.0e-8, 1.0e-20)
    tail = valid & (modes >= TAIL_START_MODE)
    if not np.any(valid) or not np.any(tail):
        raise ValueError("frozen valid or high-wavenumber band is empty")
    ratio = model[valid] / truth[valid]
    truth_total = float(np.sum(truth[valid]))
    model_total = float(np.sum(model[valid]))
    median = float(np.median(ratio))
    return {
        "valid_mode_count": int(np.sum(valid)),
        "frozen_median_modewise_ratio": median,
        "frozen_factor_four_pass": bool(0.25 <= median <= 4.0),
        "integrated_energy_ratio": model_total / truth_total,
        "tail_truth_fraction": float(np.sum(truth[tail]) / truth_total),
        "tail_model_fraction": float(np.sum(model[tail]) / model_total),
        "tail_integrated_ratio": float(
            np.sum(model[tail]) / np.sum(truth[tail])
        ),
    }


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_before_training_split_state_or_model_rollout_open"
        or contract["protocol"].get("split_code") != 1
        or contract["protocol"].get("rollout_steps") != ROLLOUT_STEPS
        or contract["protocol"].get("tail_start_radial_mode") != TAIL_START_MODE
        or len(contract["sources"].get("seeds", ())) != 3
        or contract["read_contract"].get("inference_state") is not False
        or contract["read_contract"].get("response_or_adjoint_state") is not False
    ):
        raise ValueError("training spectral-attribution contract changed")
    return contract, resolved, _file_sha256(resolved)


def _verify_sources(
    contract: Mapping[str, Any],
    dataset: Path,
) -> None:
    sources = contract["sources"]
    if (
        dataset != Path(sources["dataset"]["path"]).resolve()
        or _file_sha256(dataset / ".zmetadata")
        != sources["dataset"]["metadata_sha256"]
    ):
        raise TrainingSpectralAttributionError("dataset source changed")
    replication = sources["replication_summary"]
    replication_path = Path(replication["path"])
    if (
        not replication_path.is_file()
        or _file_sha256(replication_path) != replication["sha256"]
    ):
        raise TrainingSpectralAttributionError("replication summary changed")
    runner = sources["runner"]
    if (
        Path(runner["path"]).resolve() != Path(__file__).resolve()
        or _file_sha256(Path(__file__)) != runner["sha256"]
    ):
        raise TrainingSpectralAttributionError("attribution runner changed")
    for record in sources["seeds"]:
        for key in ("checkpoint", "normalization"):
            path = Path(record[key]["path"])
            if not path.is_file() or _file_sha256(path) != record[key]["sha256"]:
                raise TrainingSpectralAttributionError(
                    f"seed {record['seed']} {key} changed"
                )


def _records(contract: Mapping[str, Any], split: np.ndarray) -> np.ndarray:
    times = np.asarray(contract["protocol"]["start_times"], dtype=np.int32)
    expected = contract["protocol"]["start_times_sha256"]
    if _array_sha256(times) != expected:
        raise TrainingSpectralAttributionError("training start times changed")
    if len(np.unique(times)) != times.size:
        raise TrainingSpectralAttributionError("training start times repeat")
    records = np.asarray(
        [(experiment, int(time)) for experiment in range(3) for time in times],
        dtype=np.int32,
    )
    if _array_sha256(records) != contract["protocol"]["records_sha256"]:
        raise TrainingSpectralAttributionError("training records changed")
    for _, time in records:
        if not np.all(split[time : time + ROLLOUT_STEPS + 1] == 1):
            raise TrainingSpectralAttributionError(
                "a training attribution trajectory leaves split 1"
            )
    return records


def _gather_states(state: Any, records: np.ndarray, offset: int) -> np.ndarray:
    return np.stack(
        [
            np.asarray(state[int(experiment), int(time) + offset])
            for experiment, time in records
        ]
    ).astype(np.float32, copy=False)


def _gather_static(static: Any, records: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.asarray(static[int(experiment)]) for experiment, _ in records]
    ).astype(np.float32, copy=False)


def _masked_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    wet: np.ndarray,
) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(error[:, wet]))))


def _plot_attribution(
    path: Path,
    lead_days: np.ndarray,
    seeds: np.ndarray,
    median_ratio: np.ndarray,
    tail_fraction: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for column, seed in enumerate(seeds):
        axis = axes[0, column]
        image = axis.pcolormesh(
            lead_days,
            np.arange(PRESSURE_LEVELS),
            median_ratio[column].T,
            shading="nearest",
            norm=colors.LogNorm(vmin=0.25, vmax=max(64.0, float(np.nanmax(median_ratio)))),
            cmap="viridis",
        )
        axis.contour(
            lead_days,
            np.arange(PRESSURE_LEVELS),
            median_ratio[column].T,
            levels=[4.0],
            colors=["white"],
            linewidths=1.2,
        )
        axis.set_title(f"Seed {int(seed)}: frozen modewise ratio")
        axis.set_ylabel("PHIHYD vertical level")
        axis.invert_yaxis()
        figure.colorbar(image, ax=axis, label="median model/truth energy")

        axis = axes[1, column]
        image = axis.pcolormesh(
            lead_days,
            np.arange(PRESSURE_LEVELS),
            100.0 * tail_fraction[column].T,
            shading="nearest",
            cmap="magma",
        )
        axis.set_title(f"Seed {int(seed)}: model k≥10 energy share")
        axis.set_xlabel("Lead (model days)")
        axis.set_ylabel("PHIHYD vertical level")
        axis.invert_yaxis()
        figure.colorbar(image, ax=axis, label="percent of valid energy")
    figure.suptitle(
        "Training-only attribution of Model C deep-pressure spectral tail"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_day360_spectra(
    path: Path,
    modes: np.ndarray,
    seeds: np.ndarray,
    truth: np.ndarray,
    model: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, level, title in zip(
        axes,
        (7, 14),
        ("PHIHYD mid-depth (k=7)", "PHIHYD bottom (k=14)"),
        strict=True,
    ):
        axis.semilogy(modes, truth[level], "ko-", label="training truth")
        for index, seed in enumerate(seeds):
            axis.semilogy(
                modes,
                model[index, level],
                "o-",
                label=f"seed {int(seed)}",
            )
        axis.axvspan(
            TAIL_START_MODE - 0.5,
            float(np.max(modes)) + 0.5,
            color="tab:red",
            alpha=0.08,
        )
        axis.set_title(title)
        axis.set_xlabel("Radial Fourier mode")
        axis.set_ylabel("Mean spectral energy")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Day-360 split-1 pressure spectra across accepted seeds")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("training spectral attribution requires PyTorch")
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    output = Path(output_dir).resolve()
    if output != Path(contract["output"]["directory"]).resolve():
        raise ValueError("output directory differs from frozen contract")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite attribution output: {output}")
    _verify_sources(contract, dataset)

    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = _records(contract, split)
    experiments = records[:, 0]
    initial = _gather_states(state, records, 0)
    raw_static = _gather_static(static, records)
    _, _, wet, _, wind_mean, wind_scale = _normalizers(group)
    wet = np.asarray(wet, dtype=bool)
    device = _device(device_name)

    seed_records = contract["sources"]["seeds"]
    seeds = np.asarray([record["seed"] for record in seed_records], dtype=np.int32)
    lead_days = np.arange(10, 361, 10, dtype=np.int32)
    modes = np.arange(1, 31, dtype=np.float32)
    ratio = np.empty((3, ROLLOUT_STEPS, PRESSURE_LEVELS), dtype=np.float64)
    integrated = np.empty_like(ratio)
    tail_model = np.empty_like(ratio)
    tail_truth = np.empty_like(ratio)
    primary_fields = ("surface_speed", "sst", "phihyd_surface")
    model_rmse = np.empty((3, ROLLOUT_STEPS, len(primary_fields)), dtype=np.float64)
    persistence_rmse = np.empty_like(model_rmse)
    truth_spectrum_day360 = np.empty(
        (PRESSURE_LEVELS, modes.size), dtype=np.float64
    )
    model_spectrum_day360 = np.empty(
        (3, PRESSURE_LEVELS, modes.size), dtype=np.float64
    )
    checkpoint_steps = []

    for seed_index, seed_record in enumerate(seed_records):
        checkpoint_path = Path(seed_record["checkpoint"]["path"])
        normalization_path = Path(seed_record["normalization"]["path"])
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if int(payload["optimizer_step"]) != int(seed_record["optimizer_step"]):
            raise TrainingSpectralAttributionError(
                f"seed {seed_record['seed']} optimizer step changed"
            )
        architecture = ModelCSuccessorArchitecture(**payload["architecture"])
        model = build_successor(architecture).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        checkpoint_steps.append(int(payload["optimizer_step"]))
        with np.load(normalization_path, allow_pickle=False) as normalization:
            mean = np.asarray(normalization["pointwise_mean"], dtype=np.float32)
            scale = np.asarray(normalization["pointwise_scale"], dtype=np.float32)
        stepper = PointwiseDirectStepper(
            model=model,
            device=device,
            wet=wet,
            mean=mean,
            scale=scale,
            wind_mean=float(wind_mean),
            wind_scale=float(wind_scale),
        )
        current = stepper.normalized_state(initial)
        forcing = stepper.normalized_static(raw_static, experiments)
        with torch.no_grad():
            for lead_index in range(ROLLOUT_STEPS):
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
                truth = _gather_states(state, records, lead_index + 1)
                prediction_pressure = phihyd_from_theta_eta(
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
                        prediction_pressure[:, level],
                        wet,
                    )
                    if not np.array_equal(observed_modes, modes):
                        raise TrainingSpectralAttributionError(
                            "radial-mode grid changed"
                        )
                    summary = spectral_summary(
                        modes,
                        truth_spectrum,
                        predicted_spectrum,
                    )
                    ratio[seed_index, lead_index, level] = summary[
                        "frozen_median_modewise_ratio"
                    ]
                    integrated[seed_index, lead_index, level] = summary[
                        "integrated_energy_ratio"
                    ]
                    tail_model[seed_index, lead_index, level] = summary[
                        "tail_model_fraction"
                    ]
                    tail_truth[seed_index, lead_index, level] = summary[
                        "tail_truth_fraction"
                    ]
                    if lead_index == ROLLOUT_STEPS - 1:
                        truth_spectrum_day360[level] = truth_spectrum.mean(axis=0)
                        model_spectrum_day360[
                            seed_index, level
                        ] = predicted_spectrum.mean(axis=0)

                prediction_fields = derived_fields(prediction, wet)
                truth_fields = derived_fields(truth, wet)
                persistence_fields = derived_fields(initial, wet)
                for field_index, field in enumerate(primary_fields):
                    model_rmse[seed_index, lead_index, field_index] = _masked_rmse(
                        prediction_fields[field],
                        truth_fields[field],
                        wet,
                    )
                    persistence_rmse[
                        seed_index, lead_index, field_index
                    ] = _masked_rmse(
                        persistence_fields[field],
                        truth_fields[field],
                        wet,
                    )

        del model, stepper, current, forcing
        if device.type == "cuda":
            torch.cuda.empty_cache()

    attribution = {}
    reproducible_mid_bottom = True
    for seed_index, seed in enumerate(seeds):
        seed_result: dict[str, Any] = {}
        for level, name in ((7, "phihyd_mid"), (14, "phihyd_bottom")):
            failed = np.flatnonzero(ratio[seed_index, :, level] > 4.0)
            result = {
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
            reproducible_mid_bottom = (
                reproducible_mid_bottom
                and result["day360_frozen_median_modewise_ratio"] > 4.0
            )
            seed_result[name] = result
        seed_result["primary_10_to_90_rmse_ratio_to_persistence"] = {
            field: float(
                np.mean(model_rmse[seed_index, :9, field_index])
                / np.mean(persistence_rmse[seed_index, :9, field_index])
            )
            for field_index, field in enumerate(primary_fields)
        }
        attribution[str(int(seed))] = seed_result

    report = {
        "status": "complete",
        "version": VERSION,
        "purpose": "training_only_three_seed_deep_pressure_spectral_attribution",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "device": str(device),
        "training_records": records.tolist(),
        "training_records_sha256": _array_sha256(records),
        "checkpoint_optimizer_steps": checkpoint_steps,
        "seed_attribution": attribution,
        "classification": (
            "seed_consistent_training_split_tail"
            if reproducible_mid_bottom
            else "not_seed_consistent_on_training_split"
        ),
        "next_decision": contract["next_decision"][
            "seed_consistent_training_split_tail"
            if reproducible_mid_bottom
            else "not_seed_consistent_on_training_split"
        ],
        "read_contract": contract["read_contract"],
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    report["content_sha256"] = _json_sha256(report)

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary attribution output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    arrays_path = temporary / OUTPUT_NAMES[1]
    np.savez_compressed(
        arrays_path,
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
        truth_spectrum_day360=truth_spectrum_day360.astype(np.float32),
        model_spectrum_day360=model_spectrum_day360.astype(np.float32),
        training_records=records,
    )
    (temporary / OUTPUT_NAMES[0]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _plot_attribution(
        temporary / OUTPUT_NAMES[2],
        lead_days,
        seeds,
        ratio,
        tail_model,
    )
    _plot_day360_spectra(
        temporary / OUTPUT_NAMES[3],
        modes,
        seeds,
        truth_spectrum_day360,
        model_spectrum_day360,
    )
    (temporary / OUTPUT_NAMES[5]).write_text(
        "# Model C training-only pressure spectral attribution\n\n"
        "This package uses only trajectory-v2 split 1 and the three accepted "
        "pointwise-anomaly/direct-state checkpoints. It does not read inference, "
        "response, or adjoint state.\n"
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
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    manifest["content_sha256"] = _json_sha256(manifest)
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
    records = _records(contract, split)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "training_records": int(records.shape[0]),
        "rollout_days": ROLLOUT_STEPS * 10,
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
    if args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(
            args.contract,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
