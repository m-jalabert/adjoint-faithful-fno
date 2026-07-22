"""Frozen-protocol state, rollout, spectrum, and boundary evaluation for Model B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_a0 import STATE_CHANNEL_COUNT, records_for_pair_split
from .af_a0_evaluate import (
    GROUPS,
    _group_correlation,
    _group_rmse,
    _normalizers,
    _normalise,
    _target,
)
from .af_model_a_evaluate import _features
from .af_model_b import (
    ModelBLossConfig,
    binned_spectral_loss,
    build_model_b,
    loss_contract_sha256,
    model_b_architecture,
    western_boundary_mask,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


def _normalized_prediction(model: Any, features: np.ndarray, device: Any) -> np.ndarray:
    with torch.no_grad():
        feature_tensor = torch.from_numpy(features).to(device)
        predicted = feature_tensor[:, :STATE_CHANNEL_COUNT] + model(feature_tensor)
    return predicted.detach().cpu().numpy()


def _direct_metrics(
    model: Any,
    state: Any,
    static: Any,
    records: Sequence[tuple[int, int]],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
    western_boundary: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    horizon: int,
    spectral_bins: int,
    device: Any,
) -> dict[str, Any]:
    prediction, persistence, truth = [], [], []
    spectral_total, spectral_samples = 0.0, 0
    spectral_wet = torch.from_numpy(wet.astype(np.float32))[None, None].to(device)
    for start in range(0, len(records), 8):
        batch = records[start : start + 8]
        features, raw, _ = _features(state, static, batch, mean, scale, wet, wind_mean, wind_scale)
        normalized_prediction = _normalized_prediction(model, features, device)
        held = _target(state, batch, horizon)
        normalized_truth = _normalise(held, mean, scale, wet)
        with torch.no_grad():
            spectral = binned_spectral_loss(
                torch.from_numpy(normalized_prediction).to(device),
                torch.from_numpy(normalized_truth).to(device),
                spectral_wet,
                bins=spectral_bins,
            )
        spectral_total += float(spectral.cpu()) * len(batch)
        spectral_samples += len(batch)
        prediction.append(
            normalized_prediction * scale[None, :, None, None] + mean[None, :, None, None]
        )
        persistence.append(raw)
        truth.append(held)
    predicted, baseline, held = (
        np.concatenate(prediction),
        np.concatenate(persistence),
        np.concatenate(truth),
    )
    for value in (predicted, baseline, held):
        value[:, :, ~wet] = 0.0
    return {
        "pair_count": len(records),
        "model_b_rmse": _group_rmse(predicted, held, wet),
        "persistence_rmse": _group_rmse(baseline, held, wet),
        "model_b_correlation": _group_correlation(predicted, held, wet),
        "persistence_correlation": _group_correlation(baseline, held, wet),
        "western_boundary_model_b_rmse": _group_rmse(predicted, held, western_boundary),
        "western_boundary_persistence_rmse": _group_rmse(baseline, held, western_boundary),
        "model_b_normalized_spectral_loss": spectral_total / spectral_samples,
    }


def _rollout_metrics(
    model: Any,
    state: Any,
    static: Any,
    starts: Sequence[tuple[int, int]],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
    western_boundary: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    horizon: int,
    steps: int,
    device: Any,
) -> dict[str, Any]:
    features, raw_initial, forcing = _features(
        state, static, starts, mean, scale, wet, wind_mean, wind_scale
    )
    current = torch.from_numpy(features[:, :STATE_CHANNEL_COUNT]).to(device)
    forcing_tensor = torch.from_numpy(forcing).to(device)
    wet_tensor = torch.from_numpy(wet).to(device)
    output: dict[str, Any] = {
        "lead_days": [],
        "model_b_rmse": [],
        "persistence_rmse": [],
        "western_boundary_model_b_rmse": [],
        "western_boundary_persistence_rmse": [],
    }
    with torch.no_grad():
        for step in range(1, steps + 1):
            current = current + model(torch.cat((current, forcing_tensor), dim=1))
            current[:, :, ~wet_tensor] = 0.0
            predicted = (
                current.cpu().numpy() * scale[None, :, None, None] + mean[None, :, None, None]
            )
            held = np.stack(
                [
                    np.asarray(state[experiment, time + step * horizon], dtype=np.float32)
                    for experiment, time in starts
                ]
            )
            output["lead_days"].append(step * horizon)
            output["model_b_rmse"].append(_group_rmse(predicted, held, wet))
            output["persistence_rmse"].append(_group_rmse(raw_initial, held, wet))
            output["western_boundary_model_b_rmse"].append(
                _group_rmse(predicted, held, western_boundary)
            )
            output["western_boundary_persistence_rmse"].append(
                _group_rmse(raw_initial, held, western_boundary)
            )
    return output


def _load_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text())


def _plots(
    output: Path,
    direct: dict[str, Any],
    rollout: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    wet: np.ndarray,
    a0_metrics_path: Path | None,
    model_a_metrics_path: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(GROUPS)
    ratio = [direct["model_b_rmse"][name] / direct["persistence_rmse"][name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    axis.bar(np.arange(len(names)), ratio, color="#A86600")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(names)), names, rotation=12, ha="right")
    axis.set_ylabel("10-day RMSE / persistence RMSE")
    axis.set_title("Frozen Model B inference skill (all held regimes)")
    figure.savefig(output / "model_b_one_step_skill.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for name in names:
        values = [
            entry[name] / base[name]
            for entry, base in zip(rollout["model_b_rmse"], rollout["persistence_rmse"])
        ]
        axis.plot(rollout["lead_days"], values, marker="o", label=name)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xlabel("Lead (model days)")
    axis.set_ylabel("RMSE / persistence RMSE")
    axis.set_title("Frozen Model B autoregressive inference rollouts")
    axis.legend(fontsize=8)
    figure.savefig(output / "model_b_rollout_skill.png", dpi=180)
    plt.close(figure)

    fields = (("Theta surface", 30, "$^\\circ$C"), ("SSH", 45, "m"))
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    for row, (title, channel, unit) in enumerate(fields):
        values = [
            snapshot["truth"][channel],
            snapshot["model_b"][channel],
            snapshot["persistence"][channel],
        ]
        bound = max(float(np.max(np.abs(value[wet]))) for value in values)
        for column, (name, value) in enumerate(
            zip(("MITgcm target", "Model B", "Persistence"), values)
        ):
            image = axes[row, column].imshow(
                np.ma.masked_where(~wet, value),
                origin="lower",
                cmap="coolwarm",
                vmin=-bound if row else 0,
                vmax=bound if row else None,
            )
            axes[row, column].set_title(f"{title}: {name}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=0.78, label=unit)
    figure.savefig(output / "model_b_inference_snapshot.png", dpi=180)
    plt.close(figure)

    boundary_ratio = [
        direct["western_boundary_model_b_rmse"][name]
        / direct["western_boundary_persistence_rmse"][name]
        for name in names
    ]
    figure, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    axis.bar(np.arange(len(names)), boundary_ratio, color="#B13A3A")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(names)), names, rotation=12, ha="right")
    axis.set_ylabel("western-boundary RMSE / persistence RMSE")
    axis.set_title("Frozen Model B: first four wet cells from western wall")
    figure.savefig(output / "model_b_western_boundary_skill.png", dpi=180)
    plt.close(figure)

    a0, model_a = _load_metrics(a0_metrics_path), _load_metrics(model_a_metrics_path)
    if a0 is None or model_a is None:
        return
    x, width = np.arange(len(names)), 0.25
    a0_direct, a_direct = a0["direct"]["inference"], model_a["direct"]["inference"]
    ratios = {
        "A0 adapted Bire": [
            a0_direct["a0_rmse"][name] / a0_direct["persistence_rmse"][name] for name in names
        ],
        "Model A state-only": [
            a_direct["model_a_rmse"][name] / a_direct["persistence_rmse"][name] for name in names
        ],
        "Model B forward-loss": ratio,
    }
    colors = ("#2F75B5", "#218A8D", "#A86600")
    figure, axis = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    for index, ((label, values), color) in enumerate(zip(ratios.items(), colors)):
        axis.bar(x + (index - 1) * width, values, width=width, label=label, color=color)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(x, names, rotation=12, ha="right")
    axis.set_ylabel("10-day RMSE / persistence RMSE")
    axis.set_title("Frozen common-protocol comparison")
    axis.legend(fontsize=8)
    figure.savefig(output / "a0_model_a_model_b_one_step.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        for label, source, key, color in (
            ("A0", a0["rollout"], "a0_rmse", colors[0]),
            ("Model A", model_a["rollout"], "model_a_rmse", colors[1]),
            ("Model B", rollout, "model_b_rmse", colors[2]),
        ):
            values = [
                entry[name] / base[name]
                for entry, base in zip(source[key], source["persistence_rmse"])
            ]
            axis.plot(source["lead_days"], values, marker="o", label=label, color=color)
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(name)
        axis.set_ylabel("RMSE / persistence")
    for axis in axes[-1]:
        axis.set_xlabel("Lead (model days)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Frozen A0/A/B autoregressive inference comparison")
    figure.savefig(output / "a0_model_a_model_b_rollout.png", dpi=180)
    plt.close(figure)


def evaluate(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    a0_metrics_path: str | Path | None = None,
    model_a_metrics_path: str | Path | None = None,
    device_name: str = "auto",
    rollout_steps: int = 10,
    starts_per_regime: int = 16,
) -> dict[str, Any]:
    """Evaluate frozen Model B without retraining or changing its loss contract."""

    if torch is None:
        raise RuntimeError("Model B evaluation requires PyTorch")
    if rollout_steps <= 0 or starts_per_regime <= 0:
        raise ValueError("rollout steps and starts per regime must be positive")
    dataset_path = Path(dataset_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Model B evaluation output: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no CUDA device is visible")
    device = torch.device(device_name)
    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state, static = group["state"], group["static_features"]
    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    loss_config = ModelBLossConfig()
    boundary = western_boundary_mask(wet, loss_config.western_boundary_width)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = model_b_architecture()
    if payload["model_config"] != architecture.to_dict():
        raise ValueError("checkpoint architecture does not match the frozen Model A/B contract")
    if payload.get("model_b_loss_contract_sha256") != loss_contract_sha256(loss_config):
        raise ValueError(
            "checkpoint loss contract does not match the complete frozen Model B objective"
        )
    model = build_model_b(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    horizon = 10
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    direct = {
        name: _direct_metrics(
            model,
            state,
            static,
            records_for_pair_split(pair_codes, code),
            mean=mean,
            scale=scale,
            wet=wet,
            western_boundary=boundary,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
            horizon=horizon,
            spectral_bins=loss_config.spectral_bins,
            device=device,
        )
        for name, code in (("validation", 2), ("inference", 3))
    }
    valid_times = np.flatnonzero(
        (pair_codes == 3) & (np.arange(pair_codes.size) + horizon * rollout_steps < state.shape[1])
    )
    selected = np.unique(np.linspace(valid_times[0], valid_times[-1], starts_per_regime, dtype=int))
    starts = tuple((experiment, int(time)) for experiment in range(3) for time in selected)
    rollout = _rollout_metrics(
        model,
        state,
        static,
        starts,
        mean=mean,
        scale=scale,
        wet=wet,
        western_boundary=boundary,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
        horizon=horizon,
        steps=rollout_steps,
        device=device,
    )
    sample = ((0, int(selected[len(selected) // 2])),)
    features, raw, _ = _features(state, static, sample, mean, scale, wet, wind_mean, wind_scale)
    normalized = _normalized_prediction(model, features, device)[0]
    predicted = normalized * scale[:, None, None] + mean[:, None, None]
    snapshot = {
        "model_b": predicted,
        "persistence": raw[0],
        "truth": _target(state, sample, horizon)[0],
    }
    for value in snapshot.values():
        value[:, ~wet] = 0.0
    a0_path = Path(a0_metrics_path).resolve() if a0_metrics_path is not None else None
    model_a_path = (
        Path(model_a_metrics_path).resolve() if model_a_metrics_path is not None else None
    )
    metrics = {
        "status": "complete",
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "western_boundary_definition": "first four wet cells east of each row's western wall",
        "direct": direct,
        "rollout": rollout,
        "rollout_starts": [list(record) for record in starts],
        "a0_metrics_reference": str(a0_path) if a0_path is not None else None,
        "model_a_metrics_reference": str(model_a_path) if model_a_path is not None else None,
    }
    output.mkdir(parents=True)
    (output / "model_b_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(rollout["lead_days"]),
        "wet_mask": wet,
        "western_boundary_mask": boundary,
    }
    for prefix in (
        "model_b",
        "persistence",
        "western_boundary_model_b",
        "western_boundary_persistence",
    ):
        for index, value in enumerate(rollout[f"{prefix}_rmse"]):
            for name, metric in value.items():
                arrays[f"{prefix}_{index}_{name}"] = np.asarray(metric)
    np.savez_compressed(output / "model_b_metrics_arrays.npz", **arrays)
    _plots(output, direct["inference"], rollout, snapshot, wet, a0_path, model_a_path)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save frozen Model B numerical and plot outputs")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a0-metrics", type=Path)
    parser.add_argument("--model-a-metrics", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    result = evaluate(
        args.dataset,
        args.checkpoint,
        args.output_dir,
        a0_metrics_path=args.a0_metrics,
        model_a_metrics_path=args.model_a_metrics,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
