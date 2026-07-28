"""Project-facing numerical and visual evaluation for the frozen Model A FNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..af_a0 import STATE_CHANNEL_COUNT, records_for_pair_split
from ..af_a0_evaluate import GROUPS, _group_correlation, _group_rmse, _normalizers, _normalise, _target
from ..af_model_a import build_model_a, model_a_architecture, require_model_a_runtime

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


def _features(
    state: Any,
    static: Any,
    records: Sequence[tuple[int, int]],
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the exact 46-state plus five-static Model A input contract."""

    raw = np.stack([np.asarray(state[experiment, time], dtype=np.float32) for experiment, time in records])
    geometry = np.stack([np.asarray(static[experiment], dtype=np.float32) for experiment, _ in records])
    geometry[:, 0] = (geometry[:, 0] - wind_mean) / wind_scale
    geometry[:, 0, ~wet] = 0.0
    normalized = _normalise(raw, mean, scale, wet)
    return np.concatenate((normalized, geometry), axis=1), raw, geometry


def _model_prediction(model: Any, features: np.ndarray, mean: np.ndarray, scale: np.ndarray, device: Any) -> np.ndarray:
    with torch.no_grad():
        feature_tensor = torch.from_numpy(features).to(device)
        normalized = feature_tensor[:, :STATE_CHANNEL_COUNT] + model(feature_tensor)
    return normalized.detach().cpu().numpy() * scale[None, :, None, None] + mean[None, :, None, None]


def _direct_metrics(
    model: Any,
    state: Any,
    static: Any,
    records: Sequence[tuple[int, int]],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    horizon: int,
    device: Any,
) -> dict[str, Any]:
    prediction, persistence, truth = [], [], []
    for start in range(0, len(records), 8):
        batch = records[start : start + 8]
        features, raw, _ = _features(state, static, batch, mean, scale, wet, wind_mean, wind_scale)
        prediction.append(_model_prediction(model, features, mean, scale, device))
        persistence.append(raw)
        truth.append(_target(state, batch, horizon))
    predicted, baseline, held = np.concatenate(prediction), np.concatenate(persistence), np.concatenate(truth)
    for value in (predicted, baseline, held):
        value[:, :, ~wet] = 0.0
    return {
        "pair_count": len(records),
        "model_a_rmse": _group_rmse(predicted, held, wet),
        "persistence_rmse": _group_rmse(baseline, held, wet),
        "model_a_correlation": _group_correlation(predicted, held, wet),
        "persistence_correlation": _group_correlation(baseline, held, wet),
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
    wind_mean: float,
    wind_scale: float,
    horizon: int,
    steps: int,
    device: Any,
) -> dict[str, Any]:
    features, raw_initial, forcing = _features(state, static, starts, mean, scale, wet, wind_mean, wind_scale)
    current = torch.from_numpy(features[:, :STATE_CHANNEL_COUNT]).to(device)
    forcing_tensor = torch.from_numpy(forcing).to(device)
    wet_tensor = torch.from_numpy(wet).to(device)
    output: dict[str, Any] = {"lead_days": [], "model_a_rmse": [], "persistence_rmse": []}
    with torch.no_grad():
        for step in range(1, steps + 1):
            current = current + model(torch.cat((current, forcing_tensor), dim=1))
            current[:, :, ~wet_tensor] = 0.0
            predicted = current.detach().cpu().numpy() * scale[None, :, None, None] + mean[None, :, None, None]
            held = np.stack(
                [np.asarray(state[experiment, time + step * horizon], dtype=np.float32) for experiment, time in starts]
            )
            output["lead_days"].append(step * horizon)
            output["model_a_rmse"].append(_group_rmse(predicted, held, wet))
            output["persistence_rmse"].append(_group_rmse(raw_initial, held, wet))
    return output


def _plots(
    output: Path,
    direct: dict[str, Any],
    rollout: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    wet: np.ndarray,
    a0_metrics_path: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(GROUPS)
    ratio = [direct["model_a_rmse"][name] / direct["persistence_rmse"][name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    axis.bar(np.arange(len(names)), ratio, color="#218A8D")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(names)), names, rotation=12, ha="right")
    axis.set_ylabel("10-day RMSE / persistence RMSE")
    axis.set_title("Frozen Model A inference skill (all held regimes)")
    figure.savefig(output / "model_a_one_step_skill.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for name in names:
        values = [entry[name] / base[name] for entry, base in zip(rollout["model_a_rmse"], rollout["persistence_rmse"])]
        axis.plot(rollout["lead_days"], values, marker="o", label=name)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xlabel("Lead (model days)")
    axis.set_ylabel("RMSE / persistence RMSE")
    axis.set_title("Frozen Model A autoregressive inference rollouts")
    axis.legend(fontsize=8)
    figure.savefig(output / "model_a_rollout_skill.png", dpi=180)
    plt.close(figure)

    fields = (("Theta surface", 30, "$^\\circ$C"), ("SSH", 45, "m"))
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    for row, (title, channel, unit) in enumerate(fields):
        values = [snapshot["truth"][channel], snapshot["model_a"][channel], snapshot["persistence"][channel]]
        bound = max(float(np.max(np.abs(value[wet]))) for value in values)
        for column, (name, value) in enumerate(zip(("MITgcm target", "Model A", "Persistence"), values)):
            image = axes[row, column].imshow(np.ma.masked_where(~wet, value), origin="lower", cmap="coolwarm", vmin=-bound if row else 0, vmax=bound if row else None)
            axes[row, column].set_title(f"{title}: {name}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=0.78, label=unit)
    figure.savefig(output / "model_a_inference_snapshot.png", dpi=180)
    plt.close(figure)

    if a0_metrics_path is None or not a0_metrics_path.is_file():
        return
    a0 = json.loads(a0_metrics_path.read_text())
    a0_direct = a0["direct"]["inference"]
    figure, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    x, width = np.arange(len(names)), 0.36
    a0_ratio = [a0_direct["a0_rmse"][name] / a0_direct["persistence_rmse"][name] for name in names]
    a_ratio = ratio
    axis.bar(x - width / 2, a0_ratio, width=width, label="A0 adapted Bire", color="#2F75B5")
    axis.bar(x + width / 2, a_ratio, width=width, label="Model A modern", color="#218A8D")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(x, names, rotation=12, ha="right")
    axis.set_ylabel("10-day RMSE / persistence RMSE")
    axis.set_title("Frozen A0 versus Model A: held inference")
    axis.legend(fontsize=8)
    figure.savefig(output / "a0_vs_model_a_one_step.png", dpi=180)
    plt.close(figure)


def evaluate(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    a0_metrics_path: str | Path | None = None,
    device_name: str = "auto",
    rollout_steps: int = 10,
    starts_per_regime: int = 16,
) -> dict[str, Any]:
    """Save held Model A numerical metrics and project-facing plots without retraining."""

    require_model_a_runtime()
    if rollout_steps <= 0 or starts_per_regime <= 0:
        raise ValueError("rollout_steps and starts_per_regime must be positive")
    dataset_path, checkpoint_path, output = Path(dataset_path).resolve(), Path(checkpoint_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Model A evaluation output: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no CUDA device is visible")
    device = torch.device(device_name)
    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state, static = group["state"], group["static_features"]
    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = model_a_architecture()
    if payload["model_config"] != architecture.to_dict():
        raise ValueError("checkpoint architecture does not match frozen Model A contract")
    model = build_model_a(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    horizon = 10
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    direct = {
        "validation": _direct_metrics(model, state, static, records_for_pair_split(pair_codes, 2), mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, device=device),
        "inference": _direct_metrics(model, state, static, records_for_pair_split(pair_codes, 3), mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, device=device),
    }
    valid_times = np.flatnonzero((pair_codes == 3) & (np.arange(pair_codes.size) + horizon * rollout_steps < state.shape[1]))
    selected = np.unique(np.linspace(valid_times[0], valid_times[-1], starts_per_regime, dtype=int))
    starts = tuple((experiment, int(time)) for experiment in range(3) for time in selected)
    rollout = _rollout_metrics(model, state, static, starts, mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, steps=rollout_steps, device=device)
    sample = ((0, int(selected[len(selected) // 2])),)
    features, raw, _ = _features(state, static, sample, mean, scale, wet, wind_mean, wind_scale)
    predicted = _model_prediction(model, features, mean, scale, device)[0]
    target = _target(state, sample, horizon)[0]
    snapshot = {"model_a": predicted, "persistence": raw[0], "truth": target}
    for value in snapshot.values():
        value[:, ~wet] = 0.0
    a0_path = Path(a0_metrics_path).resolve() if a0_metrics_path is not None else None
    metrics = {"status": "complete", "dataset": str(dataset_path), "checkpoint": str(checkpoint_path), "device": str(device), "direct": direct, "rollout": rollout, "rollout_starts": [list(record) for record in starts], "a0_metrics_reference": str(a0_path) if a0_path is not None else None}
    output.mkdir(parents=True)
    (output / "model_a_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    arrays: dict[str, np.ndarray] = {"lead_days": np.asarray(rollout["lead_days"]), "wet_mask": wet}
    for index, value in enumerate(rollout["model_a_rmse"]):
        for name, metric in value.items():
            arrays[f"model_a_{index}_{name}"] = np.asarray(metric)
    for index, value in enumerate(rollout["persistence_rmse"]):
        for name, metric in value.items():
            arrays[f"persistence_{index}_{name}"] = np.asarray(metric)
    np.savez_compressed(output / "model_a_metrics_arrays.npz", **arrays)
    _plots(output, direct["inference"], rollout, snapshot, wet, a0_path)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save frozen Model A numerical and plot outputs")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a0-metrics", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    result = evaluate(args.dataset, args.checkpoint, args.output_dir, a0_metrics_path=args.a0_metrics, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
