"""Save numerical and visual evaluation products for the frozen A0 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_a0 import STATE_CHANNEL_COUNT, WIND_FEATURE_INDEX, a0_architecture, records_for_pair_split
from .af_data import STATE_CHANNELS
from .fno import build_paper_fno, require_fno_dependencies

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


GROUPS = {
    "U (m s$^{-1}$)": slice(0, 15),
    "V (m s$^{-1}$)": slice(15, 30),
    "Theta ($^\\circ$C)": slice(30, 45),
    "Eta (m)": slice(45, 46),
}


def _require_runtime() -> None:
    require_fno_dependencies()
    if torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("A0 evaluation requires the project PyTorch environment")


def _normalizers(group: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if tuple(group.attrs.get("state_channels", ())) != STATE_CHANNELS:
        raise ValueError("unexpected A0 state-channel contract")
    mean = np.asarray(group["state_mean"][:], dtype=np.float32)
    scale = np.asarray(group["state_scale"][:], dtype=np.float32)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    wind = np.asarray(group["static_features"][:, WIND_FEATURE_INDEX], dtype=np.float32)
    wind_values = wind[:, wet]
    return mean, scale, wet, wind, float(wind_values.mean()), float(wind_values.std())


def _normalise(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray, wet: np.ndarray) -> np.ndarray:
    value = (raw - mean[None, :, None, None]) / scale[None, :, None, None]
    value[:, :, ~wet] = 0.0
    return np.ascontiguousarray(value, dtype=np.float32)


def _inputs(
    state: Any,
    static: Any,
    records: Sequence[tuple[int, int]],
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.stack([np.asarray(state[experiment, time], dtype=np.float32) for experiment, time in records])
    wind = np.stack([np.asarray(static[experiment, WIND_FEATURE_INDEX], dtype=np.float32) for experiment, _ in records])
    wind = (wind - wind_mean) / wind_scale
    wind[:, ~wet] = 0.0
    return np.concatenate((_normalise(raw, mean, scale, wet), wind[:, None]), axis=1), raw


def _target(state: Any, records: Sequence[tuple[int, int]], horizon: int) -> np.ndarray:
    return np.stack([np.asarray(state[experiment, time + horizon], dtype=np.float32) for experiment, time in records])


def _group_rmse(prediction: np.ndarray, truth: np.ndarray, wet: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.sqrt(np.mean(np.square((prediction[:, fields] - truth[:, fields])[:, :, wet]))))
        for name, fields in GROUPS.items()
    }


def _group_correlation(prediction: np.ndarray, truth: np.ndarray, wet: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, fields in GROUPS.items():
        left = prediction[:, fields][:, :, wet].ravel().astype(np.float64)
        right = truth[:, fields][:, :, wet].ravel().astype(np.float64)
        left -= left.mean()
        right -= right.mean()
        result[name] = float(np.dot(left, right) / np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return result


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
    batch_size: int,
) -> dict[str, Any]:
    prediction: list[np.ndarray] = []
    persistence: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            features, raw_input = _inputs(state, static, batch, mean, scale, wet, wind_mean, wind_scale)
            output = model(torch.from_numpy(features).to(device)).detach().cpu().numpy()
            prediction.append(output * scale[None, :, None, None] + mean[None, :, None, None])
            persistence.append(raw_input)
            truth.append(_target(state, batch, horizon))
    predicted = np.concatenate(prediction)
    held = np.concatenate(truth)
    persist = np.concatenate(persistence)
    for value in (predicted, persist, held):
        value[:, :, ~wet] = 0.0
    return {
        "pair_count": len(records),
        "a0_rmse": _group_rmse(predicted, held, wet),
        "persistence_rmse": _group_rmse(persist, held, wet),
        "a0_correlation": _group_correlation(predicted, held, wet),
        "persistence_correlation": _group_correlation(persist, held, wet),
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
    features, raw_initial = _inputs(state, static, starts, mean, scale, wet, wind_mean, wind_scale)
    current = torch.from_numpy(features[:, :STATE_CHANNEL_COUNT]).to(device)
    forcing = torch.from_numpy(features[:, STATE_CHANNEL_COUNT:]).to(device)
    output: dict[str, Any] = {"lead_days": [], "a0_rmse": [], "persistence_rmse": []}
    with torch.no_grad():
        for step in range(1, steps + 1):
            current = model(torch.cat((current, forcing), dim=1))
            current[:, :, ~torch.from_numpy(wet).to(device)] = 0.0
            predicted = current.detach().cpu().numpy() * scale[None, :, None, None] + mean[None, :, None, None]
            held = np.stack(
                [np.asarray(state[experiment, time + step * horizon], dtype=np.float32) for experiment, time in starts]
            )
            output["lead_days"].append(step * horizon)
            output["a0_rmse"].append(_group_rmse(predicted, held, wet))
            output["persistence_rmse"].append(_group_rmse(raw_initial, held, wet))
    return output


def _plots(output: Path, direct: dict[str, Any], rollout: dict[str, Any], snapshot: dict[str, np.ndarray], wet: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(GROUPS)
    ratio = [direct["a0_rmse"][name] / direct["persistence_rmse"][name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    axis.bar(np.arange(len(names)), ratio, color="#2F75B5")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(names)), names, rotation=12, ha="right")
    axis.set_ylabel("10-day RMSE / persistence RMSE")
    axis.set_title("Frozen A0 inference skill (all held regimes)")
    figure.savefig(output / "a0_one_step_skill.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for name in names:
        values = [entry[name] / base[name] for entry, base in zip(rollout["a0_rmse"], rollout["persistence_rmse"])]
        axis.plot(rollout["lead_days"], values, marker="o", label=name)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xlabel("Lead (model days)")
    axis.set_ylabel("RMSE / persistence RMSE")
    axis.set_title("Frozen A0 autoregressive inference rollouts")
    axis.legend(fontsize=8)
    figure.savefig(output / "a0_rollout_skill.png", dpi=180)
    plt.close(figure)

    fields = (("Theta surface", 30, "$^\\circ$C"), ("SSH", 45, "m"))
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    for row, (title, channel, unit) in enumerate(fields):
        values = [snapshot["truth"][channel], snapshot["a0"][channel], snapshot["persistence"][channel]]
        bound = max(float(np.max(np.abs(value[wet]))) for value in values)
        for column, (name, value) in enumerate(zip(("MITgcm target", "A0", "Persistence"), values)):
            image = axes[row, column].imshow(np.ma.masked_where(~wet, value), origin="lower", cmap="coolwarm", vmin=-bound if row else 0, vmax=bound if row else None)
            axes[row, column].set_title(f"{title}: {name}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=0.78, label=unit)
    figure.savefig(output / "a0_inference_snapshot.png", dpi=180)
    plt.close(figure)


def evaluate(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    rollout_steps: int = 10,
    starts_per_regime: int = 16,
) -> dict[str, Any]:
    _require_runtime()
    if rollout_steps <= 0 or starts_per_regime <= 0:
        raise ValueError("rollout_steps and starts_per_regime must be positive")
    dataset_path = Path(dataset_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing A0 evaluation output: {output}")
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
    architecture = a0_architecture()
    if payload["model_config"] != architecture.to_dict():
        raise ValueError("checkpoint architecture does not match the frozen A0 contract")
    model = build_paper_fno(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    horizon = 10
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    direct = {
        "validation": _direct_metrics(model, state, static, records_for_pair_split(pair_codes, 2), mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, device=device, batch_size=8),
        "inference": _direct_metrics(model, state, static, records_for_pair_split(pair_codes, 3), mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, device=device, batch_size=8),
    }
    valid_times = np.flatnonzero((pair_codes == 3) & (np.arange(pair_codes.size) + horizon * rollout_steps < state.shape[1]))
    selected_times = np.unique(np.linspace(valid_times[0], valid_times[-1], starts_per_regime, dtype=int))
    starts = tuple((experiment, int(time)) for experiment in range(3) for time in selected_times)
    rollout = _rollout_metrics(model, state, static, starts, mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, steps=rollout_steps, device=device)
    sample = ((0, int(selected_times[len(selected_times) // 2])),)
    features, raw = _inputs(state, static, sample, mean, scale, wet, wind_mean, wind_scale)
    with torch.no_grad():
        predicted = model(torch.from_numpy(features).to(device)).detach().cpu().numpy()[0]
    target = _target(state, sample, horizon)[0]
    snapshot = {"a0": predicted * scale[:, None, None] + mean[:, None, None], "persistence": raw[0], "truth": target}
    for value in snapshot.values():
        value[:, ~wet] = 0.0
    metrics = {"status": "complete", "dataset": str(dataset_path), "checkpoint": str(checkpoint_path), "device": str(device), "direct": direct, "rollout": rollout, "rollout_starts": [list(record) for record in starts]}
    output.mkdir(parents=True)
    (output / "a0_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    arrays: dict[str, np.ndarray] = {"lead_days": np.asarray(rollout["lead_days"]), "wet_mask": wet}
    for index, value in enumerate(rollout["a0_rmse"]):
        for name, metric in value.items():
            arrays[f"a0_{index}_{name}"] = np.asarray(metric)
    for index, value in enumerate(rollout["persistence_rmse"]):
        for name, metric in value.items():
            arrays[f"persistence_{index}_{name}"] = np.asarray(metric)
    np.savez_compressed(output / "a0_metrics_arrays.npz", **arrays)
    _plots(output, direct["inference"], rollout, snapshot, wet)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save frozen A0 numerical and plot outputs")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    result = evaluate(args.dataset, args.checkpoint, args.output_dir, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
