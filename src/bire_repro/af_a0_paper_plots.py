"""Paper-style, tutorial-appropriate diagnostics for the frozen A0 baseline.

The Bire et al. figures use a different 0.25-degree configuration, variables,
and forecast intervals.  This module therefore preserves the *diagnostic
forms* rather than claiming a figure-for-figure replication: 15-member
inference ensembles, 10--200 day anomaly-correlation curves, spatial RMSE
maps, multi-lead snapshots, and low/control/high wind-regime skill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_a0 import STATE_CHANNEL_COUNT, a0_architecture
from .af_a0_evaluate import GROUPS, _inputs, _normalizers, _target
from .fno import build_paper_fno, require_fno_dependencies

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


SURFACE_THETA = 30
ETA = 45
REGIMES = ("control (S0)", "low wind (S1)", "high wind (S2)")
SPATIAL_LEADS = (10, 30, 100)
SNAPSHOT_LEADS = (60, 200)


def _require_runtime() -> None:
    require_fno_dependencies()
    if torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("paper-style A0 plots require the project PyTorch environment")


def _rmse_samples(prediction: np.ndarray, truth: np.ndarray, fields: slice, wet: np.ndarray) -> np.ndarray:
    """RMSE of each ensemble member over selected channels and wet cells."""

    error = (prediction[:, fields] - truth[:, fields])[:, :, wet]
    return np.sqrt(np.mean(np.square(error), axis=(1, 2)))


def anomaly_correlation_samples(
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    fields: slice,
    wet: np.ndarray,
) -> np.ndarray:
    """Spatial ACC per member after removing the training global channel mean."""

    left = (prediction[:, fields] - climatology[fields, None, None])[:, :, wet].astype(np.float64)
    right = (truth[:, fields] - climatology[fields, None, None])[:, :, wet].astype(np.float64)
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(np.square(left), axis=1) * np.sum(np.square(right), axis=1))
    return np.divide(np.sum(left * right, axis=1), denominator, out=np.zeros_like(denominator), where=denominator > 0)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _direct_regime_skill(
    model: Any,
    state: Any,
    static: Any,
    records_by_regime: Sequence[Sequence[tuple[int, int]]],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    horizon: int,
    device: Any,
) -> dict[str, dict[str, dict[str, float]]]:
    """Held ten-day RMSE by forcing regime, with persistence as comparator."""

    result: dict[str, dict[str, dict[str, float]]] = {}
    with torch.no_grad():
        for name, records in zip(REGIMES, records_by_regime):
            squared = {group: [0.0, 0] for group in GROUPS}
            persistent = {group: [0.0, 0] for group in GROUPS}
            for start in range(0, len(records), 8):
                batch = records[start : start + 8]
                features, raw = _inputs(state, static, batch, mean, scale, wet, wind_mean, wind_scale)
                predicted = model(torch.from_numpy(features).to(device)).detach().cpu().numpy()
                predicted = predicted * scale[None, :, None, None] + mean[None, :, None, None]
                truth = _target(state, batch, horizon)
                for group, fields in GROUPS.items():
                    error = (predicted[:, fields] - truth[:, fields])[:, :, wet]
                    baseline = (raw[:, fields] - truth[:, fields])[:, :, wet]
                    squared[group][0] += float(np.sum(np.square(error)))
                    squared[group][1] += int(error.size)
                    persistent[group][0] += float(np.sum(np.square(baseline)))
                    persistent[group][1] += int(baseline.size)
            result[name] = {
                group: {
                    "a0_rmse": float(np.sqrt(total / count)),
                    "persistence_rmse": float(np.sqrt(persistent[group][0] / persistent[group][1])),
                }
                for group, (total, count) in squared.items()
            }
    return result


def _ensemble_rollout(
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
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    """Run one 15-member-per-regime ensemble and retain paper-style fields."""

    features, initial_raw = _inputs(state, static, starts, mean, scale, wet, wind_mean, wind_scale)
    current = torch.from_numpy(features[:, :STATE_CHANNEL_COUNT]).to(device)
    forcing = torch.from_numpy(features[:, STATE_CHANNEL_COUNT:]).to(device)
    wet_torch = torch.from_numpy(wet).to(device)
    diagnostics: dict[str, Any] = {
        "lead_days": [],
        "a0_acc": {name: [] for name in GROUPS},
        "persistence_acc": {name: [] for name in GROUPS},
        "a0_rmse": {name: [] for name in GROUPS},
        "persistence_rmse": {name: [] for name in GROUPS},
    }
    spatial: dict[str, np.ndarray] = {}
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for step in range(1, steps + 1):
            current = model(torch.cat((current, forcing), dim=1))
            current[:, :, ~wet_torch] = 0.0
            predicted = current.detach().cpu().numpy() * scale[None, :, None, None] + mean[None, :, None, None]
            truth = np.stack(
                [np.asarray(state[experiment, time + step * horizon], dtype=np.float32) for experiment, time in starts]
            )
            lead = step * horizon
            diagnostics["lead_days"].append(lead)
            for name, fields in GROUPS.items():
                diagnostics["a0_acc"][name].append(_summary(anomaly_correlation_samples(predicted, truth, mean, fields, wet)))
                diagnostics["persistence_acc"][name].append(_summary(anomaly_correlation_samples(initial_raw, truth, mean, fields, wet)))
                diagnostics["a0_rmse"][name].append(_summary(_rmse_samples(predicted, truth, fields, wet)))
                diagnostics["persistence_rmse"][name].append(_summary(_rmse_samples(initial_raw, truth, fields, wet)))
            if lead in SPATIAL_LEADS:
                for label, channel in (("theta_surface", SURFACE_THETA), ("eta", ETA)):
                    spatial[f"a0_{label}_day{lead}"] = np.sqrt(
                        np.mean(np.square(predicted[:, channel] - truth[:, channel]), axis=0)
                    )
                    spatial[f"persistence_{label}_day{lead}"] = np.sqrt(
                        np.mean(np.square(initial_raw[:, channel] - truth[:, channel]), axis=0)
                    )
            if lead in SNAPSHOT_LEADS:
                # Middle control member: fixed and documented, not a cherry-picked case.
                index = len(starts) // 6
                snapshots[lead] = {
                    "truth": truth[index].copy(),
                    "a0": predicted[index].copy(),
                    "persistence": initial_raw[index].copy(),
                }
    return diagnostics, spatial, snapshots


def _plot_acc(output: Path, diagnostics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
    leads = np.asarray(diagnostics["lead_days"])
    for axis, name in zip(axes.flat, GROUPS):
        for method, color, style in (("a0", "#2F75B5", "-"), ("persistence", "#222222", "--")):
            values = diagnostics[f"{method}_acc"][name]
            average = np.asarray([entry["mean"] for entry in values])
            low = np.asarray([entry["p10"] for entry in values])
            high = np.asarray([entry["p90"] for entry in values])
            axis.plot(leads, average, color=color, linestyle=style, label="A0" if method == "a0" else "Persistence")
            axis.fill_between(leads, low, high, color=color, alpha=0.14)
        axis.set_title(name)
        axis.grid(alpha=0.35)
        axis.set_ylim(-1.0, 1.02)
    axes[1, 0].set_xlabel("Lead (model days)")
    axes[1, 1].set_xlabel("Lead (model days)")
    axes[0, 0].set_ylabel("Anomaly correlation")
    axes[1, 0].set_ylabel("Anomaly correlation")
    axes[0, 0].legend(loc="lower left")
    figure.suptitle("Frozen A0 ensemble anomaly correlation: 15 starts per regime")
    figure.savefig(output / "a0_acc_vs_lead.png", dpi=180)
    plt.close(figure)


def _plot_spatial_rmse(output: Path, spatial: dict[str, np.ndarray], wet: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (("theta_surface", "Surface theta RMSE ($^\\circ$C)"), ("eta", "SSH RMSE (m)"))
    figure, axes = plt.subplots(2, 6, figsize=(15, 5.5), constrained_layout=True)
    for row, (label, title) in enumerate(labels):
        limit = max(float(np.max(value[wet])) for key, value in spatial.items() if f"_{label}_" in key)
        for column, lead in enumerate(SPATIAL_LEADS):
            for offset, method in enumerate(("a0", "persistence")):
                axis = axes[row, 2 * column + offset]
                image = axis.imshow(np.ma.masked_where(~wet, spatial[f"{method}_{label}_day{lead}"]), origin="lower", cmap="magma", vmin=0, vmax=limit)
                axis.set_title(f"{title.split(' RMSE')[0]}: {method}\nday {lead}", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
                if offset == 1:
                    figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle("Frozen A0 spatial RMSE maps (45 held inference rollouts)")
    figure.savefig(output / "a0_spatial_rmse_maps.png", dpi=180)
    plt.close(figure)


def _plot_snapshots(output: Path, snapshots: dict[int, dict[str, np.ndarray]], wet: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (("Surface theta", SURFACE_THETA, "$^\\circ$C", "coolwarm"), ("SSH", ETA, "m", "coolwarm"))
    figure, axes = plt.subplots(2, 6, figsize=(16, 5.4), constrained_layout=True)
    for row, (field, channel, unit, cmap) in enumerate(fields):
        values = [entry[model][channel] for lead in SNAPSHOT_LEADS for model, entry in (("truth", snapshots[lead]), ("a0", snapshots[lead]), ("persistence", snapshots[lead]))]
        bound = max(float(np.max(np.abs(value[wet]))) for value in values)
        for lead_index, lead in enumerate(SNAPSHOT_LEADS):
            for model_index, model in enumerate(("truth", "a0", "persistence")):
                axis = axes[row, 3 * lead_index + model_index]
                image = axis.imshow(np.ma.masked_where(~wet, snapshots[lead][model][channel]), origin="lower", cmap=cmap, vmin=-bound if field == "SSH" else 0, vmax=bound)
                axis.set_title(f"{field}: {model}\nday {lead}", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
                if model_index == 2:
                    figure.colorbar(image, ax=axis, shrink=0.82, label=unit)
    figure.suptitle("Frozen A0 autoregressive snapshots: fixed middle control member")
    figure.savefig(output / "a0_rollout_snapshots.png", dpi=180)
    plt.close(figure)


def _plot_regime_skill(output: Path, direct: dict[str, dict[str, dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    x = np.arange(len(GROUPS))
    width = 0.22
    for index, regime in enumerate(REGIMES):
        ratio = [direct[regime][group]["a0_rmse"] / direct[regime][group]["persistence_rmse"] for group in GROUPS]
        axis.bar(x + (index - 1) * width, ratio, width=width, label=regime)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(x, list(GROUPS), rotation=12, ha="right")
    axis.set_ylabel("10-day RMSE / persistence RMSE")
    axis.set_title("Frozen A0 held skill by wind regime")
    axis.legend(fontsize=8)
    figure.savefig(output / "a0_regime_skill.png", dpi=180)
    plt.close(figure)


def evaluate(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    starts_per_regime: int = 15,
    rollout_steps: int = 20,
) -> dict[str, Any]:
    """Create a distinct, paper-style A0 output package without modifying A0 itself."""

    _require_runtime()
    if starts_per_regime != 15 or rollout_steps != 20:
        raise ValueError("paper-style A0 evaluation is fixed to 15 starts per regime and 200 days")
    dataset_path, checkpoint_path, output = Path(dataset_path).resolve(), Path(checkpoint_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing paper-style A0 outputs: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA paper-style evaluation requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state, static = group["state"], group["static_features"]
    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = a0_architecture()
    if payload["model_config"] != architecture.to_dict():
        raise ValueError("checkpoint does not match the frozen A0 architecture contract")
    model = build_paper_fno(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    horizon = 10
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    valid_times = np.flatnonzero((pair_codes == 3) & (np.arange(pair_codes.size) + rollout_steps * horizon < state.shape[1]))
    if valid_times.size < starts_per_regime:
        raise ValueError("inference split has too few starts for the fixed 15-member ensemble")
    selected = np.unique(np.linspace(valid_times[0], valid_times[-1], starts_per_regime, dtype=int))
    if selected.size != starts_per_regime:
        raise ValueError("could not choose 15 unique chronological inference starts")
    starts = tuple((experiment, int(time)) for experiment in range(3) for time in selected)
    direct_records = [tuple((experiment, int(time)) for time in np.flatnonzero(pair_codes == 3)) for experiment in range(3)]
    direct = _direct_regime_skill(model, state, static, direct_records, mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, device=device)
    diagnostics, spatial, snapshots = _ensemble_rollout(model, state, static, starts, mean=mean, scale=scale, wet=wet, wind_mean=wind_mean, wind_scale=wind_scale, horizon=horizon, steps=rollout_steps, device=device)
    for values in list(spatial.values()) + [value[channel] for snapshot in snapshots.values() for value in snapshot.values() for channel in (SURFACE_THETA, ETA)]:
        values[~wet] = 0.0
    metrics = {
        "status": "complete",
        "purpose": "Bire-style diagnostic forms on the distinct 1-degree 10-day AF-FNO tutorial configuration",
        "paper_reference": {"figures": [3, 5, 6, 7, 9], "not_claimed": "numerical figure replication"},
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "ensemble": {"starts_per_regime": starts_per_regime, "regimes": list(REGIMES), "lead_days": diagnostics["lead_days"], "starts": [list(record) for record in starts]},
        "acc_anomaly_reference": "training global per-channel mean; spatial mean removed per sample",
        "direct_regime_skill": direct,
        "rollout": diagnostics,
        "spatial_rmse_leads_days": list(SPATIAL_LEADS),
        "snapshot_leads_days": list(SNAPSHOT_LEADS),
    }
    output.mkdir(parents=True)
    (output / "a0_paper_style_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    arrays: dict[str, np.ndarray] = {"wet_mask": wet, "lead_days": np.asarray(diagnostics["lead_days"]), "ensemble_starts": np.asarray(starts, dtype=np.int32)}
    arrays.update({key: value.astype(np.float32) for key, value in spatial.items()})
    for group_name in GROUPS:
        for method in ("a0", "persistence"):
            for metric in ("acc", "rmse"):
                entries = diagnostics[f"{method}_{metric}"][group_name]
                arrays[f"{method}_{metric}_{group_name.split()[0]}"] = np.asarray([[entry["mean"], entry["p10"], entry["p90"]] for entry in entries], dtype=np.float32)
    np.savez_compressed(output / "a0_paper_style_arrays.npz", **arrays)
    _plot_acc(output, diagnostics)
    _plot_spatial_rmse(output, spatial, wet)
    _plot_snapshots(output, snapshots, wet)
    _plot_regime_skill(output, direct)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create paper-style frozen A0 evaluation figures")
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
