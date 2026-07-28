"""Deterministic plot generators for the paper's Figures 2--11.

The plotting code follows what is visible in the published panels when captions conflict.
All quantitative panels also save their numerical values beside the image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import canonical_store_path
from .metrics import (
    anomaly_correlation,
    latitude_weights,
    legacy_unweighted_mse,
    percentile_band,
    ssh_anomaly,
    weighted_rmse,
)


class PlotError(RuntimeError):
    """Raised when a required data or rollout product is unavailable."""


def _zarr():
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise PlotError("plot commands require the project analysis environment") from exc
    return zarr


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.grid": False,
        }
    )


@dataclass(frozen=True)
class RolloutRecord:
    path: Path
    experiment_id: int
    lag_days: int
    stage: str
    resolution: str
    stride: int


class RolloutCatalog:
    """Attribute-driven rollout discovery; filenames are deliberately irrelevant."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise PlotError(f"rollout root does not exist: {self.root}")
        self.records = self._scan()

    def _scan(self) -> list[RolloutRecord]:
        paths = sorted(self.root.rglob("*.zarr"))
        if not paths:
            raise PlotError(
                f"no rollout Zarr groups with experiment_id/lag_days attributes under {self.root}"
            )
        zarr = _zarr()
        records: list[RolloutRecord] = []
        for path in paths:
            try:
                attrs = dict(zarr.open_group(str(path), mode="r").attrs)
                experiment_id = int(attrs["experiment_id"])
                lag_days = int(attrs["lag_days"])
            except (KeyError, TypeError, ValueError):
                continue
            if attrs.get("complete") is False:
                continue
            checkpoint = str(attrs.get("checkpoint", "")).lower()
            stage = str(
                attrs.get("stage", attrs.get("checkpoint_stage", attrs.get("training_stage", "")))
            ).lower()
            if not stage:
                stage = "pretrained" if "pretrain" in checkpoint else "finetuned"
            if stage in {"fine", "finetune", "fine_tuned"}:
                stage = "finetuned"
            if stage in {"pre", "pretrain"}:
                stage = "pretrained"
            stride = int(attrs.get("stride", attrs.get("spatial_stride", 1)))
            resolution = str(attrs.get("resolution", "full" if stride == 1 else "low")).lower()
            records.append(RolloutRecord(path, experiment_id, lag_days, stage, resolution, stride))
        if not records:
            raise PlotError(
                f"no rollout Zarr groups with experiment_id/lag_days attributes under {self.root}"
            )
        return records

    def find(
        self,
        experiment_id: int,
        lag_days: int,
        stage: str = "finetuned",
        resolution: str = "full",
    ) -> "Rollout":
        stage = stage.lower()
        resolution = resolution.lower()
        matches = [
            record
            for record in self.records
            if record.experiment_id == experiment_id
            and record.lag_days == lag_days
            and record.stage == stage
            and record.resolution == resolution
        ]
        if len(matches) != 1:
            found = ", ".join(
                f"exp={r.experiment_id}/lag={r.lag_days}/{r.stage}/{r.resolution}:{r.path}"
                for r in self.records
            )
            raise PlotError(
                f"expected one exp={experiment_id}, lag={lag_days}, {stage}, {resolution} rollout; "
                f"found {len(matches)}. Catalog: {found}"
            )
        return Rollout(matches[0])


class Rollout:
    def __init__(self, record: RolloutRecord):
        self.record = record
        self.group = _zarr().open_group(str(record.path), mode="r")
        required = {"prediction", "truth", "initial_index", "day"}
        missing = required - set(self.group.array_keys())
        if missing:
            raise PlotError(f"{record.path} is missing rollout arrays {sorted(missing)}")
        self.prediction = self.group["prediction"]
        self.truth = self.group["truth"]
        self.initial_index = np.asarray(self.group["initial_index"][:], dtype=int)
        self.day = np.asarray(self.group["day"][:], dtype=int)
        if self.prediction.ndim != 5 or self.truth.ndim != 5:
            raise PlotError(f"rollout arrays must be [member,time,channel,y,x]: {record.path}")
        if self.prediction.shape[:3] != self.truth.shape[:3]:
            raise PlotError(f"prediction/truth shape mismatch: {record.path}")

    def member(self, initial_index: int) -> int:
        matches = np.flatnonzero(self.initial_index == int(initial_index))
        if len(matches) != 1:
            raise PlotError(
                f"initial index {initial_index} occurs {len(matches)} times in {self.record.path}"
            )
        return int(matches[0])

    def day_index(self, day: int, exact: bool = False) -> int:
        matches = np.flatnonzero(self.day == int(day))
        if len(matches):
            return int(matches[0])
        if exact:
            raise PlotError(f"day {day} unavailable in {self.record.path}; available={self.day.tolist()}")
        return int(np.argmin(np.abs(self.day - day)))

    def indices_through(self, horizon: int) -> np.ndarray:
        indices = np.flatnonzero(self.day <= int(horizon))
        if not len(indices):
            raise PlotError(f"no rollout samples through day {horizon}: {self.record.path}")
        return indices

    def field(
        self, kind: str, member: int, diagnostic: str, indices: Iterable[int] | slice
    ) -> np.ndarray:
        source = self.prediction if kind == "prediction" else self.truth
        if isinstance(indices, slice):
            selection = indices
            post = None
        else:
            requested = np.asarray(list(indices), dtype=int)
            if requested.size == 0:
                raise PlotError("empty rollout selection")
            selection = slice(0, int(requested.max()) + 1)
            post = requested

        def channel(index: int) -> np.ndarray:
            values = np.asarray(source[member, selection, index, :, :], dtype="f4")
            return values if post is None else values[post]

        if diagnostic == "speed":
            values = np.hypot(channel(0), channel(2))
        elif diagnostic == "U":
            values = channel(0)
        elif diagnostic == "V":
            values = channel(2)
        elif diagnostic == "SST":
            values = channel(4)
        elif diagnostic == "pressure":
            values = channel(6)
            values = values - np.nanmean(values, axis=(-2, -1), keepdims=True)
        elif diagnostic == "psi":
            values = channel(9) / 1.0e6
        else:
            raise PlotError(f"unknown diagnostic {diagnostic}")
        return values


class PlotContext:
    def __init__(
        self,
        config: Mapping[str, Any],
        store_path: str | Path | None = None,
        rollout_root: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        self.config = config
        self.store_path = Path(store_path or canonical_store_path(config)).resolve()
        self.root = _zarr().open_group(str(self.store_path), mode="r")
        self.state = self.root["state"]
        self.lat = np.asarray(self.root["lat_deg_n"][:])
        self.lon = np.asarray(self.root["lon_deg_e"][:])
        self.rollout_root = Path(rollout_root or config["paths"]["rollouts"])
        self._catalog: RolloutCatalog | None = None
        self.output_dir = Path(output_dir or config["paths"]["figures"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._climatology: dict[tuple[str, int], np.ndarray] = {}

    @property
    def catalog(self) -> RolloutCatalog:
        if self._catalog is None:
            self._catalog = RolloutCatalog(self.rollout_root)
        return self._catalog

    def wet_mask(self, stride: int = 1) -> np.ndarray:
        surface_temperature = np.asarray(self.state[2, 0, 4, ::stride, ::stride])
        return np.isfinite(surface_temperature) & (np.abs(surface_temperature) > 1.0e-12)

    def climatology(self, diagnostic: str, stride: int = 1) -> np.ndarray:
        key = (diagnostic, stride)
        if key in self._climatology:
            return self._climatology[key]
        nt = self.state.shape[1]
        total = None
        count = 0
        for start in range(0, nt, 32):
            stop = min(start + 32, nt)
            if diagnostic == "speed":
                u = np.asarray(self.state[2, start:stop, 0, ::stride, ::stride])
                v = np.asarray(self.state[2, start:stop, 2, ::stride, ::stride])
                values = np.hypot(u, v)
            else:
                channel = {"U": 0, "V": 2, "SST": 4, "pressure": 6, "psi": 9}[diagnostic]
                values = np.asarray(self.state[2, start:stop, channel, ::stride, ::stride])
                if diagnostic == "pressure":
                    values = values - np.nanmean(values, axis=(-2, -1), keepdims=True)
                if diagnostic == "psi":
                    values = values / 1.0e6
            subtotal = np.nansum(values, axis=0, dtype="f8")
            total = subtotal if total is None else total + subtotal
            count += values.shape[0]
        result = total / count
        self._climatology[key] = result
        return result

    def save(self, figure: plt.Figure, number: int) -> Path:
        stem = self.output_dir / f"figure{number:02d}"
        figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        return stem.with_suffix(".png")


def _map(
    axis,
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    title: str,
    cmap: str = "RdBu_r",
    vlim: float | None = None,
    levels: int = 41,
):
    if vlim is None:
        vlim = float(np.nanpercentile(np.abs(field), 99))
        vlim = max(vlim, np.finfo(float).eps)
    image = axis.contourf(lon, lat, field, np.linspace(-vlim, vlim, levels), cmap=cmap, extend="both")
    axis.set_title(title)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    return image


def figure2(context: PlotContext) -> Path:
    """Control streamfunction, SSH anomaly, and SST at archived index 387."""
    _style()
    day = int(context.config["evaluation"]["figure2_index"])
    state = context.state[2]
    psi = np.asarray(state[day, 9]) / 1.0e6
    ssh = ssh_anomaly(np.asarray(state[day, 6]), context.config["mitgcm"]["gravity_m_s2"])
    sst = np.asarray(state[day, 4])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    images = [
        _map(axes[0], psi, context.lon, context.lat, "(a) Barotropic streamfunction", vlim=50),
        _map(axes[1], ssh, context.lon, context.lat, "(b) SSH anomaly", vlim=1),
        axes[2].contourf(context.lon, context.lat, sst, 31, cmap="plasma"),
    ]
    axes[2].set(title="(c) SST", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    units = ["Sv", "m", "°C"]
    for axis, image, unit in zip(axes, images, units):
        fig.colorbar(image, ax=axis, orientation="horizontal", pad=0.16, label=unit)
    fig.suptitle(f"Control experiment, production day {day}")
    return context.save(fig, 2)


def figure3(context: PlotContext) -> Path:
    """Ten-day iterative prediction at 0.25° and 2° resolution."""
    _style()
    initial = int(context.config["evaluation"]["figure2_index"])
    full = context.catalog.find(3, 10, "finetuned", "full")
    low = context.catalog.find(3, 10, "finetuned", "low")
    member_full, member_low = full.member(initial), low.member(initial)
    requested_days = [0, 10, 20, 30, 40]
    full_indices = [full.day_index(day, exact=True) for day in requested_days]
    low_indices = [low.day_index(day, exact=True) for day in requested_days]
    truth = full.field("truth", member_full, "psi", full_indices)
    prediction = full.field("prediction", member_full, "psi", full_indices)
    low_prediction = low.field("prediction", member_low, "psi", low_indices)
    stride = low.record.stride or int(context.config["fno"]["spatial_stride_low_resolution"])
    coarse_truth = truth[:, ::stride, ::stride]
    low_lon, low_lat = context.lon[::stride], context.lat[::stride]
    fig, axes = plt.subplots(5, 5, figsize=(11.5, 11), constrained_layout=True)
    row_labels = ["Ground truth", "Prediction", "GT − prediction", "Prediction (2°)", "Coarsened GT − 2°"]
    for column, day in enumerate(requested_days):
        _map(axes[0, column], truth[column], context.lon, context.lat, f"day {day}", vlim=50)
        if column == 0:
            for row in range(1, 5):
                axes[row, column].axis("off")
            continue
        _map(axes[1, column], prediction[column], context.lon, context.lat, "", vlim=50)
        _map(axes[2, column], truth[column] - prediction[column], context.lon, context.lat, "", vlim=10)
        _map(axes[3, column], low_prediction[column], low_lon, low_lat, "", vlim=50)
        _map(
            axes[4, column],
            coarse_truth[column] - low_prediction[column],
            low_lon,
            low_lat,
            "",
            vlim=10,
        )
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(label, (-0.35, 0.5), xycoords="axes fraction", rotation=90, va="center")
    fig.suptitle("Figure 3 reconstruction: barotropic streamfunction (Sv), Δt = 10 days")
    return context.save(fig, 3)


def _curve_values(
    context: PlotContext, reader: Rollout, diagnostic: str, horizon: int
) -> dict[str, np.ndarray]:
    indices = reader.indices_through(horizon)
    days = reader.day[indices]
    stride = reader.record.stride
    lat = context.lat[::stride]
    nx = reader.prediction.shape[-1]
    weights = latitude_weights(lat, nx=nx)
    mask = context.wet_mask(stride)
    climatology = context.climatology(diagnostic, stride)
    prediction_values = []
    climatology_values = []
    persistence_values = []
    legacy_values = []
    for member in range(reader.prediction.shape[0]):
        prediction = reader.field("prediction", member, diagnostic, indices)
        truth = reader.field("truth", member, diagnostic, indices)
        prediction_values.append(weighted_rmse(prediction, truth, weights, mask))
        climatology_values.append(weighted_rmse(climatology[None], truth, weights, mask))
        persistence_values.append(weighted_rmse(truth[0:1], truth, weights, mask))
        legacy_values.append(legacy_unweighted_mse(prediction, truth))
    return {
        "day": days,
        "prediction": np.asarray(prediction_values),
        "climatology": np.asarray(climatology_values),
        "persistence": np.asarray(persistence_values),
        "legacy_unweighted_mse": np.asarray(legacy_values),
    }


def _plot_rmse(context: PlotContext, horizon: int, number: int) -> Path:
    _style()
    lags = [5, 10, 30]
    diagnostics = [("speed", "Surface speed (m s⁻¹)"), ("SST", "SST (°C)"), ("pressure", "P/ρ anomaly (m² s⁻²)")]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2), sharex="col", constrained_layout=True)
    payload: dict[str, np.ndarray] = {}
    for column, lag in enumerate(lags):
        reader = context.catalog.find(3, lag, "finetuned", "full")
        for row, (diagnostic, label) in enumerate(diagnostics):
            values = _curve_values(context, reader, diagnostic, horizon)
            axis = axes[row, column]
            colors = {"prediction": "red", "climatology": "black", "persistence": "blue"}
            for key in ("prediction", "climatology", "persistence"):
                mean, low, high = percentile_band(values[key])
                axis.plot(values["day"], mean, color=colors[key], label=key.capitalize())
                axis.fill_between(values["day"], low, high, color=colors[key], alpha=0.18, linewidth=0)
                payload[f"lag{lag}_{diagnostic}_{key}"] = values[key]
            payload[f"lag{lag}_{diagnostic}_legacy_unweighted_mse"] = values[
                "legacy_unweighted_mse"
            ]
            payload[f"lag{lag}_day"] = values["day"]
            axis.grid(True, alpha=0.35)
            axis.set_title(f"Δt = {lag} days")
            if column == 0:
                axis.set_ylabel(label)
            if row == 2:
                axis.set_xlabel("Forecast time (days)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        f"Figure {number}: area-weighted true RMSE, finetuned 15-member ensembles"
    )
    np.savez_compressed(context.output_dir / f"figure{number:02d}_metrics.npz", **payload)
    return context.save(fig, number)


def figure4(context: PlotContext) -> Path:
    return _plot_rmse(context, int(context.config["evaluation"]["short_horizon_days"]), 4)


def figure8(context: PlotContext) -> Path:
    return _plot_rmse(context, int(context.config["evaluation"]["long_horizon_days"]), 8)


def figure5(context: PlotContext) -> Path:
    """Spatial RMSE over days 0--200 for one archived ensemble member."""
    _style()
    initial = int(context.config["evaluation"]["figure5_index"])
    horizon = int(context.config["evaluation"]["short_horizon_days"])
    lags = [5, 10, 30]
    diagnostics = [("psi", "Barotropic streamfunction (Sv)"), ("SST", "SST (°C)")]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.1), constrained_layout=True)
    payload: dict[str, np.ndarray] = {}
    for column, lag in enumerate(lags):
        reader = context.catalog.find(3, lag, "finetuned", "full")
        member = reader.member(initial)
        indices = reader.indices_through(horizon)
        for row, (diagnostic, label) in enumerate(diagnostics):
            prediction = reader.field("prediction", member, diagnostic, indices)
            truth = reader.field("truth", member, diagnostic, indices)
            spatial_rmse = np.sqrt(np.nanmean(np.square(prediction - truth), axis=0))
            payload[f"lag{lag}_{diagnostic}_spatial_rmse"] = spatial_rmse
            vmax = float(np.nanpercentile(spatial_rmse, 99.5))
            image = axes[row, column].contourf(
                context.lon,
                context.lat,
                spatial_rmse,
                np.linspace(0, max(vmax, 1e-12), 31),
                cmap="Reds",
                extend="max",
            )
            axes[row, column].set_title(f"Δt = {lag} days")
            axes[row, column].set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
            fig.colorbar(image, ax=axes[row, column], label=label)
    np.savez_compressed(context.output_dir / "figure05_metrics.npz", **payload)
    fig.suptitle(f"Figure 5: spatial RMSE over forecast days 0–{horizon}, initial index {initial}")
    return context.save(fig, 5)


def _acc_values(
    context: PlotContext, reader: Rollout, diagnostic: str, horizon: int
) -> dict[str, np.ndarray]:
    indices = reader.indices_through(horizon)
    stride = reader.record.stride
    weights = latitude_weights(context.lat[::stride], nx=reader.prediction.shape[-1])
    mask = context.wet_mask(stride)
    climatology = context.climatology(diagnostic, stride)
    acc = []
    for member in range(reader.prediction.shape[0]):
        prediction = reader.field("prediction", member, diagnostic, indices)
        truth = reader.field("truth", member, diagnostic, indices)
        acc.append(anomaly_correlation(prediction, truth, climatology, weights, mask))
    return {"day": reader.day[indices], "acc": np.asarray(acc)}


def figure6(context: PlotContext) -> Path:
    """Pretrained versus two-step-finetuned anomaly-correlation skill."""
    _style()
    horizon = int(context.config["evaluation"]["short_horizon_days"])
    lags = [5, 10, 30]
    diagnostics = [
        ("U", "Surface U"),
        ("V", "Surface V"),
        ("SST", "SST"),
        ("pressure", "P/ρ anomaly"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 10.5), sharex="col", sharey=True, constrained_layout=True)
    payload: dict[str, np.ndarray] = {}
    for column, lag in enumerate(lags):
        readers = {
            "Pretrained": context.catalog.find(3, lag, "pretrained", "full"),
            "Finetuned": context.catalog.find(3, lag, "finetuned", "full"),
        }
        for row, (diagnostic, label) in enumerate(diagnostics):
            axis = axes[row, column]
            for stage_label, color in (("Pretrained", "black"), ("Finetuned", "red")):
                values = _acc_values(context, readers[stage_label], diagnostic, horizon)
                mean, low, high = percentile_band(values["acc"])
                axis.plot(values["day"], mean, color=color, label=stage_label)
                axis.fill_between(values["day"], low, high, color=color, alpha=0.18, linewidth=0)
                payload[f"lag{lag}_{diagnostic}_{stage_label.lower()}_acc"] = values["acc"]
                payload[f"lag{lag}_day"] = values["day"]
            axis.set_ylim(0.4, 1.01)
            axis.grid(True, alpha=0.35)
            axis.set_title(f"Δt = {lag} days")
            if column == 0:
                axis.set_ylabel(f"{label} ACC")
            if row == 3:
                axis.set_xlabel("Forecast time (days)")
    axes[0, 0].legend(frameon=False)
    np.savez_compressed(context.output_dir / "figure06_metrics.npz", **payload)
    fig.suptitle("Figure 6: cosine-latitude-weighted ACC (caption row order corrected)")
    return context.save(fig, 6)


def figure7(context: PlotContext) -> Path:
    """Barotropic streamfunction maps at short and long forecast horizons."""
    _style()
    initial = int(context.config["evaluation"]["figure7_index"])
    lags = [5, 10, 30]
    targets = [60, 2000]
    readers = {lag: context.catalog.find(3, lag, "finetuned", "full") for lag in lags}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    titles = ["Ground truth", "Δt = 5 days", "Δt = 10 days", "Δt = 30 days"]
    actual_days: dict[str, int] = {}
    for row, target in enumerate(targets):
        truth_reader = readers[5]
        truth_member = truth_reader.member(initial)
        truth_index = truth_reader.day_index(target)
        truth = truth_reader.field("truth", truth_member, "psi", [truth_index])[0]
        _map(axes[row, 0], truth, context.lon, context.lat, f"{titles[0]} (day {truth_reader.day[truth_index]})", vlim=50)
        actual_days[f"truth_{target}"] = int(truth_reader.day[truth_index])
        for column, lag in enumerate(lags, start=1):
            reader = readers[lag]
            member = reader.member(initial)
            index = reader.day_index(target)
            field = reader.field("prediction", member, "psi", [index])[0]
            actual = int(reader.day[index])
            _map(axes[row, column], field, context.lon, context.lat, f"{titles[column]} (day {actual})", vlim=50)
            actual_days[f"lag{lag}_{target}"] = actual
    (context.output_dir / "figure07_actual_days.json").write_text(
        json.dumps(actual_days, indent=2, sort_keys=True) + "\n"
    )
    fig.suptitle(f"Figure 7: barotropic streamfunction (Sv), initial index {initial}")
    return context.save(fig, 7)


def _relative_mitgcm_mean(
    context: PlotContext, experiment_id: int, initial: int, start: int, stop: int
) -> np.ndarray:
    absolute_start = initial + start
    absolute_stop = initial + stop
    if absolute_stop > context.state.shape[1]:
        raise PlotError(
            f"MITgcm window [{absolute_start},{absolute_stop}) exceeds {context.state.shape[1]} records"
        )
    return np.asarray(
        context.state[experiment_id - 1, absolute_start:absolute_stop, 9, :, :], dtype="f4"
    ).mean(axis=0) / 1.0e6


def _rollout_mean(reader: Rollout, member: int, start: int, stop: int) -> np.ndarray:
    indices = np.flatnonzero((reader.day >= start) & (reader.day < stop))
    if not len(indices):
        raise PlotError(f"rollout has no samples in [{start},{stop}): {reader.record.path}")
    return np.nanmean(reader.field("prediction", member, "psi", indices), axis=0)


def _gyre_extrema(
    field: np.ndarray, lon: np.ndarray, wet_mask: np.ndarray, east_of: float
) -> tuple[float, float]:
    region = wet_mask & (lon[None, :] >= east_of) & np.isfinite(field)
    if not np.any(region):
        raise PlotError("gyre-extrema mask is empty")
    return float(np.max(field[region])), float(-np.min(field[region]))


def figure9(context: PlotContext) -> Path:
    """Gyre extrema versus wind forcing for all five experiments."""
    _style()
    initial = int(context.config["evaluation"]["figure7_index"])
    early = (
        int(context.config["evaluation"]["figure9_early_start"]),
        int(context.config["evaluation"]["figure9_early_stop"]),
    )
    late = (
        int(context.config["evaluation"]["figure9_late_start"]),
        int(context.config["evaluation"]["figure9_late_stop"]),
    )
    east = float(context.config["evaluation"]["east_of_lon_deg"])
    wet = context.wet_mask()
    tau = np.asarray([item["tau0_n_m2"] for item in context.config["experiments"]])
    rho0 = float(context.config["evaluation"]["theory_rho0_kg_m3"])
    beta0 = float(context.config["evaluation"]["theory_beta0_m_inv_s"])
    theory = 2.0 * np.pi * tau / (rho0 * beta0) / 1.0e6
    mit = {"early_pos": [], "early_neg": [], "late_pos": [], "late_neg": []}
    for exp_id in range(1, 6):
        for label, window in (("early", early), ("late", late)):
            field = _relative_mitgcm_mean(context, exp_id, initial, *window)
            positive, negative = _gyre_extrema(field, context.lon, wet, east)
            mit[f"{label}_pos"].append(positive)
            mit[f"{label}_neg"].append(negative)
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.5), sharey=True, constrained_layout=True)
    payload: dict[str, np.ndarray] = {"tau0": tau, "theory": theory}
    payload.update({f"mitgcm_{key}": np.asarray(value) for key, value in mit.items()})
    for axis, lag in zip(axes, [5, 10, 30]):
        fno = {"early_pos": [], "early_neg": [], "late_pos": [], "late_neg": []}
        for exp_id in range(1, 6):
            reader = context.catalog.find(exp_id, lag, "finetuned", "full")
            member = reader.member(initial)
            for label, window in (("early", early), ("late", late)):
                positive, negative = _gyre_extrema(
                    _rollout_mean(reader, member, *window), context.lon, wet, east
                )
                fno[f"{label}_pos"].append(positive)
                fno[f"{label}_neg"].append(negative)
        axis.plot(tau, theory, color="black", label="Ψmax theory")
        axis.plot(tau, mit["late_pos"], "o:r", label="Ψmax MITgcm")
        axis.plot(tau, mit["late_neg"], "o:b", label="−Ψmin MITgcm")
        axis.plot(tau, fno["early_pos"], "d-r", label="Ψmax FNO initial 500 d")
        axis.plot(tau, fno["late_pos"], "s-r", label="Ψmax FNO final 500 d")
        axis.plot(tau, fno["early_neg"], "d-b", label="−Ψmin FNO initial 500 d")
        axis.plot(tau, fno["late_neg"], "s-b", label="−Ψmin FNO final 500 d")
        axis.set(title=f"Δt = {lag} days", xlabel="τ₀ (N m⁻²)")
        axis.grid(True, alpha=0.45)
        payload.update({f"lag{lag}_{key}": np.asarray(value) for key, value in fno.items()})
    axes[0].set_ylabel("Gyre transport (Sv)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    np.savez_compressed(context.output_dir / "figure09_metrics.npz", **payload)
    fig.suptitle("Figure 9: gyre extrema east of 10°E")
    return context.save(fig, 9)


def figure10(context: PlotContext) -> Path:
    """Final-500-day barotropic streamfunction means."""
    _style()
    initial = int(context.config["evaluation"]["figure7_index"])
    start = int(context.config["evaluation"]["figure9_late_start"])
    stop = int(context.config["evaluation"]["figure9_late_stop"])
    lags = [5, 10, 30]
    fig, axes = plt.subplots(4, 5, figsize=(13, 9), constrained_layout=True, sharex=True, sharey=True)
    labels = ["MITgcm", "FNO Δt=5 d", "FNO Δt=10 d", "FNO Δt=30 d"]
    image = None
    for column, item in enumerate(context.config["experiments"]):
        exp_id = int(item["id"])
        fields = [_relative_mitgcm_mean(context, exp_id, initial, start, stop)]
        for lag in lags:
            reader = context.catalog.find(exp_id, lag, "finetuned", "full")
            fields.append(_rollout_mean(reader, reader.member(initial), start, stop))
        for row, field in enumerate(fields):
            image = axes[row, column].contourf(
                context.lon,
                context.lat,
                field,
                np.linspace(-40, 40, 41),
                cmap="RdBu_r",
                extend="both",
            )
            axes[row, column].contour(context.lon, context.lat, field, colors="black", levels=9, linewidths=0.35)
            if row == 0:
                axes[row, column].set_title(f"τ₀={item['tau0_n_m2']:.4g}")
            if column == 0:
                axes[row, column].set_ylabel(labels[row] + "\nLatitude (°N)")
            if row == 3:
                axes[row, column].set_xlabel("Longitude (°E)")
    fig.colorbar(image, ax=axes, label="Barotropic streamfunction (Sv)", shrink=0.7)
    fig.suptitle(f"Figure 10: mean forecast days {start}–{stop}")
    return context.save(fig, 10)


def figure11(context: PlotContext) -> Path:
    """Longitude-time streamfunction sections averaged over 25--35°N."""
    _style()
    initial = int(context.config["evaluation"]["figure7_index"])
    start = int(context.config["evaluation"]["figure11_day_start"])
    stop = int(context.config["evaluation"]["figure11_day_stop"])
    lat_min = float(context.config["evaluation"]["figure11_lat_min_deg"])
    lat_max = float(context.config["evaluation"]["figure11_lat_max_deg"])
    lat_mask = (context.lat >= lat_min) & (context.lat <= lat_max)
    truth_days = np.arange(start, stop + 1)
    truth = np.asarray(context.state[2, initial + start : initial + stop + 1, 9, :, :]) / 1.0e6
    truth_section = np.nanmean(truth[:, lat_mask, :], axis=1)
    panels: list[tuple[str, np.ndarray, np.ndarray]] = [("Ground truth", truth_days, truth_section)]
    for lag in [5, 10, 30]:
        reader = context.catalog.find(3, lag, "finetuned", "full")
        indices = np.flatnonzero((reader.day >= start) & (reader.day <= stop))
        prediction = reader.field("prediction", reader.member(initial), "psi", indices)
        panels.append((f"Δt = {lag} days", reader.day[indices], np.nanmean(prediction[:, lat_mask, :], axis=1)))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for axis, (title, days, values) in zip(axes.flat, panels):
        image = axis.pcolormesh(context.lon, days, values, cmap="RdBu_r", vmin=-50, vmax=50, shading="auto")
        axis.set(title=title, xlabel="Longitude (°E)", ylabel="Forecast time (days)")
    fig.colorbar(image, ax=axes, label="Barotropic streamfunction (Sv)", shrink=0.75)
    fig.suptitle(f"Figure 11: {lat_min:g}–{lat_max:g}°N mean")
    return context.save(fig, 11)


FIGURES = {
    2: figure2,
    3: figure3,
    4: figure4,
    5: figure5,
    6: figure6,
    7: figure7,
    8: figure8,
    9: figure9,
    10: figure10,
    11: figure11,
}


def generate(
    config: Mapping[str, Any],
    numbers: Iterable[int] = tuple(FIGURES),
    store_path: str | Path | None = None,
    rollout_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    context = PlotContext(config, store_path, rollout_root, output_dir)
    outputs = []
    for number in numbers:
        if int(number) not in FIGURES:
            raise PlotError(f"unsupported paper figure {number}; choose 2 through 11")
        outputs.append(FIGURES[int(number)](context))
    return outputs
