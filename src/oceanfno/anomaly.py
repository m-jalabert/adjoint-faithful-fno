"""Streamfunction-anomaly companions to figures 3 and 7.

One operation, applied identically to truth and to the prediction:

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

``psi_bar_S0`` is MITgcm's own two-dimensional time-mean barotropic
streamfunction over training days 0--5999. The model's own mean is never
subtracted, so a bias in the stationary circulation cannot hide inside the
anomaly. With the standing gyre removed, an anomaly RMS ratio below one means
the transients have been damped and above one that they have been manufactured.

Three plates are published: the early-lead anomaly grid (figure 3a), the day-60 /
day-2,000 anomaly pair (figure 7a), and the removed reference mean itself. The
package reads the *sealed* figure arrays and model-visible MITgcm training state
only: it loads no weights, rolls nothing out, and modifies no published figure.

Entry points::

    python -m oceanfno.anomaly finalize  --contract config/...json
    python -m oceanfno.anomaly preflight --contract config/...json
    python -m oceanfno.anomaly run       --contract config/...json
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

from . import figures, plots
from .runtime import _file_sha256, _json_sha256, json_safe
from .dataset import DATASET_VERSION, TRAIN_RANGE, western_boundary_mask
from .diagnostics import derived_fields
from .model import MANIFEST_NAME, README_NAME
from .plots import FIGURE_3_LEADS, FIGURE_7_LEADS, _finite_bound, _masked, _style

VERSION = "model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1"

CONTRACT_STATUS = (
    "frozen_after_the_production_figure_package_and_before_any_anomaly_metric"
)

FIGURE_PACKAGE_VERSION = figures.VERSION

REGIME = "S0"

REGIME_INDEX = 0

TAU0 = 0.1

FIGURE_3A = "model_c_bire_figure3a_streamfunction_anomaly_1deg_s0_dt10.png"

FIGURE_7A = "model_c_bire_figure7a_streamfunction_anomaly_day060_day2000_s0.png"

FIGURE_NAMES = (FIGURE_3A, FIGURE_7A)

REFERENCE_FIGURE = "model_c_bire_reference_time_mean_streamfunction_s0.png"

REPORT_NAME = "model_c_bire_s0_anomaly_report.json"

ARRAYS_NAME = "model_c_bire_s0_anomaly_arrays.npz"

ANOMALY_LABEL = "Barotropic streamfunction anomaly $\\psi'$ (Sv)"

DIFFERENCE_LABEL = "Truth $-$ model anomaly (Sv)"

MEAN_LABEL = "Time-mean $\\overline{\\psi}$ (Sv)"

PENDING = "PENDING_AFTER_FIGURES"

#: All four sealed digests are deferred until the figure package exists.
#: The figure *contract* is included deliberately: `figures finalize` stamps the
#: selected optimizer step and the training digests into that file as a declared
#: pipeline step, so its bytes are not knowable when this contract is written.
#: What binds this package to one figure package is the pinned *path* plus the
#: full provenance chain re-verified on every load; the fill below still refuses
#: to overwrite a digest that was already filled with something else.
PENDING_PATHS: tuple[tuple[str, ...], ...] = (
    ("artifacts", "figure_package_contract", "sha256"),
    ("artifacts", "figure_package_report", "sha256"),
    ("artifacts", "figure_package_arrays", "sha256"),
    ("artifacts", "figure_package_manifest", "sha256"),
)

SEALED_KEYS = (
    "figure_package_contract",
    "figure_package_report",
    "figure_package_arrays",
    "figure_package_manifest",
)

_EXPECTED_REQUIRED = (
    FIGURE_3A,
    FIGURE_7A,
    REFERENCE_FIGURE,
    REPORT_NAME,
    ARRAYS_NAME,
    README_NAME,
    MANIFEST_NAME,
)

_EXPECTED_DIAGNOSTICS = (
    "anomaly_rms_ratio",
    "amplitude_normalized_meridional_and_zonal_first_difference_rms",
    "hann_directional_power_fraction_above_0p2_cycles_per_cell",
    "western_first_4_wet_cells_rms_and_boundary_to_interior_ratio",
)

_REQUIRED_ARRAYS = frozenset(
    {
        "figure3_truth_streamfunction",
        "figure3_model_streamfunction",
        "figure7_truth_streamfunction",
        "figure7_model_streamfunction",
        "longitude_deg",
        "latitude_deg",
        "wet_mask",
        "start_draw_order",
    }
)

_REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/oceanfno/anomaly.py",
        "src/oceanfno/dataset.py",
        "src/oceanfno/diagnostics.py",
        "src/oceanfno/figures.py",
        "src/oceanfno/plots.py",
        "src/oceanfno/runtime.py",
    }
)


class AnomalyContractError(RuntimeError):
    """Raised when the anomaly package loses its sealed figure provenance."""


def _integer(value: Any, fallback: int = -1) -> int:
    try:
        if isinstance(value, bool):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _read(contract: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = contract
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def unfilled_fields(contract: Mapping[str, Any]) -> list[str]:
    """Return figure artifacts explicitly deferred until publication."""

    return [
        ".".join(path) for path in PENDING_PATHS if _read(contract, path) in (None, PENDING)
    ]


def training_mean_streamfunction(
    group: Any,
    wet: np.ndarray,
    *,
    experiment: int = REGIME_INDEX,
    chunk_days: int = 60,
) -> tuple[np.ndarray, int]:
    """MITgcm time-mean barotropic streamfunction over the S0 training block.

    Averages the *derived* streamfunction field chunk by chunk, the same
    construction :func:`oceanfno.validation.train_only_climatology` uses for its
    per-regime climatology, so this field and that one agree.
    """

    start, stop = TRAIN_RANGE
    state = group["state"]
    total = np.zeros(wet.shape, dtype=np.float64)
    count = 0
    for begin in range(start, stop, chunk_days):
        end = min(begin + chunk_days, stop)
        raw = np.asarray(state[experiment, begin:end], dtype=np.float32)
        total += derived_fields(raw, wet)["streamfunction"].sum(axis=0, dtype=np.float64)
        count += int(raw.shape[0])
    if count != stop - start:
        raise AnomalyContractError(
            f"the training mean covered {count} days, not {stop - start}"
        )
    mean = (total / count).astype(np.float32)
    mean[~wet] = 0.0
    if not np.all(np.isfinite(mean)):
        raise AnomalyContractError("the time-mean streamfunction is not finite")
    return mean, count


def _sealed_figure_provenance(contract: Mapping[str, Any]) -> dict[str, str]:
    """Verify the exact contract/report/arrays chain and the sealed manifest."""

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
        or figure_contract.get("selected_model", {}).get("version")
        != figures.TRAINING_VERSION
        or report.get("status") != "complete"
        or report.get("version") != FIGURE_PACKAGE_VERSION
        or report.get("regime") != REGIME
        or tuple(report.get("start_draw_order", ()))
        != tuple(int(value) for value in figures.declared_inference_starts())
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
            != tuple(int(value) for value in figures.declared_inference_starts())
        ):
            raise AnomalyContractError("the figure array protocol changed")
    return digests


def finalize(contract_path: str | Path) -> dict[str, Any]:
    """Fill the deferred sealed-package hashes, idempotently.

    ``_sealed_figure_provenance`` has already proved that the four files form
    one self-consistent package --- the report names this contract and carries
    its digest, the manifest seals the report and arrays, and the arrays carry
    the declared protocol --- before any digest is written here.
    """

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
            + " -- run `python -m oceanfno.anomaly finalize` first"
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
        or reference.get("source") != "mitgcm"
        or tuple(reference.get("days", ())) != TRAIN_RANGE
        or reference.get("regime") != REGIME
        or reference.get("subtracted_from") != "both_truth_and_prediction"
        or reference.get("model_own_mean_used") is not False
        or reference.get("is_two_dimensional_field") is not True
        or reference.get("not_a_scalar_spatial_mean") is not True
        or contract.get("modifies_published_figures") is not False
        or contract.get("adds_only") is not True
        or not str(output.get("project_root", "")).endswith(VERSION)
        or not str(output.get("scratch_root", "")).endswith(VERSION)
        or output.get("overwrite") is not False
        or output.get("one_folder_per_regime") is not True
        or tuple(output.get("required", ())) != _EXPECTED_REQUIRED
    ):
        raise AnomalyContractError("the anomaly contract changed")

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


def wet_rms(field: np.ndarray, wet: np.ndarray) -> float:
    """Root-mean-square over wet cells only."""

    values = np.asarray(field, dtype=np.float64)[wet]
    return float(np.sqrt(np.mean(np.square(values))))


def variability_summary(
    truth: np.ndarray,
    model: np.ndarray,
    leads: Sequence[int],
    wet: np.ndarray,
) -> dict[str, Any]:
    """Per-lead transient variability of truth and prediction about the same mean."""

    record: dict[str, Any] = {}
    for index, lead in enumerate(leads):
        truth_rms = wet_rms(truth[index], wet)
        model_rms = wet_rms(model[index], wet)
        record[str(int(lead))] = {
            "truth_anomaly_rms_sv": truth_rms,
            "model_anomaly_rms_sv": model_rms,
            "anomaly_rms_ratio": (
                (model_rms / truth_rms) if truth_rms > 0.0 else float("inf")
            ),
            "anomaly_error_rms_sv": wet_rms(truth[index] - model[index], wet),
            "truth_anomaly_range_sv": [
                float(truth[index][wet].min()),
                float(truth[index][wet].max()),
            ],
            "model_anomaly_range_sv": [
                float(model[index][wet].min()),
                float(model[index][wet].max()),
            ],
        }
    return record


def _normalized_first_difference(
    field: np.ndarray,
    wet: np.ndarray,
    *,
    axis: int,
) -> float:
    """Wet-pair first-difference RMS normalized by the field's wet RMS."""

    if axis not in (0, 1):
        raise ValueError("the streamfunction roughness axis must be 0 or 1")
    pair = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
    prior = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
    valid = wet[pair] & wet[prior]
    scale = wet_rms(field, wet)
    if scale == 0.0 or not np.any(valid):
        return float("nan")
    difference = (
        np.asarray(field, dtype=np.float64)[pair]
        - np.asarray(field, dtype=np.float64)[prior]
    )
    return float(np.sqrt(np.mean(np.square(difference[valid]))) / scale)


def _directional_high_wavenumber_fraction(
    field: np.ndarray,
    wet: np.ndarray,
    *,
    threshold: float = 0.2,
) -> dict[str, float]:
    """Hann-tapered fractions above a directional cycles-per-cell threshold."""

    rows, columns = np.where(wet)
    crop = np.asarray(field, dtype=np.float64)[
        rows.min() : rows.max() + 1,
        columns.min() : columns.max() + 1,
    ]
    crop = crop - float(crop.mean())
    window = np.hanning(crop.shape[0])[:, None] * np.hanning(crop.shape[1])[None, :]
    power = np.abs(np.fft.rfft2(crop * window)) ** 2
    total = float(power.sum())
    if total == 0.0:
        return {"meridional": float("nan"), "zonal": float("nan")}
    fy = np.fft.fftfreq(crop.shape[0])[:, None]
    fx = np.fft.rfftfreq(crop.shape[1])[None, :]
    return {
        "meridional": float(
            power[np.broadcast_to(np.abs(fy) > threshold, power.shape)].sum() / total
        ),
        "zonal": float(power[np.broadcast_to(fx > threshold, power.shape)].sum() / total),
    }


def day2000_structure_summary(
    truth: np.ndarray,
    model: np.ndarray,
    wet: np.ndarray,
) -> dict[str, Any]:
    """Directional sharpness and western-wall concentration after mean removal."""

    boundary = western_boundary_mask(wet, 4)
    interior = wet & ~boundary
    truth_boundary = wet_rms(truth, boundary)
    model_boundary = wet_rms(model, boundary)
    truth_interior = wet_rms(truth, interior)
    model_interior = wet_rms(model, interior)
    truth_spectrum = _directional_high_wavenumber_fraction(truth, wet)
    model_spectrum = _directional_high_wavenumber_fraction(model, wet)
    return {
        "basis": "member_0_day_2000_streamfunction_anomaly_about_the_S0_MITgcm_training_mean",
        "normalized_first_difference_rms": {
            "truth_meridional": _normalized_first_difference(truth, wet, axis=0),
            "model_meridional": _normalized_first_difference(model, wet, axis=0),
            "truth_zonal": _normalized_first_difference(truth, wet, axis=1),
            "model_zonal": _normalized_first_difference(model, wet, axis=1),
        },
        "hann_directional_power_fraction_above_0p2_cycles_per_cell": {
            "truth_meridional": truth_spectrum["meridional"],
            "model_meridional": model_spectrum["meridional"],
            "truth_zonal": truth_spectrum["zonal"],
            "model_zonal": model_spectrum["zonal"],
        },
        "western_first_4_wet_cells": {
            "truth_rms_sv": truth_boundary,
            "model_rms_sv": model_boundary,
            "model_to_truth_rms_ratio": (
                model_boundary / truth_boundary if truth_boundary > 0.0 else float("inf")
            ),
            "truth_boundary_to_interior_rms_ratio": (
                truth_boundary / truth_interior if truth_interior > 0.0 else float("inf")
            ),
            "model_boundary_to_interior_rms_ratio": (
                model_boundary / model_interior if model_interior > 0.0 else float("inf")
            ),
        },
    }


def _plot_anomaly_grid(
    output: Path,
    truth: np.ndarray,
    model: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    """Figure 3a: the early-lead plate, stationary circulation removed."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difference = truth - model
    bound = _finite_bound((truth, model))
    difference_bound = _finite_bound((difference,))
    figure, axes = plt.subplots(
        3,
        len(FIGURE_3_LEADS),
        figsize=(11.0, 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    state_image = difference_image = None
    for column, lead in enumerate(FIGURE_3_LEADS):
        state_image = axes[0, column].pcolormesh(
            longitude, latitude, _masked(truth[column], wet),
            cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
        )
        axes[1, column].pcolormesh(
            longitude, latitude, _masked(model[column], wet),
            cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
        )
        difference_image = axes[2, column].pcolormesh(
            longitude, latitude, _masked(difference[column], wet),
            cmap="RdBu_r", vmin=-difference_bound, vmax=difference_bound, shading="auto",
        )
        axes[0, column].set_title(f"Day {lead}")
        axes[2, column].set_xlabel("Longitude (°)")
    axes[0, 0].set_ylabel("MITgcm $\\psi'$\nLatitude (°)")
    axes[1, 0].set_ylabel("Emulator $\\psi'$\nLatitude (°)")
    axes[2, 0].set_ylabel("Truth − model\nLatitude (°)")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    figure.colorbar(
        state_image, ax=axes[:2].ravel().tolist(), label=ANOMALY_LABEL, shrink=0.82
    )
    figure.colorbar(
        difference_image, ax=axes[2].ravel().tolist(), label=DIFFERENCE_LABEL, shrink=0.82
    )
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; anomaly about the MITgcm "
        r"training-block mean $\overline{\psi}_{S0}$ (days 0–5999)"
    )
    figure.savefig(output / FIGURE_3A, bbox_inches="tight")
    plt.close(figure)


def _plot_anomaly_long(
    output: Path,
    truth: np.ndarray,
    model: np.ndarray,
    mean_field: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    """Figure 7a: day 60 and day 2,000 anomalies, with the removed mean shown once."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bound = _finite_bound((truth, model))
    figure, axes = plt.subplots(
        len(FIGURE_7_LEADS),
        3,
        figsize=(10.4, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    anomaly_image = None
    for row, lead in enumerate(FIGURE_7_LEADS):
        anomaly_image = axes[row, 0].pcolormesh(
            longitude, latitude, _masked(truth[row], wet),
            cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
        )
        axes[row, 1].pcolormesh(
            longitude, latitude, _masked(model[row], wet),
            cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
        )
        axes[row, 2].pcolormesh(
            longitude, latitude, _masked(truth[row] - model[row], wet),
            cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
        )
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
    axes[0, 0].set_title("MITgcm $\\psi'$")
    axes[0, 1].set_title("Emulator $\\psi'$")
    axes[0, 2].set_title("Truth − model")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
        axis.set_xlabel("")
    for axis in axes[-1]:
        axis.set_xlabel("Longitude (°)")
    figure.colorbar(
        anomaly_image, ax=axes.ravel().tolist(), label=ANOMALY_LABEL, shrink=0.75
    )
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; $\psi' = \psi - \overline{\psi}_{S0}$, "
        r"the same MITgcm training mean removed from both columns"
    )
    figure.savefig(output / FIGURE_7A, bbox_inches="tight")
    plt.close(figure)

    # The removed field, published once so the anomalies can be read against it.
    figure, axis = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    mean_image = axis.pcolormesh(
        longitude, latitude, _masked(mean_field, wet),
        cmap="RdBu_r",
        vmin=-_finite_bound((mean_field,)), vmax=_finite_bound((mean_field,)),
        shading="auto",
    )
    axis.set_aspect("equal")
    axis.set_facecolor("0.86")
    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")
    axis.set_title(r"$\overline{\psi}_{S0}$, MITgcm days 0–5999")
    figure.colorbar(mean_image, ax=axis, label=MEAN_LABEL, shrink=0.85)
    figure.savefig(output / REFERENCE_FIGURE, bbox_inches="tight")
    plt.close(figure)


def _readme(report: Mapping[str, Any]) -> str:
    day60 = report["variability"]["figure7"]["60"]
    day2000 = report["variability"]["figure7"]["2000"]
    structure = report["day2000_structure"]
    gradients = structure["normalized_first_difference_rms"]
    spectrum = structure["hann_directional_power_fraction_above_0p2_cycles_per_cell"]
    boundary = structure["western_first_4_wet_cells"]
    return f"""# Production emulator streamfunction anomalies, S0 — companions to figures 3 and 7

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

`psi_bar_S0` is MITgcm's two-dimensional time-mean barotropic streamfunction
over training days {TRAIN_RANGE[0]}--{TRAIN_RANGE[1] - 1}. The identical field is
subtracted from truth and from the model; the model's own mean is never
subtracted, so a bias in the standing gyre cannot hide inside the anomaly.

| lead | truth anomaly RMS | model anomaly RMS | ratio |
| --- | --- | --- | --- |
| day 60 | {day60['truth_anomaly_rms_sv']:.3f} Sv | {day60['model_anomaly_rms_sv']:.3f} Sv | {day60['anomaly_rms_ratio']:.3f} |
| day 2,000 | {day2000['truth_anomaly_rms_sv']:.3f} Sv | {day2000['model_anomaly_rms_sv']:.3f} Sv | {day2000['anomaly_rms_ratio']:.3f} |

At day 2,000 the model's normalized first-difference RMS is
{gradients['model_meridional']:.3f} meridionally and {gradients['model_zonal']:.3f}
zonally, against {gradients['truth_meridional']:.3f} and
{gradients['truth_zonal']:.3f} in truth; the directional high-wavenumber power
fractions are {spectrum['model_meridional']:.5f} and {spectrum['model_zonal']:.5f}
against {spectrum['truth_meridional']:.5f} and {spectrum['truth_zonal']:.5f}.

The western-boundary concentration is reported explicitly, because no gate scores
it: the model's first-four-wet-cell to interior anomaly RMS ratio is
{boundary['model_boundary_to_interior_rms_ratio']:.3f} against
{boundary['truth_boundary_to_interior_rms_ratio']:.3f} in truth.

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

    # The one operation this package exists to perform, applied identically to
    # both sides so no model bias in the mean circulation can hide.
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

        report = {
            "status": "complete",
            "version": VERSION,
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
        "reference_days": list(TRAIN_RANGE),
        "reference_source": "mitgcm_training_block_only",
        "same_reference_for_truth_and_prediction": True,
        "figure_names": list(FIGURE_NAMES),
        "modifies_published_figures": False,
        "reads_model_weights": False,
        "rolls_nothing_out": True,
        "figure_package_arrays": contract["artifacts"]["figure_package_arrays"]["path"],
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
