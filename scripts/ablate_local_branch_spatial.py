#!/usr/bin/env python
"""Inference-only spatial gate on the local 3x3 branch of the 32x32 checkpoint.

The scalar sweep showed the local branch is useful but too active in the basin
interior.  This script tests a wall/interior split instead of a constant:

    z = G(x) + gamma(d) L(x),
    gamma(d) = gamma_int + (gamma_wall - gamma_int) exp(-d / L),

where ``d`` is the existing ``distance_to_wall_normalized`` static feature
restored to wet-cell units, and ``L`` is the western-boundary width (4 cells).
Nothing is retrained and nothing on disk is rewritten: the gate multiplies the
local branch's output through a forward hook, which is removed afterwards.

Because ``gamma`` is now spatial it can no longer be folded into ``local.weight``
as the scalar sweep did -- the hook is applied to the convolution's output, which
is exactly ``gamma(d) L(x)``.

Two references (flat gamma = 1 and flat gamma = 0.5) are rolled alongside the
three gates so the comparison is internal to one run.  Flat gamma = 1 must
reproduce the published 32x32 report or the script aborts.

New in this script relative to the scalar sweep:

  * a direct SSH / eta RMSE and ACC, rather than inferring the short-lead
    penalty through the reconstructed surface pressure,
  * interior RMS reported beside the western four-cell RMS,
  * absolute zonal high-wavenumber power beside the published share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from oceanfno import anomaly, figures, plots  # noqa: E402
from oceanfno.dataset import _normalizers, western_boundary_mask  # noqa: E402
from oceanfno.runtime import _device, torch  # noqa: E402
from oceanfno.validation import train_only_climatology  # noqa: E402


REGIME_INDEX = 0
DAY_200 = 200
DAY_2000 = 2000
BOUNDARY_WIDTH = 4
GATE_LENGTH_CELLS = 4.0

#: (label, gamma_wall, gamma_int).  ``None`` interior marks a flat reference.
CONFIGURATIONS: tuple[tuple[str, float, float | None], ...] = (
    ("flat 1.0", 1.0, None),
    ("flat 0.5", 0.5, None),
    ("gate 1.0/0.5", 1.0, 0.5),
    ("gate 1.0/0.25", 1.0, 0.25),
    ("gate 0.75/0.5", 0.75, 0.5),
)

PUBLISHED_FLAT1 = {
    "day2000_anomaly_rms_ratio": 3.6113588257932676,
    "western_first_4_wet_cells_rms_sv": 2.066642273122052,
    "zonal_high_wavenumber_fraction": 0.05201215477476319,
}
REPRODUCTION_TOLERANCE = 1.0e-9

#: Categorical slots in fixed order; the two flat references share the ink pair.
CONFIG_COLORS = {
    "flat 1.0": "#2a78d6",
    "flat 0.5": "#eb6834",
    "gate 1.0/0.5": "#1baf7a",
    "gate 1.0/0.25": "#eda100",
    "gate 0.75/0.5": "#4a3aa7",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"

#: The frozen suite does not reduce SSH; these extend its field tuples so the
#: unmodified ``evaluate_regime`` computes it. ``derived_fields`` already
#: returns "ssh", so this adds a reduction, not a new definition.
EXTENDED_RMSE_FIELDS = (*plots.RMSE_FIELDS, "ssh")
EXTENDED_ACC_FIELDS = (*plots.ACC_FIELDS, "ssh")


class SpatialAblationError(RuntimeError):
    """Raised when the spatial-gate ablation cannot run against verified inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_contract(contract_path: Path) -> dict[str, Any]:
    """Verify every declared artifact by content before any weight is touched."""

    contract = json.loads(contract_path.read_text())
    verified: dict[str, str] = {}
    for key in (
        "selected_checkpoint",
        "comparator_checkpoint",
        "selected_normalization",
        "selected_report",
    ):
        declared = contract["artifacts"][key]
        path = Path(str(declared["path"]))
        if not path.is_file():
            raise SpatialAblationError(f"{key} is absent at {path}")
        actual = _sha256(path)
        if actual != declared.get("sha256"):
            raise SpatialAblationError(
                f"{key} hash changed: contract declares {declared.get('sha256')}, "
                f"{path} hashes to {actual}"
            )
        verified[key] = actual
    architecture = contract["selected_model"]["architecture"]
    if tuple(architecture["n_modes"]) != (32, 32):
        raise SpatialAblationError(f"expected the 32x32 arm, found {architecture['n_modes']}")
    if int(architecture["local_kernel_size"]) != 3:
        raise SpatialAblationError("the contract does not declare a local 3x3 branch")
    return {"contract": contract, "verified_sha256": verified}


def wall_distance_cells(wet: np.ndarray) -> np.ndarray:
    """Distance to the nearest wall in wet cells, matching the static feature.

    The dataset stores ``distance_transform_edt(wet)`` divided by its wet
    maximum; this returns the same transform before that normalization, so
    ``L`` is expressed in cells exactly as the western-boundary width is.
    """

    return np.asarray(distance_transform_edt(np.asarray(wet, dtype=bool)), dtype=np.float64)


def spatial_gate(
    wet: np.ndarray,
    gamma_wall: float,
    gamma_int: float | None,
    length_cells: float = GATE_LENGTH_CELLS,
) -> np.ndarray:
    """Return gamma(d) on the grid; a flat gamma when ``gamma_int`` is ``None``."""

    wet = np.asarray(wet, dtype=bool)
    if gamma_int is None:
        return np.full(wet.shape, float(gamma_wall), dtype=np.float32)
    if length_cells <= 0.0:
        raise SpatialAblationError("the gate length scale must be positive")
    distance = wall_distance_cells(wet)
    gate = gamma_int + (gamma_wall - gamma_int) * np.exp(-distance / length_cells)
    # Land carries no local correction anywhere in the objective; keep it neutral.
    return np.where(wet, gate, float(gamma_int)).astype(np.float32)


def gate_description(wet: np.ndarray, gate: np.ndarray) -> dict[str, float]:
    """Report the gamma actually achieved in the band and the interior."""

    wet = np.asarray(wet, dtype=bool)
    boundary = western_boundary_mask(wet, BOUNDARY_WIDTH)
    interior = wet & ~boundary
    return {
        "min_wet": float(gate[wet].min()),
        "max_wet": float(gate[wet].max()),
        "mean_western_band": float(gate[boundary].mean()),
        "mean_interior": float(gate[interior].mean()),
        "at_first_wet_cell": float(gate[wall_distance_cells(wet) == 1.0].mean()),
    }


def _metrics_from_arrays(
    arrays: Mapping[str, np.ndarray], group: Any, wet: np.ndarray
) -> dict[str, Any]:
    """Reduce one rollout, including the direct SSH and interior diagnostics."""

    lead_days = list(plots.LEAD_DAYS)
    short_days = list(plots.SHORT_LEAD_DAYS)
    day200 = short_days.index(DAY_200)
    day200_long = lead_days.index(DAY_200)

    rmse_200 = {
        field: float(np.mean(arrays[f"rmse__model__{field}"][:, day200_long]))
        for field in EXTENDED_RMSE_FIELDS
    }
    acc_200 = {
        field: float(np.mean(arrays[f"acc__selected__{field}"][:, day200]))
        for field in EXTENDED_ACC_FIELDS
    }

    mean_field, _ = anomaly.training_mean_streamfunction(group, wet)
    truth = np.asarray(arrays["figure7_truth_streamfunction"], dtype=np.float64) - mean_field
    model = np.asarray(arrays["figure7_model_streamfunction"], dtype=np.float64) - mean_field
    variability = anomaly.variability_summary(truth, model, plots.FIGURE_7_LEADS, wet)
    structure = anomaly.day2000_structure_summary(truth[-1], model[-1], wet)

    boundary = western_boundary_mask(wet, BOUNDARY_WIDTH)
    interior = wet & ~boundary
    absolute = absolute_directional_power(model[-1], wet)
    day2000 = variability[str(DAY_2000)]
    return {
        "day200": {"rmse": rmse_200, "acc": acc_200},
        "day2000_anomaly_rms_ratio": day2000["anomaly_rms_ratio"],
        "day2000_model_anomaly_rms_sv": day2000["model_anomaly_rms_sv"],
        "day2000_truth_anomaly_rms_sv": day2000["truth_anomaly_rms_sv"],
        "western_first_4_wet_cells_rms_sv": structure["western_first_4_wet_cells"]["model_rms_sv"],
        "western_truth_rms_sv": structure["western_first_4_wet_cells"]["truth_rms_sv"],
        "western_model_to_truth_rms_ratio": structure["western_first_4_wet_cells"][
            "model_to_truth_rms_ratio"
        ],
        "interior_rms_sv": anomaly.wet_rms(model[-1], interior),
        "interior_truth_rms_sv": anomaly.wet_rms(truth[-1], interior),
        "boundary_to_interior_rms_ratio": structure["western_first_4_wet_cells"][
            "model_boundary_to_interior_rms_ratio"
        ],
        "truth_boundary_to_interior_rms_ratio": structure["western_first_4_wet_cells"][
            "truth_boundary_to_interior_rms_ratio"
        ],
        "zonal_high_wavenumber_fraction": structure[
            "hann_directional_power_fraction_above_0p2_cycles_per_cell"
        ]["model_zonal"],
        "truth_zonal_high_wavenumber_fraction": structure[
            "hann_directional_power_fraction_above_0p2_cycles_per_cell"
        ]["truth_zonal"],
        "zonal_high_wavenumber_power": absolute["zonal"],
        "meridional_high_wavenumber_power": absolute["meridional"],
        "total_anomaly_power": absolute["total"],
        "day2000_structure": structure,
        "curves": {
            "short_lead_days": short_days,
            "lead_days": lead_days,
            "rmse": {
                field: arrays[f"rmse__model__{field}"].mean(axis=0).tolist()
                for field in EXTENDED_RMSE_FIELDS
            },
            "acc": {
                field: arrays[f"acc__selected__{field}"].mean(axis=0).tolist()
                for field in EXTENDED_ACC_FIELDS
            },
            "normalized_max_abs": arrays["normalized_max_abs"].max(axis=0).tolist(),
        },
    }


def absolute_directional_power(
    field: np.ndarray, wet: np.ndarray, *, threshold: float = 0.2
) -> dict[str, float]:
    """Absolute directional high-wavenumber power (the suite's spectrum, un-normalized)."""

    rows, columns = np.where(wet)
    crop = np.asarray(field, dtype=np.float64)[
        rows.min() : rows.max() + 1, columns.min() : columns.max() + 1
    ]
    crop = crop - float(crop.mean())
    window = np.hanning(crop.shape[0])[:, None] * np.hanning(crop.shape[1])[None, :]
    power = np.abs(np.fft.rfft2(crop * window)) ** 2
    fy = np.fft.fftfreq(crop.shape[0])[:, None]
    fx = np.fft.rfftfreq(crop.shape[1])[None, :]
    return {
        "total": float(power.sum()),
        "meridional": float(power[np.broadcast_to(np.abs(fy) > threshold, power.shape)].sum()),
        "zonal": float(power[np.broadcast_to(fx > threshold, power.shape)].sum()),
    }


def run_ablation(contract_path: Path, device_name: str = "auto") -> dict[str, Any]:
    """Roll the checkpoint once per configuration and reduce."""

    if torch is None:
        raise SpatialAblationError("the spatial-gate ablation requires PyTorch")
    started = time.monotonic()
    audited = audit_contract(contract_path)
    contract = audited["contract"]
    device = _device(device_name)

    dataset = Path(contract["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    _, _, _, _, wind_mean, wind_scale = _normalizers(group)
    starts = figures.declared_inference_starts()
    climatology_state, climatology_derived, _ = train_only_climatology(state, wet)

    selected = figures._stepper(contract, "selected_checkpoint", device, wet, wind_mean, wind_scale)
    comparator = figures._stepper(
        contract, "comparator_checkpoint", device, wet, wind_mean, wind_scale
    )

    # Extend the reduced field set so the unmodified evaluate_regime also
    # reduces SSH. derived_fields already produces it; only the tuples change.
    original_rmse, original_acc = figures.RMSE_FIELDS, figures.ACC_FIELDS
    figures.RMSE_FIELDS = EXTENDED_RMSE_FIELDS
    figures.ACC_FIELDS = EXTENDED_ACC_FIELDS

    results: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    saved_fields: dict[str, np.ndarray] = {}
    try:
        for label, gamma_wall, gamma_int in CONFIGURATIONS:
            gate = spatial_gate(wet, gamma_wall, gamma_int)
            gates[label] = {
                "gamma_wall": gamma_wall,
                "gamma_interior": gamma_int,
                "length_cells": None if gamma_int is None else GATE_LENGTH_CELLS,
                "flat": gamma_int is None,
                **gate_description(wet, gate),
            }
            gate_tensor = torch.from_numpy(gate).to(device)[None, None]
            handle = selected.model.local.register_forward_hook(
                lambda module, inputs, output, g=gate_tensor: output * g
            )
            try:
                print(f"\nrolling {label} ...", flush=True)
                arrays = figures.evaluate_regime(
                    selected, comparator, state, static, REGIME_INDEX, starts,
                    climatology_state, climatology_derived, wet,
                )
            finally:
                handle.remove()
            metrics = _metrics_from_arrays(arrays, group, wet)
            results[label] = metrics
            saved_fields[f"{label}__day2000_model_streamfunction"] = arrays[
                "figure7_model_streamfunction"
            ][-1]
            truth_day2000 = np.asarray(
                arrays["figure7_truth_streamfunction"][-1], dtype=np.float64
            )
            if "truth_day2000_streamfunction" in saved_fields:
                if not np.array_equal(saved_fields["truth_day2000_streamfunction"], truth_day2000):
                    raise SpatialAblationError("truth changed between configurations")
            else:
                saved_fields["truth_day2000_streamfunction"] = truth_day2000
            print(
                f"  day-200 RMSE(ssh) {metrics['day200']['rmse']['ssh']:.5f}  "
                f"RMSE(phihyd) {metrics['day200']['rmse']['phihyd_surface']:.5f}  "
                f"ACC(ssh) {metrics['day200']['acc']['ssh']:.4f}  |  "
                f"anom {metrics['day2000_anomaly_rms_ratio']:.3f}  "
                f"west {metrics['western_first_4_wet_cells_rms_sv']:.3f}  "
                f"int {metrics['interior_rms_sv']:.3f}  "
                f"kx {metrics['zonal_high_wavenumber_power']:.4g}",
                flush=True,
            )
    finally:
        figures.RMSE_FIELDS, figures.ACC_FIELDS = original_rmse, original_acc

    baseline = results["flat 1.0"]
    discrepancies = {
        name: (baseline[name], expected)
        for name, expected in PUBLISHED_FLAT1.items()
        if abs(baseline[name] - expected) > REPRODUCTION_TOLERANCE * max(abs(expected), 1.0)
    }
    if discrepancies:
        raise SpatialAblationError(
            "flat gamma = 1 did not reproduce the published 32x32 report: "
            + "; ".join(f"{k}: got {g!r}, published {e!r}" for k, (g, e) in discrepancies.items())
        )

    mean_field, _ = anomaly.training_mean_streamfunction(group, wet)
    truth_absolute = absolute_directional_power(
        saved_fields["truth_day2000_streamfunction"] - mean_field, wet
    )
    return {
        "report": {
            "version": "local_branch_spatial_gate_ablation_v1",
            "kind": "inference_only_ablation",
            "arm": "bire_protocol_rollout_ft_y32_x32",
            "formula": "z = G(x) + gamma(d) L(x),  gamma(d) = g_int + (g_wall - g_int) exp(-d/L)",
            "length_cells": GATE_LENGTH_CELLS,
            "distance_field": "distance_transform_edt(wet), the static feature before normalization",
            "boundary_width_cells": BOUNDARY_WIDTH,
            "configurations": [label for label, _, _ in CONFIGURATIONS],
            "gates": gates,
            "retrained": False,
            "checkpoint_modified_on_disk": False,
            "gate_application": "forward hook on model.local, output multiplied by gamma(d)",
            "added_reductions": ["ssh"],
            "contract": str(contract_path.resolve()),
            "verified_sha256": audited["verified_sha256"],
            "published_flat1_reproduced": PUBLISHED_FLAT1,
            "truth_absolute_power": truth_absolute,
            "members": int(starts.size),
            "results": results,
            "elapsed_seconds": time.monotonic() - started,
        },
        "fields": {
            **saved_fields,
            "reference_time_mean_streamfunction": mean_field,
            "wet_mask": wet.astype(np.uint8),
            "longitude_deg": longitude,
            "latitude_deg": latitude,
            **{
                f"gate__{label}": spatial_gate(wet, gw, gi)
                for label, gw, gi in CONFIGURATIONS
            },
        },
    }


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK_PRIMARY,
        }
    )


def _frame(axis) -> None:
    axis.grid(color="0.88", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(length=3)


def _bar_panel(
    axis,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    unit: str,
    truth: float | None = None,
    log: bool = False,
) -> None:
    axis.bar(range(len(labels)), values, color=[CONFIG_COLORS[k] for k in labels], width=0.66)
    if log:
        axis.set_yscale("log")
    if truth is not None:
        axis.axhline(truth, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
        axis.annotate(
            f"truth {truth:.3g}",
            xy=(0.99, truth),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=INK_SECONDARY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
    for index, value in enumerate(values):
        axis.annotate(
            f"{value:.3g}",
            xy=(index, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK_SECONDARY,
        )
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels([k.replace(" ", "\n", 1) for k in labels], fontsize=7)
    axis.set_title(title, color=INK_PRIMARY, loc="left")
    axis.set_ylabel(unit)
    if not log:
        top = max(list(values) + ([truth] if truth is not None else []))
        axis.set_ylim(0.0, top * 1.30)
    _frame(axis)


def plot_metrics(report: Mapping[str, Any], output: Path) -> Path:
    """The metrics that decide the wall/interior split."""

    import matplotlib.pyplot as plt

    _style()
    labels = list(report["configurations"])
    results = report["results"]
    reference = results["flat 1.0"]
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4), constrained_layout=True)
    panels = (
        ("day-200 RMSE, SSH", lambda m: m["day200"]["rmse"]["ssh"], "m", None, False),
        ("day-200 ACC, SSH", lambda m: m["day200"]["acc"]["ssh"], "", None, False),
        (
            "day-200 RMSE, phihyd",
            lambda m: m["day200"]["rmse"]["phihyd_surface"],
            "",
            None,
            False,
        ),
        (
            "day-200 ACC, phihyd",
            lambda m: m["day200"]["acc"]["phihyd_surface"],
            "",
            None,
            False,
        ),
        (
            "day-2000 anomaly RMS ratio",
            lambda m: m["day2000_anomaly_rms_ratio"],
            "model / truth",
            1.0,
            False,
        ),
        (
            "western 4-cell RMS",
            lambda m: m["western_first_4_wet_cells_rms_sv"],
            "Sv",
            reference["western_truth_rms_sv"],
            False,
        ),
        (
            "interior RMS",
            lambda m: m["interior_rms_sv"],
            "Sv",
            reference["interior_truth_rms_sv"],
            False,
        ),
        (
            "absolute zonal high-$k$ power",
            lambda m: m["zonal_high_wavenumber_power"],
            "power",
            report["truth_absolute_power"]["zonal"],
            True,
        ),
    )
    for axis, (title, getter, unit, truth, log) in zip(axes.ravel(), panels):
        _bar_panel(axis, labels, [getter(results[k]) for k in labels], title, unit, truth, log)
    figure.suptitle(
        "Spatially gated local branch: $z=G(x)+\\gamma(d)L(x)$, "
        "$\\gamma(d)=\\gamma_{int}+(\\gamma_{wall}-\\gamma_{int})e^{-d/4}$\n"
        "inference only, no retraining, 15 S0 members",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_gate_and_maps(
    report: Mapping[str, Any], fields: Mapping[str, np.ndarray], output: Path
) -> Path:
    """The gate profile, then the day-2000 anomaly for truth and each configuration."""

    import matplotlib.pyplot as plt

    _style()
    wet = np.asarray(fields["wet_mask"], dtype=bool)
    longitude = np.asarray(fields["longitude_deg"])
    latitude = np.asarray(fields["latitude_deg"])
    mean_field = np.asarray(fields["reference_time_mean_streamfunction"])
    labels = list(report["configurations"])

    figure = plt.figure(figsize=(13.6, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)

    profile = figure.add_subplot(grid[0, 0])
    distance = wall_distance_cells(wet)
    order = np.argsort(distance[wet])
    for label in labels:
        gate = np.asarray(fields[f"gate__{label}"])
        profile.plot(
            distance[wet][order],
            gate[wet][order],
            color=CONFIG_COLORS[label],
            linewidth=1.6,
            label=label,
        )
    profile.axvspan(0.0, float(BOUNDARY_WIDTH), color=INK_MUTED, alpha=0.12, linewidth=0)
    profile.annotate(
        "western 4-cell band",
        xy=(BOUNDARY_WIDTH, profile.get_ylim()[1]),
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK_SECONDARY,
    )
    profile.set_xlabel("distance to wall (wet cells)")
    profile.set_ylabel(r"$\gamma(d)$")
    profile.set_title("the gate", color=INK_PRIMARY, loc="left")
    profile.legend(frameon=False, fontsize=7.5, loc="upper right")
    _frame(profile)

    panels = [("MITgcm truth", np.asarray(fields["truth_day2000_streamfunction"]) - mean_field)]
    for label in labels:
        panels.append(
            (label, np.asarray(fields[f"{label}__day2000_model_streamfunction"]) - mean_field)
        )
    bound = float(np.percentile(np.abs(np.stack([v[wet] for _, v in panels])), 99.5))
    positions = [grid[0, 1], grid[0, 2], grid[0, 3], grid[1, 0], grid[1, 1], grid[1, 2]]
    if len(panels) > len(positions):
        raise SpatialAblationError(
            f"{len(panels)} maps do not fit the {len(positions)}-slot layout"
        )
    mesh = None
    for position, (title, value) in zip(positions, panels):
        axis = figure.add_subplot(position)
        mesh = axis.pcolormesh(
            longitude, latitude, np.where(wet, value, np.nan), cmap="RdBu_r",
            vmin=-bound, vmax=bound, shading="auto",
        )
        axis.set_title(title, color=INK_PRIMARY, loc="left")
        axis.annotate(
            f"RMS {float(np.sqrt(np.mean(np.square(value[wet])))):.3g} Sv",
            xy=(0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7.5, color=INK_SECONDARY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        )
    if mesh is not None:
        # A narrow strip inside the free slot, not the whole slot.
        strip = grid[1, 3].subgridspec(3, 9)
        figure.colorbar(mesh, cax=figure.add_subplot(strip[0:3, 1]), label="anomaly (Sv)")
    figure.suptitle(
        "Day-2,000 streamfunction anomaly under the wall/interior gate",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_short_lead(report: Mapping[str, Any], output: Path) -> Path:
    """Short-lead SSH and pressure curves, where the scalar gate paid its cost."""

    import matplotlib.pyplot as plt

    _style()
    labels = list(report["configurations"])
    results = report["results"]
    lead = np.asarray(results["flat 1.0"]["curves"]["lead_days"], dtype=float)
    short = np.asarray(results["flat 1.0"]["curves"]["short_lead_days"], dtype=float)
    mask = lead <= DAY_200
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 5.8), constrained_layout=True)
    rmse_panels = ("ssh", "phihyd_surface", "sst")
    acc_panels = ("ssh", "phihyd_surface", "surface_u")
    for column, field in enumerate(rmse_panels):
        axis = axes[0, column]
        for label in labels:
            curve = np.asarray(results[label]["curves"]["rmse"][field], dtype=float)
            axis.plot(lead[mask], curve[mask], color=CONFIG_COLORS[label], linewidth=1.5, label=label)
        axis.set_title(f"RMSE  {field}", color=INK_PRIMARY, loc="left")
        axis.set_xlim(0.0, float(DAY_200))
        axis.set_xlabel("lead time (days)")
        _frame(axis)
    for column, field in enumerate(acc_panels):
        axis = axes[1, column]
        for label in labels:
            curve = np.asarray(results[label]["curves"]["acc"][field], dtype=float)
            axis.plot(short, curve, color=CONFIG_COLORS[label], linewidth=1.5, label=label)
        axis.set_title(f"ACC  {field}", color=INK_PRIMARY, loc="left")
        axis.set_xlim(0.0, float(DAY_200))
        axis.set_ylim(0.5, 1.0)
        axis.set_xlabel("lead time (days)")
        _frame(axis)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(
        "Short-lead skill under the spatial gate (15 S0 members)",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT
        / "config"
        / "model_c_bire_protocol_rollout_ft_y32_x32_s0_figures_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "af_fno"
        / "C"
        / "local_branch_spatial_gate_ablation_v1",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = run_ablation(args.contract.resolve(), args.device)
    report, fields = produced["report"], produced["fields"]

    (output_dir / "local_branch_spatial_gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(output_dir / "local_branch_spatial_gate_arrays.npz", **fields)
    paths = [
        plot_metrics(report, output_dir / "spatial_gate_metrics.png"),
        plot_short_lead(report, output_dir / "spatial_gate_short_lead.png"),
        plot_gate_and_maps(report, fields, output_dir / "spatial_gate_day2000_anomaly.png"),
    ]

    header = (
        f"{'configuration':>15}{'RMSE200 ssh':>13}{'ACC200 ssh':>12}{'RMSE200 phi':>13}"
        f"{'anomRatio':>11}{'westRMS':>9}{'intRMS':>8}{'kx power':>11}"
    )
    print("\n" + "=" * len(header))
    print(header)
    for label in report["configurations"]:
        m = report["results"][label]
        print(
            f"{label:>15}{m['day200']['rmse']['ssh']:13.5f}{m['day200']['acc']['ssh']:12.4f}"
            f"{m['day200']['rmse']['phihyd_surface']:13.5f}"
            f"{m['day2000_anomaly_rms_ratio']:11.3f}"
            f"{m['western_first_4_wet_cells_rms_sv']:9.3f}"
            f"{m['interior_rms_sv']:8.3f}{m['zonal_high_wavenumber_power']:11.4g}"
        )
    reference = report["results"]["flat 1.0"]
    print("=" * len(header))
    print(
        f"{'TRUTH':>15}{'-':>13}{'-':>12}{'-':>13}{1.0:11.3f}"
        f"{reference['western_truth_rms_sv']:9.3f}"
        f"{reference['interior_truth_rms_sv']:8.3f}"
        f"{report['truth_absolute_power']['zonal']:11.4g}"
    )
    print("\nflat gamma = 1 reproduced the published 32x32 report exactly.")
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
