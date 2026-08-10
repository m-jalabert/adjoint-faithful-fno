#!/usr/bin/env python
"""Inference-only ablation of the local 3x3 branch on the 32x32 checkpoint.

The 32x32 model evaluates

    z = G(x) + gamma L(x),

where ``G`` is the three-block FNO on the Bire sine/cosine position encoding and
``L`` is the bias-free 49 -> 46 3x3 convolution on the raw external input.  The
trained model is ``gamma = 1``.  This script sweeps ``gamma`` at inference only:
no retraining, no checkpoint selection, and nothing on disk is modified.

Because ``L`` is a bias-free convolution, scaling it by ``gamma`` is exactly
scaling ``local.weight``, so the ablation needs no change to ``src/oceanfno``
and the published checkpoint is never rewritten -- the weight is scaled in
memory on a freshly loaded model for each ``gamma``.

The question is whether the unrestricted local convolution is what concentrates
spurious energy at the western wall.  Four quantities decide it, all computed
with the frozen suite's own functions so they are directly comparable with the
published numbers:

    day-200 RMSE and ACC              (short-lead skill)
    day-2000 anomaly RMS ratio        (long-horizon amplitude)
    western first-4-wet-cell RMS      (the wall itself, Sv)
    zonal high-wavenumber fraction    (where the spurious energy sits)

``gamma = 1`` reproduces the published report values exactly; the script asserts
this before reporting anything, so a harness error cannot masquerade as a
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from oceanfno import anomaly, figures, plots  # noqa: E402
from oceanfno.dataset import _normalizers  # noqa: E402
from oceanfno.runtime import _device, torch  # noqa: E402
from oceanfno.validation import train_only_climatology  # noqa: E402


GAMMAS = (1.0, 0.5, 0.0)
REGIME = "S0"
REGIME_INDEX = 0
DAY_200 = 200
DAY_2000 = 2000

#: Published 32x32 (gamma = 1) values the harness must reproduce bit-for-bit.
PUBLISHED_GAMMA1 = {
    "day2000_anomaly_rms_ratio": 3.6113588257932676,
    "western_first_4_wet_cells_rms_sv": 2.066642273122052,
    "zonal_high_wavenumber_fraction": 0.05201215477476319,
}
REPRODUCTION_TOLERANCE = 1.0e-9

#: Validated categorical slots 1-3, in fixed order: gamma = 1, 0.5, 0.
GAMMA_COLORS = {1.0: "#2a78d6", 0.5: "#eb6834", 0.0: "#1baf7a"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"


class AblationError(RuntimeError):
    """Raised when the ablation cannot be run against verified provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(declared: Path) -> Path:
    """Resolve a declared artifact, allowing for the per-arm output regrouping.

    The published arm directories were regrouped under a per-arm folder after
    the figure runs, so the frozen contract's ``selected_report`` path is stale.
    The content is unchanged and is hash-verified by the caller; only the
    location moved.
    """

    if declared.is_file():
        return declared
    arm = "bire_protocol_rollout_ft_y32_x32"
    moved = Path(str(declared).replace(f"/C/{arm}_v1/", f"/C/{arm}/{arm}_v1/"))
    if moved.is_file():
        return moved
    raise AblationError(f"artifact is absent at {declared} and at {moved}")


def audit_contract(contract_path: Path) -> dict[str, Any]:
    """Load the frozen figure contract and verify every artifact by content.

    This is stricter than the suite's own loader on the point that matters here:
    each checkpoint file is hashed and compared with the contract, so the
    weights being ablated are provably the published ones.
    """

    contract = json.loads(contract_path.read_text())
    verified: dict[str, str] = {}
    for key in (
        "selected_checkpoint",
        "comparator_checkpoint",
        "selected_normalization",
        "selected_report",
    ):
        declared = contract["artifacts"][key]
        path = _resolve(Path(str(declared["path"])))
        actual = _sha256(path)
        if actual != declared.get("sha256"):
            raise AblationError(
                f"{key} hash changed: contract declares {declared.get('sha256')}, "
                f"{path} hashes to {actual}"
            )
        verified[key] = actual
    architecture = contract["selected_model"]["architecture"]
    if tuple(architecture["n_modes"]) != (32, 32):
        raise AblationError(f"this ablation is for the 32x32 arm, found {architecture['n_modes']}")
    if int(architecture["local_kernel_size"]) != 3:
        raise AblationError("the contract does not declare a local 3x3 branch")
    return {"contract": contract, "verified_sha256": verified}


def _metrics_from_arrays(
    arrays: Mapping[str, np.ndarray],
    group: Any,
    wet: np.ndarray,
) -> dict[str, Any]:
    """Reduce one rollout to the four deciding quantities, plus the curves."""

    lead_days = list(plots.LEAD_DAYS)
    short_days = list(plots.SHORT_LEAD_DAYS)
    day200 = short_days.index(DAY_200)
    day200_long = lead_days.index(DAY_200)
    day2000 = lead_days.index(DAY_2000)

    rmse_200 = {
        field: float(np.mean(arrays[f"rmse__model__{field}"][:, day200_long]))
        for field in plots.RMSE_FIELDS
    }
    acc_200 = {
        field: float(np.mean(arrays[f"acc__selected__{field}"][:, day200]))
        for field in plots.ACC_FIELDS
    }

    # The anomaly package's one operation, applied identically to both sides.
    mean_field, mean_days = anomaly.training_mean_streamfunction(group, wet)
    figure7_truth = np.asarray(arrays["figure7_truth_streamfunction"], dtype=np.float64) - mean_field
    figure7_model = np.asarray(arrays["figure7_model_streamfunction"], dtype=np.float64) - mean_field
    variability = anomaly.variability_summary(
        figure7_truth, figure7_model, plots.FIGURE_7_LEADS, wet
    )
    structure = anomaly.day2000_structure_summary(figure7_truth[-1], figure7_model[-1], wet)

    # variability_summary keys by str(int(lead)), not by the integer itself.
    day2000_variability = variability[str(DAY_2000)]
    return {
        "day200": {"rmse": rmse_200, "acc": acc_200},
        "day2000_anomaly_rms_ratio": day2000_variability["anomaly_rms_ratio"],
        "day2000_model_anomaly_rms_sv": day2000_variability["model_anomaly_rms_sv"],
        "day2000_truth_anomaly_rms_sv": day2000_variability["truth_anomaly_rms_sv"],
        "western_first_4_wet_cells_rms_sv": structure["western_first_4_wet_cells"]["model_rms_sv"],
        "western_truth_rms_sv": structure["western_first_4_wet_cells"]["truth_rms_sv"],
        "western_model_to_truth_rms_ratio": structure["western_first_4_wet_cells"][
            "model_to_truth_rms_ratio"
        ],
        "zonal_high_wavenumber_fraction": structure[
            "hann_directional_power_fraction_above_0p2_cycles_per_cell"
        ]["model_zonal"],
        "meridional_high_wavenumber_fraction": structure[
            "hann_directional_power_fraction_above_0p2_cycles_per_cell"
        ]["model_meridional"],
        "truth_zonal_high_wavenumber_fraction": structure[
            "hann_directional_power_fraction_above_0p2_cycles_per_cell"
        ]["truth_zonal"],
        "day2000_structure": structure,
        "variability": variability,
        "training_mean_days": mean_days,
        "curves": {
            "short_lead_days": short_days,
            "lead_days": lead_days,
            "rmse": {
                field: arrays[f"rmse__model__{field}"].mean(axis=0).tolist()
                for field in plots.RMSE_FIELDS
            },
            "acc": {
                field: arrays[f"acc__selected__{field}"].mean(axis=0).tolist()
                for field in plots.ACC_FIELDS
            },
            "normalized_max_abs": arrays["normalized_max_abs"].max(axis=0).tolist(),
            "finite_fraction": arrays["finite"].mean(axis=0).tolist(),
        },
        "day2000_finite_fraction": float(arrays["finite"][:, day2000].mean()),
        "day2000_normalized_max_abs": float(arrays["normalized_max_abs"][:, day2000].max()),
    }


def run_ablation(contract_path: Path, device_name: str = "auto") -> dict[str, Any]:
    """Roll the 32x32 checkpoint to day 2,000 once per gamma and reduce."""

    if torch is None:
        raise AblationError("the local-branch ablation requires PyTorch")
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
    base_weight = selected.model.local.weight.detach().clone()
    local_weight_rms = float(base_weight.pow(2).mean().sqrt())

    results: dict[str, Any] = {}
    saved_fields: dict[str, np.ndarray] = {}
    for gamma in GAMMAS:
        print(f"\nrolling gamma = {gamma} ...", flush=True)
        with torch.no_grad():
            selected.model.local.weight.copy_(base_weight * gamma)
        arrays = figures.evaluate_regime(
            selected,
            comparator,
            state,
            static,
            REGIME_INDEX,
            starts,
            climatology_state,
            climatology_derived,
            wet,
        )
        metrics = _metrics_from_arrays(arrays, group, wet)
        results[str(gamma)] = metrics
        saved_fields[f"gamma_{gamma}_day2000_model_streamfunction"] = arrays[
            "figure7_model_streamfunction"
        ][-1]
        # Truth does not depend on the model; keep it once and assert it is stable.
        truth_day2000 = np.asarray(arrays["figure7_truth_streamfunction"][-1], dtype=np.float64)
        if "truth_day2000_streamfunction" in saved_fields:
            if not np.array_equal(saved_fields["truth_day2000_streamfunction"], truth_day2000):
                raise AblationError("truth changed between gamma sweeps")
        else:
            saved_fields["truth_day2000_streamfunction"] = truth_day2000
        print(
            f"  day-200 RMSE(sst) {metrics['day200']['rmse']['sst']:.4f}   "
            f"ACC(sst) {metrics['day200']['acc']['sst']:.4f}   "
            f"day-2000 anomaly ratio {metrics['day2000_anomaly_rms_ratio']:.4f}   "
            f"western RMS {metrics['western_first_4_wet_cells_rms_sv']:.4f} Sv   "
            f"zonal high-k {metrics['zonal_high_wavenumber_fraction']:.5f}",
            flush=True,
        )
    with torch.no_grad():
        selected.model.local.weight.copy_(base_weight)

    # The gate: gamma = 1 is the published model, so it must reproduce exactly.
    baseline = results["1.0"]
    discrepancies = {
        name: (baseline[name], expected)
        for name, expected in PUBLISHED_GAMMA1.items()
        if abs(baseline[name] - expected) > REPRODUCTION_TOLERANCE * max(abs(expected), 1.0)
    }
    if discrepancies:
        raise AblationError(
            "gamma = 1 did not reproduce the published 32x32 report: "
            + "; ".join(f"{k}: got {g!r}, published {e!r}" for k, (g, e) in discrepancies.items())
        )

    mean_field, _ = anomaly.training_mean_streamfunction(group, wet)
    return {
        "report": {
            "version": "local_branch_gamma_ablation_v1",
            "kind": "inference_only_ablation",
            "arm": "bire_protocol_rollout_ft_y32_x32",
            "regime": REGIME,
            "formula": "z = G(x) + gamma * L(x)",
            "gammas": list(GAMMAS),
            "retrained": False,
            "checkpoint_modified_on_disk": False,
            "local_branch": {
                "kernel_size": 3,
                "bias": False,
                "in_channels": 49,
                "out_channels": 46,
                "weight_rms": local_weight_rms,
                "implementation": "gamma applied by scaling local.weight in memory",
            },
            "contract": str(contract_path.resolve()),
            "verified_sha256": audited["verified_sha256"],
            "published_gamma1_reproduced": PUBLISHED_GAMMA1,
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
        },
    }


def absolute_directional_power(
    field: np.ndarray, wet: np.ndarray, *, threshold: float = 0.2
) -> dict[str, float]:
    """Absolute (un-normalized) directional high-wavenumber power.

    Identical cropping, mean removal, Hann taper and ``rfft2`` to the suite's
    ``_directional_high_wavenumber_fraction``; only the final division by total
    power is omitted.  The published fraction is a *share* of total power, so
    when a change alters total anomaly power it can move opposite to the
    absolute high-wavenumber content.  Both are reported.
    """

    rows, columns = np.where(wet)
    crop = np.asarray(field, dtype=np.float64)[
        rows.min() : rows.max() + 1, columns.min() : columns.max() + 1
    ]
    crop = crop - float(crop.mean())
    window = np.hanning(crop.shape[0])[:, None] * np.hanning(crop.shape[1])[None, :]
    power = np.abs(np.fft.rfft2(crop * window)) ** 2
    fy = np.fft.fftfreq(crop.shape[0])[:, None]
    fx = np.fft.rfftfreq(crop.shape[1])[None, :]
    total = float(power.sum())
    return {
        "total": total,
        "meridional": float(power[np.broadcast_to(np.abs(fy) > threshold, power.shape)].sum()),
        "zonal": float(power[np.broadcast_to(fx > threshold, power.shape)].sum()),
    }


def rescore(output_dir: Path) -> dict[str, Any]:
    """Add absolute directional power to a finished ablation, from saved fields."""

    report_path = output_dir / "local_branch_gamma_ablation_report.json"
    report = json.loads(report_path.read_text())
    with np.load(output_dir / "local_branch_gamma_ablation_arrays.npz") as stored:
        fields = {name: stored[name] for name in stored.files}
    wet = np.asarray(fields["wet_mask"], dtype=bool)
    mean_field = np.asarray(fields["reference_time_mean_streamfunction"])

    absolute = {
        "truth": absolute_directional_power(
            np.asarray(fields["truth_day2000_streamfunction"]) - mean_field, wet
        )
    }
    for gamma in report["gammas"]:
        absolute[str(gamma)] = absolute_directional_power(
            np.asarray(fields[f"gamma_{gamma}_day2000_model_streamfunction"]) - mean_field, wet
        )
    report["absolute_high_wavenumber_power"] = absolute
    report["absolute_power_note"] = (
        "the published fraction normalizes by total power; these are the same "
        "Hann-windowed spectra without that division, so the absolute trend is "
        "separable from the change in total anomaly power"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def plot_spectrum(report: Mapping[str, Any], output: Path) -> Path:
    """Zonal high-k as a share of power and in absolute terms, side by side."""

    import matplotlib.pyplot as plt

    _style()
    gammas = [float(g) for g in report["gammas"]]
    absolute = report["absolute_high_wavenumber_power"]
    results = report["results"]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), constrained_layout=True)

    panels = (
        (
            "zonal high-$k$ share of power",
            [results[str(g)]["zonal_high_wavenumber_fraction"] for g in gammas],
            results["1.0"]["truth_zonal_high_wavenumber_fraction"],
            "fraction",
            False,
        ),
        (
            "zonal high-$k$ absolute power",
            [absolute[str(g)]["zonal"] for g in gammas],
            absolute["truth"]["zonal"],
            "power",
            True,
        ),
        (
            "total anomaly power",
            [absolute[str(g)]["total"] for g in gammas],
            absolute["truth"]["total"],
            "power",
            True,
        ),
    )
    for axis, (title, values, truth, unit, log) in zip(axes, panels):
        axis.bar(
            range(len(gammas)),
            values,
            color=[GAMMA_COLORS[g] for g in gammas],
            width=0.62,
        )
        if log:
            axis.set_yscale("log")
        axis.axhline(truth, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
        axis.annotate(
            f"truth {truth:.4g}",
            xy=(0.98, truth),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.5,
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
                fontsize=8,
                color=INK_SECONDARY,
            )
        axis.set_xticks(range(len(gammas)))
        axis.set_xticklabels([rf"$\gamma$={g:g}" for g in gammas])
        axis.set_title(title, color=INK_PRIMARY, loc="left")
        axis.set_ylabel(unit)
        _frame(axis)
    figure.suptitle(
        "The high-$k$ share rises at $\\gamma$=0.5 only because total anomaly power fell;\n"
        "absolute zonal high-$k$ power decreases monotonically as $\\gamma$ is reduced",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


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


def plot_metrics(report: Mapping[str, Any], output: Path) -> Path:
    """The four deciding quantities against gamma, each with its truth mark."""

    import matplotlib.pyplot as plt

    _style()
    results = report["results"]
    gammas = [float(g) for g in report["gammas"]]
    panels = (
        ("day-200 RMSE, SST", lambda m: m["day200"]["rmse"]["sst"], "degC", None),
        ("day-200 ACC, SST", lambda m: m["day200"]["acc"]["sst"], "", None),
        (
            "day-2000 anomaly RMS ratio",
            lambda m: m["day2000_anomaly_rms_ratio"],
            "model / truth",
            1.0,
        ),
        (
            "western 4-cell streamfunction RMS",
            lambda m: m["western_first_4_wet_cells_rms_sv"],
            "Sv",
            results["1.0"]["western_truth_rms_sv"],
        ),
        (
            "zonal high-$k$ power fraction",
            lambda m: m["zonal_high_wavenumber_fraction"],
            "",
            results["1.0"]["truth_zonal_high_wavenumber_fraction"],
        ),
        (
            "day-200 ACC, surface $u$",
            lambda m: m["day200"]["acc"]["surface_u"],
            "",
            None,
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), constrained_layout=True)
    for axis, (title, getter, unit, truth) in zip(axes.ravel(), panels):
        values = [getter(results[str(g)]) for g in gammas]
        colors = [GAMMA_COLORS[g] for g in gammas]
        axis.bar(range(len(gammas)), values, color=colors, width=0.62)
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.4g}",
                xy=(index, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_SECONDARY,
            )
        if truth is not None:
            axis.axhline(truth, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
            axis.annotate(
                f"truth {truth:.4g}",
                xy=(0.98, truth),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=INK_SECONDARY,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
            )
        axis.set_xticks(range(len(gammas)))
        axis.set_xticklabels([rf"$\gamma$={g:g}" for g in gammas])
        axis.set_title(title, color=INK_PRIMARY, loc="left")
        if unit:
            axis.set_ylabel(unit)
        axis.set_ylim(0.0, max(values + ([truth] if truth is not None else [])) * 1.28)
        _frame(axis)
    figure.suptitle(
        "Inference-only ablation of the local 3x3 branch on the 32x32 checkpoint\n"
        r"$z=G(x)+\gamma L(x)$, no retraining, 15 S0 members",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_short_lead(report: Mapping[str, Any], output: Path) -> Path:
    """Short-lead skill curves, to show what reducing gamma costs before day 200."""

    import matplotlib.pyplot as plt

    _style()
    results = report["results"]
    gammas = [float(g) for g in report["gammas"]]
    rmse_fields = ("sst", "surface_speed", "phihyd_surface")
    acc_fields = ("sst", "surface_u")
    # No shared x: five panels are the 0-200 day window, the last runs to 2,000.
    figure, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), constrained_layout=True)
    short_days = np.asarray(results["1.0"]["curves"]["short_lead_days"], dtype=float)
    lead_days = np.asarray(results["1.0"]["curves"]["lead_days"], dtype=float)
    short_mask = lead_days <= DAY_200
    for column, field in enumerate(rmse_fields):
        axis = axes[0, column]
        for gamma in gammas:
            curve = np.asarray(results[str(gamma)]["curves"]["rmse"][field], dtype=float)
            axis.plot(
                lead_days[short_mask],
                curve[short_mask],
                color=GAMMA_COLORS[gamma],
                linewidth=1.6,
                label=rf"$\gamma$={gamma:g}",
            )
        axis.set_title(f"RMSE  {field}", color=INK_PRIMARY, loc="left")
        axis.set_xlim(0.0, float(DAY_200))
        _frame(axis)
    for column, field in enumerate(acc_fields):
        axis = axes[1, column]
        for gamma in gammas:
            curve = np.asarray(results[str(gamma)]["curves"]["acc"][field], dtype=float)
            axis.plot(
                short_days, curve, color=GAMMA_COLORS[gamma], linewidth=1.6,
                label=rf"$\gamma$={gamma:g}",
            )
        axis.set_title(f"ACC  {field}", color=INK_PRIMARY, loc="left")
        axis.set_xlim(0.0, float(DAY_200))
        axis.set_ylim(0.5, 1.0)
        _frame(axis)
    axis = axes[1, 2]
    for gamma in gammas:
        curve = np.asarray(
            results[str(gamma)]["curves"]["normalized_max_abs"], dtype=float
        )
        axis.semilogy(
            lead_days, curve, color=GAMMA_COLORS[gamma], linewidth=1.6,
            label=rf"$\gamma$={gamma:g}",
        )
    axis.set_title("max |normalized state|, to day 2,000", color=INK_PRIMARY, loc="left")
    axis.set_xlim(0.0, float(lead_days[-1]))
    _frame(axis)
    for row in axes:
        for axis in row:
            axis.set_xlabel("lead time (days)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(
        "Short-lead skill under the local-branch ablation (15 S0 members)",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_day2000_maps(
    report: Mapping[str, Any], fields: Mapping[str, np.ndarray], output: Path
) -> Path:
    """Day-2000 streamfunction anomaly maps, truth and each gamma."""

    import matplotlib.pyplot as plt

    _style()
    wet = np.asarray(fields["wet_mask"], dtype=bool)
    longitude = np.asarray(fields["longitude_deg"])
    latitude = np.asarray(fields["latitude_deg"])
    mean_field = np.asarray(fields["reference_time_mean_streamfunction"])
    gammas = [float(g) for g in report["gammas"]]

    panels = [
        (
            "MITgcm truth",
            np.asarray(fields["truth_day2000_streamfunction"]) - mean_field,
        )
    ]
    for gamma in gammas:
        panels.append(
            (
                rf"$\gamma$ = {gamma:g}",
                np.asarray(fields[f"gamma_{gamma}_day2000_model_streamfunction"]) - mean_field,
            )
        )
    bound = float(
        np.percentile(np.abs(np.stack([value[wet] for _, value in panels])), 99.5)
    )
    figure, axes = plt.subplots(1, 4, figsize=(12.4, 3.5), constrained_layout=True)
    for axis, (title, value) in zip(axes, panels):
        masked = np.where(wet, value, np.nan)
        mesh = axis.pcolormesh(
            longitude, latitude, masked, cmap="RdBu_r", vmin=-bound, vmax=bound,
            shading="auto",
        )
        axis.set_title(title, color=INK_PRIMARY, loc="left")
        axis.set_xlabel("longitude (deg)")
        rms = float(np.sqrt(np.mean(np.square(value[wet]))))
        axis.annotate(
            f"RMS {rms:.3g} Sv",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )
    axes[0].set_ylabel("latitude (deg)")
    figure.colorbar(mesh, ax=axes, shrink=0.86, label="streamfunction anomaly (Sv)")
    figure.suptitle(
        "Day-2,000 barotropic streamfunction anomaly: does removing the local branch "
        "clean up the western wall?",
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
        default=PROJECT_ROOT / "outputs" / "af_fno" / "C" / "local_branch_gamma_ablation_v1",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = run_ablation(args.contract.resolve(), args.device)
    report, fields = produced["report"], produced["fields"]

    (output_dir / "local_branch_gamma_ablation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(output_dir / "local_branch_gamma_ablation_arrays.npz", **fields)
    figure_paths = [
        plot_metrics(report, output_dir / "local_ablation_metrics.png"),
        plot_short_lead(report, output_dir / "local_ablation_short_lead.png"),
        plot_day2000_maps(report, fields, output_dir / "local_ablation_day2000_anomaly.png"),
    ]

    print("\n" + "=" * 96)
    print(f"{'gamma':>7} {'RMSE200(sst)':>13} {'ACC200(sst)':>12} {'anomRMSratio':>13}"
          f" {'westRMS(Sv)':>12} {'zonal high-k':>13}")
    for gamma in report["gammas"]:
        m = report["results"][str(gamma)]
        print(
            f"{gamma:>7g} {m['day200']['rmse']['sst']:13.5f} {m['day200']['acc']['sst']:12.5f}"
            f" {m['day2000_anomaly_rms_ratio']:13.5f}"
            f" {m['western_first_4_wet_cells_rms_sv']:12.5f}"
            f" {m['zonal_high_wavenumber_fraction']:13.6f}"
        )
    print("=" * 96)
    print("gamma = 1 reproduced the published 32x32 report exactly.")
    for path in figure_paths:
        print(f"wrote {path}")
    print(f"wrote {output_dir / 'local_branch_gamma_ablation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
