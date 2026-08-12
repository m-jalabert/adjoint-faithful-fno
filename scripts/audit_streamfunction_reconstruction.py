#!/usr/bin/env python
"""Three-way streamfunction reconstruction audit of the continuity model.

The published barotropic streamfunction is reconstructed from the predicted
velocities *after* the forecast, by depth-integrating the 15 ``U`` levels and
then cumulatively integrating southward-to-northward:

    U_BT(x,y) = sum_k U_k(x,y) dz_k
    psi(x,y_j) = -sum_{m<=j} U_BT(x,y_m) dy

Because that cumulative sum runs independently down every longitude column, a
small *coherent* error in depth-integrated ``U`` at one longitude is integrated
across the whole basin and appears as a north-south stripe in ``psi``.  The
stripe survives the anomaly figures exactly, because both truth and prediction
have the same MITgcm mean subtracted and it cancels:

    psi_hat' - psi' = psi_hat - psi

so the anomaly plots cannot tell us whether the stripes come from the one-way
reconstruction or from a genuinely wrong barotropic transport.

This script separates those two possibilities without retraining anything.  It
rolls the selected checkpoint exactly as the frozen figure engine does, and at
each captured lead builds three independent streamfunctions from the *same*
prediction:

    psi_U    meridional cumulative integral of -U_BT      (the published one)
    psi_V    zonal cumulative integral of +V_BT           (independent)
    psi_UV   least-squares fit to both transports at once (no accumulation)

The discriminator is where the error is coherent.  A cumulative integral along
``y`` converts a per-column transport bias into a column-constant offset, so:

    psi_U striped in columns, psi_UV clean  -> one-way reconstruction artifact
    all three wrong                         -> genuinely wrong transport

Nothing on disk belonging to the model is modified: no retraining, no
checkpoint selection, no rewriting of the published figure package.  As a
harness check the script asserts that its own ``psi_U`` reproduces the
published ``figure7_model_streamfunction`` bit for bit before reporting
anything, so a wiring error cannot masquerade as a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from oceanfno import _figures_core as base
from oceanfno import figures as figmod
from oceanfno import plots
from oceanfno.dataset import read_mitgcm_2d
from oceanfno.diagnostics import (
    EARTH_RADIUS_M,
    depth_integrated_transport,
    streamfunction_from_u,
    streamfunction_from_v,
    streamfunction_least_squares,
)
from oceanfno.runtime import torch
from oceanfno.validation import _gather

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_continuity_s0_figures_v1.json"
CAPTURE_LEADS = (0, 10, 20, 30, 40, 60, 200, 1000, 2000)
MAP_LEADS = (60, 2000)
RECONSTRUCTIONS = ("psi_U", "psi_V", "psi_UV")
LABEL = {
    "psi_U": r"$\psi_U$  (published, $\int\,dy$)",
    "psi_V": r"$\psi_V$  ($\int\,dx$)",
    "psi_UV": r"$\psi_{UV}$  (least squares)",
}


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------
def capture_transports(contract: Mapping[str, Any], device_name: str) -> dict[str, Any]:
    """Roll the selected checkpoint and capture both transports at each lead."""

    device = base._device(device_name)
    group = zarr.open_consolidated(str(Path(contract["dataset"]["path"]).resolve()), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    _, _, _, _, wind_mean, wind_scale = base._normalizers(group)
    starts = base.declared_inference_starts()

    with np.load(Path(contract["artifacts"]["selected_normalization"]["path"])) as stored:
        point_mean = np.asarray(stored["pointwise_mean"], dtype=np.float32)
        point_scale = np.asarray(stored["pointwise_scale"], dtype=np.float32)
    static_block, _ = base.physical_static_block(
        contract["artifacts"], group, point_mean, point_scale
    )
    stepper = base._stepper(
        contract, "selected_checkpoint", device, wet, wind_mean, wind_scale, static_block
    )

    records = np.stack(
        [np.full(starts.shape, base.REGIME_INDEX["S0"], dtype=np.int64), starts], axis=1
    )
    initial = _gather(state, records, 0)
    current = stepper.normalized_state(initial)
    static_normalized = stepper.normalized_static(static, records[:, 0])
    if getattr(stepper, "requires_history", False):
        stepper.begin(_gather(state, records, -base.HORIZON_DAYS))

    captured: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for lead in base.LEAD_DAYS:
            if lead:
                current = stepper.step(current, static_normalized)
                prediction = stepper.physical(current)
            else:
                prediction = initial.copy()
            if lead not in CAPTURE_LEADS:
                continue
            truth = _gather(state, records, lead)
            model_u, model_v = depth_integrated_transport(prediction)
            truth_u, truth_v = depth_integrated_transport(truth)
            captured[lead] = {
                "model_u_bt": model_u.astype(np.float32),
                "model_v_bt": model_v.astype(np.float32),
                "truth_u_bt": truth_u.astype(np.float32),
                "truth_v_bt": truth_v.astype(np.float32),
            }
    return {
        "captured": captured,
        "wet": wet,
        "starts": starts,
        "longitude": np.asarray(group["longitude_deg"][:], dtype=np.float32),
        "latitude": np.asarray(group["latitude_deg"][:], dtype=np.float32),
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _demean(field: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Remove the wet-cell mean; the streamfunction gauge is arbitrary."""

    out = np.array(field, dtype=np.float64, copy=True)
    out[:, ~wet] = np.nan
    out -= np.nanmean(out.reshape(out.shape[0], -1), axis=1)[:, None, None]
    return out


def coherence_fractions(error: np.ndarray, wet: np.ndarray) -> tuple[float, float]:
    """Return (column-coherent, row-coherent) fractions of the error variance.

    A north-south stripe is constant down a column, so averaging the error over
    ``y`` retains it while averaging incoherent error suppresses it.  The two
    numbers are the variance of the column means and of the row means, each
    divided by the total error variance.
    """

    masked = np.array(error, dtype=np.float64)
    masked[:, ~wet] = np.nan
    total = np.nanvar(masked.reshape(masked.shape[0], -1), axis=1)
    column = np.nanvar(np.nanmean(masked, axis=1), axis=1)
    row = np.nanvar(np.nanmean(masked, axis=2), axis=1)
    good = total > 0
    if not np.any(good):
        # Lead 0 is the initial condition, where the error is identically zero
        # and the coherence of a zero field is undefined rather than large.
        return (None, None)
    return (
        float(np.mean(column[good] / total[good])),
        float(np.mean(row[good] / total[good])),
    )


def directional_roughness(error: np.ndarray, wet: np.ndarray) -> tuple[float, float]:
    """RMS first difference of the error field, zonal and meridional."""

    masked = np.array(error, dtype=np.float64)
    masked[:, ~wet] = np.nan
    zonal = np.sqrt(np.nanmean(np.square(np.diff(masked, axis=2))))
    meridional = np.sqrt(np.nanmean(np.square(np.diff(masked, axis=1))))
    return float(zonal), float(meridional)


def reconstruct(
    u_bt: np.ndarray, v_bt: np.ndarray, wet: np.ndarray, dx: np.ndarray, dy: float
) -> dict[str, np.ndarray]:
    return {
        "psi_U": streamfunction_from_u(u_bt, wet, dy),
        "psi_V": streamfunction_from_v(v_bt, wet, dx),
        "psi_UV": streamfunction_least_squares(u_bt, v_bt, wet, dx, dy),
    }


def analyse(capture: Mapping[str, Any], dx: np.ndarray) -> dict[str, Any]:
    wet = capture["wet"]
    dy = EARTH_RADIUS_M * np.deg2rad(1.0)
    per_lead: dict[str, Any] = {}
    fields: dict[int, Any] = {}

    for lead, block in sorted(capture["captured"].items()):
        model = reconstruct(block["model_u_bt"], block["model_v_bt"], wet, dx, dy)
        truth = reconstruct(block["truth_u_bt"], block["truth_v_bt"], wet, dx, dy)
        entry: dict[str, Any] = {}
        for name in RECONSTRUCTIONS:
            error = _demean(model[name], wet) - _demean(truth[name], wet)
            column, row = coherence_fractions(error, wet)
            zonal, meridional = directional_roughness(error, wet)
            truth_rms = float(np.sqrt(np.nanmean(np.square(_demean(truth[name], wet)))))
            entry[name] = {
                "error_rms_sv": float(np.sqrt(np.nanmean(np.square(error)))),
                "truth_rms_sv": truth_rms,
                "column_coherent_fraction": column,
                "row_coherent_fraction": row,
                "error_zonal_first_difference_rms": zonal,
                "error_meridional_first_difference_rms": meridional,
            }
        du = block["model_u_bt"] - block["truth_u_bt"]
        dv = block["model_v_bt"] - block["truth_v_bt"]
        entry["transport"] = {
            "u_bt_error_rms_m2_s": float(np.sqrt(np.mean(np.square(du[:, wet])))),
            "v_bt_error_rms_m2_s": float(np.sqrt(np.mean(np.square(dv[:, wet])))),
            "u_bt_truth_rms_m2_s": float(np.sqrt(np.mean(np.square(block["truth_u_bt"][:, wet])))),
            "v_bt_truth_rms_m2_s": float(np.sqrt(np.mean(np.square(block["truth_v_bt"][:, wet])))),
        }
        entry["transport"]["u_bt_relative_error"] = (
            entry["transport"]["u_bt_error_rms_m2_s"] / entry["transport"]["u_bt_truth_rms_m2_s"]
        )
        entry["transport"]["v_bt_relative_error"] = (
            entry["transport"]["v_bt_error_rms_m2_s"] / entry["transport"]["v_bt_truth_rms_m2_s"]
        )
        per_lead[str(lead)] = entry
        if lead in MAP_LEADS:
            fields[lead] = {"model": model, "truth": truth, "du": du, "dv": dv}
    return {"per_lead": per_lead, "fields": fields}


#: Axis each reconstruction accumulates along; ``psi_UV`` accumulates along
#: neither, and is shown against ``y`` purely for comparison with ``psi_U``.
INTEGRATION_AXIS = {"psi_U": "along_y", "psi_V": "along_x", "psi_UV": "along_y"}


def accumulation_profile(
    capture: Mapping[str, Any], dx: np.ndarray, lead: int
) -> dict[str, dict[str, list[float]]]:
    """RMS error profiles along both axes, per reconstruction.

    A one-way cumulative integral should grow with distance from where it
    starts, so each reconstruction is read along its *own* integration axis.
    """

    wet = capture["wet"]
    dy = EARTH_RADIUS_M * np.deg2rad(1.0)
    block = capture["captured"][lead]
    model = reconstruct(block["model_u_bt"], block["model_v_bt"], wet, dx, dy)
    truth = reconstruct(block["truth_u_bt"], block["truth_v_bt"], wet, dx, dy)
    profile: dict[str, dict[str, list[float]]] = {}
    for name in RECONSTRUCTIONS:
        error = _demean(model[name], wet) - _demean(truth[name], wet)
        with np.errstate(invalid="ignore"):
            along_y = np.sqrt(np.nanmean(np.square(error), axis=(0, 2)))
            along_x = np.sqrt(np.nanmean(np.square(error), axis=(0, 1)))
        profile[name] = {
            "along_y": [None if np.isnan(v) else float(v) for v in along_y],
            "along_x": [None if np.isnan(v) else float(v) for v in along_x],
        }
    return profile


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _panel(axis, field, wet, title, limit, cmap):
    shown = np.array(field, dtype=np.float64)
    shown[~wet] = np.nan
    mesh = axis.pcolormesh(
        shown, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        shading="auto", rasterized=True,
    )
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])
    return mesh


def figure_three_streamfunctions(analysis, capture, lead, path):
    wet = capture["wet"]
    block = analysis["fields"][lead]
    figure, axes = plt.subplots(3, 3, figsize=(10.2, 10.0))
    for row, name in enumerate(RECONSTRUCTIONS):
        truth = _demean(block["truth"][name], wet)[0]
        model = _demean(block["model"][name], wet)[0]
        error = model - truth
        scale = float(np.nanmax(np.abs(truth))) or 1.0
        escale = float(np.nanmax(np.abs(error))) or 1.0
        _panel(axes[row, 0], truth, wet, f"{LABEL[name]}  truth", scale, "RdBu_r")
        _panel(axes[row, 1], model, wet, f"{LABEL[name]}  model", scale, "RdBu_r")
        mesh = _panel(axes[row, 2], error, wet,
                      f"error   rms {np.sqrt(np.nanmean(error**2)):.3f} Sv", escale, "PuOr_r")
        plt.colorbar(mesh, ax=axes[row, 2], fraction=0.046, label="Sv")
    figure.suptitle(
        f"Three streamfunctions from one prediction  |  day {lead}  |  member 0  |  S0",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_diagnostics(analysis, capture, profile, path):
    wet = capture["wet"]
    leads = sorted(int(k) for k in analysis["per_lead"])
    figure = plt.figure(figsize=(12.6, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)

    axis = figure.add_subplot(grid[0, 0])
    for name in RECONSTRUCTIONS:
        series = profile[name][INTEGRATION_AXIS[name]]
        axis.plot([np.nan if v is None else v for v in series],
                  label=LABEL[name], linewidth=1.6)
    axis.set_xlabel("index along that reconstruction's own integration path")
    axis.set_ylabel("RMS error across it  (Sv)")
    axis.set_title("Error accumulates along the integral, day 2000", fontsize=10)
    axis.legend(fontsize=8); axis.grid(alpha=.25)

    axis = figure.add_subplot(grid[0, 1])
    for name in RECONSTRUCTIONS:
        axis.plot(leads, [analysis["per_lead"][str(l)][name]["error_rms_sv"] for l in leads],
                  marker="o", ms=3, label=LABEL[name], linewidth=1.6)
    axis.set_xscale("symlog", linthresh=10)
    axis.set_xlabel("lead (days)"); axis.set_ylabel("RMS error (Sv)")
    axis.set_title("Streamfunction error vs lead", fontsize=10)
    axis.legend(fontsize=8); axis.grid(alpha=.25)

    axis = figure.add_subplot(grid[0, 2])
    width = 0.35
    positions = np.arange(len(RECONSTRUCTIONS))
    last = analysis["per_lead"][str(leads[-1])]
    axis.bar(positions - width/2, [last[n]["column_coherent_fraction"] or 0.0 for n in RECONSTRUCTIONS],
             width, label="column-coherent (vertical stripes)")
    axis.bar(positions + width/2, [last[n]["row_coherent_fraction"] or 0.0 for n in RECONSTRUCTIONS],
             width, label="row-coherent (horizontal stripes)")
    axis.set_xticks(positions); axis.set_xticklabels(["$\\psi_U$", "$\\psi_V$", "$\\psi_{UV}$"])
    axis.set_ylabel("fraction of error variance")
    axis.set_title(f"Where the error is coherent, day {leads[-1]}", fontsize=10)
    axis.legend(fontsize=8); axis.grid(alpha=.25, axis="y")

    block = analysis["fields"][MAP_LEADS[-1]]
    for column, (key, name) in enumerate((("du", "$\\delta U_{BT}$"), ("dv", "$\\delta V_{BT}$"))):
        axis = figure.add_subplot(grid[1, column])
        field = block[key][0]
        limit = float(np.nanpercentile(np.abs(field[wet]), 99)) or 1.0
        mesh = _panel(axis, field, wet, f"{name}  day {MAP_LEADS[-1]}  (m$^2$ s$^{{-1}}$)",
                      limit, "PuOr_r")
        plt.colorbar(mesh, ax=axis, fraction=0.046)

    axis = figure.add_subplot(grid[1, 2])
    axis.plot(leads, [analysis["per_lead"][str(l)]["transport"]["u_bt_relative_error"] for l in leads],
              marker="o", ms=3, label="$\\delta U_{BT}$ / $U_{BT}$", linewidth=1.6)
    axis.plot(leads, [analysis["per_lead"][str(l)]["transport"]["v_bt_relative_error"] for l in leads],
              marker="s", ms=3, label="$\\delta V_{BT}$ / $V_{BT}$", linewidth=1.6)
    axis.set_xscale("symlog", linthresh=10)
    axis.set_xlabel("lead (days)"); axis.set_ylabel("relative RMS error")
    axis.set_title("Barotropic transport error", fontsize=10)
    axis.legend(fontsize=8); axis.grid(alpha=.25)

    figure.suptitle("Streamfunction reconstruction audit  |  continuity model  |  S0", fontsize=12)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--published-arrays", type=Path, default=None,
        help="figure-package arrays .npz used to confirm psi_U reproduces the published figure",
    )
    args = parser.parse_args(argv)

    figmod._install()
    contract, resolved, _ = figmod.load_contract(args.contract, verify_sources=False)
    capture = capture_transports(contract, args.device)
    wet = capture["wet"]
    dx = np.asarray(
        read_mitgcm_2d(contract["artifacts"]["mitgcm_zonal_spacing"]["path"]), dtype=np.float64
    )

    # Harness check: this rollout must reproduce the published streamfunction.
    # The figure package stores psi for member 0 at the figure-7 leads, so any
    # drift in checkpoint, statics, normalizers or start draw shows up here
    # before a single number below is believed.
    dy = EARTH_RADIUS_M * np.deg2rad(1.0)
    control_max = None
    if args.published_arrays is not None and args.published_arrays.is_file():
        with np.load(args.published_arrays) as stored:
            expected = np.asarray(stored["figure7_model_streamfunction"], dtype=np.float64)
        for index, lead in enumerate(plots.FIGURE_7_LEADS):
            mine = streamfunction_from_u(capture["captured"][lead]["model_u_bt"], wet, dy)[0]
            difference = float(np.max(np.abs(mine.astype(np.float64) - expected[index])))
            control_max = difference if control_max is None else max(control_max, difference)
        if control_max > 1.0e-4:
            raise SystemExit(
                f"control failed: psi_U differs from the published figure by {control_max:.3e} Sv"
            )
        print(f"control: psi_U reproduces the published figure to {control_max:.2e} Sv")
    else:
        print("control: published arrays not supplied, skipping reproduction check")

    analysis = analyse(capture, dx)
    profile = accumulation_profile(capture, dx, 2000)

    args.output.mkdir(parents=True, exist_ok=True)
    figures = [
        figure_three_streamfunctions(
            analysis, capture, 2000, args.output / "streamfunction_audit_three_psi_day2000.png"),
        figure_three_streamfunctions(
            analysis, capture, 60, args.output / "streamfunction_audit_three_psi_day0060.png"),
        figure_diagnostics(
            analysis, capture, profile, args.output / "streamfunction_audit_diagnostics.png"),
    ]

    report = {
        "version": "streamfunction_reconstruction_audit_v1",
        "purpose": "separate one-way reconstruction artefact from genuinely wrong barotropic transport",
        "model": contract["selected_model"]["version"],
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "contract": str(resolved),
        "regime": "S0",
        "member_count": int(capture["starts"].size),
        "map_member": 0,
        "capture_leads": list(CAPTURE_LEADS),
        "reconstructions": {
            "psi_U": "meridional cumulative integral of -U_BT (published)",
            "psi_V": "zonal cumulative integral of +V_BT",
            "psi_UV": "least-squares fit to both transports, wet-mean gauge",
        },
        "control_psi_U_max_difference_from_published_sv": control_max,
        "reads_model_weights": True,
        "modifies_published_figures": False,
        "retrains_nothing": True,
        "per_lead": analysis["per_lead"],
        "accumulation_profile_day2000": profile,
        "figures": [path.name for path in figures],
    }
    (args.output / "streamfunction_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )

    leads = sorted(int(k) for k in analysis["per_lead"])
    print("=" * 92)
    print(f"{'lead':>6} {'psi_U rms':>10} {'psi_V rms':>10} {'psi_UV rms':>11} "
          f"{'U col-coh':>10} {'UV col-coh':>11} {'dU_BT rel':>10}")
    print("-" * 92)
    def _fmt(value: float | None, width: int) -> str:
        return f"{'--':>{width}}" if value is None else f"{value:>{width}.4f}"

    for lead in leads:
        e = analysis["per_lead"][str(lead)]
        print(f"{lead:>6} {e['psi_U']['error_rms_sv']:>10.4f} {e['psi_V']['error_rms_sv']:>10.4f} "
              f"{e['psi_UV']['error_rms_sv']:>11.4f} "
              f"{_fmt(e['psi_U']['column_coherent_fraction'], 10)} "
              f"{_fmt(e['psi_UV']['column_coherent_fraction'], 11)} "
              f"{e['transport']['u_bt_relative_error']:>10.4f}")
    print("=" * 92)
    for path in figures:
        print(f"wrote {path}")
    print(f"wrote {args.output / 'streamfunction_audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
