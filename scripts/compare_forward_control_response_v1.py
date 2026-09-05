"""Forward-model comparison figures: nominal control (B) vs response (C).

Report figures for the adjoint-faithful study's paired forward arms
(``model_c_adjoint_faithful_nominal_control_v1`` vs
``model_c_adjoint_faithful_response_v1``, same seed). Pure re-plotting: both
arms already have complete, frozen ``*_s0_figures_v1`` and ``*_s0_anomaly_v1``
packages on disk, so nothing here runs inference or touches a checkpoint --
it only reads the arrays and report scalars those packages already wrote and
overlays them. Plotting primitives (styling, percentile bands, colour
bounds) are reused from ``oceanfno.plots`` rather than reimplemented.

Usage:

    python scripts/compare_forward_control_response_v1.py
    python scripts/compare_forward_control_response_v1.py --seed 20260724
    python scripts/compare_forward_control_response_v1.py --force   # overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oceanfno import plots as base_plots

REGIME = "S0"
DEFAULT_SEED = 20260911

CONTROL_VERSION = "model_c_adjoint_faithful_nominal_control_v1"
RESPONSE_VERSION = "model_c_adjoint_faithful_response_v1"

ARM_STYLE = {
    "control": {"label": "Control ($\\lambda_{resp}=0$)", "color": "tab:red"},
    "response": {"label": "Response ($\\lambda_{resp}=10^{-3}$)", "color": "tab:purple"},
}
BASELINE_STYLE = {
    "climatology": {"label": "Climatology", "color": "0.35"},
    "persistence": {"label": "Persistence", "color": "tab:blue"},
}


class CompareForwardError(RuntimeError):
    """Raised when the two arms' packages cannot be legitimately compared."""


def _arm_dirs(version: str, seed: int) -> dict[str, Path]:
    tag = f"{version}_seed_{seed}_s0"
    return {
        "figures": _ROOT / "outputs" / "af_fno" / "C" / f"{tag}_figures_v1" / REGIME,
        "anomaly": _ROOT / "outputs" / "af_fno" / "C" / f"{tag}_anomaly_v1" / REGIME,
        "figures_config": _ROOT / "config" / f"{tag}_figures_v1.json",
    }


def load_arm(version: str, seed: int) -> dict[str, Any]:
    dirs = _arm_dirs(version, seed)
    for label, path in dirs.items():
        if not path.exists():
            raise CompareForwardError(f"{version} seed {seed}: {label} missing at {path}")

    figures_config = json.loads(dirs["figures_config"].read_text())
    with np.load(dirs["figures"] / "model_c_bire_s0_figures_arrays.npz") as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    with np.load(dirs["anomaly"] / "model_c_bire_s0_anomaly_arrays.npz") as z:
        anomaly_arrays = {k: np.asarray(z[k]) for k in z.files}
    figures_report = json.loads((dirs["figures"] / "model_c_bire_s0_figures_report.json").read_text())
    anomaly_report = json.loads((dirs["anomaly"] / "model_c_bire_s0_anomaly_report.json").read_text())

    return {
        "version": version,
        "seed": seed,
        "arrays": arrays,
        "anomaly": anomaly_arrays,
        "figures_report": figures_report,
        "anomaly_report": anomaly_report,
        "checkpoint_sha256": figures_config["artifacts"]["selected_checkpoint"]["sha256"],
        "optimizer_step": figures_config["selected_model"]["optimizer_step"],
        "sources": {k: str(v) for k, v in dirs.items()},
    }


def check_shared_contract(control: dict[str, Any], response: dict[str, Any]) -> None:
    """The two arms must share truth, grid and members, or nothing below is subtractable."""

    problems = []
    if not np.array_equal(control["arrays"]["wet_mask"], response["arrays"]["wet_mask"]):
        problems.append("wet masks differ")
    if not np.array_equal(control["arrays"]["start_draw_order"], response["arrays"]["start_draw_order"]):
        problems.append("inference member start days differ")
    if not np.array_equal(control["arrays"]["lead_days"], response["arrays"]["lead_days"]):
        problems.append("lead-day grids differ")
    if not np.allclose(
        control["arrays"]["figure7_truth_streamfunction"],
        response["arrays"]["figure7_truth_streamfunction"],
    ):
        problems.append("figure7 MITgcm truth streamfunction differs between arms")
    if not np.allclose(control["anomaly"]["figure7_truth"], response["anomaly"]["figure7_truth"]):
        problems.append("figure7a MITgcm truth anomaly differs between arms")
    if not np.allclose(
        control["anomaly"]["reference_time_mean_streamfunction"],
        response["anomaly"]["reference_time_mean_streamfunction"],
    ):
        problems.append("the removed S0 time-mean streamfunction differs between arms")
    if problems:
        raise CompareForwardError("control and response are not comparable: " + "; ".join(problems))


# ===========================================================================
# Figure 1 --- RMSE, 0-2000 days, both arms over persistence/climatology
# ===========================================================================


def figure_rmse_long(output: Path, control: dict[str, Any], response: dict[str, Any]) -> None:
    base_plots._style()
    leads = np.asarray(control["arrays"]["lead_days"], dtype=float)
    figure, axes = plt.subplots(3, 1, figsize=(6.2, 9.6), sharex=True, constrained_layout=True)
    for axis, field in zip(axes, base_plots.RMSE_FIELDS):
        for method, style in BASELINE_STYLE.items():
            summary = base_plots.percentile_curve(control["arrays"][f"rmse__{method}__{field}"])
            axis.plot(leads, summary["mean"], color=style["color"], linewidth=1.4, label=style["label"])
            axis.fill_between(leads, summary["p10"], summary["p90"], color=style["color"], alpha=0.12, linewidth=0)
        for arm_name, arm in (("control", control), ("response", response)):
            style = ARM_STYLE[arm_name]
            summary = base_plots.percentile_curve(arm["arrays"][f"rmse__model__{field}"])
            axis.plot(leads, summary["mean"], color=style["color"], linewidth=1.7, label=style["label"])
            axis.fill_between(leads, summary["p10"], summary["p90"], color=style["color"], alpha=0.15, linewidth=0)
        axis.set_ylabel(base_plots.FIELD_LABELS[field])
        axis.grid(color="0.85", linewidth=0.6)
    axes[0].set_title(
        r"$\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days; 15 inference initial conditions; "
        f"seed {control['seed']}"
    )
    axes[-1].set_xlabel("Time (days)")
    axes[-1].set_xlim(0, 2000)
    axes[-1].legend(loc="best", ncol=2, fontsize=7)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# Figure 2 --- streamfunction, day 60 / day 2,000, truth | control | response
# ===========================================================================


def figure_streamfunction_maps(output: Path, control: dict[str, Any], response: dict[str, Any]) -> None:
    base_plots._style()
    truth = np.asarray(control["arrays"]["figure7_truth_streamfunction"])
    control_field = np.asarray(control["arrays"]["figure7_model_streamfunction"])
    response_field = np.asarray(response["arrays"]["figure7_model_streamfunction"])
    wet = control["arrays"]["wet_mask"].astype(bool)
    longitude, latitude = control["arrays"]["longitude_deg"], control["arrays"]["latitude_deg"]

    bound = base_plots._finite_bound((truth, control_field, response_field))
    columns = (("MITgcm ground truth", truth), ("Control", control_field), ("Response", response_field))

    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.9), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for row, lead in enumerate(base_plots.FIGURE_7_LEADS):
        for column, (_, field) in enumerate(columns):
            image = axes[row, column].pcolormesh(
                longitude, latitude, base_plots._masked(field[row], wet),
                cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_facecolor("0.86")
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
    for column, (title, _) in enumerate(columns):
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("Longitude (°)")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Barotropic streamfunction (Sv)", shrink=0.84)
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days; " f"seed {control['seed']}"
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# Figure 3 --- streamfunction anomaly, day 60 / day 2,000, + truth-minus-arm
# ===========================================================================


def figure_streamfunction_anomaly_maps(output: Path, control: dict[str, Any], response: dict[str, Any]) -> None:
    base_plots._style()
    truth = np.asarray(control["anomaly"]["figure7_truth"])
    control_field = np.asarray(control["anomaly"]["figure7_model"])
    response_field = np.asarray(response["anomaly"]["figure7_model"])
    wet = control["anomaly"]["wet_mask"].astype(bool)
    longitude, latitude = control["anomaly"]["longitude_deg"], control["anomaly"]["latitude_deg"]

    diff_control = truth - control_field
    diff_response = truth - response_field
    state_bound = base_plots._finite_bound((truth, control_field, response_field))
    diff_bound = base_plots._finite_bound((diff_control, diff_response))

    state_columns = (("MITgcm $\\psi'$", truth), ("Control $\\psi'$", control_field), ("Response $\\psi'$", response_field))
    diff_columns = (("Truth − control", diff_control), ("Truth − response", diff_response))

    figure, axes = plt.subplots(2, 5, figsize=(16.2, 7.0), sharex=True, sharey=True, constrained_layout=True)
    state_image = diff_image = None
    for row, lead in enumerate(base_plots.FIGURE_7_LEADS):
        for column, (_, field) in enumerate(state_columns):
            state_image = axes[row, column].pcolormesh(
                longitude, latitude, base_plots._masked(field[row], wet),
                cmap="RdBu_r", vmin=-state_bound, vmax=state_bound, shading="auto",
            )
        for column, (_, field) in enumerate(diff_columns):
            diff_image = axes[row, 3 + column].pcolormesh(
                longitude, latitude, base_plots._masked(field[row], wet),
                cmap="RdBu_r", vmin=-diff_bound, vmax=diff_bound, shading="auto",
            )
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
    for column, (title, _) in enumerate(state_columns):
        axes[0, column].set_title(title)
    for column, (title, _) in enumerate(diff_columns):
        axes[0, 3 + column].set_title(title)
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    for column in range(5):
        axes[-1, column].set_xlabel("Longitude (°)")
    figure.colorbar(state_image, ax=axes[:, :3].ravel().tolist(), label="Streamfunction anomaly (Sv)", shrink=0.75)
    figure.colorbar(diff_image, ax=axes[:, 3:].ravel().tolist(), label="Truth − model (Sv)", shrink=0.75)
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; $\psi' = \psi - \overline{\psi}_{S0}$, "
        f"the same MITgcm training mean removed everywhere; seed {control['seed']}"
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# Figure 4 --- day-2000 RMSE relative to climatology (the ratio the long-
# horizon gate never scores; see docs on gate coverage 90-360 days)
# ===========================================================================


def figure_day2000_climatology_ratio(output: Path, control: dict[str, Any], response: dict[str, Any]) -> None:
    base_plots._style()
    fields = base_plots.RMSE_FIELDS
    ratios = {"control": [], "response": []}
    for field in fields:
        for arm_name, arm in (("control", control), ("response", response)):
            rmse = arm["figures_report"]["summary"]["rmse"][field]
            ratios[arm_name].append(rmse["model"]["day2000_mean"] / rmse["climatology"]["day2000_mean"])

    x = np.arange(len(fields))
    width = 0.36
    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for offset, arm_name in zip((-width / 2, width / 2), ("control", "response")):
        style = ARM_STYLE[arm_name]
        bars = axis.bar(x + offset, ratios[arm_name], width, color=style["color"], label=style["label"])
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axis.axhline(1.0, color="0.3", linewidth=1.0, linestyle="--")
    axis.text(x[-1] + width, 1.0, "= climatology skill", va="center", ha="left", fontsize=7, color="0.3")
    axis.set_xticks(x, [base_plots.FIELD_LABELS[f] for f in fields], fontsize=7.5)
    axis.set_ylabel("Day-2000 RMSE(model) / RMSE(climatology)")
    axis.set_title(f"Below 1.0 beats climatology at day 2,000; seed {control['seed']}, S0")
    axis.grid(axis="y", color="0.88", linewidth=0.6)
    axis.legend(loc="best")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# Figure 5 --- paired per-member day-2000 RMSE (phihyd_surface)
# ===========================================================================


def figure_day2000_member_spread(output: Path, control: dict[str, Any], response: dict[str, Any]) -> None:
    base_plots._style()
    field = "phihyd_surface"
    lead_days = np.asarray(control["arrays"]["lead_days"])
    index = int(np.where(lead_days == 2000)[0][0])
    control_values = control["arrays"][f"rmse__model__{field}"][:, index]
    response_values = response["arrays"][f"rmse__model__{field}"][:, index]
    starts = control["arrays"]["start_draw_order"]

    order = np.argsort(response_values - control_values)[::-1]
    labels = [f"start {int(s)}" for s in starts[order]]
    y = np.arange(len(order))

    climatology_day2000 = control["figures_report"]["summary"]["rmse"][field]["climatology"]["day2000_mean"]
    persistence_day2000 = control["figures_report"]["summary"]["rmse"][field]["persistence"]["day2000_mean"]

    figure, axis = plt.subplots(figsize=(6.0, 5.6), constrained_layout=True)
    for i, idx in enumerate(order):
        axis.plot(
            [control_values[idx], response_values[idx]], [i, i],
            color="0.75", linewidth=1.0, zorder=1,
        )
    axis.scatter(control_values[order], y, color=ARM_STYLE["control"]["color"], label=ARM_STYLE["control"]["label"], zorder=2, s=28)
    axis.scatter(response_values[order], y, color=ARM_STYLE["response"]["color"], label=ARM_STYLE["response"]["label"], zorder=2, s=28)
    axis.axvline(climatology_day2000, color=BASELINE_STYLE["climatology"]["color"], linestyle="--", linewidth=1.0, label="Climatology")
    axis.axvline(persistence_day2000, color=BASELINE_STYLE["persistence"]["color"], linestyle="--", linewidth=1.0, label="Persistence")
    axis.set_yticks(y, labels, fontsize=7)
    axis.set_xlabel(f"Day-2000 {base_plots.FIELD_LABELS[field]}")
    axis.set_title(f"Per-member day-2000 RMSE, paired by initial condition; seed {control['seed']}, S0")
    axis.grid(axis="x", color="0.88", linewidth=0.6)
    axis.legend(loc="lower right", fontsize=7.5)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# Figure 6 --- day-2000 streamfunction-anomaly structure diagnostics
# ===========================================================================


def figure_day2000_structure(output: Path, control: dict[str, Any], response: dict[str, Any]) -> None:
    base_plots._style()
    control_var = control["anomaly_report"]["variability"]["figure7"]["2000"]
    response_var = response["anomaly_report"]["variability"]["figure7"]["2000"]
    control_struct = control["anomaly_report"]["day2000_structure"]
    response_struct = response["anomaly_report"]["day2000_structure"]

    categories = ("Truth", "Control", "Response")
    colors = ("0.4", ARM_STYLE["control"]["color"], ARM_STYLE["response"]["color"])

    rms = [control_var["truth_anomaly_rms_sv"], control_var["model_anomaly_rms_sv"], response_var["model_anomaly_rms_sv"]]
    boundary_ratio = [
        control_struct["western_first_4_wet_cells"]["truth_boundary_to_interior_rms_ratio"],
        control_struct["western_first_4_wet_cells"]["model_boundary_to_interior_rms_ratio"],
        response_struct["western_first_4_wet_cells"]["model_boundary_to_interior_rms_ratio"],
    ]
    zonal = [
        control_struct["hann_directional_power_fraction_above_0p2_cycles_per_cell"]["truth_zonal"],
        control_struct["hann_directional_power_fraction_above_0p2_cycles_per_cell"]["model_zonal"],
        response_struct["hann_directional_power_fraction_above_0p2_cycles_per_cell"]["model_zonal"],
    ]
    meridional = [
        control_struct["hann_directional_power_fraction_above_0p2_cycles_per_cell"]["truth_meridional"],
        control_struct["hann_directional_power_fraction_above_0p2_cycles_per_cell"]["model_meridional"],
        response_struct["hann_directional_power_fraction_above_0p2_cycles_per_cell"]["model_meridional"],
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)
    x = np.arange(3)

    axes[0].bar(x, rms, color=colors)
    axes[0].set_xticks(x, categories)
    axes[0].set_ylabel("Day-2000 anomaly RMS (Sv)")
    axes[0].set_title("Whole-basin $\\psi'$ amplitude")

    axes[1].bar(x, boundary_ratio, color=colors)
    axes[1].set_xticks(x, categories)
    axes[1].set_ylabel("Western boundary / interior RMS ratio")
    axes[1].set_title("Boundary-current concentration")

    width = 0.36
    axes[2].bar(x - width / 2, zonal, width, color="tab:cyan", label="Zonal")
    axes[2].bar(x + width / 2, meridional, width, color="tab:pink", label="Meridional")
    axes[2].set_xticks(x, categories)
    axes[2].set_ylabel("Power fraction $>0.2$ cycles/cell")
    axes[2].set_title("High-wavenumber content")
    axes[2].legend(loc="best", fontsize=7.5)

    for axis in axes:
        axis.grid(axis="y", color="0.88", linewidth=0.6)
    figure.suptitle(f"Day-2,000 streamfunction-anomaly structure; seed {control['seed']}, S0")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# Figure 7 --- short-horizon (0-200 day) ACC, control vs response
# ===========================================================================


def figure_short_horizon_acc(output: Path, control: dict[str, Any], response: dict[str, Any]) -> None:
    base_plots._style()
    leads = np.asarray(control["arrays"]["short_lead_days"], dtype=float)
    figure, axes = plt.subplots(4, 1, figsize=(5.6, 10.6), sharex=True, constrained_layout=True)
    for axis, field in zip(axes, base_plots.ACC_FIELDS):
        for arm_name, arm in (("control", control), ("response", response)):
            style = ARM_STYLE[arm_name]
            curve = base_plots.percentile_curve(arm["arrays"][f"acc__model__{field}"])
            axis.plot(leads, curve["mean"], color=style["color"], linewidth=1.6, label=style["label"])
            axis.fill_between(leads, curve["p10"], curve["p90"], color=style["color"], alpha=0.17, linewidth=0)
        axis.axhline(0.0, color="0.65", linewidth=0.6)
        axis.set_ylim(-1.0, 1.02)
        axis.set_ylabel(f"{base_plots.FIELD_LABELS[field]}\nACC")
        axis.grid(color="0.85", linewidth=0.6)
    axes[0].set_title(f"S0 anomaly correlation; 15 members; $\\Delta t=10$ days; seed {control['seed']}")
    axes[-1].set_xlabel("Time (days)")
    axes[-1].set_xlim(0, 200)
    axes[-1].legend(loc="best")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# Orchestration
# ===========================================================================

FIGURES = (
    ("model_c_bire_compare_seed{seed}_rmse_0_2000_days_s0.png", figure_rmse_long,
     "figure8-style: RMSE 0-2000d, persistence/climatology/control/response, 3 fields"),
    ("model_c_bire_compare_seed{seed}_streamfunction_day060_day2000_s0.png", figure_streamfunction_maps,
     "figure7-style: truth | control | response streamfunction at day 60 and day 2000"),
    ("model_c_bire_compare_seed{seed}_streamfunction_anomaly_day060_day2000_s0.png", figure_streamfunction_anomaly_maps,
     "figure7a-style: truth psi' | control psi' | response psi' + truth-minus-arm, day 60 and day 2000"),
    ("model_c_bire_compare_seed{seed}_day2000_climatology_ratio_s0.png", figure_day2000_climatology_ratio,
     "day-2000 RMSE(model)/RMSE(climatology), the ratio the long-horizon gate never scores"),
    ("model_c_bire_compare_seed{seed}_day2000_member_spread_s0.png", figure_day2000_member_spread,
     "paired per-member day-2000 phihyd_surface RMSE, control vs response"),
    ("model_c_bire_compare_seed{seed}_day2000_structure_diagnostics_s0.png", figure_day2000_structure,
     "day-2000 psi' amplitude, western-boundary/interior ratio, directional high-k power fraction"),
    ("model_c_bire_compare_seed{seed}_short_horizon_acc_s0.png", figure_short_horizon_acc,
     "figure6-style: ACC 0-200d, control vs response, 4 fields"),
)


def run(seed: int, output_dir: Path, *, force: bool) -> dict[str, Any]:
    control = load_arm(CONTROL_VERSION, seed)
    response = load_arm(RESPONSE_VERSION, seed)
    check_shared_contract(control, response)

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise CompareForwardError(f"{output_dir} already exists and is non-empty; pass --force to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name_template, function, description in FIGURES:
        name = name_template.format(seed=seed)
        path = output_dir / name
        function(path, control, response)
        written.append({"file": name, "content": description})
        print(f"[compare-forward] wrote {path}")

    manifest = {
        "version": f"model_c_adjoint_faithful_control_vs_response_v1_seed_{seed}_s0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime": REGIME,
        "seed": seed,
        "note": (
            "pure re-plot of the already-frozen per-arm figures_v1/anomaly_v1 packages; "
            "no inference or checkpoint access performed by this script"
        ),
        "arms": {
            "control": {
                "training_version": control["version"],
                "checkpoint_sha256": control["checkpoint_sha256"],
                "optimizer_step": control["optimizer_step"],
                "sources": control["sources"],
            },
            "response": {
                "training_version": response["version"],
                "checkpoint_sha256": response["checkpoint_sha256"],
                "optimizer_step": response["optimizer_step"],
                "sources": response["sources"],
            },
        },
        "figures": written,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    readme = (
        f"# Control vs response forward comparison --- seed {seed}, {REGIME}\n\n"
        f"`{CONTROL_VERSION}` vs `{RESPONSE_VERSION}`, both seed {seed}. The two "
        "arms are identical except the response arm trains jointly with the "
        "signed forward-perturbation-response loss (lambda_resp=1e-3 vs 0); see "
        "`config/model_c_adjoint_faithful_response_v1.json` for the exact delta.\n\n"
        "Every figure here re-plots arrays and report scalars already written by "
        "each arm's frozen `*_s0_figures_v1` / `*_s0_anomaly_v1` package "
        "(`scripts/compare_forward_control_response_v1.py`). No new inference.\n\n"
        "## Figures\n\n"
        + "\n".join(f"- `{item['file']}` --- {item['content']}" for item in written)
        + "\n\nSee `manifest.json` for exact source paths and checkpoint hashes.\n"
    )
    (output_dir / "README.md").write_text(readme)

    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to outputs/af_fno/adjoint/C/model_c_adjoint_faithful_control_vs_response_v1_seed_<seed>_s0",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing non-empty output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output_dir = arguments.output_dir or (
        _ROOT / "outputs" / "af_fno" / "adjoint" / "C"
        / f"model_c_adjoint_faithful_control_vs_response_v1_seed_{arguments.seed}_s0"
    )
    manifest = run(arguments.seed, output_dir, force=arguments.force)
    print(json.dumps({"output_dir": str(output_dir), "figures": [f["file"] for f in manifest["figures"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
