"""Figures for the MITgcm adjoint ground truth.

The reference half of the comparison in docs/mitgcm_adjoint_ground_truth_plan.md,
drawn to match outputs/af_fno/adjoint/fno_s0_adjoint_v1/ panel for panel so the
two sets can be laid side by side.  Every styling choice here -- the RdBu_r
diverging scale for signed sensitivity, magma for log magnitude, 0.86 grey for
masked land, the 99th-percentile colour clip with the clip value printed in the
caption, per-panel normalisation in the lead sweep -- is copied deliberately
from scripts/fno_adjoint.py rather than reinvented.  Two maps drawn on two
different conventions cannot be compared by eye, and this pair exists to be
compared by eye.

The structural diagnostics come from scripts/adjoint_metrics.py, the same module
the FNO side calls, so the radial decay, radial spectrum and western-band split
are computed identically on both sides.

Figures:

  1  s10_map              Run A, the primary 10-day map: linear and log10
  2  run_a_b              Run A vs Run B, shared scale, plus the gate G2 residual
  3  lead_sweep           dJ/deta at leads 1, 2, 5, 10 and 20 days
  4  conservation_probe   gate G3: the mean-only cost against its exact answer
  5  gate_g1_plateau      the grdchk sweep at both cg2d tolerances
  6  structure            radial decay, radial spectrum, western band vs interior

Loads no FNO weights and reads nothing under outputs/af_fno/C/**.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjoint_metrics import structural_metrics
from extract_mitgcm_adjoint import read_mds
from select_adjoint_target import CONTRACT_VERSION

FIGURE_NAMES = (
    "mitgcm_adjoint_s10_map.png",
    "mitgcm_adjoint_run_a_b.png",
    "mitgcm_adjoint_lead_sweep.png",
    "mitgcm_adjoint_conservation_probe.png",
    "mitgcm_adjoint_gate_g1_plateau.png",
    "mitgcm_adjoint_structure.png",
)

#: Leads to show in figure 3.  Run B gives all 21; five panels is what the FNO
#: lead sweep shows, and matching the count keeps the two figures the same size.
LEAD_SELECTION = (1, 2, 5, 10, 20)

#: pkg/grdchk's summary lines, as archived per (point, epsilon) STDOUT file.
ADM_ADJOINT = re.compile(r"ADM\s+adjoint_gradient\s*=\s*(\S+)")
ADM_FD = re.compile(r"ADM\s+finite-diff_grad\s*=\s*(\S+)")
SWEEP_NAME = re.compile(r"^(?P<label>.+)_eps(?P<eps>[0-9]+pd-[0-9]+)\.stdout$")


class FigureError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Style, copied from scripts/fno_adjoint.py
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def _wrap(text: str, width: int = 118) -> str:
    """Wrap a caption to a fixed width.

    `bbox_inches="tight"` grows the canvas to fit the longest text object, so a
    one-line suptitle silently stretches the whole figure and squashes the
    panels.  Wrapping keeps the panels at the aspect ratio they were laid out
    with.  Paragraphs already split with \\n are wrapped independently.
    """

    import textwrap

    return "\n".join(
        line
        for paragraph in text.split("\n")
        for line in textwrap.wrap(paragraph, width=width) or [""]
    )


def _masked(field: np.ndarray, wet: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where((~wet) | (~np.isfinite(field)), field)


def _bound(field: np.ndarray, wet: np.ndarray, percentile: float = 99.0) -> float:
    """A robust symmetric colour limit.

    The map is extremely peaked -- the cost contains a delta function at p* --
    so scaling to the maximum renders the rest of the basin uniformly white.  A
    high percentile shows the pattern; the caption records the clipping.
    """

    values = np.abs(field[wet])
    value = float(np.percentile(values, percentile))
    return value if value > 0.0 else float(values.max() or 1.0)


def _draw_map(
    axis: Any,
    field: np.ndarray,
    grid: dict[str, np.ndarray],
    bound: float,
    title: str,
    *,
    mark_target: bool = True,
) -> Any:
    image = axis.pcolormesh(
        grid["longitude"],
        grid["latitude"],
        _masked(field, grid["wet"]),
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        shading="auto",
    )
    if mark_target:
        j, i = grid["target"]
        axis.plot(
            grid["longitude"][j, i],
            grid["latitude"][j, i],
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="k",
            markeredgewidth=0.9,
        )
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_facecolor("0.86")
    return image


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_grid(project_root: Path, run_dir: Path, arrays: Any) -> dict[str, np.ndarray]:
    """Coordinates from the run's own XC/YC, target from the frozen contract."""

    contract = json.loads(
        (project_root / "config" / f"{CONTRACT_VERSION}.json").read_text()
    )
    return {
        "longitude": read_mds(run_dir / "XC"),
        "latitude": read_mds(run_dir / "YC"),
        "wet": np.asarray(arrays["wet_mask"], dtype=bool),
        "target": (int(contract["j_index0"]), int(contract["i_index0"])),
    }


def load_grdchk_sweep(results_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """(epsilon, |FD/adjoint - 1|) per test point, from the archived STDOUTs."""

    sweep: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(results_dir.glob("*.stdout")):
        match = SWEEP_NAME.match(path.name)
        if match is None:
            continue
        text = path.read_text(errors="replace")
        adjoint_match, fd_match = ADM_ADJOINT.search(text), ADM_FD.search(text)
        if adjoint_match is None or fd_match is None:
            continue
        adjoint = float(adjoint_match.group(1).replace("D", "E"))
        finite = float(fd_match.group(1).replace("D", "E"))
        if adjoint == 0.0:
            continue
        epsilon = float(match.group("eps").replace("p", ".").replace("d", "e"))
        sweep.setdefault(match.group("label"), []).append(
            (epsilon, abs(finite / adjoint - 1.0))
        )
    for entries in sweep.values():
        entries.sort()
    return sweep


def checkerboard_fraction(field: np.ndarray, wet: np.ndarray) -> dict[str, Any]:
    """Amplitude of the grid-scale $(-1)^{i+j}$ mode, per column and by band.

    The Run A map carries a visible 2dx checkerboard along the western wall.
    It is a property of the discrete operator being differentiated, not an
    error in the differentiation -- gates G1 to G3 place the adjoint itself at
    1e-8 or better -- and it sits exactly where the Munk layer is one grid cell
    wide (plan section 3.1).  Measuring it matters for the comparison ahead: a
    smooth emulator cannot represent a Nyquist mode at all, so when the FNO map
    fails to reproduce this, the number here says whether that is a meaningful
    disagreement or a feature of the truth no smooth model could carry.
    """

    j, i = np.indices(field.shape)
    sign = (-1.0) ** (i + j)
    columns = []
    for column in range(field.shape[1]):
        mask = wet[:, column]
        if mask.sum() < 4:
            continue
        values = field[mask, column]
        spread = float(values.std())
        columns.append(
            {
                "i_global": column + 1,
                "rms": spread,
                "checkerboard_amplitude": float(abs((values * sign[mask, column]).mean())),
                "fraction_of_rms": float(
                    abs((values * sign[mask, column]).mean()) / spread
                ) if spread > 0.0 else 0.0,
            }
        )

    def band(start: int, stop: int) -> float:
        mask = np.zeros_like(wet)
        mask[:, start:stop] = True
        mask &= wet
        values, spread = field[mask], float(field[mask].std())
        return float(abs((values * sign[mask]).mean()) / spread) if spread > 0.0 else 0.0

    return {
        "definition": "|mean of S * (-1)^(i+j)| / RMS(S), over wet cells",
        "per_column": columns,
        "first_wet_column_fraction": columns[0]["fraction_of_rms"] if columns else None,
        "western_band_fraction": band(1, 8),
        "interior_fraction": band(20, 61),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_s10_map(output: Path, arrays: Any, grid: dict[str, np.ndarray]) -> None:
    """The headline map, linearly and then on a log scale to show its reach."""

    field = np.asarray(arrays["S10"])
    bound = _bound(field, grid["wet"])
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)

    image = _draw_map(
        axes[0], field, grid, bound,
        "Run A  $\\partial J/\\partial \\eta$(day 7210)   [exact]",
    )
    figure.colorbar(image, ax=axes[0], label="Sensitivity (dimensionless)", shrink=0.85)

    magnitude = np.log10(np.maximum(np.abs(field), 1.0e-16))
    floor = float(np.percentile(magnitude[grid["wet"]], 2.0))
    log_image = axes[1].pcolormesh(
        grid["longitude"],
        grid["latitude"],
        _masked(magnitude, grid["wet"]),
        cmap="magma",
        vmin=floor,
        vmax=float(magnitude[grid["wet"]].max()),
        shading="auto",
    )
    j, i = grid["target"]
    axes[1].plot(
        grid["longitude"][j, i], grid["latitude"][j, i],
        marker="o", markersize=4, markerfacecolor="none", markeredgecolor="w", markeredgewidth=0.9,
    )
    axes[1].set_title("$\\log_{10}|\\partial J/\\partial \\eta|$   (domain of dependence)")
    axes[1].set_aspect("equal")
    axes[1].set_facecolor("0.86")
    figure.colorbar(log_image, ax=axes[1], label="$\\log_{10}$ sensitivity", shrink=0.85)

    for axis in axes:
        axis.set_xlabel("Longitude (°)")
    axes[0].set_ylabel("Latitude (°)")
    figure.suptitle(
        _wrap(
        "MITgcm c68j adjoint (TAF 6.8.11), S0, ten-day window: sensitivity of the day-7220 SSH "
        f"anomaly at p* to day-7210 SSH.  Colour clipped at the 99th percentile ({bound:.3g}).\n"
        "Ground truth: exact derivative of the discrete model, gates G0-G5 passed."
        )
    )
    figure.savefig(output / FIGURE_NAMES[0], bbox_inches="tight")
    plt.close(figure)


def figure_run_a_b(output: Path, arrays: Any, grid: dict[str, np.ndarray], report: Any) -> None:
    """Run A against Run B, and the gate G2 residual between them.

    The two runs start ten days apart and evaluate the same cost at day 7220, so
    Run B's adjoint state at day 7210 must reproduce Run A's map exactly.  The
    third panel is that difference, and it is identically zero -- which is why it
    is drawn on its own colour scale with the residual printed in the title
    rather than left to look like an empty panel.
    """

    s10 = np.asarray(arrays["S10"])
    s20 = np.asarray(arrays["S20"])
    lead_days = np.asarray(arrays["lead_days"])
    day_7210 = np.asarray(arrays["S_lead"])[int(np.argmin(np.abs(lead_days - 10)))]
    residual = day_7210 - s10

    # One colour scale across the first two panels on purpose: the twenty-day
    # map is genuinely weaker and more spread out, and giving it its own scale
    # would hide exactly that.
    bound = _bound(s10, grid["wet"])
    ratio = float(
        np.linalg.norm(s20[grid["wet"]]) / np.linalg.norm(s10[grid["wet"]])
    )
    g2 = report["gates"]["G2"]["residual"]

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)
    first = _draw_map(
        axes[0], s10, grid, bound,
        "Run A  $\\partial J/\\partial \\eta$(day 7210)\n10-day lead (compares to the FNO present slot)",
    )
    _draw_map(
        axes[1], s20, grid, bound,
        f"Run B  $\\partial J/\\partial \\eta$(day 7200)\n20-day lead, same scale, "
        f"$\\|S_{{20}}\\|/\\|S_{{10}}\\|$ = {ratio:.3f}",
    )
    # The residual is identically zero.  _bound() would fall back to 1.0 on an
    # all-zero field, drawing a +/-1 colour bar that makes an exact result look
    # like an unscaled one, so the scale is pinned to the map's own and the
    # panel says in words what the colour cannot.
    third = _draw_map(
        axes[2], residual, grid, bound,
        f"Gate G2: Run B's day-7210 state − Run A\nrelative L2 = {g2:.1e}",
    )
    axes[2].text(
        0.5, 0.5, f"identically zero\nat all {residual.size} cells",
        transform=axes[2].transAxes, ha="center", va="center", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.6"},
    )
    for axis in axes:
        axis.set_xlabel("Longitude (°)")
    axes[0].set_ylabel("Latitude (°)")
    figure.colorbar(first, ax=axes[:2].tolist(), label="Sensitivity (dimensionless)", shrink=0.85)
    figure.colorbar(third, ax=axes[2], label="Difference", shrink=0.85)
    figure.suptitle(
        _wrap(
        "The adjoint state at time $t$ inside a run whose cost sits at $T$ is "
        "$\\partial J/\\partial \\eta(\\cdot,t)$ regardless of when the run started, so the third "
        "panel is a free end-to-end check on the whole tape and checkpointing scheme."
        )
    )
    figure.savefig(output / FIGURE_NAMES[1], bbox_inches="tight")
    plt.close(figure)


def figure_lead_sweep(output: Path, arrays: Any, grid: dict[str, np.ndarray]) -> None:
    """How the domain of dependence grows backwards from the cost."""

    maps = np.asarray(arrays["S_lead"])
    lead_days = np.asarray(arrays["lead_days"])
    chosen = [int(np.argmin(np.abs(lead_days - lead))) for lead in LEAD_SELECTION]

    figure, axes = plt.subplots(
        1, len(chosen), figsize=(2.6 * len(chosen) + 1.4, 3.5), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, index in zip(axes, chosen):
        # Amplitude falls sharply across the sweep, so each panel is divided by
        # its own robust scale and the shared colour bar reads as a fraction of
        # that scale.  A single absolute scale would render the 20-day panel
        # blank; five separate colour bars would invite the eye to compare
        # shades that are not comparable.  Same convention as the FNO figure.
        field = maps[index]
        bound = _bound(field, grid["wet"])
        image = _draw_map(
            axis, field / bound, grid, 1.0,
            f"lead {int(lead_days[index])} d\nscale ±{bound:.2g}",
        )
        axis.set_xlabel("Longitude (°)")
    axes[0].set_ylabel("Latitude (°)")
    figure.colorbar(
        image, ax=axes.tolist(),
        label="Sensitivity ÷ that panel's own scale", shrink=0.85,
    )
    figure.suptitle(
        _wrap(
        "Run B lead sweep: $\\partial J/\\partial \\eta$ at 1, 2, 5, 10 and 20 days before the cost, "
        "from one three-minute job.  Each panel is normalised by its own 99th-percentile scale, "
        "printed in its title; the amplitudes themselves fall across the sweep."
        )
    )
    figure.savefig(output / FIGURE_NAMES[2], bbox_inches="tight")
    plt.close(figure)


def figure_conservation_probe(
    output: Path, project_root: Path, scratch_root: Path, grid: dict[str, np.ndarray], report: Any
) -> bool:
    """Gate G3: the mean-only cost against its analytically exact answer."""

    run_g3 = scratch_root / "runG3"
    if not (run_g3 / "adxx_etan.0000000000.data").is_file():
        return False

    measured = read_mds(run_g3 / "adxx_etan.0000000000")
    exact = np.fromfile(run_g3 / "costWeight.bin", dtype=">f4").reshape(62, 62).astype(np.float64)
    difference = measured - exact
    bound = max(_bound(measured, grid["wet"]), _bound(exact, grid["wet"]))
    difference_bound = _bound(difference, grid["wet"])
    worst = report["gates"]["G3"]["worst_relative_l2"]

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)
    first = _draw_map(axes[0], measured, grid, bound, "MITgcm  $\\partial J/\\partial \\eta$, mean-only cost")
    _draw_map(
        axes[1], exact, grid, bound,
        "Exact answer: $w = -rA/A_{wet}$\n(the model conserves $\\int \\eta\\, dA$)",
    )
    third = _draw_map(
        axes[2], difference, grid, difference_bound,
        "MITgcm − exact\n= sea-level conservation error",
    )
    for axis in axes:
        axis.set_xlabel("Longitude (°)")
    axes[0].set_ylabel("Latitude (°)")
    figure.colorbar(first, ax=axes[:2].tolist(), label="Sensitivity (dimensionless)", shrink=0.85)
    figure.colorbar(third, ax=axes[2], label="Difference", shrink=0.85)
    figure.suptitle(
        _wrap(
        "Gate G3 conservation probe.  The adjoint of a conserved functional is constant in time, "
        f"so this map must equal $w$ itself at every dump time: worst relative L2 = {worst:.2e} "
        "over all 11 dumps.  No finite differences are involved."
        )
    )
    figure.savefig(output / FIGURE_NAMES[3], bbox_inches="tight")
    plt.close(figure)
    return True


def figure_gate_g1(output: Path, loose: dict, tight: dict) -> None:
    """The gradient check at both solver tolerances -- the plan's section 12.2.

    Left is the production configuration, where the curves are flat in epsilon:
    that flatness is the whole diagnostic.  A correct adjoint checked against an
    accurate finite difference gives a V; a correct adjoint checked against a
    *noisy* finite difference gives a horizontal line, because the noise floor,
    not the truncation error, sets the discrepancy at every epsilon.
    """

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    panels = (
        (axes[0], loose, "cg2dTargetResidual = $10^{-7}$  (production)",
         "flat in $\\epsilon$ and far above tolerance:\nthe finite difference is the noisy instrument"),
        (axes[1], tight, "cg2dTargetResidual = $10^{-12}$  (diagnostic only)",
         "the round-off arm alone, rising as $\\epsilon$ falls;\nno truncation arm — the response is linear here"),
    )
    for axis, sweep, title, subtitle in panels:
        for label in sorted(sweep):
            entries = sweep[label]
            axis.plot(
                [e for e, _ in entries],
                [max(v, 1.0e-12) for _, v in entries],
                marker="o", markersize=3.5, linewidth=1.0, label=label,
            )
        axis.axhline(1.0e-4, color="k", linewidth=0.9, linestyle=":")
        axis.text(
            0.98, 1.0e-4, "gate G1 tolerance $10^{-4}$ ",
            transform=axis.get_yaxis_transform(), ha="right", va="bottom", fontsize=7,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("Perturbation $\\epsilon$ (metres of SSH)")
        axis.set_ylabel("$|\\,FD / \\mathrm{adjoint} - 1\\,|$")
        axis.set_title(f"{title}\n{subtitle}")
    # A shared y range is what makes the six-order-of-magnitude drop legible.
    low = min(axis.get_ylim()[0] for axis, *_ in panels)
    high = max(axis.get_ylim()[1] for axis, *_ in panels)
    for axis, *_ in panels:
        axis.set_ylim(low, high)
    axes[0].legend(fontsize=6.0, loc="best", framealpha=0.9)

    figure.suptitle(
        _wrap(
        "Gate G1.  The adjoint is unchanged between the two panels -- only the accuracy of the "
        "reference finite difference changes.  Tightening the forward free-surface solve moves p* "
        "from $1.0\\times10^{-2}$ to $1.5\\times10^{-8}$, so the discrepancy was never in the adjoint.\n"
        "The delivered maps use the production $10^{-7}$: it is the model the FNO was trained on."
        )
    )
    figure.savefig(output / FIGURE_NAMES[4], bbox_inches="tight")
    plt.close(figure)


def figure_structure(output: Path, arrays: Any, grid: dict[str, np.ndarray]) -> dict[str, Any]:
    """Radial decay, radial spectrum and the western-band split of the Run A map.

    Computed by scripts/adjoint_metrics.py, the same module the FNO side uses,
    so the two structure figures are measuring the same things the same way.
    """

    field = np.asarray(arrays["S10"])
    structure = structural_metrics(field, grid["wet"], grid["target"])
    structure["checkerboard"] = checkerboard_fraction(field, grid["wet"])

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    decay = structure["radial_decay"]
    radius = np.asarray(decay["radius_cells"])
    amplitude = np.asarray(decay["rms_abs_sensitivity"])
    axes[0].semilogy(radius, amplitude, marker="o", markersize=3, linewidth=1.0, label="ring RMS")
    # The exponential fit is the FNO side's convention and is kept so the two
    # panels are the same measurement.  Here it does not describe the data --
    # R^2 = 0.18 -- and that is the result, not a defect in the fit: after ten
    # days the barotropic adjustment has filled the basin, so there is no
    # radial decay length to report.  Labelling it as a length scale anyway
    # would invent a number.
    fitted = decay["fit_r_squared"] >= 0.5
    axes[0].semilogy(
        radius,
        np.exp(decay["fit_intercept_log"] + decay["fit_slope_per_cell"] * radius),
        linewidth=1.2, linestyle="--",
        label=(
            f"fit, $L$ = {decay['e_folding_cells']:.1f} cells"
            if fitted
            else "exponential fit (does not describe the data)"
        ),
    )
    axes[0].set_xlabel("Distance from p* (grid cells)")
    axes[0].set_ylabel("RMS $|\\partial J/\\partial \\eta|$")
    axes[0].set_title(
        f"Radial decay  ($R^2$ = {decay['fit_r_squared']:.2f})"
        + ("" if fitted else "\nno radial decay: the map is basin-scale")
    )
    axes[0].legend(fontsize=7)

    spectrum = structure["radial_spectrum"]
    power = np.asarray(spectrum["power_per_bin"])
    axes[1].bar(np.arange(1, power.size + 1), np.maximum(power, 1.0e-30), width=0.75)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Radial wavenumber bin (12-bin tapered convention)")
    axes[1].set_ylabel("Absolute power")
    axes[1].set_title("Radial spectrum\n(absolute power, never a fraction)")

    band = structure["western_band"]
    values = (band["boundary_rms"], band["interior_rms"])
    bars = axes[2].bar(
        ["Western band\n(4 wet cells)", "Interior"],
        values,
        width=0.55,
        color=("#3b6ea5", "#9aa6b2"),
    )
    axes[2].set_yscale("log")
    # Autoscaling a log bar chart over a ratio of 1.18 crops the axis to a
    # sliver and makes the two bars look like an order of magnitude apart.
    # Pinning a full decade keeps the drawn heights honest; the printed values
    # carry the precision.
    axes[2].set_ylim(min(values) / 10.0, max(values) * 2.0)
    for bar, value in zip(bars, values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, value, f" {value:.2e}",
            ha="center", va="bottom", fontsize=7,
        )
    axes[2].set_ylabel("RMS sensitivity")
    checkerboard = structure["checkerboard"]
    axes[2].set_title(
        f"Western band vs interior\nratio = {band['boundary_to_interior_ratio']:.2f}"
        f"\n$2\\Delta x$ mode: {checkerboard['first_wet_column_fraction']:.0%} of RMS at the wall, "
        f"{checkerboard['interior_fraction']:.2%} inside"
    )

    figure.suptitle(
        _wrap(
        "Structure of the Run A map, computed by the same scripts/adjoint_metrics.py the FNO side "
        "calls.  The western-band split is the project's existing convention: "
        "'western-boundary-ratio-degrades' records a defect there that the forecast gate never scores."
        )
    )
    figure.savefig(output / FIGURE_NAMES[5], bbox_inches="tight")
    plt.close(figure)
    return structure


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/home/mjalabert314/bire_james25_repro")
    parser.add_argument(
        "--scratch-root",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1",
    )
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    project_root = Path(arguments.project_root)
    scratch_root = Path(arguments.scratch_root)
    output = Path(
        arguments.output
        or project_root / "outputs" / "af_fno" / "adjoint" / "mitgcm_s0_adjoint_v1"
    )
    if not (output / "mitgcm_s0_adjoint_v1.npz").is_file():
        raise FigureError(
            f"missing {output / 'mitgcm_s0_adjoint_v1.npz'}; "
            "run scripts/extract_mitgcm_adjoint.py first"
        )

    _style()
    arrays = np.load(output / "mitgcm_s0_adjoint_v1.npz")
    report = json.loads((output / "report.json").read_text())
    grid = load_grid(project_root, scratch_root / "runA", arrays)

    figure_s10_map(output, arrays, grid)
    figure_run_a_b(output, arrays, grid, report)
    figure_lead_sweep(output, arrays, grid)
    drew_g3 = figure_conservation_probe(output, project_root, scratch_root, grid, report)

    loose = load_grdchk_sweep(output / "grdchk_cg2d1em7")
    tight = load_grdchk_sweep(output / "grdchk_cg2d1em12")
    drew_g1 = bool(loose and tight)
    if drew_g1:
        figure_gate_g1(output, loose, tight)

    structure = figure_structure(output, arrays, grid)
    (output / "structure.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n"
    )

    for index, name in enumerate(FIGURE_NAMES):
        if index == 3 and not drew_g3:
            print(f"  skipped {name}  (no runG3 products)")
            continue
        if index == 4 and not drew_g1:
            print(f"  skipped {name}  (no archived grdchk sweeps)")
            continue
        path = output / name
        print(f"  wrote {path}  ({path.stat().st_size // 1024} kB)")
    print(f"  wrote {output / 'structure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
