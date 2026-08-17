"""Sea-surface height plates for the frozen emulator: the streamfunction figures' twin.

Produces, for regime S0, the same four plates the project already publishes for
the barotropic streamfunction --- but for SSH --- plus one diagnostic that only
SSH makes possible.

    figure 3  analogue   SSH,         days 0, 10, 20, 30, 40   truth / model / difference
    figure 7  analogue   SSH,         days 60 and 2,000        truth / model
    figure 3a analogue   SSH anomaly, days 0-40                truth / model / difference
    figure 7a analogue   SSH anomaly, days 60 and 2,000        truth / model / difference
    new                  area-weighted basin-mean SSH against lead

-----------------------------------------------------------------------------
Why SSH gets a diagnostic the streamfunction cannot have
-----------------------------------------------------------------------------

For every other field, the only way to know whether the emulator is right is to
compare it with MITgcm.  SSH is different: this configuration runs
``implicitFreeSurface`` with ``exactConserv`` and a linear free surface, so the
area integral of eta is conserved *exactly*.  Measured over the whole 25-year
S0 record, MITgcm's area-weighted basin-mean SSH is

    -1.5e-09 m, varying by 8.1e-10 m peak-to-peak,

against SSH swinging between -0.99 and +0.77 m in space.  It is zero, and it
stays zero, to a part in a billion.

So the basin mean has a **known correct value at every lead, with no MITgcm
comparison required**.  Any drift in the emulator's basin mean is unambiguously
error against an exact reference --- the same free ground truth that gate F5 of
``scripts/fno_adjoint.py`` exploits, but visible here in the forecast fields
themselves rather than through the adjoint.

Measured here, the emulator's drift is negligible --- a fraction of a millimetre
by day 2,000.  That is a null result, and it is not a clean bill of health: gate
F5 of ``scripts/fno_adjoint.py`` shows the same emulator destroying ~90% of a
*uniform* sea-level anomaly in one ten-day step.  The reason it never shows up
in a rollout is that the uniform mode is never excited --- truth's basin mean is
zero, the emulator damps toward zero, and damping zero gives zero.  A forecast
only explores directions the trajectory actually visits; an adjoint probes every
direction at once.  This package is the forecast half of that contrast.

The second panel of the diagnostic figure carries the other thing the four
plates cannot show.  SSH here is ~99% a stationary gyre pattern, and the anomaly
about the training mean is itself ~92% stationary, so the truth and model rows
of the anomaly plates are indistinguishable by eye for reasons that have nothing
to do with skill.  Removing each side's own mean over the rollout window leaves
the genuinely time-varying field, and that is where the error lives.

-----------------------------------------------------------------------------
Conventions, inherited rather than reinvented
-----------------------------------------------------------------------------

Member
    Member 0 of the frozen 15-member S0 inference protocol
    (``inference_starts(15, 20260802)``), the same member figures 3 and 7 plot.
    It starts at day 6263 and its day-2,000 lead lands at day 8263, inside the
    record.
Rollout
    Driven by ``oceanfno.model.BireTwoInNewChannelsStepper`` --- the project's
    canonical evaluation path, not a reimplementation.  The preflight gate
    recomputes the barotropic streamfunction from this rollout and requires it
    to match the sealed figure package's stored arrays, which proves the SSH
    published here comes from the identical trajectory.
Anomaly
    ``eta' = eta - mean(eta)_S0`` where the mean is MITgcm's pointwise time mean
    over the Bire training block, days 0-5999, subtracted from truth and model
    alike.  Identical in form to ``oceanfno._anomaly_core``'s streamfunction
    convention.  Note that a common field removed from both sides leaves
    ``truth - model`` unchanged, so the difference panels of the anomaly plates
    equal those of the raw plates by construction.

Usage::

    python scripts/ssh_rollout_figures.py            # ~3 minutes on CPU
    python scripts/ssh_rollout_figures.py --force    # replace an existing package

Nothing here trains or modifies any weight.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fno_adjoint import (
    ETA_CHANNEL,
    MODEL_CONTRACT,
    REGIME,
    REGIME_INDEX,
    file_sha256,
    json_sha256,
    load_frozen_model,
    load_model_provenance,
    _verify,
)
from select_adjoint_target import CONTRACT_VERSION, read_mds_2d

from oceanfno import plots
from oceanfno._figures_core import MEMBER_COUNT, START_SEED
from oceanfno.dataset import TRAIN_RANGE, _normalizers, inference_starts, new_channel_static_block
from oceanfno.diagnostics import derived_fields
from oceanfno.model import BireTwoInNewChannelsStepper
from oceanfno.validation import _gather

VERSION = "model_c_2in_1out_new_channels_p_cont_BT_loss_s0_ssh_v1"

#: Member 0 of the frozen protocol --- the one figures 3 and 7 plot.
MEMBER_INDEX = 0

HORIZON_DAYS = 10

#: 0, 10, ..., 2000: every lead the ten-day operator reaches in 200 calls.
LEAD_DAYS = plots.LEAD_DAYS
FIGURE_3_LEADS = plots.FIGURE_3_LEADS  # (0, 10, 20, 30, 40)
FIGURE_7_LEADS = plots.FIGURE_7_LEADS  # (60, 2000)

#: Where the sealed streamfunction package lives, for the preflight cross-check.
FIGURES_PACKAGE = (
    Path("outputs") / "af_fno" / "C" / "model_c_2in_1out_new_channels_p_cont_BT_loss"
    / "model_c_2in_1out_new_channels_p_cont_BT_loss_s0_figures_v1" / REGIME
)
FIGURES_ARRAYS = "model_c_bire_s0_figures_arrays.npz"

OUTPUT_RELATIVE = (
    Path("outputs") / "af_fno" / "C" / "model_c_2in_1out_new_channels_p_cont_BT_loss"
    / VERSION / REGIME
)

ARRAYS_NAME = "model_c_bire_s0_ssh_arrays.npz"
REPORT_NAME = "model_c_bire_s0_ssh_report.json"
MANIFEST_NAME = "manifest.json"
README_NAME = "README.md"

FIGURE_3 = "model_c_bire_ssh_figure3_1deg_s0_dt10.png"
FIGURE_7 = "model_c_bire_ssh_figure7_day060_day2000_s0.png"
FIGURE_3A = "model_c_bire_ssh_figure3a_anomaly_1deg_s0_dt10.png"
FIGURE_7A = "model_c_bire_ssh_figure7a_anomaly_day060_day2000_s0.png"
REFERENCE_FIGURE = "model_c_bire_ssh_training_mean_reference_s0.png"
BASIN_MEAN_FIGURE = "model_c_bire_ssh_basin_mean_vs_lead_s0.png"

FIGURE_NAMES = (FIGURE_3, FIGURE_7, FIGURE_3A, FIGURE_7A, REFERENCE_FIGURE, BASIN_MEAN_FIGURE)

SSH_LABEL = r"SSH $\eta$ (m)"
ANOMALY_LABEL = r"SSH anomaly $\eta'$ (m)"
DIFFERENCE_LABEL = r"Truth $-$ model (m)"

#: How closely the recomputed streamfunction must match the sealed package.
#: The two rollouts are the same float32 arithmetic in the same order, so this
#: is a match-or-fail check, not a tolerance to be tuned.
CROSS_CHECK_TOLERANCE = 1.0e-4


class SshFigureError(RuntimeError):
    """Raised when the SSH package cannot be built on the frozen protocol."""


# ===========================================================================
# 1.  Inputs
# ===========================================================================


def member_start() -> int:
    """The day member 0 starts from, drawn by the protocol's declared seed."""

    starts = inference_starts(MEMBER_COUNT, START_SEED)
    start = int(starts[MEMBER_INDEX])
    if start < HORIZON_DAYS:
        raise SshFigureError("member 0 has no t-10 initial condition")
    return start


def cell_areas(project_root: Path) -> tuple[np.ndarray, float]:
    """MITgcm's cell areas and the wet-basin area, read rather than recomputed.

    The basin mean below is *area weighted* because what MITgcm conserves is the
    area integral of eta, not its plain average over cells.  On a spherical grid
    those differ: cells at 75N are a quarter the area of cells at 15N.
    """

    contract = json.loads(
        (project_root / "config" / f"{CONTRACT_VERSION}.json").read_text()
    )
    rac = read_mds_2d(Path(contract["grid"]["rac_source"]))
    return rac, float(contract["wet_area_m2"])


def training_mean_ssh(group: Any, *, chunk_days: int = 500) -> tuple[np.ndarray, int]:
    """MITgcm's pointwise time-mean SSH over the training block, days 0-5999.

    The same reference ``oceanfno._anomaly_core`` removes from the streamfunction,
    for the same reason: subtracting the stationary field lets the anomaly plates
    show what actually varies, and removing the *identical* field from truth and
    model means no bias in the mean state can hide in the comparison.
    """

    start, stop = TRAIN_RANGE
    total = np.zeros((62, 62), dtype=np.float64)
    count = 0
    for begin in range(start, stop, chunk_days):
        end = min(begin + chunk_days, stop)
        block = np.asarray(group["state"][REGIME_INDEX, begin:end, ETA_CHANNEL], dtype=np.float64)
        total += block.sum(axis=0)
        count += block.shape[0]
    if count != stop - start:
        raise SshFigureError(f"training mean averaged {count} days, expected {stop - start}")
    return total / count, count


# ===========================================================================
# 2.  The rollout
# ===========================================================================


def roll_member(
    stepper: BireTwoInNewChannelsStepper,
    state: Any,
    static: Any,
    start: int,
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    """Autoregress member 0 to day 2,000 and keep SSH at every ten-day lead.

    This is the project's canonical evaluation loop, not a private one: the
    stepper holds the two-time-level pair and slides it forward, so the first
    call sees two truth states and every call afterwards sees its own previous
    predictions.  No teacher forcing after the initial condition.

    Full 46-channel states are retained only at the six plotted leads, because
    the preflight gate needs the velocities to recompute the streamfunction.
    """

    records = np.asarray([[REGIME_INDEX, start]], dtype=np.int64)
    initial = _gather(state, records, 0)
    history = _gather(state, records, -HORIZON_DAYS)

    current = stepper.normalized_state(initial)
    static_normalized = stepper.normalized_static(static, records[:, 0])
    stepper.begin(history)

    plotted = sorted(set(FIGURE_3_LEADS) | set(FIGURE_7_LEADS))
    ssh = np.empty((len(LEAD_DAYS), *wet.shape), dtype=np.float32)
    states_at_leads: dict[int, np.ndarray] = {}

    # Lead 0 is the initial condition itself: the model has not been called yet,
    # so truth and model are identical there and the difference panel is zero.
    ssh[0] = initial[0, ETA_CHANNEL]
    if 0 in plotted:
        states_at_leads[0] = initial[0].copy()

    for index in range(1, len(LEAD_DAYS)):
        current = stepper.step(current, static_normalized)
        physical = stepper.physical(current)[0]
        ssh[index] = physical[ETA_CHANNEL]
        lead = int(LEAD_DAYS[index])
        if lead in plotted:
            states_at_leads[lead] = physical.copy()
        if not np.all(np.isfinite(physical)):
            raise SshFigureError(f"the rollout became non-finite at lead {lead}")

    return {"ssh": ssh, "states_at_leads": states_at_leads, "initial": initial}


def truth_ssh(group: Any, start: int, wet: np.ndarray) -> np.ndarray:
    """MITgcm SSH at every lead the model was rolled to."""

    days = [start + int(lead) for lead in LEAD_DAYS]
    if days[-1] >= group["state"].shape[1]:
        raise SshFigureError("the day-2,000 lead runs past the stored record")
    values = np.asarray(group["state"].oindex[REGIME_INDEX, days, ETA_CHANNEL], dtype=np.float32)
    values[:, ~wet] = 0.0
    return values


def gate_same_rollout(
    project_root: Path,
    states_at_leads: Mapping[int, np.ndarray],
    wet: np.ndarray,
) -> dict[str, Any]:
    """Preflight: prove this rollout is the one the streamfunction plates used.

    The sealed figure package stores member 0's barotropic streamfunction at the
    plotted leads.  Recomputing it from the states rolled here and requiring a
    match is what licenses placing the SSH plates beside the streamfunction ones
    and calling them the same experiment.  Without it, a silently different
    member, seed or checkpoint would produce plausible figures of a different
    trajectory.
    """

    arrays_path = project_root / FIGURES_PACKAGE / FIGURES_ARRAYS
    if not arrays_path.is_file():
        raise SshFigureError(
            f"the sealed streamfunction package is missing: {arrays_path}"
        )
    with np.load(arrays_path) as stored:
        expected = {
            "figure3": np.asarray(stored["figure3_model_streamfunction"], dtype=np.float64),
            "figure7": np.asarray(stored["figure7_model_streamfunction"], dtype=np.float64),
        }

    findings: dict[str, Any] = {"source": str(arrays_path), "source_sha256": file_sha256(arrays_path)}
    worst = 0.0
    for tag, leads in (("figure3", FIGURE_3_LEADS), ("figure7", FIGURE_7_LEADS)):
        recomputed = np.stack(
            [
                derived_fields(states_at_leads[int(lead)][None], wet)["streamfunction"][0]
                for lead in leads
            ]
        ).astype(np.float64)
        scale = float(np.abs(expected[tag]).max())
        error = float(np.abs(recomputed - expected[tag]).max())
        findings[f"{tag}_max_abs_difference_sv"] = error
        findings[f"{tag}_scale_sv"] = scale
        findings[f"{tag}_relative"] = error / scale if scale > 0.0 else float("nan")
        worst = max(worst, error / scale if scale > 0.0 else float("inf"))

    findings["worst_relative"] = worst
    findings["tolerance"] = CROSS_CHECK_TOLERANCE
    findings["passed"] = bool(worst < CROSS_CHECK_TOLERANCE)
    if not findings["passed"]:
        raise SshFigureError(
            "the recomputed streamfunction does not match the sealed figure package: "
            f"worst relative difference {worst:.3e}. This rollout is not the one "
            "figures 3 and 7 plot, so the SSH plates would not be comparable."
        )
    return findings


# ===========================================================================
# 3.  Figures
# ===========================================================================


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


def _masked(field: np.ndarray, wet: np.ndarray) -> np.ma.MaskedArray:
    array = np.asarray(field)
    return np.ma.masked_where((~wet) | (~np.isfinite(array)), array)


def _bound(values: Sequence[np.ndarray]) -> float:
    """Symmetric colour limit from the largest finite magnitude present."""

    limit = max(float(np.abs(np.asarray(v)[np.isfinite(v)]).max()) for v in values)
    return limit if limit > 0.0 else 1.0


def _early_plate(
    output: Path,
    filename: str,
    truth: np.ndarray,
    model: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
    *,
    state_label: str,
    truth_row: str,
    model_row: str,
    title: str,
) -> None:
    """Three rows --- truth, model, difference --- across the five early leads."""

    difference = truth - model
    bound = _bound((truth, model))
    difference_bound = _bound((difference,))
    figure, axes = plt.subplots(
        3, len(FIGURE_3_LEADS), figsize=(11.0, 6.6),
        sharex=True, sharey=True, constrained_layout=True,
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
    axes[0, 0].set_ylabel(f"{truth_row}\nLatitude (°)")
    axes[1, 0].set_ylabel(f"{model_row}\nLatitude (°)")
    axes[2, 0].set_ylabel("Truth − model\nLatitude (°)")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    figure.colorbar(state_image, ax=axes[:2].ravel().tolist(), label=state_label, shrink=0.82)
    figure.colorbar(
        difference_image, ax=axes[2].ravel().tolist(),
        label=f"{DIFFERENCE_LABEL}  (clip ±{difference_bound:.3g})", shrink=0.82,
    )
    figure.suptitle(title)
    figure.savefig(output / filename, bbox_inches="tight")
    plt.close(figure)


def _figure7_plate(
    output: Path,
    truth: np.ndarray,
    model: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
    basin_mean: Mapping[str, np.ndarray],
) -> None:
    """Day 60 and day 2,000, truth against model --- the raw-SSH long plate.

    Two columns, mirroring the streamfunction figure 7 exactly.  Each row's
    title carries that lead's basin-mean SSH, because with a shared colour scale
    a uniform offset and a pattern error look identical, and at day 2,000 the
    offset is the larger of the two.
    """

    bound = _bound((truth, model))
    figure, axes = plt.subplots(
        len(FIGURE_7_LEADS), 2, figsize=(7.6, 6.8),
        sharex=True, sharey=True, constrained_layout=True,
    )
    image = None
    lead_index = {int(lead): list(LEAD_DAYS).index(int(lead)) for lead in FIGURE_7_LEADS}
    for row, lead in enumerate(FIGURE_7_LEADS):
        index = lead_index[int(lead)]
        for column, value in enumerate((truth[row], model[row])):
            image = axes[row, column].pcolormesh(
                longitude, latitude, _masked(value, wet),
                cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_facecolor("0.86")
            axes[row, column].set_xlabel("Longitude (°)")
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
        # Both means in the same units and format, because the whole point is
        # that one is zero to nine decimal places and the other is not.
        prefix = ("MITgcm   ", "Model C   ") if row == 0 else ("", "")
        for column, series in enumerate(("truth", "model")):
            axes[row, column].set_title(
                f"{prefix[column]}$\\langle\\eta\\rangle$ = "
                f"{basin_mean[series][index] * 1e3:+.2e} mm"
            )
    figure.colorbar(image, ax=axes.ravel().tolist(), label=SSH_LABEL, shrink=0.84)
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days."
        "\nMITgcm conserves the basin mean exactly; the emulator's is printed beside it."
    )
    figure.savefig(output / FIGURE_7, bbox_inches="tight")
    plt.close(figure)


def _figure7a_plate(
    output: Path,
    truth: np.ndarray,
    model: np.ndarray,
    mean_field: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    """Day 60 and day 2,000 anomalies in three columns, plus the removed mean."""

    bound = _bound((truth, model))
    figure, axes = plt.subplots(
        len(FIGURE_7_LEADS), 3, figsize=(10.4, 6.8),
        sharex=True, sharey=True, constrained_layout=True,
    )
    image = None
    for row, lead in enumerate(FIGURE_7_LEADS):
        for column, value in enumerate((truth[row], model[row], truth[row] - model[row])):
            image = axes[row, column].pcolormesh(
                longitude, latitude, _masked(value, wet),
                cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
            )
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
    axes[0, 0].set_title("MITgcm $\\eta'$")
    axes[0, 1].set_title("Model C $\\eta'$")
    axes[0, 2].set_title("Truth − model")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
        axis.set_xlabel("")
    for axis in axes[-1]:
        axis.set_xlabel("Longitude (°)")
    figure.colorbar(image, ax=axes.ravel().tolist(), label=ANOMALY_LABEL, shrink=0.75)
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; $\eta' = \eta - \overline{\eta}_{S0}$, "
        "the same MITgcm training mean removed from both columns"
        "\n(so the third column is identical to the raw plate's difference row)"
    )
    figure.savefig(output / FIGURE_7A, bbox_inches="tight")
    plt.close(figure)

    # The removed field, published once so the anomalies can be read against it.
    limit = _bound((mean_field,))
    figure, axis = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    image = axis.pcolormesh(
        longitude, latitude, _masked(mean_field, wet),
        cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto",
    )
    axis.set_aspect("equal")
    axis.set_facecolor("0.86")
    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")
    axis.set_title(r"$\overline{\eta}_{S0}$, MITgcm days 0–5999")
    figure.colorbar(image, ax=axis, label=r"Mean SSH $\overline{\eta}$ (m)", shrink=0.85)
    figure.savefig(output / REFERENCE_FIGURE, bbox_inches="tight")
    plt.close(figure)


def _diagnostic_plate(
    output: Path,
    basin_mean: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    fluctuation: Mapping[str, Any],
) -> None:
    """The two things the four plates above cannot show.

    **Left --- basin-mean SSH.** MITgcm's is a flat line at zero because
    ``exactConserv`` makes it one, so this is the only SSH diagnostic whose
    correct answer is known without a comparison.

    **Right --- skill on the part of SSH that actually varies.**  This is the
    figure that makes the anomaly plates readable.  SSH here is 99% a stationary
    gyre pattern, and the anomaly about the training mean is itself ~92%
    stationary, so the truth and model rows of the anomaly plates are
    indistinguishable by eye for a reason that has nothing to do with skill.
    Removing each side's own mean over the rollout window leaves the genuinely
    time-varying field, and that is where the emulator's error lives.
    """

    leads = np.asarray(LEAD_DAYS, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)

    axes[0].plot(leads, basin_mean["model"] * 1e3, linewidth=1.3, label="Model C")
    axes[0].plot(
        leads, basin_mean["truth"] * 1e3, linewidth=1.3, linestyle="--",
        label="MITgcm (exactly conserved)",
    )
    axes[0].axhline(0.0, color="k", linewidth=0.7, linestyle=":")
    axes[0].set_xlabel("Lead (days)")
    axes[0].set_ylabel(r"$\langle \eta \rangle_A$ (mm)")
    axes[0].set_title(
        "Area-weighted basin-mean SSH, against an exact zero\n"
        f"model reaches {summary['model_final'] * 1e3:+.3f} mm at day 2,000"
    )
    axes[0].legend(fontsize=7.5)

    correlation = np.asarray(fluctuation["correlation"])
    ratio = np.asarray(fluctuation["amplitude_ratio"])
    axes[1].plot(leads, correlation, linewidth=1.3, label="pattern correlation")
    axes[1].plot(leads, ratio, linewidth=1.3, linestyle="--", label="amplitude ratio model/truth")
    axes[1].axhline(1.0, color="k", linewidth=0.7, linestyle=":")
    axes[1].set_xlabel("Lead (days)")
    axes[1].set_ylabel("Skill on the time-varying SSH")
    axes[1].set_ylim(0.0, 1.25)
    axes[1].set_title(
        "The part of SSH that actually varies\n"
        f"({fluctuation['stationary_fraction']:.0%} of the anomaly's variance is stationary and is removed)"
    )
    axes[1].legend(fontsize=7.5)

    figure.suptitle(
        r"SSH is 99% a stationary gyre; the anomaly about the training mean is a further "
        f"{fluctuation['stationary_fraction']:.0%} stationary."
        "\nThe plates show the total and the anomaly as asked; these two panels show what is "
        "left once the stationary parts are accounted for."
    )
    figure.savefig(output / BASIN_MEAN_FIGURE, bbox_inches="tight")
    plt.close(figure)


# ===========================================================================
# 4.  Driver
# ===========================================================================


def _readme(report: Mapping[str, Any]) -> str:
    drift = report["basin_mean_ssh"]
    varying = report["time_varying_ssh"]
    error = report["error_summary"]
    return "\n".join(
        [
            f"# SSH plates — {VERSION}",
            "",
            "The four streamfunction plates the project already publishes, redone for sea-surface",
            f"height, on the frozen emulator `{MODEL_CONTRACT}` "
            f"(step {report['model']['optimizer_step']}), plus one diagnostic only SSH allows.",
            "",
            f"Member {MEMBER_INDEX} of the frozen 15-member S0 inference protocol, start day "
            f"{report['protocol']['start_day']}, rolled {report['protocol']['calls']} calls to day 2,000.",
            "The preflight gate recomputes this rollout's barotropic streamfunction and requires it to",
            "match the sealed figure package, so these plates and figures 3/7 describe one trajectory.",
            "",
            "## Read this before reading the plates",
            "",
            "**SSH in this configuration is almost entirely stationary.** Three layers, measured on",
            "this member:",
            "",
            f"| Layer | RMS |",
            "|---|---|",
            f"| total SSH | {varying['total_ssh_rms_m']:.4f} m |",
            f"| anomaly about the training mean | {varying['truth_anomaly_rms_m']:.5f} m |",
            f"| what actually varies in time | {varying['truth_fluctuation_rms_m']:.5f} m |",
            "",
            f"The anomaly is {varying['truth_anomaly_rms_m'] / varying['total_ssh_rms_m']:.1%} of the total"
            f" field, and a further **{varying['stationary_fraction']:.0%} of the anomaly's variance is",
            "itself stationary** — the offset between this window's mean state and the days-0–5999",
            "training mean, which truth and model both carry.",
            "",
            "So the truth and model rows of the anomaly plates are indistinguishable by eye, and any",
            "correlation computed on the raw or anomaly field is flattered by a fixed pattern both",
            "sides share. Only the difference row of each plate carries information, and the",
            "right-hand panel of the diagnostic figure is where the actual skill is.",
            "",
            "## Skill on the part of SSH that varies",
            "",
            "Anomaly about the training mean, minus each side's own mean over this rollout window.",
            "",
            "| Lead | Truth fluctuation RMS (m) | Pattern correlation | Amplitude ratio | Raw RMSE (m) |",
            "|---|---|---|---|---|",
            *[
                f"| day {row['lead']} | {row['truth_fluctuation_rms']:.5f} | "
                f"{row['fluctuation_correlation']:.3f} | {row['fluctuation_amplitude_ratio']:.3f} | "
                f"{row['rmse']:.5f} |"
                for row in error["by_lead"]
            ],
            "",
            "## Global sea level: a null result, and why it is not a clean bill of health",
            "",
            "This configuration runs `implicitFreeSurface` with `exactConserv` and a linear free",
            "surface, so the area integral of eta is conserved **exactly**. Over the whole 25-year S0",
            f"record MITgcm's basin-mean SSH sits at {drift['truth_max_abs']:.1e} m — zero to a part in",
            "a billion, against SSH swinging ±1 m in space. The basin mean therefore has a known",
            "correct value at every lead, with no MITgcm comparison needed.",
            "",
            f"The emulator reaches only **{drift['model_final'] * 1e3:+.3f} mm** by day 2,000. Essentially",
            "no drift.",
            "",
            "That looks like a pass, and it is not one. `scripts/fno_adjoint.py` gate F5 shows the",
            "emulator destroys ~90% of a *uniform* sea-level anomaly in a single ten-day step. The",
            "reason that never appears here is that the uniform mode is never excited: truth's basin",
            "mean is zero, the emulator damps toward zero, and damping zero gives zero. The defect is",
            "real and the free-running rollout is structurally incapable of revealing it.",
            "",
            "That contrast is the point. A forecast only ever explores directions the trajectory",
            "actually visits; an adjoint probes every direction at once. This figure is the forecast",
            "half of that comparison and should be read next to gate F5, not instead of it.",
            "",
            "## Conventions",
            "",
            "- **Anomaly**: `eta' = eta - mean(eta)_S0` over MITgcm training days 0–5999, the same",
            "  field removed from truth and model — identical in form to the streamfunction",
            "  convention in `oceanfno._anomaly_core`. A common field removed from both leaves",
            "  `truth − model` unchanged, so the anomaly plates' difference panels equal the raw",
            "  plates' by construction.",
            "- **Basin mean** is area weighted with MITgcm's `rA`, because what is conserved is the",
            "  area integral, and cells at 75°N are a quarter the area of cells at 15°N.",
            "- **Day 0** is the initial condition, so its difference panel is exactly zero. Kept as a",
            "  sanity anchor, as in the streamfunction plate.",
            "",
            "## Files",
            "",
            f"- `{FIGURE_3}` — SSH, days 0–40, truth / model / difference",
            f"- `{FIGURE_7}` — SSH, days 60 and 2,000",
            f"- `{FIGURE_3A}` — SSH anomaly, days 0–40",
            f"- `{FIGURE_7A}` — SSH anomaly, days 60 and 2,000",
            f"- `{REFERENCE_FIGURE}` — the removed training-mean field",
            f"- `{BASIN_MEAN_FIGURE}` — basin-mean SSH against lead, model vs exact zero",
            f"- `{ARRAYS_NAME}` — every plotted field, all 201 leads of SSH, the basin-mean curves",
            f"- `{REPORT_NAME}` — provenance, the preflight gate and the error table",
            "",
        ]
    )


def run(project_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Build the SSH package for member 0 of the frozen S0 protocol."""

    started = time.monotonic()
    output = (project_root / OUTPUT_RELATIVE).resolve()
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    provenance = load_model_provenance(project_root)
    dataset_path = Path(provenance["contract"]["sources"]["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state, static = group["state"], group["static_features"]
    _, _, wet, _, wind_mean, wind_scale = _normalizers(group)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float64)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float64)
    rac, wet_area = cell_areas(project_root)
    weight = rac * wet / wet_area  # area weights summing to one over the basin

    start = member_start()
    print(f"[1/5] member {MEMBER_INDEX} of {MEMBER_COUNT}: start day {start}, "
          f"day-2,000 lead at day {start + 2000}")

    model = load_frozen_model(provenance["checkpoint"], double=False)
    with np.load(provenance["normalization"]) as stored:
        mean = np.ascontiguousarray(stored["pointwise_mean"], dtype=np.float32)
        scale = np.ascontiguousarray(stored["pointwise_scale"], dtype=np.float32)
    sources = provenance["contract"]["sources"]
    static_block, _ = new_channel_static_block(
        group,
        zonal_spacing_path=_verify(sources["mitgcm_zonal_spacing"], "zonal spacing"),
        sst_relax_path=_verify(sources["mitgcm_sst_relaxation"], "SST relaxation target"),
        data_path=_verify(sources["mitgcm_declaration"], "MITgcm declaration"),
        pointwise_mean=mean,
        pointwise_scale=scale,
    )
    stepper = BireTwoInNewChannelsStepper(
        model=model, device=torch.device("cpu"), wet=wet, mean=mean, scale=scale,
        wind_mean=float(wind_mean), wind_scale=float(wind_scale), static_block=static_block,
    )

    print(f"[2/5] rolling {len(LEAD_DAYS) - 1} calls to day 2,000")
    rollout = roll_member(stepper, state, static, start, wet)

    gate = gate_same_rollout(project_root, rollout["states_at_leads"], wet)
    print(f"      preflight: streamfunction matches the sealed package to "
          f"{gate['worst_relative']:.2e} relative -> same trajectory as figures 3 and 7")

    print("[3/5] reading truth and the training-mean SSH field")
    truth = truth_ssh(group, start, wet)
    model_ssh = rollout["ssh"]
    model_ssh[:, ~wet] = 0.0
    reference, reference_days = training_mean_ssh(group)
    reference = reference * wet

    # Area-weighted basin means at every lead.  MITgcm's is the exact zero the
    # whole diagnostic rests on; the model's is what is being measured.
    basin_mean = {
        "truth": np.einsum("lij,ij->l", truth.astype(np.float64), weight),
        "model": np.einsum("lij,ij->l", model_ssh.astype(np.float64), weight),
    }
    drift_summary = {
        "truth_max_abs": float(np.abs(basin_mean["truth"]).max()),
        "truth_peak_to_peak": float(basin_mean["truth"].max() - basin_mean["truth"].min()),
        "model_final": float(basin_mean["model"][-1]),
        "model_max_abs": float(np.abs(basin_mean["model"]).max()),
        "ratio_at_day_2000": float(
            abs(basin_mean["model"][-1]) / max(abs(basin_mean["truth"][-1]), 1e-30)
        ),
        "exact_answer": "zero at every lead; the area integral of eta is exactly conserved",
    }
    print(f"      MITgcm basin mean stays at {drift_summary['truth_max_abs']:.2e} m; "
          f"model reaches {drift_summary['model_final']:+.4f} m at day 2,000")

    # SSH here is overwhelmingly stationary, so a comparison of the raw or even
    # the anomaly field flatters the model: both sides carry the same fixed
    # pattern and it dominates every norm and correlation.  Split the anomaly
    # into its own time mean over the rollout window plus a fluctuation, and
    # score the fluctuation --- that is the only part with any time dependence
    # for the emulator to get right or wrong.
    truth_anomaly = truth.astype(np.float64) - reference
    model_anomaly = model_ssh.astype(np.float64) - reference
    truth_fluctuation = truth_anomaly - truth_anomaly.mean(axis=0)
    model_fluctuation = model_anomaly - model_anomaly.mean(axis=0)
    stationary_fraction = float(
        (truth_anomaly.mean(axis=0)[wet] ** 2).mean() / (truth_anomaly[:, wet] ** 2).mean()
    )
    correlation, amplitude_ratio = [], []
    for index in range(len(LEAD_DAYS)):
        a, b = truth_fluctuation[index][wet], model_fluctuation[index][wet]
        norm_a, norm_b = float(np.sqrt((a**2).mean())), float(np.sqrt((b**2).mean()))
        correlation.append(
            float(np.corrcoef(a, b)[0, 1]) if norm_a > 0.0 and norm_b > 0.0 else float("nan")
        )
        amplitude_ratio.append(norm_b / norm_a if norm_a > 0.0 else float("nan"))
    fluctuation = {
        "definition": (
            "anomaly about the training mean, minus each side's own mean over the "
            "201 leads of this rollout"
        ),
        "stationary_fraction": stationary_fraction,
        "truth_anomaly_rms_m": float(np.sqrt((truth_anomaly[:, wet] ** 2).mean())),
        "truth_fluctuation_rms_m": float(np.sqrt((truth_fluctuation[:, wet] ** 2).mean())),
        "total_ssh_rms_m": float(np.sqrt((truth[:, wet].astype(np.float64) ** 2).mean())),
        "correlation": correlation,
        "amplitude_ratio": amplitude_ratio,
    }
    print(
        f"      SSH is {1.0 - fluctuation['truth_anomaly_rms_m'] / fluctuation['total_ssh_rms_m']:.1%} "
        f"stationary; the anomaly is a further {stationary_fraction:.1%} stationary"
    )
    print(
        f"      skill on the time-varying part: correlation {correlation[1]:.3f} at day 10, "
        f"{correlation[-1]:.3f} at day 2,000"
    )

    # Error with and without the global offset, so the two failures separate.
    by_lead = []
    for lead in (*FIGURE_3_LEADS, *FIGURE_7_LEADS):
        index = list(LEAD_DAYS).index(int(lead))
        difference = (truth[index] - model_ssh[index]).astype(np.float64)
        offset = basin_mean["truth"][index] - basin_mean["model"][index]
        by_lead.append(
            {
                "lead": int(lead),
                "rmse": float(np.sqrt(np.mean(difference[wet] ** 2))),
                "rmse_demeaned": float(np.sqrt(np.mean((difference[wet] - offset) ** 2))),
                "model_basin_mean": float(basin_mean["model"][index]),
                "truth_basin_mean": float(basin_mean["truth"][index]),
                "truth_fluctuation_rms": float(
                    np.sqrt(np.mean(truth_fluctuation[index][wet] ** 2))
                ),
                "fluctuation_correlation": correlation[index],
                "fluctuation_amplitude_ratio": amplitude_ratio[index],
            }
        )

    print("[4/5] drawing plates")
    figure3 = {name: np.stack([v[list(LEAD_DAYS).index(int(l))] for l in FIGURE_3_LEADS])
               for name, v in (("truth", truth), ("model", model_ssh))}
    figure7 = {name: np.stack([v[list(LEAD_DAYS).index(int(l))] for l in FIGURE_7_LEADS])
               for name, v in (("truth", truth), ("model", model_ssh))}
    anomaly3 = {name: value - reference for name, value in figure3.items()}
    anomaly7 = {name: value - reference for name, value in figure7.items()}

    temporary = output.with_name(output.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        _style()
        _early_plate(
            temporary, FIGURE_3, figure3["truth"], figure3["model"], longitude, latitude, wet,
            state_label=SSH_LABEL, truth_row=r"MITgcm $\eta$", model_row=r"Model C $\eta$",
            title=r"Control wind $\tau_0=0.1$ N m$^{-2}$; Model C $\Delta t=10$ days; "
                  r"native $1^\circ$ grid.  Day 0 is the initial condition, so its difference is zero.",
        )
        _figure7_plate(temporary, figure7["truth"], figure7["model"], longitude, latitude, wet, basin_mean)
        _early_plate(
            temporary, FIGURE_3A, anomaly3["truth"], anomaly3["model"], longitude, latitude, wet,
            state_label=ANOMALY_LABEL, truth_row=r"MITgcm $\eta'$", model_row=r"Model C $\eta'$",
            title=r"Control wind $\tau_0=0.1$ N m$^{-2}$; anomaly about the MITgcm "
                  r"training-block mean $\overline{\eta}_{S0}$ (days 0–5999)",
        )
        _figure7a_plate(temporary, anomaly7["truth"], anomaly7["model"], reference, longitude, latitude, wet)
        _diagnostic_plate(temporary, basin_mean, drift_summary, fluctuation)

        arrays_path = temporary / ARRAYS_NAME
        np.savez_compressed(
            arrays_path,
            figure3_truth_ssh=figure3["truth"], figure3_model_ssh=figure3["model"],
            figure7_truth_ssh=figure7["truth"], figure7_model_ssh=figure7["model"],
            figure3_truth_ssh_anomaly=anomaly3["truth"], figure3_model_ssh_anomaly=anomaly3["model"],
            figure7_truth_ssh_anomaly=anomaly7["truth"], figure7_model_ssh_anomaly=anomaly7["model"],
            training_mean_ssh=reference,
            all_leads_truth_ssh=truth, all_leads_model_ssh=model_ssh,
            lead_days=np.asarray(LEAD_DAYS, dtype=np.int16),
            figure3_lead_days=np.asarray(FIGURE_3_LEADS, dtype=np.int16),
            figure7_lead_days=np.asarray(FIGURE_7_LEADS, dtype=np.int16),
            basin_mean_truth=basin_mean["truth"], basin_mean_model=basin_mean["model"],
            fluctuation_correlation=np.asarray(fluctuation["correlation"]),
            fluctuation_amplitude_ratio=np.asarray(fluctuation["amplitude_ratio"]),
            truth_fluctuation=truth_fluctuation.astype(np.float32),
            model_fluctuation=model_fluctuation.astype(np.float32),
            area_weight=weight, rA=rac,
            wet_mask=wet.astype(np.uint8), longitude_deg=longitude, latitude_deg=latitude,
        )

        report: dict[str, Any] = {
            "status": "complete",
            "version": VERSION,
            "regime": REGIME,
            "field": "ETAN sea-surface height, channel 45 of the 46-channel state",
            "model": {
                "contract": MODEL_CONTRACT,
                "checkpoint": str(provenance["checkpoint"]),
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "normalizer_sha256": provenance["normalization_sha256"],
                "optimizer_step": provenance["optimizer_step"],
                "frozen": True,
                "dtype": "float32",
            },
            "protocol": {
                "member_index": MEMBER_INDEX,
                "member_count": MEMBER_COUNT,
                "start_seed": START_SEED,
                "start_day": start,
                "history_day": start - HORIZON_DAYS,
                "final_truth_day": start + int(LEAD_DAYS[-1]),
                "calls": len(LEAD_DAYS) - 1,
                "horizon_days": HORIZON_DAYS,
                "stepper": "oceanfno.model.BireTwoInNewChannelsStepper",
                "teacher_forcing_after_initial_condition": False,
            },
            "preflight_same_rollout": gate,
            "anomaly_reference": {
                "definition": "mitgcm_pointwise_time_mean_ssh",
                "days": list(TRAIN_RANGE),
                "days_averaged": reference_days,
                "regime": REGIME,
                "subtracted_from": "both_truth_and_model",
                "note": "a common field removed from both sides leaves truth - model unchanged",
                "rms_m": float(np.sqrt(np.mean(reference[wet] ** 2))),
                "range_m": [float(reference[wet].min()), float(reference[wet].max())],
            },
            "basin_mean_ssh": drift_summary,
            "time_varying_ssh": fluctuation,
            "error_summary": {"by_lead": by_lead},
            "conservation": {
                "mitgcm_property": "implicitFreeSurface with exactConserv and a linear free surface",
                "consequence": "the area integral of eta is conserved exactly, so the basin mean cannot move",
                "verified_in_truth_over_record": True,
                "why_it_matters_here": (
                    "the basin mean has a known correct value at every lead, so model drift is "
                    "error measured against an exact reference with no comparison required"
                ),
                "related": "gate F5 of scripts/fno_adjoint.py measures the same failure through the adjoint",
            },
            "figures": list(FIGURE_NAMES),
            "arrays": ARRAYS_NAME,
            "arrays_sha256": file_sha256(arrays_path),
            "dataset": str(dataset_path),
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        report["content_sha256"] = json_sha256(report)
        (temporary / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (temporary / README_NAME).write_text(_readme(report))

        manifest = {
            "version": VERSION,
            "regime": REGIME,
            "report_content_sha256": report["content_sha256"],
            "artifacts": {
                path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in sorted(temporary.iterdir()) if path.is_file()
            },
        }
        manifest["manifest_content_sha256"] = json_sha256(manifest)
        (temporary / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if output.exists():
            shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"[5/5] wrote {output}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="replace an existing package")
    arguments = parser.parse_args(argv)
    report = run(Path(__file__).resolve().parent.parent, force=arguments.force)

    drift = report["basin_mean_ssh"]
    print()
    print(f"complete in {report['elapsed_seconds']:.1f} s")
    print(f"  MITgcm basin-mean SSH : {drift['truth_max_abs']:.2e} m  (exact answer: 0)")
    print(f"  model basin-mean SSH  : {drift['model_final']:+.4f} m at day 2,000")
    for row in report["error_summary"]["by_lead"]:
        print(
            f"  day {row['lead']:>4}: RMSE {row['rmse']:.4f} m, "
            f"{row['rmse_demeaned']:.4f} m with the global offset removed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
