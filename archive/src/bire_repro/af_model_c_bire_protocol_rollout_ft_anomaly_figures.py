"""Streamfunction-anomaly companions to figures 3 and 7 of the rollout fine-tune.

Figures 3 and 7 plot the **total** barotropic streamfunction, which on this
basin is dominated by the stationary double gyre::

    psi(x, y, t) = psi_bar(x, y) + psi'(x, y, t)
                   \\_____ _____/   \\______ ______/
                         v                 v
                 stationary circulation   what actually varies

On a +/-33 Sv colour scale the two mean gyres fill the plate, so a model whose
transient variability is wrong by a few Sv still looks convincing.  Removing the
stationary field rescales the plate to the part the model had to learn, which is
where accumulated striping, missing or excessive eddy activity, and phase error
against MITgcm become visible.

This package **adds** two figures; it does not modify the existing ones.  The
total-streamfunction plates stay exactly as published, and the total field
remains the right basis for amplitude diagnostics --- the day-2,000 minimum of
-32.90 Sv is a statement about mean-circulation intensity and is reported from
the total field, not from these anomalies.  The two answer different questions.

The reference field
-------------------
``psi_bar`` is the **MITgcm** time mean over the S0 **training** block, days
0--5999::

    psi_bar_S0(x, y) = (1 / N_train) sum_{t in S0 training} psi_MITgcm(x, y, t)

Three properties of that choice matter and are enforced here:

* it is the full two-dimensional field at every grid point, not a scalar;
* the **same** field is subtracted from truth and from prediction.  Subtracting
  the FNO's own time mean would silently absorb any bias in the mean circulation
  the model carries, which is precisely the error worth seeing;
* it is computed from training days only, so nothing from the validation or
  inference blocks enters the reference.  This package reads no model weights
  and rolls nothing out; it consumes the sealed figure package's published
  arrays, whose SHA-256 the contract pins.

Held-evaluation only: no training, no checkpoint selection, no promotion.
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

from .af_bire_protocol_split import TRAIN_RANGE
from .af_data_v3 import DATASET_VERSION
from .af_forward_complete import derived_fields
from .af_model_c_bire_aligned_full_state import (
    MANIFEST_NAME,
    README_NAME,
    _json_sha256,
)
from .af_model_c_bire_s0_figures import (
    ARRAYS_NAME as SUITE_ARRAYS_NAME,
)
from .af_model_c_bire_s0_figures import (
    FIGURE_3_LEADS,
    FIGURE_7_LEADS,
    _finite_bound,
    _masked,
    _style,
)
from .af_model_c_overfit import _file_sha256

VERSION = "model_c_bire_protocol_rollout_ft_s0_anomaly_v1"
CONTRACT_STATUS = (
    "frozen_after_the_rollout_ft_figure_package_and_before_any_anomaly_metric"
)

REGIME = "S0"
REGIME_INDEX = 0
TAU0 = 0.1

FIGURE_3A = "model_c_bire_figure3a_streamfunction_anomaly_1deg_s0_dt10.png"
FIGURE_7A = "model_c_bire_figure7a_streamfunction_anomaly_day060_day2000_s0.png"
FIGURE_NAMES = (FIGURE_3A, FIGURE_7A)
REPORT_NAME = "model_c_bire_rollout_ft_anomaly_report.json"
ARRAYS_NAME = "model_c_bire_rollout_ft_anomaly_arrays.npz"

#: Colour-bar text.  The published figures say "Barotropic streamfunction (Sv)";
#: these must not be mistaken for them at a glance.
ANOMALY_LABEL = "Barotropic streamfunction anomaly $\\psi'$ (Sv)"
DIFFERENCE_LABEL = "Truth $-$ model anomaly (Sv)"
MEAN_LABEL = "Time-mean $\\overline{\\psi}$ (Sv)"


class BireProtocolRolloutFineTuneAnomalyError(RuntimeError):
    """Raised when the anomaly figure contract is violated."""


def training_mean_streamfunction(
    group: Any,
    wet: np.ndarray,
    *,
    experiment: int = REGIME_INDEX,
    chunk_days: int = 60,
) -> tuple[np.ndarray, int]:
    """MITgcm time-mean barotropic streamfunction over the S0 training block.

    Averages the *derived* streamfunction field chunk by chunk, which is the
    same construction :func:`af_model_c_bire_protocol.train_only_climatology`
    uses for its per-regime climatology, so this field and that one agree.
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
        raise BireProtocolRolloutFineTuneAnomalyError(
            f"the training mean covered {count} days, not {stop - start}"
        )
    mean = (total / count).astype(np.float32)
    mean[~wet] = 0.0
    if not np.all(np.isfinite(mean)):
        raise BireProtocolRolloutFineTuneAnomalyError("the time-mean streamfunction is not finite")
    return mean, count


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the anomaly contract frozen after the figure package it consumes."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    reference = contract.get("reference", {})
    protocol = contract.get("protocol", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or contract.get("dataset", {}).get("version") != DATASET_VERSION
        or protocol.get("primary_regime") != REGIME
        or tuple(protocol.get("figure3_lead_days", ())) != FIGURE_3_LEADS
        or tuple(protocol.get("figure7_lead_days", ())) != FIGURE_7_LEADS
        or tuple(protocol.get("figure_names", ())) != FIGURE_NAMES
        or reference.get("source") != "mitgcm"
        or tuple(reference.get("days", ())) != TRAIN_RANGE
        or reference.get("subtracted_from") != "both_truth_and_prediction"
        or reference.get("model_own_mean_used") is not False
        or reference.get("is_two_dimensional_field") is not True
        or contract.get("modifies_published_figures") is not False
        or Path(contract["artifacts"]["figure_package_arrays"]["path"]).name
        != SUITE_ARRAYS_NAME
    ):
        raise BireProtocolRolloutFineTuneAnomalyError("anomaly figure contract changed")
    if verify_sources:
        for label, specification in contract["artifacts"].items():
            path_ = Path(specification["path"]).resolve()
            target = path_ / ".zmetadata" if path_.is_dir() else path_
            if not target.is_file() or _file_sha256(target) != specification["sha256"]:
                raise BireProtocolRolloutFineTuneAnomalyError(f"{label} changed on disk")
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolRolloutFineTuneAnomalyError(f"source changed: {relative}")
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
    """Per-lead transient variability of truth and prediction about the same mean.

    ``anomaly_rms_ratio`` is the diagnostic the total-field standard-deviation
    ratio cannot give: with the stationary gyre removed, a ratio below one means
    the model has damped the transients and a ratio above one means it has
    manufactured them.
    """

    record: dict[str, Any] = {}
    for index, lead in enumerate(leads):
        truth_rms = wet_rms(truth[index], wet)
        model_rms = wet_rms(model[index], wet)
        record[str(int(lead))] = {
            "truth_anomaly_rms_sv": truth_rms,
            "model_anomaly_rms_sv": model_rms,
            "anomaly_rms_ratio": (model_rms / truth_rms) if truth_rms > 0.0 else float("inf"),
            "anomaly_error_rms_sv": wet_rms(truth[index] - model[index], wet),
            "truth_anomaly_range_sv": [
                float(truth[index][wet].min()), float(truth[index][wet].max())
            ],
            "model_anomaly_range_sv": [
                float(model[index][wet].min()), float(model[index][wet].max())
            ],
        }
    return record


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
    axes[0, 0].set_ylabel("MITgcm $\\psi'$\nLatitude (°)")
    axes[1, 0].set_ylabel("Model C $\\psi'$\nLatitude (°)")
    axes[2, 0].set_ylabel("Truth − model\nLatitude (°)")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    figure.colorbar(state_image, ax=axes[:2].ravel().tolist(), label=ANOMALY_LABEL, shrink=0.82)
    figure.colorbar(difference_image, ax=axes[2].ravel().tolist(), label=DIFFERENCE_LABEL, shrink=0.82)
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
        len(FIGURE_7_LEADS), 3, figsize=(10.4, 6.8),
        sharex=True, sharey=True, constrained_layout=True,
    )
    anomaly_image = mean_image = None
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
    axes[0, 1].set_title("Model C $\\psi'$")
    axes[0, 2].set_title("Truth − model")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
        axis.set_xlabel("")
    for axis in axes[-1]:
        axis.set_xlabel("Longitude (°)")
    figure.colorbar(anomaly_image, ax=axes.ravel().tolist(), label=ANOMALY_LABEL, shrink=0.75)
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
    figure.savefig(output / "model_c_bire_reference_time_mean_streamfunction_s0.png", bbox_inches="tight")
    plt.close(figure)


def _readme(report: Mapping[str, Any]) -> str:
    day2000 = report["variability"]["figure7"]["2000"]
    day60 = report["variability"]["figure7"]["60"]
    return f"""# Streamfunction anomalies, S0 — companions to figures 3 and 7

These two plates **add to** the published figure package; they replace nothing.
`{FIGURE_3A}` and `{FIGURE_7A}` show

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

with `psi_bar_S0` the **MITgcm** time-mean barotropic streamfunction over the S0
**training** block, days {TRAIN_RANGE[0]}–{TRAIN_RANGE[1] - 1}, averaged over
{report['reference']['days_averaged']:,} days. The same two-dimensional field is
subtracted from truth and from the model. The model's own time mean is
deliberately *not* used: that would absorb any bias in the mean circulation the
model carries, which is the error most worth seeing.

The reference field itself is published as
`model_c_bire_reference_time_mean_streamfunction_s0.png`, range
{report['reference']['range_sv'][0]:.2f} to {report['reference']['range_sv'][1]:.2f} Sv,
RMS {report['reference']['rms_sv']:.2f} Sv.

## Why the total-field plates were not enough

On a ±33 Sv scale the stationary double gyre fills the plate. The transients
this model actually had to learn are roughly
{day2000['truth_anomaly_rms_sv']:.2f} Sv RMS at day 2,000 — about
{100.0 * day2000['truth_anomaly_rms_sv'] / report['reference']['rms_sv']:.0f}% of
the mean field's own RMS. A model can carry the mean gyres correctly and still
get the variability badly wrong without the total-field plate showing it.

| lead | truth psi' RMS | model psi' RMS | ratio |
| --- | --- | --- | --- |
| day 60 | {day60['truth_anomaly_rms_sv']:.3f} Sv | {day60['model_anomaly_rms_sv']:.3f} Sv | {day60['anomaly_rms_ratio']:.3f} |
| day 2,000 | {day2000['truth_anomaly_rms_sv']:.3f} Sv | {day2000['model_anomaly_rms_sv']:.3f} Sv | {day2000['anomaly_rms_ratio']:.3f} |

A ratio below one means damped transients; above one, manufactured ones.

## What this does not change

Amplitude diagnostics stay on the **total** field. The acceptance gate's
day-2,000 streamfunction minimum of −32.90 Sv is a statement about
mean-circulation intensity and is unaffected by anything here. These plates
answer a different question: whether the model reproduces variability *about*
that mean.

Member 0 of the 15-member S0 ensemble, the same member figures 3 and 7 plot.

Report content SHA-256: `{report['content_sha256']}`.
"""


def run(contract_path: str | Path) -> dict[str, Any]:
    """Publish the two anomaly plates and the variability summary."""

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
            project_tmp, anomalies["figure3_truth"], anomalies["figure3_model"],
            longitude, latitude, wet,
        )
        _plot_anomaly_long(
            project_tmp, anomalies["figure7_truth"], anomalies["figure7_model"],
            mean_field, longitude, latitude, wet,
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
                    anomalies["figure3_truth"], anomalies["figure3_model"],
                    FIGURE_3_LEADS, wet,
                ),
                "figure7": variability_summary(
                    anomalies["figure7_truth"], anomalies["figure7_model"],
                    FIGURE_7_LEADS, wet,
                ),
            },
            "total_field_note": (
                "amplitude diagnostics including the day-2000 streamfunction minimum "
                "remain defined on the total field and are unaffected by this package"
            ),
            "modifies_published_figures": False,
            "figures": list(FIGURE_NAMES),
            "arrays": str(scratch / ARRAYS_NAME),
            "arrays_sha256": _file_sha256(arrays_path),
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
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
                for path in sorted(project_tmp.iterdir()) if path.is_file()
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
        "figure_package_arrays": contract["artifacts"]["figure_package_arrays"]["path"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preflight(args.contract) if args.command == "preflight" else run(args.contract)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
