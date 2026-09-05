"""Field-map comparison figures: nominal control (B) vs response (C), beyond streamfunction.

Companion to ``compare_forward_control_response_v1.py``. That script only
re-plots arrays the frozen ``*_s0_figures_v1``/``*_s0_anomaly_v1`` packages
already saved, which is streamfunction snapshots only -- ``evaluate_regime``
in ``oceanfno.figures`` computes every field per lead but only ever persists
the streamfunction ones to disk (see figure3/figure7 handling). Getting day
60 / day 2,000 maps of any other field therefore requires re-running the two
published checkpoints, which is what this script does: one member (the same
start day 60/2,000 already publishes, i.e. member 0's start), autoregressed
ten days at a time exactly as ``evaluate_regime`` does, capturing full 2-D
fields only at lead 60 and lead 2,000 rather than reducing every lead to a
scalar. No training, no checkpoint mutation -- ``ProductionStepper.step`` is
called under ``torch.no_grad()``, same as the frozen evaluation.

Fields plotted (see ``oceanfno.diagnostics.derived_fields``):

    phihyd_surface  surface dynamic pressure / rho -- the field with the
                    largest control-vs-response day-2,000 RMSE gap already
                    surfaced in figure 4 of the compare-forward package
    sst             sea surface temperature -- second-largest day-2,000 gap
    ssh             sea surface height (channel 45, "eta") -- the exact
                    field the adjoint-sensitivity study (fno_adjoint_ft90.py)
                    differentiates; never plotted spatially anywhere else in
                    this figure suite

Usage:

    python scripts/compare_field_maps_control_response_v1.py
    python scripts/compare_field_maps_control_response_v1.py --seed 20260724
    python scripts/compare_field_maps_control_response_v1.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
for path in (_SRC, _HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oceanfno import figures as base_figures
from oceanfno import figures_response
from oceanfno import plots as base_plots
from oceanfno.runtime import _device, torch
from oceanfno.train import physical_static_block
from oceanfno.validation import _gather

import compare_forward_control_response_v1 as fwd

REGIME_INDEX = 0
CAPTURE_LEADS = (60, 2000)
CACHE_NAME = "model_c_bire_compare_field_rollout_cache.npz"

#: (state key, label, unit, colour mode). "diverging" centres 0; "sequential"
#: spans the data's own min/max -- SST has no meaningful zero crossing here.
FIELD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("phihyd_surface", "Surface $P/\\rho$", "m$^2$ s$^{-2}$", "diverging"),
    ("sst", "SST", "$^\\circ$C", "sequential"),
    ("ssh", "Sea surface height $\\eta$", "m", "diverging"),
)


class FieldMapError(RuntimeError):
    """Raised when the two arms' rollouts cannot be legitimately compared."""


def prepare_stepper(config: dict[str, Any], training_version: str, group: Any, wet: np.ndarray, device: Any) -> Any:
    with np.load(Path(config["artifacts"]["selected_normalization"]["path"])) as stored:
        point_mean = np.asarray(stored["pointwise_mean"], dtype=np.float32)
        point_scale = np.asarray(stored["pointwise_scale"], dtype=np.float32)
    statics, _ = physical_static_block(config["artifacts"], group, point_mean, point_scale)
    # figures._stepper is hard-bound to the historical production lineage's
    # checkpoint identity; figures_response._stepper is the study adapter
    # that takes the arm's own training version instead (see its docstring).
    return figures_response._stepper(config, device, wet, statics, training_version)


def fields_at(state_slice: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.asarray(v[0]) for k, v in base_figures._fields(state_slice, wet).items()}


def truth_member(state: Any, start: int, wet: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    records = np.array([[REGIME_INDEX, start]], dtype=np.int64)
    return {lead: fields_at(_gather(state, records, lead), wet) for lead in CAPTURE_LEADS}


def rollout_member(
    stepper: Any, state: Any, static: Any, start: int, wet: np.ndarray
) -> dict[int, dict[str, np.ndarray]]:
    records = np.array([[REGIME_INDEX, start]], dtype=np.int64)
    initial = _gather(state, records, 0)
    current = stepper.normalized_state(initial)
    forcing = stepper.normalized_static(static, records[:, 0])
    captured: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for lead in base_plots.LEAD_DAYS:
            if lead:
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
            else:
                prediction = initial.copy()
            if lead in CAPTURE_LEADS:
                captured[lead] = fields_at(prediction, wet)
    return captured


def figure_field_maps(
    output: Path,
    key: str,
    label: str,
    unit: str,
    mode: str,
    truth: dict[int, dict[str, np.ndarray]],
    control: dict[int, dict[str, np.ndarray]],
    response: dict[int, dict[str, np.ndarray]],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
    seed: int,
) -> None:
    base_plots._style()
    truth_by_lead = {lead: truth[lead][key] for lead in CAPTURE_LEADS}
    control_by_lead = {lead: control[lead][key] for lead in CAPTURE_LEADS}
    response_by_lead = {lead: response[lead][key] for lead in CAPTURE_LEADS}
    diff_control = {lead: truth_by_lead[lead] - control_by_lead[lead] for lead in CAPTURE_LEADS}
    diff_response = {lead: truth_by_lead[lead] - response_by_lead[lead] for lead in CAPTURE_LEADS}

    state_values = tuple(truth_by_lead.values()) + tuple(control_by_lead.values()) + tuple(response_by_lead.values())
    if mode == "sequential":
        vmin = min(float(np.min(v[wet])) for v in state_values)
        vmax = max(float(np.max(v[wet])) for v in state_values)
        state_kwargs = {"cmap": "viridis", "vmin": vmin, "vmax": vmax}
    else:
        bound = base_plots._finite_bound(state_values)
        state_kwargs = {"cmap": "RdBu_r", "vmin": -bound, "vmax": bound}
    diff_bound = base_plots._finite_bound(tuple(diff_control.values()) + tuple(diff_response.values()))
    diff_kwargs = {"cmap": "RdBu_r", "vmin": -diff_bound, "vmax": diff_bound}

    state_columns = (("MITgcm truth", truth_by_lead), ("Control", control_by_lead), ("Response", response_by_lead))
    diff_columns = (("Truth − control", diff_control), ("Truth − response", diff_response))

    figure, axes = plt.subplots(2, 5, figsize=(16.2, 7.0), sharex=True, sharey=True, constrained_layout=True)
    state_image = diff_image = None
    for row, lead in enumerate(CAPTURE_LEADS):
        for column, (_, data) in enumerate(state_columns):
            state_image = axes[row, column].pcolormesh(
                longitude, latitude, base_plots._masked(data[lead], wet), shading="auto", **state_kwargs
            )
        for column, (_, data) in enumerate(diff_columns):
            diff_image = axes[row, 3 + column].pcolormesh(
                longitude, latitude, base_plots._masked(data[lead], wet), shading="auto", **diff_kwargs
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
    figure.colorbar(state_image, ax=axes[:, :3].ravel().tolist(), label=f"{label} ({unit})", shrink=0.75)
    figure.colorbar(diff_image, ax=axes[:, 3:].ravel().tolist(), label=f"Truth − model ({unit})", shrink=0.75)
    figure.suptitle(
        f"{label}, member start day {truth.get('_start', '?')}; "
        r"$\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days; " f"seed {seed}, S0"
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def run(seed: int, device_name: str) -> dict[str, Any]:
    started = time.monotonic()
    control_dirs = fwd._arm_dirs(fwd.CONTROL_VERSION, seed)
    response_dirs = fwd._arm_dirs(fwd.RESPONSE_VERSION, seed)
    control_config = json.loads(control_dirs["figures_config"].read_text())
    response_config = json.loads(response_dirs["figures_config"].read_text())

    output_dir = (
        _ROOT / "outputs" / "af_fno" / "adjoint" / "C"
        / f"model_c_adjoint_faithful_control_vs_response_v1_seed_{seed}_s0"
    )
    if not output_dir.is_dir():
        raise FieldMapError(f"{output_dir} does not exist -- run compare_forward_control_response_v1.py first")

    dataset_path = Path(control_config["dataset"]["path"]).resolve()
    if dataset_path != Path(response_config["dataset"]["path"]).resolve():
        raise FieldMapError("control and response arms declare different datasets")
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)

    with np.load(control_dirs["figures"] / "model_c_bire_s0_figures_arrays.npz") as z:
        if not np.array_equal(z["wet_mask"].astype(bool), wet):
            raise FieldMapError("the dataset's wet mask does not match the published figures package")
        start = int(z["start_draw_order"][0])

    device = _device(device_name)
    print(f"[compare-field-maps] device={device}, member start day={start}, seed={seed}")

    truth = truth_member(state, start, wet)
    truth["_start"] = start

    control_stepper = prepare_stepper(control_config, fwd.CONTROL_VERSION, group, wet, device)
    control_fields = rollout_member(control_stepper, state, static, start, wet)
    del control_stepper
    print(f"[compare-field-maps] control rollout done, {time.monotonic() - started:.1f}s elapsed")

    response_stepper = prepare_stepper(response_config, fwd.RESPONSE_VERSION, group, wet, device)
    response_fields = rollout_member(response_stepper, state, static, start, wet)
    del response_stepper
    print(f"[compare-field-maps] response rollout done, {time.monotonic() - started:.1f}s elapsed")

    # Cached so a follow-up (e.g. anomaly variants) can reuse these fields
    # without repeating the ~7-minute rollout -- the same reason the
    # anomaly.py package reads the figures_v1 arrays.npz instead of
    # re-running inference itself.
    cache_path = output_dir / CACHE_NAME
    cache_payload: dict[str, np.ndarray] = {"start": np.asarray(start)}
    for source_name, source in (("truth", truth), ("control", control_fields), ("response", response_fields)):
        for lead in CAPTURE_LEADS:
            for key, value in source[lead].items():
                cache_payload[f"{source_name}__{lead}__{key}"] = value.astype(np.float32)
    np.savez_compressed(cache_path, **cache_payload)
    print(f"[compare-field-maps] wrote rollout cache {cache_path}")

    written = []
    for key, label, unit, mode in FIELD_SPECS:
        name = f"model_c_bire_compare_seed{seed}_{key}_day060_day2000_s0.png"
        path = output_dir / name
        figure_field_maps(path, key, label, unit, mode, truth, control_fields, response_fields, longitude, latitude, wet, seed)
        written.append({"file": name, "content": f"{label}: truth | control | response + truth-minus-arm, day 60 and day 2000"})
        print(f"[compare-field-maps] wrote {path}")

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("figures", [])
    existing_names = {item["file"] for item in manifest["figures"]}
    for item in written:
        if item["file"] not in existing_names:
            manifest["figures"].append(item)
    manifest.setdefault("supplementary_runs", [])
    manifest["supplementary_runs"].append(
        {
            "script": "scripts/compare_field_maps_control_response_v1.py",
            "member_start_day": start,
            "dataset": str(dataset_path),
            "elapsed_seconds": time.monotonic() - started,
            "cache": CACHE_NAME,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    readme_path = output_dir / "README.md"
    readme = readme_path.read_text()
    addendum = (
        "\n## Additional field-map figures\n\n"
        "Same truth | control | response + truth-minus-arm layout as the "
        "streamfunction-anomaly figure above, for fields other than "
        "streamfunction. Unlike the other figures in this folder, these "
        "required re-running both published checkpoints (member start day "
        f"{start}, the same member figure 7/7a plot) out to day 2,000 -- "
        "`evaluate_regime` computes these fields per lead but only persists "
        "streamfunction snapshots to the frozen `*_figures_v1` packages. "
        "See `scripts/compare_field_maps_control_response_v1.py`.\n\n"
        + "\n".join(f"- `{item['file']}` --- {item['content']}" for item in written)
        + "\n"
    )
    if addendum.strip() not in readme:
        readme_path.write_text(readme + addendum)

    return {"output_dir": str(output_dir), "figures": [item["file"] for item in written]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=fwd.DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(arguments.seed, arguments.device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
