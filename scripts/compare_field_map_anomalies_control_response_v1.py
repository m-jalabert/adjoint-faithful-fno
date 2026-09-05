"""Anomaly-map comparison figures: control (B) vs response (C), phihyd_surface/sst/ssh.

Companion to ``compare_field_maps_control_response_v1.py``, mirroring what
``anomaly.py`` already does for streamfunction (figure 7a): the same MITgcm
S0 training-block time mean (days 0-5,999) is subtracted from truth, control
and response alike before differencing, so mean gyre structure common to all
three does not dominate the colour scale -- only the transient/eddy part
remains, the same convention as ``model_c_bire_figure7a_streamfunction_
anomaly_day060_day2000_s0.png``.

Reuses the raw day-60/day-2,000 fields ``compare_field_maps_control_response_
v1.py`` cached (``model_c_bire_compare_field_rollout_cache.npz``) rather than
re-running either checkpoint. The only new compute here is the training-mean
pass over the truth trajectory (no model involved), generalizing
``anomaly.training_mean_streamfunction`` to phihyd_surface/sst/ssh in one
pass over the same 6,000-day block.

Usage:

    python scripts/compare_field_map_anomalies_control_response_v1.py
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

from oceanfno.dataset import TRAIN_RANGE
from oceanfno.diagnostics import derived_fields

import compare_forward_control_response_v1 as fwd
import compare_field_maps_control_response_v1 as field_maps

REGIME_INDEX = field_maps.REGIME_INDEX
CAPTURE_LEADS = field_maps.CAPTURE_LEADS
FIELD_KEYS = tuple(spec[0] for spec in field_maps.FIELD_SPECS)


class AnomalyMapError(RuntimeError):
    """Raised when the cached rollout or the training mean cannot be trusted."""


def training_mean_fields(
    group: Any, wet: np.ndarray, fields: tuple[str, ...], *, experiment: int = REGIME_INDEX, chunk_days: int = 60
) -> tuple[dict[str, np.ndarray], int]:
    """MITgcm time-mean of each field over the S0 training block, one pass.

    Same construction as ``anomaly.training_mean_streamfunction`` (chunked,
    ``derived_fields`` over each chunk, divide by day count) generalized to
    several fields at once so the 6,000-day block is only read once.
    """

    start, stop = TRAIN_RANGE
    state = group["state"]
    totals = {field: np.zeros(wet.shape, dtype=np.float64) for field in fields}
    count = 0
    for begin in range(start, stop, chunk_days):
        end = min(begin + chunk_days, stop)
        raw = np.asarray(state[experiment, begin:end], dtype=np.float32)
        chunk_fields = derived_fields(raw, wet)
        for field in fields:
            totals[field] += chunk_fields[field].sum(axis=0, dtype=np.float64)
        count += int(raw.shape[0])
    if count != stop - start:
        raise AnomalyMapError(f"the training mean covered {count} days, not {stop - start}")
    means = {}
    for field in fields:
        mean = (totals[field] / count).astype(np.float32)
        mean[~wet] = 0.0
        if not np.all(np.isfinite(mean)):
            raise AnomalyMapError(f"the time-mean {field} is not finite")
        means[field] = mean
    return means, count


def load_rollout_cache(
    output_dir: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]], int]:
    cache_path = output_dir / field_maps.CACHE_NAME
    if not cache_path.is_file():
        raise AnomalyMapError(f"{cache_path} is missing -- run compare_field_maps_control_response_v1.py first")
    sources: dict[str, dict[int, dict[str, np.ndarray]]] = {"truth": {}, "control": {}, "response": {}}
    with np.load(cache_path) as stored:
        start = int(stored["start"])
        for key in stored.files:
            if key == "start":
                continue
            source_name, lead, field = key.split("__", 2)
            sources[source_name].setdefault(int(lead), {})[field] = np.asarray(stored[key])
    return sources["truth"], sources["control"], sources["response"], start


def figure_reference_means(
    output: Path,
    means: dict[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
    seed: int,
    days: int,
) -> None:
    """The removed field, published once per field so the anomalies can be read against it."""

    field_maps.base_plots._style()
    figure, axes = plt.subplots(1, len(FIELD_KEYS), figsize=(4.8 * len(FIELD_KEYS), 4.4), constrained_layout=True)
    for axis, (key, label, unit, mode) in zip(axes, field_maps.FIELD_SPECS):
        value = means[key]
        # The mean field itself is *not* an anomaly -- SST in particular has
        # no meaningful zero crossing, so it keeps the same sequential
        # treatment as the raw (non-anomaly) SST maps rather than a
        # zero-centred diverging scale.
        if mode == "sequential":
            plot_kwargs = {"cmap": "viridis", "vmin": float(value[wet].min()), "vmax": float(value[wet].max())}
        else:
            bound = field_maps.base_plots._finite_bound((value,))
            plot_kwargs = {"cmap": "RdBu_r", "vmin": -bound, "vmax": bound}
        image = axis.pcolormesh(
            longitude, latitude, field_maps.base_plots._masked(value, wet),
            shading="auto", **plot_kwargs,
        )
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
        axis.set_xlabel("Longitude (°)")
        axis.set_title(label)
        figure.colorbar(image, ax=axis, label=f"{label} ({unit})", shrink=0.85)
    axes[0].set_ylabel("Latitude (°)")
    figure.suptitle(f"MITgcm S0 training-block time mean, {days} days; seed {seed}")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def run(seed: int) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = (
        _ROOT / "outputs" / "af_fno" / "adjoint" / "C"
        / f"model_c_adjoint_faithful_control_vs_response_v1_seed_{seed}_s0"
    )
    if not output_dir.is_dir():
        raise AnomalyMapError(f"{output_dir} does not exist")

    truth, control_fields, response_fields, start = load_rollout_cache(output_dir)

    control_dirs = fwd._arm_dirs(fwd.CONTROL_VERSION, seed)
    control_config = json.loads(control_dirs["figures_config"].read_text())
    dataset_path = Path(control_config["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)

    print(f"[compare-field-anomalies] computing {TRAIN_RANGE[1]-TRAIN_RANGE[0]}-day training means for {FIELD_KEYS}")
    means, days = training_mean_fields(group, wet, FIELD_KEYS)
    print(f"[compare-field-anomalies] training means done, {time.monotonic() - started:.1f}s elapsed")

    def anomaly_source(source: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, np.ndarray]]:
        return {lead: {key: source[lead][key] - means[key] for key in FIELD_KEYS} for lead in CAPTURE_LEADS}

    truth_anomaly = anomaly_source(truth)
    truth_anomaly["_start"] = start
    control_anomaly = anomaly_source(control_fields)
    response_anomaly = anomaly_source(response_fields)

    written = []
    for key, label, unit, _mode in field_maps.FIELD_SPECS:
        name = f"model_c_bire_compare_seed{seed}_{key}_anomaly_day060_day2000_s0.png"
        path = output_dir / name
        field_maps.figure_field_maps(
            path, key, f"{label}$'$", unit, "diverging",
            truth_anomaly, control_anomaly, response_anomaly, longitude, latitude, wet, seed,
        )
        written.append(
            {
                "file": name,
                "content": f"{label} anomaly (S0 training mean removed): truth | control | response + truth-minus-arm, day 60/2000",
            }
        )
        print(f"[compare-field-anomalies] wrote {path}")

    reference_name = f"model_c_bire_compare_seed{seed}_reference_time_mean_fields_s0.png"
    reference_path = output_dir / reference_name
    figure_reference_means(reference_path, means, longitude, latitude, wet, seed, days)
    written.append({"file": reference_name, "content": "the removed S0 training-mean field, one panel per quantity"})
    print(f"[compare-field-anomalies] wrote {reference_path}")

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
            "script": "scripts/compare_field_map_anomalies_control_response_v1.py",
            "reference_mean": {"days": list(TRAIN_RANGE), "days_averaged": days, "regime": "S0"},
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    readme_path = output_dir / "README.md"
    readme = readme_path.read_text()
    addendum = (
        "\n## Anomaly field-map figures\n\n"
        "Same layout as the field-map figures above, but with the MITgcm S0 "
        f"training-block time mean ({days} days) subtracted from truth, "
        "control and response alike first -- the same convention as "
        "`model_c_bire_figure7a_streamfunction_anomaly_day060_day2000_s0.png`. "
        "Built from the cached rollout, no new inference. See "
        "`scripts/compare_field_map_anomalies_control_response_v1.py`.\n\n"
        + "\n".join(f"- `{item['file']}` --- {item['content']}" for item in written)
        + "\n"
    )
    if addendum.strip() not in readme:
        readme_path.write_text(readme + addendum)

    return {"output_dir": str(output_dir), "figures": [item["file"] for item in written]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=fwd.DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(arguments.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
