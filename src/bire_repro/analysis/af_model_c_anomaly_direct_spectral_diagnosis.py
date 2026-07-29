"""Diagnose the frozen Model C day-360 deep-pressure spectral failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = "model_c_anomaly_direct_spectral_diagnosis_v1"
OUTPUT_NAMES = (
    "spectral_diagnosis.json",
    "spectral_modes.csv",
    "model_c_day360_deep_pressure_spectral_diagnosis.png",
    "manifest.json",
    "README.md",
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def diagnose_field(
    modes: np.ndarray,
    truth_members: np.ndarray,
    model_members: np.ndarray,
) -> dict[str, Any]:
    """Return frozen-gate and scale-aware diagnostics for one field."""

    modes = np.asarray(modes, dtype=np.float64)
    truth = np.asarray(truth_members, dtype=np.float64).mean(axis=0)
    model = np.asarray(model_members, dtype=np.float64).mean(axis=0)
    if truth.shape != modes.shape or model.shape != modes.shape:
        raise ValueError("spectral arrays do not match radial modes")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(model)):
        raise ValueError("spectral arrays contain non-finite values")
    valid = truth > max(float(np.max(truth)) * 1.0e-8, 1.0e-20)
    if not np.any(valid):
        raise ValueError("no modes satisfy the frozen spectral validity rule")
    ratio = np.divide(
        model,
        truth,
        out=np.full_like(model, np.nan),
        where=truth > 0.0,
    )
    large = valid & (modes >= 1.0) & (modes <= 5.0)
    tail = valid & (modes >= 10.0)
    if not np.any(large) or not np.any(tail):
        raise ValueError("required large-scale or tail spectral band is absent")

    truth_total = float(np.sum(truth[valid]))
    model_total = float(np.sum(model[valid]))
    result = {
        "valid_mode_count": int(np.sum(valid)),
        "valid_modes": [int(value) for value in modes[valid]],
        "frozen_median_modewise_energy_ratio": float(np.median(ratio[valid])),
        "frozen_factor_four_pass": bool(
            0.25 <= float(np.median(ratio[valid])) <= 4.0
        ),
        "integrated_energy_ratio": model_total / truth_total,
        "large_scale_k1_to_k5": {
            "median_modewise_energy_ratio": float(np.median(ratio[large])),
            "integrated_energy_ratio": float(
                np.sum(model[large]) / np.sum(truth[large])
            ),
            "truth_fraction_of_valid_energy": float(
                np.sum(truth[large]) / truth_total
            ),
            "model_fraction_of_valid_energy": float(
                np.sum(model[large]) / model_total
            ),
        },
        "tail_k10_plus": {
            "median_modewise_energy_ratio": float(np.median(ratio[tail])),
            "integrated_energy_ratio": float(
                np.sum(model[tail]) / np.sum(truth[tail])
            ),
            "truth_fraction_of_valid_energy": float(
                np.sum(truth[tail]) / truth_total
            ),
            "model_fraction_of_valid_energy": float(
                np.sum(model[tail]) / model_total
            ),
        },
        "per_mode": [
            {
                "mode": int(mode),
                "truth_energy": float(truth[index]),
                "model_energy": float(model[index]),
                "model_over_truth": (
                    float(ratio[index]) if np.isfinite(ratio[index]) else None
                ),
                "frozen_valid": bool(valid[index]),
                "band": (
                    "large_k1_to_k5"
                    if large[index]
                    else "tail_k10_plus"
                    if tail[index]
                    else "intermediate_or_invalid"
                ),
            }
            for index, mode in enumerate(modes)
        ],
    }
    return result


def _verify_source(
    contract: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    source = contract["source"]
    directory = Path(source["directory"])
    paths = tuple(
        directory / source[key]["name"]
        for key in ("metrics", "arrays", "manifest")
    )
    for key, path in zip(("metrics", "arrays", "manifest"), paths, strict=True):
        if not path.is_file():
            raise FileNotFoundError(path)
        if _file_sha256(path) != source[key]["sha256"]:
            raise ValueError(f"immutable forward {key} hash changed")
    return paths


def _plot(
    path: Path,
    modes: np.ndarray,
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "phihyd_mid": "PHIHYD mid-depth (k=7)",
        "phihyd_bottom": "PHIHYD bottom (k=14)",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for column, field in enumerate(("phihyd_mid", "phihyd_bottom")):
        rows = results[field]["per_mode"]
        truth = np.asarray([row["truth_energy"] for row in rows])
        model = np.asarray([row["model_energy"] for row in rows])
        ratio = np.asarray(
            [
                np.nan if row["model_over_truth"] is None else row["model_over_truth"]
                for row in rows
            ]
        )
        valid = np.asarray([row["frozen_valid"] for row in rows])

        axis = axes[0, column]
        axis.semilogy(modes, truth, "o-", label="MITgcm truth")
        axis.semilogy(modes, model, "o-", label="Model C")
        axis.axvspan(9.5, float(np.max(modes)) + 0.5, color="tab:red", alpha=0.08)
        axis.set_title(titles[field])
        axis.set_xlabel("Radial Fourier mode")
        axis.set_ylabel("Mean spectral energy")
        axis.grid(alpha=0.25)
        axis.legend()

        axis = axes[1, column]
        axis.semilogy(modes[valid], ratio[valid], "o-", color="tab:purple")
        axis.axhspan(0.25, 4.0, color="tab:green", alpha=0.12, label="frozen bounds")
        axis.axvspan(9.5, float(np.max(modes)) + 0.5, color="tab:red", alpha=0.08)
        axis.set_xlabel("Radial Fourier mode")
        axis.set_ylabel("Model / truth energy")
        axis.grid(alpha=0.25)
        diagnosis = results[field]
        tail = diagnosis["tail_k10_plus"]
        text = (
            f"frozen median = {diagnosis['frozen_median_modewise_energy_ratio']:.2f}×\n"
            f"integrated = {diagnosis['integrated_energy_ratio']:.3f}×\n"
            f"k≥10 model share = {100.0 * tail['model_fraction_of_valid_energy']:.3f}%"
        )
        axis.text(
            0.03,
            0.97,
            text,
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    figure.suptitle(
        "Model C day-360 deep-pressure spectral diagnosis\n"
        "Frozen gate remains failed; excess is confined to a tiny high-k tail"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(contract_path: str | Path) -> dict[str, Any]:
    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise ValueError("spectral diagnosis contract version changed")
    metrics_path, arrays_path, source_manifest_path = _verify_source(contract)
    metrics = json.loads(metrics_path.read_text())
    if metrics["forward_gate"]["status"] != "fail":
        raise ValueError("source forward gate is not the frozen failed gate")

    with np.load(arrays_path, allow_pickle=False) as arrays:
        modes = np.asarray(arrays["spectral_modes"], dtype=np.float64)
        results = {
            field: diagnose_field(
                modes,
                arrays[f"spectrum_{field}_truth_day360"],
                arrays[f"spectrum_{field}_model_day360"],
            )
            for field in contract["analysis"]["fields"]
        }

    tail_limit = float(
        contract["interpretation_rule"][
            "negligible_tail_threshold_fraction_of_total_model_energy"
        ]
    )
    integrated_low, integrated_high = contract["interpretation_rule"][
        "basin_scale_integrated_energy_ratio_bounds"
    ]
    localized = all(
        result["tail_k10_plus"]["model_fraction_of_valid_energy"] < tail_limit
        and integrated_low <= result["integrated_energy_ratio"] <= integrated_high
        for result in results.values()
    )
    diagnosis = {
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": _file_sha256(resolved),
        "source_forward_gate": {
            "status": metrics["forward_gate"]["status"],
            "complete_adjoint_campaign_may_open": metrics["forward_gate"][
                "complete_adjoint_campaign_may_open"
            ],
            "spectral_energy_ratio_model_over_truth": metrics["forward_gate"][
                "spectral_energy_ratio_model_over_truth"
            ],
        },
        "field_diagnosis": results,
        "classification": (
            "localized_small_amplitude_high_wavenumber_tail"
            if localized
            else "material_or_basin_scale_spectral_error"
        ),
        "frozen_gate_reinterpreted": False,
        "complete_adjoint_campaign_may_open": False,
        "next_action": contract["next_decision"][
            "localized_tail" if localized else "basin_scale_or_material_energy_error"
        ],
    }
    diagnosis["content_sha256"] = _json_sha256(diagnosis)

    output = Path(contract["output"]["directory"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite spectral diagnosis: {output}")
    output.mkdir(parents=True)
    diagnosis_path = output / OUTPUT_NAMES[0]
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")

    csv_path = output / OUTPUT_NAMES[1]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "field",
                "mode",
                "truth_energy",
                "model_energy",
                "model_over_truth",
                "frozen_valid",
                "band",
            ),
        )
        writer.writeheader()
        for field, result in results.items():
            for row in result["per_mode"]:
                writer.writerow({"field": field, **row})

    figure_path = output / OUTPUT_NAMES[2]
    _plot(figure_path, modes, results)
    readme_path = output / OUTPUT_NAMES[4]
    readme_path.write_text(
        "# Model C day-360 deep-pressure spectral diagnosis\n\n"
        "This immutable posthoc package diagnoses the sole failed frozen "
        "long-forward criterion. It does not alter or rescue the original gate. "
        f"Classification: `{diagnosis['classification']}`.\n"
    )
    manifest = {
        "version": VERSION,
        "status": "complete",
        "contract": str(resolved),
        "contract_sha256": _file_sha256(resolved),
        "source": {
            "metrics_sha256": _file_sha256(metrics_path),
            "arrays_sha256": _file_sha256(arrays_path),
            "manifest_sha256": _file_sha256(source_manifest_path),
        },
        "artifacts": {
            name: _file_sha256(output / name)
            for name in OUTPUT_NAMES
            if name != "manifest.json"
        },
        "inference_state_opened": False,
        "source_inference_outputs_read": True,
        "frozen_gate_reinterpreted": False,
    }
    manifest["content_sha256"] = _json_sha256(manifest)
    (output / OUTPUT_NAMES[3]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return diagnosis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    diagnosis = run(args.contract)
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
