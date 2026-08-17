"""Compare the FNO's sensitivity maps against MITgcm's TAF adjoint.

Implements section 6 of docs/fno_adjoint_plan.md.  Reads the MITgcm ``.npz``
written by section 10 of the ground-truth plan and the FNO ``.npz`` written by
``scripts/fno_adjoint.py``, and scores one against the other.

    python scripts/compare_adjoint_maps.py

-----------------------------------------------------------------------------
Read this before reading any number this script prints
-----------------------------------------------------------------------------

**Do not run this before MITgcm's gate G1 passes.**  An unvalidated
``adxx_etan`` is not ground truth, and comparing against it produces a
confident-looking table of meaningless numbers.  The script refuses to run
unless the MITgcm report declares G1 passed.

**No pass/fail threshold is declared on the primary metrics.**  This is the
first measurement of this quantity for this model.  Inventing a threshold
before seeing the number would be reverse-engineering a verdict.  Gates F1 to
F4 on the FNO side and G0 to G5 on the MITgcm side guard the *pipelines*; the
science below is reported, not graded.

**E1 is the primary map**, declared before looking.  ``E1 + E2`` answers a
different question --- the response to an offset applied at both input times ---
and is reported as a secondary hypothesis only.

The pairings are fixed by the two plans:

===========================  ==============================  ================
FNO array                    MITgcm array                    What it is
===========================  ==============================  ================
``S_fno_present``            ``S10``                         Run A, 10-day
``S_fno_20day``              ``S20``                         Run B, 20-day
``S_fno_history``            --                              FNO only
``S_fno_mean_only``          the weight field itself         exact reference
===========================  ==============================  ================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjoint_metrics import (
    primary_metrics,
    spectrum_ratio,
    structural_metrics,
)

FNO_DIRECTORY = Path("outputs") / "af_fno" / "adjoint" / "fno_s0_adjoint_v1"
MITGCM_DIRECTORY = Path("outputs") / "af_fno" / "adjoint" / "mitgcm_s0_adjoint_v1"
OUTPUT_DIRECTORY = Path("outputs") / "af_fno" / "adjoint" / "comparison_s0_v1"

REPORT_NAME = "comparison_report.json"
FIGURE_NAMES = ("adjoint_comparison_maps.png", "adjoint_comparison_structure.png")

#: Which FNO map is scored against which MITgcm map.
PAIRINGS = (
    ("E1_present_vs_run_a", "S_fno_present", "S10", "10-day, present slot"),
    ("E3_twenty_day_vs_run_b", "S_fno_20day", "S20", "20-day, two chained calls"),
)


class AdjointComparisonError(RuntimeError):
    """Raised when the two sides cannot be compared as they stand."""


def _load(directory: Path, arrays_name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one side's arrays and its report, or say precisely what is missing."""

    arrays_path = directory / arrays_name
    report_path = directory / "report.json"
    if not arrays_path.is_file():
        raise AdjointComparisonError(f"missing arrays: {arrays_path}")
    if not report_path.is_file():
        raise AdjointComparisonError(f"missing report: {report_path}")
    with np.load(arrays_path) as stored:
        arrays = {key: np.asarray(stored[key]) for key in stored.files}
    return arrays, json.loads(report_path.read_text())


def check_shared_contract(
    fno: Mapping[str, np.ndarray],
    mitgcm: Mapping[str, np.ndarray],
    fno_report: Mapping[str, Any],
    mitgcm_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse to compare two maps that were not built on the same contract.

    Any convention that differs between the two sides shows up as a difference
    in the maps and would be mistaken for a model error.  The weight-field hash
    is the sharpest of these checks: if the two sides read different bytes for
    ``w`` they are answering different questions, and every metric below is
    meaningless.  The plan calls for this to be an assertion, not a note.
    """

    findings: dict[str, Any] = {}

    fno_target = tuple(int(v) for v in np.asarray(fno["target_ij"]).ravel())
    mitgcm_target = tuple(int(v) for v in np.asarray(mitgcm["target_ij"]).ravel())
    if fno_target != mitgcm_target:
        raise AdjointComparisonError(
            f"the two sides disagree about p*: FNO {fno_target}, MITgcm {mitgcm_target}"
        )
    findings["target_ij"] = list(fno_target)

    fno_wet = np.asarray(fno["wet_mask"], dtype=bool)
    mitgcm_wet = np.asarray(mitgcm["wet_mask"], dtype=bool)
    if fno_wet.shape != mitgcm_wet.shape or not np.array_equal(fno_wet, mitgcm_wet):
        raise AdjointComparisonError("the two sides disagree about the wet mask")
    findings["wet_cell_count"] = int(fno_wet.sum())

    area_difference = float(
        np.abs(np.asarray(fno["rA"], dtype=np.float64) - np.asarray(mitgcm["rA"], dtype=np.float64)).max()
    )
    if area_difference > 1.0e-6 * float(np.asarray(fno["rA"]).max()):
        raise AdjointComparisonError("the two sides disagree about the cell areas")
    findings["max_abs_cell_area_difference_m2"] = area_difference

    fno_digest = fno_report["shared_contract"]["cost_weight_sha256"]["ssh_anomaly"]
    mitgcm_digest = _mitgcm_weight_digest(mitgcm_report)
    if mitgcm_digest is not None and fno_digest != mitgcm_digest:
        raise AdjointComparisonError(
            "the two sides read different cost weight files:\n"
            f"  FNO    {fno_digest}\n  MITgcm {mitgcm_digest}\n"
            "the comparison would silently become a convention test"
        )
    findings["cost_weight_sha256"] = fno_digest
    findings["cost_weight_hash_cross_checked"] = mitgcm_digest is not None

    gate_g1 = _mitgcm_gate_g1(mitgcm_report)
    if gate_g1 is not True:
        raise AdjointComparisonError(
            "MITgcm gate G1 (the gradient check) is not recorded as passed; an unvalidated "
            "adxx_etan is not ground truth. Until it clears, the FNO maps stand alone."
        )
    findings["mitgcm_gate_g1_passed"] = True
    return findings


def _mitgcm_weight_digest(report: Mapping[str, Any]) -> str | None:
    """Find the weight-field hash wherever the MITgcm extractor recorded it."""

    for path in (
        ("shared_contract", "cost_weight_sha256", "ssh_anomaly"),
        ("cost_weight_sha256",),
        ("run_manifest", "cost_weight_sha256"),
    ):
        value: Any = report
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, str):
            return value
    return None


def _mitgcm_gate_g1(report: Mapping[str, Any]) -> Any:
    """Read the MITgcm gradient-check verdict, tolerating either report layout."""

    gates = report.get("gates") or report.get("acceptance_gates") or {}
    entry = gates.get("G1") if isinstance(gates, Mapping) else None
    if isinstance(entry, Mapping):
        return entry.get("passed")
    return entry


def compare(
    fno_map: np.ndarray,
    mitgcm_map: np.ndarray,
    wet: np.ndarray,
    target: tuple[int, int],
) -> dict[str, Any]:
    """Every metric for one pairing: primary, then structural, then the difference."""

    fno_structure = structural_metrics(fno_map, wet, target)
    mitgcm_structure = structural_metrics(mitgcm_map, wet, target)
    return {
        "primary": primary_metrics(fno_map, mitgcm_map, wet),
        "fno_structure": fno_structure,
        "mitgcm_structure": mitgcm_structure,
        "structural_comparison": {
            # MITgcm is exactly zero on land, so its leakage is the reference
            # against which the FNO's is stated -- no modelling assumption needed.
            "land_leakage_fno_max_abs": fno_structure["land_leakage"]["max_abs_dry"],
            "land_leakage_mitgcm_max_abs": mitgcm_structure["land_leakage"]["max_abs_dry"],
            "e_folding_cells_fno": fno_structure["radial_decay"]["e_folding_cells"],
            "e_folding_cells_mitgcm": mitgcm_structure["radial_decay"]["e_folding_cells"],
            "western_ratio_fno": fno_structure["western_band"]["boundary_to_interior_ratio"],
            "western_ratio_mitgcm": mitgcm_structure["western_band"]["boundary_to_interior_ratio"],
            "spectral_power_ratio_per_bin": spectrum_ratio(
                fno_structure["radial_spectrum"], mitgcm_structure["radial_spectrum"]
            ),
        },
        "difference_in_western_band": _band_error(fno_map, mitgcm_map, wet),
    }


def _band_error(fno_map: np.ndarray, mitgcm_map: np.ndarray, wet: np.ndarray) -> dict[str, float]:
    """Where the disagreement lives: western boundary band versus interior.

    ``western-boundary-ratio-degrades`` records a day-2000 defect in that band
    that the forecast acceptance gate never scores.  If the sensitivity maps
    disagree most there too, the adjoint has localised the same defect.
    """

    from oceanfno.dataset import western_boundary_mask

    mask = np.asarray(wet, dtype=bool)
    band = western_boundary_mask(mask, 4) & mask
    interior = mask & ~band
    difference = np.asarray(fno_map, dtype=np.float64) - np.asarray(mitgcm_map, dtype=np.float64)

    def relative(selection: np.ndarray) -> float:
        reference = float(np.linalg.norm(mitgcm_map[selection]))
        return float(np.linalg.norm(difference[selection]) / reference) if reference > 0.0 else float("nan")

    return {
        "relative_l2_western_band": relative(band),
        "relative_l2_interior": relative(interior),
        "fraction_of_squared_error_in_band": float(
            np.sum(difference[band] ** 2) / np.sum(difference[mask] ** 2)
        )
        if np.sum(difference[mask] ** 2) > 0.0
        else float("nan"),
        "band_fraction_of_wet_cells": float(band.sum() / mask.sum()),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
         "figure.dpi": 120, "savefig.dpi": 180}
    )


def _masked(field: np.ndarray, wet: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where((~wet) | (~np.isfinite(field)), field)


def figure_maps(
    output: Path,
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray, str]],
    wet: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    results: Mapping[str, Any],
) -> None:
    """One row per pairing: MITgcm, FNO, difference."""

    rows = len(pairs)
    figure, axes = plt.subplots(
        rows, 3, figsize=(12.6, 4.2 * rows), squeeze=False, constrained_layout=True
    )
    for row, (name, (fno_map, mitgcm_map, label)) in enumerate(pairs.items()):
        bound = float(np.percentile(np.abs(mitgcm_map[wet]), 99.0)) or 1.0
        difference = fno_map - mitgcm_map
        difference_bound = float(np.percentile(np.abs(difference[wet]), 99.0)) or bound
        panels = (
            (mitgcm_map, bound, f"MITgcm TAF adjoint\n{label}"),
            (fno_map, bound, f"Frozen FNO\n{label}"),
            (difference, difference_bound, "FNO − MITgcm"),
        )
        images = []
        for column, (field, limit, title) in enumerate(panels):
            axis = axes[row, column]
            images.append(
                axis.pcolormesh(
                    longitude, latitude, _masked(field, wet),
                    cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto",
                )
            )
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_facecolor("0.86")
            axis.set_xlabel("Longitude (°)")
        metrics = results[name]["primary"]
        axes[row, 0].set_ylabel(
            f"r = {metrics['pattern_correlation']:.3f}\n"
            f"rel. L2 = {metrics['relative_l2']:.3f}\n"
            f"amp = {metrics['amplitude_ratio']:.3f}\nLatitude (°)"
        )
        figure.colorbar(images[0], ax=axes[row, :2].tolist(), label="Sensitivity", shrink=0.85)
        figure.colorbar(images[2], ax=axes[row, 2], label="Difference", shrink=0.85)
    figure.suptitle(
        "Adjoint sensitivity of the day-7220 SSH anomaly at p*.  Colour clipped at the 99th "
        "percentile of the MITgcm map.  No pass/fail threshold is declared on these metrics."
    )
    figure.savefig(output / FIGURE_NAMES[0], bbox_inches="tight")
    plt.close(figure)


def figure_structure(output: Path, results: Mapping[str, Any]) -> None:
    """Radial decay and radial spectrum, both sides overlaid."""

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    for name, result in results.items():
        for side, style in (("fno", "-"), ("mitgcm", "--")):
            decay = result[f"{side}_structure"]["radial_decay"]
            axes[0].semilogy(
                decay["radius_cells"], decay["rms_abs_sensitivity"],
                linestyle=style, linewidth=1.1,
                label=f"{side.upper()} {name}  (L = {decay['e_folding_cells']:.1f} cells)",
            )
            spectrum = result[f"{side}_structure"]["radial_spectrum"]
            power = np.asarray(spectrum["power_per_bin"])
            axes[1].semilogy(
                np.arange(1, power.size + 1), np.maximum(power, 1.0e-30),
                linestyle=style, marker="o", markersize=3, linewidth=1.1,
                label=f"{side.upper()} {name}",
            )
    axes[0].set_xlabel("Distance from p* (grid cells)")
    axes[0].set_ylabel("RMS $|\\partial J/\\partial \\eta|$")
    axes[0].set_title("Radial decay: effective range of influence")
    axes[0].legend(fontsize=6.5)
    axes[1].set_xlabel("Radial wavenumber bin (12-bin tapered convention)")
    axes[1].set_ylabel("Absolute power")
    axes[1].set_title("Radial spectrum — absolute power, never a fraction")
    axes[1].legend(fontsize=6.5)
    figure.savefig(output / FIGURE_NAMES[1], bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(project_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Score both pairings, write the report and the two figures."""

    fno_arrays, fno_report = _load(project_root / FNO_DIRECTORY, "fno_s0_adjoint_arrays.npz")
    mitgcm_arrays, mitgcm_report = _load(
        project_root / MITGCM_DIRECTORY, "mitgcm_s0_adjoint_arrays.npz"
    )
    shared = check_shared_contract(fno_arrays, mitgcm_arrays, fno_report, mitgcm_report)

    wet = np.asarray(fno_arrays["wet_mask"], dtype=bool)
    target = (int(shared["target_ij"][0]), int(shared["target_ij"][1]))

    results, pairs = {}, {}
    for name, fno_key, mitgcm_key, label in PAIRINGS:
        if fno_key not in fno_arrays or mitgcm_key not in mitgcm_arrays:
            print(f"  skipping {name}: {fno_key} or {mitgcm_key} is absent")
            continue
        fno_map = np.asarray(fno_arrays[fno_key], dtype=np.float64)
        mitgcm_map = np.asarray(mitgcm_arrays[mitgcm_key], dtype=np.float64)
        results[name] = compare(fno_map, mitgcm_map, wet, target)
        pairs[name] = (fno_map, mitgcm_map, label)
        primary = results[name]["primary"]
        print(
            f"  {name}: r = {primary['pattern_correlation']:+.4f}  "
            f"rel L2 = {primary['relative_l2']:.4f}  "
            f"amplitude = {primary['amplitude_ratio']:.4f}  "
            f"sign agreement = {primary['sign_agreement']:.3f}"
        )
    if not results:
        raise AdjointComparisonError("no pairing could be formed from the two archives")

    # Secondary hypothesis, reported only after E1 has been scored on its own.
    if "S_fno_history" in fno_arrays and "S10" in mitgcm_arrays:
        total = np.asarray(fno_arrays["S_fno_present"], dtype=np.float64) + np.asarray(
            fno_arrays["S_fno_history"], dtype=np.float64
        )
        results["secondary_E1_plus_E2_vs_run_a"] = {
            "primary": primary_metrics(total, np.asarray(mitgcm_arrays["S10"], dtype=np.float64), wet),
            "status": (
                "secondary hypothesis about how the operator distributes dependence across its "
                "two input slots; E1 alone is the declared primary comparison"
            ),
        }

    output = (project_root / OUTPUT_DIRECTORY).resolve()
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    output.mkdir(parents=True, exist_ok=True)

    _style()
    figure_maps(
        output, pairs, wet,
        np.asarray(fno_arrays["longitude_deg"]), np.asarray(fno_arrays["latitude_deg"]),
        results,
    )
    figure_structure(output, {k: v for k, v in results.items() if "fno_structure" in v})

    report = {
        "status": "complete",
        "version": "comparison_s0_v1",
        "plan": "docs/fno_adjoint_plan.md section 6",
        "shared_contract": shared,
        "fno_report_content_sha256": fno_report.get("content_sha256"),
        "mitgcm_report_content_sha256": mitgcm_report.get("content_sha256"),
        "results": results,
        "no_threshold_declared": (
            "this is the first measurement of this quantity for this model; the primary metrics "
            "are reported, not graded"
        ),
        "figures": list(FIGURE_NAMES),
    }
    (output / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {output}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="replace an existing comparison")
    arguments = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    try:
        run(project_root, force=arguments.force)
    except AdjointComparisonError as error:
        print(f"comparison not possible yet:\n  {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
