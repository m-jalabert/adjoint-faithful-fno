"""Compare the emulator's sensitivity maps against MITgcm's, Phase A.
Implements step 12 of docs/Adjoint_study_Phase_A.md.  Reads the two .npz
packages written by scripts/fno_adjoint_ft90.py and
scripts/extract_mitgcm_adjoint_phase_a.py and produces the section 7 tables and
the section 8 diagnostics.

-----------------------------------------------------------------------------
What is compared against what
-----------------------------------------------------------------------------

``S_forced`` is the primary comparison.  MITgcm's adjoint is linearized about
the truth trajectory by construction, so the forced chain puts the emulator's
Jacobian at exactly the same points and the difference is Jacobian error alone.
``S_free`` is reported alongside as the operationally relevant map, and the
gap between them is the trajectory-drift term --- measured on the emulator side
as 0.09 relative L2 at ninety days, so it is not what dominates.

-----------------------------------------------------------------------------
Refuses to run before gate G1-90
-----------------------------------------------------------------------------

An unvalidated ``adxx_etan`` at a new window is not ground truth.  The
executable's v1 gates certify the *build*; they do not certify a ninety-day
tape.  ``--before-g1`` overrides, and stamps every output as provisional.

    python scripts/compare_adjoint_maps_phase_a.py
    python scripts/compare_adjoint_maps_phase_a.py --before-g1   # provisional
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import adjoint_metrics as metrics  # noqa: E402

FNO_RELATIVE = Path("outputs/af_fno/adjoint/fno_ft90_s0_adjoint_v1")
MITGCM_RELATIVE = Path("outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2")
OUTPUT_RELATIVE = Path("outputs/af_fno/adjoint/comparison_phase_a_v1")

#: The emulator's spectral path keeps |k| <= 16 of 37 on its 74x74 padded grid.
OPERATOR_CUTOFF_CELLS = 74.0 / 16.0


class ComparisonError(RuntimeError):
    """Raised when the two sides cannot be compared as they stand."""


def load(path: Path, label: str) -> dict[str, np.ndarray]:
    archive = path / f"{path.name}.npz"
    if not archive.is_file():
        archive = next(path.glob("*.npz"), None)
    if archive is None or not archive.is_file():
        raise ComparisonError(f"no .npz in {path} --- run the {label} extractor first")
    with np.load(archive) as stored:
        return {key: stored[key] for key in stored.files}


def check_shared_contract(fno: dict[str, np.ndarray], mitgcm: dict[str, np.ndarray]) -> dict[str, Any]:
    """The two sides must agree about the basin, the target and the weights.
    Checked by comparing the arrays both sides carried through, not by
    re-deriving anything.  A disagreement here means the maps are not
    subtractable and every metric below would be measuring the convention.
    """
    problems = []
    if not np.array_equal(fno["wet_mask"], mitgcm["wet_mask"]):
        problems.append("wet masks differ")
    if not np.array_equal(fno["target_ij"], mitgcm["target_ij"]):
        problems.append("target cells differ")
    if not np.array_equal(fno["lead_days"], mitgcm["lead_days"]):
        problems.append("lead days differ")
    for name in ("w_ssh_anomaly", "w_ssh_anomaly_kernel", "w_mean_only"):
        if name in fno and name in mitgcm and not np.array_equal(fno[name], mitgcm[name]):
            problems.append(f"{name} differs between the two sides")
    if problems:
        raise ComparisonError("; ".join(problems))
    return {
        "wet_cells": int(fno["wet_mask"].astype(bool).sum()),
        "target_ij": fno["target_ij"].tolist(),
        "lead_days": fno["lead_days"].tolist(),
        "weight_fields_identical": True,
    }


def compare_maps(
    emulator: np.ndarray, reference: np.ndarray, wet: np.ndarray, target: tuple[int, int]
) -> dict[str, Any]:
    result = dict(metrics.primary_metrics(emulator, reference, wet))
    result["emulator"] = metrics.structural_metrics(emulator, wet, target)
    result["reference"] = metrics.structural_metrics(reference, wet, target)
    result["spectrum_ratio"] = metrics.spectrum_ratio(
        result["emulator"]["radial_spectrum"], result["reference"]["radial_spectrum"]
    )
    result["boundary_ratio_emulator"] = result["emulator"]["western_band"]["boundary_to_interior_ratio"]
    result["boundary_ratio_reference"] = result["reference"]["western_band"]["boundary_to_interior_ratio"]
    result["norm_emulator"] = float(np.linalg.norm(emulator[wet]))
    result["norm_reference"] = float(np.linalg.norm(reference[wet]))
    return result


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
    return np.ma.masked_where((~wet) | (~np.isfinite(field)), field)


def _bound(field: np.ndarray, wet: np.ndarray, percentile: float = 99.0) -> float:
    values = np.abs(field[wet])
    value = float(np.percentile(values, percentile))
    return value if value > 0.0 else float(values.max() or 1.0)


def figure_side_by_side(
    output: Path,
    emulator: np.ndarray,
    reference: np.ndarray,
    wet: np.ndarray,
    target: tuple[int, int],
    lead: int,
    comparison: dict[str, Any],
) -> None:
    """Reference, emulator and difference, on one shared colour scale.
    The first two panels share a scale on purpose: an emulator map rendered on
    its own limits would look right however wrong its amplitude was, and
    amplitude is one of the four primary metrics.
    """
    _style()
    bound = _bound(reference, wet)
    difference = emulator - reference
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, field, title, limit in (
        (axes[0], reference, f"MITgcm / TAF\nmax|S| = {np.abs(reference[wet]).max():.3e}", bound),
        (axes[1], emulator, f"emulator (S_forced)\nmax|S| = {np.abs(emulator[wet]).max():.3e}", bound),
        (axes[2], difference, "emulator - MITgcm", _bound(difference, wet)),
    ):
        image = axis.pcolormesh(
            _masked(field, wet), cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto"
        )
        axis.plot(target[1] + 0.5, target[0] + 0.5, "ko", ms=4, mfc="none", mew=0.9)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(
        f"lead {lead} d.  pattern correlation {comparison['pattern_correlation']:.4f}, "
        f"relative L2 {comparison['relative_l2']:.4f}, amplitude ratio "
        f"{comparison['amplitude_ratio']:.4f}, sign agreement {comparison['sign_agreement']:.4f}\n"
        "first two panels share a colour scale; both models have global instantaneous "
        "domain of dependence, so no disagreement is attributable to one being local",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


def figure_metric_curves(
    output: Path, comparisons: dict[str, Any], amplitude: dict[str, Any], leads: list[int]
) -> None:
    """How the agreement evolves with lead --- section 8.3."""

    _style()
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)
    primary = comparisons.get("ssh_anomaly", {})

    axis = axes[0]
    for chain, marker in (("forced", "o-"), ("free", "s--")):
        if chain in primary:
            axis.plot(leads, [primary[chain][str(l)]["pattern_correlation"] for l in leads], marker, label=chain)
    axis.set_xlabel("lead [days]")
    axis.set_ylabel("pattern correlation")
    axis.set_title("does the emulator put sensitivity\nin the right places?")
    axis.axhline(0.0, color="0.6", lw=0.8)
    axis.legend()
    axis.grid(alpha=0.3)

    axis = axes[1]
    for chain, marker in (("forced", "o-"), ("free", "s--")):
        if chain in primary:
            axis.plot(leads, [primary[chain][str(l)]["relative_l2"] for l in leads], marker, label=chain)
    axis.set_xlabel("lead [days]")
    axis.set_ylabel("relative L2")
    axis.set_title("where in the ninety days\nthe agreement breaks")
    axis.grid(alpha=0.3)
    axis.legend()

    axis = axes[2]
    if amplitude:
        axis.semilogy(leads, [amplitude[str(l)]["mitgcm_norm"] for l in leads], "o-", label="MITgcm")
        axis.semilogy(leads, [amplitude[str(l)]["emulator_norm"] for l in leads], "s--", label="emulator")
    axis.set_xlabel("lead [days]")
    axis.set_ylabel("||S|| over wet cells")
    axis.set_title("amplitude against truth\n(does the true adjoint decay too?)")
    axis.grid(alpha=0.3, which="both")
    axis.legend()

    figure.savefig(output)
    plt.close(figure)


def figure_spectra(
    output: Path, comparisons: dict[str, Any], leads: list[int]
) -> None:
    """Absolute power per radial bin, with the operator's own cutoff drawn on.
    local-branch-gamma-ablation is the reason this is absolute power and never
    a fraction: the high-wavenumber fraction can fall simply because the
    low-wavenumber power grew.
    """
    _style()
    primary = comparisons.get("ssh_anomaly", {}).get("forced", {})
    if not primary:
        return
    figure, axes = plt.subplots(1, len(leads), figsize=(3.5 * len(leads), 4.0),
                               constrained_layout=True, squeeze=False)
    for axis, lead in zip(axes.ravel(), leads):
        record = primary[str(lead)]
        for label, spectrum, marker in (
            ("MITgcm", record["reference"]["radial_spectrum"], "o-"),
            ("emulator", record["emulator"]["radial_spectrum"], "s--"),
        ):
            axis.loglog(
                spectrum["bin_upper_wavenumber"],
                np.maximum(spectrum["power_per_bin"], 1e-300),
                marker,
                ms=3,
                label=label,
            )
        axis.axvline(1.0 / OPERATOR_CUTOFF_CELLS, color="k", ls="--", lw=1.0)
        axis.set_xlabel("radial wavenumber [1/cell]")
        axis.set_title(f"lead {lead} d")
        axis.grid(alpha=0.3, which="both")
    axes.ravel()[0].set_ylabel("absolute power")
    axes.ravel()[0].legend()
    figure.suptitle(
        "Radial spectra of |S|.  Dashed line is the emulator's own mode cutoff at 4.63 "
        "cells:\na deficit to its right is the truncation, a deficit to its left is the "
        "learned dynamics.",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--before-g1", action="store_true",
                        help="compare before gate G1-90 has passed; marks everything provisional")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    project_root = (
        Path(arguments.project_root).resolve()
        if arguments.project_root
        else Path(__file__).resolve().parent.parent
    )
    output = (project_root / OUTPUT_RELATIVE).resolve()
    if output.exists() and not arguments.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    fno_report = json.loads((project_root / FNO_RELATIVE / "report.json").read_text())
    mitgcm_report = json.loads((project_root / MITGCM_RELATIVE / "report.json").read_text())

    g1 = mitgcm_report.get("gates", {}).get("G1")
    if not (g1 and g1.get("passed")):
        message = (
            "gate G1-90 has not passed, so adxx_etan at this window is not yet ground "
            "truth.  Pass --before-g1 to compare anyway; every output is then stamped "
            "provisional."
        )
        if not arguments.before_g1:
            raise ComparisonError(message)
        print(f"WARNING: {message}")

    fno = load(project_root / FNO_RELATIVE, "emulator")
    mitgcm = load(project_root / MITGCM_RELATIVE, "MITgcm")
    shared = check_shared_contract(fno, mitgcm)

    wet = fno["wet_mask"].astype(bool)
    target = (int(fno["target_ij"][0]), int(fno["target_ij"][1]))
    leads = [int(lead) for lead in fno["lead_days"]]

    comparisons: dict[str, Any] = {}
    for objective in ("ssh_anomaly", "ssh_anomaly_kernel"):
        reference_key = f"S_{objective}"
        if reference_key not in mitgcm:
            continue
        comparisons[objective] = {}
        for chain in ("forced", "free"):
            emulator_key = f"S_{objective}_{chain}"
            if emulator_key not in fno:
                continue
            comparisons[objective][chain] = {
                str(lead): compare_maps(
                    fno[emulator_key][index], mitgcm[reference_key][index], wet, target
                )
                for index, lead in enumerate(leads)
            }

    # Section 5.3: which of the two errors dominates.  Only meaningful once both
    # chains have been compared against the same reference.
    decomposition = {}
    primary = comparisons.get("ssh_anomaly", {})
    if "forced" in primary and "free" in primary:
        for lead in leads:
            jacobian = primary["forced"][str(lead)]["relative_l2"]
            total = primary["free"][str(lead)]["relative_l2"]
            drift = metrics.relative_l2(
                fno["S_ssh_anomaly_free"][leads.index(lead)],
                fno["S_ssh_anomaly_forced"][leads.index(lead)],
                wet,
            )
            decomposition[str(lead)] = {
                "jacobian_error_S_forced_vs_mitgcm": jacobian,
                "trajectory_error_S_free_vs_S_forced": drift,
                "total_S_free_vs_mitgcm": total,
                "dominant": "jacobian" if jacobian > drift else "trajectory",
            }

    # Section 8.2: does the true adjoint decay the way the emulator's does?
    amplitude = {}
    if "S_ssh_anomaly" in mitgcm and "S_ssh_anomaly_forced" in fno:
        for index, lead in enumerate(leads):
            amplitude[str(lead)] = {
                "emulator_norm": float(np.linalg.norm(fno["S_ssh_anomaly_forced"][index][wet])),
                "mitgcm_norm": float(np.linalg.norm(mitgcm["S_ssh_anomaly"][index][wet])),
            }
        base = amplitude[str(leads[0])]
        for lead in leads:
            record = amplitude[str(lead)]
            record["emulator_ratio_to_lead_10"] = record["emulator_norm"] / base["emulator_norm"]
            record["mitgcm_ratio_to_lead_10"] = record["mitgcm_norm"] / base["mitgcm_norm"]
            record["decay_ratio"] = (
                record["emulator_ratio_to_lead_10"] / record["mitgcm_ratio_to_lead_10"]
            )

    # Section 8.1: the period-2 probe, now with a reference to compare against.
    period_2 = {}
    if "S_backward" in mitgcm and "S_backward_ssh_anomaly_free" in fno:
        reference_sweep = mitgcm["S_backward"]
        days = mitgcm.get("backward_days")
        emulator_sweep = fno["S_backward_ssh_anomaly_free"]
        emulator_days = fno.get("backward_days")
        if days is not None and emulator_days is not None:
            index = {int(day): position for position, day in enumerate(days)}
            matched = [index[int(day)] for day in emulator_days if int(day) in index]
            if len(matched) == emulator_sweep.shape[0]:
                aligned = reference_sweep[matched]
                period_2 = {
                    "emulator": [
                        metrics.pattern_correlation(emulator_sweep[k], emulator_sweep[k + 1], wet)
                        for k in range(emulator_sweep.shape[0] - 1)
                    ],
                    "mitgcm": [
                        metrics.pattern_correlation(aligned[k], aligned[k + 1], wet)
                        for k in range(aligned.shape[0] - 1)
                    ],
                    "note": (
                        "strongly negative in the emulator but not in MITgcm would be a "
                        "period-2 computational mode; both positive is the healthy case"
                    ),
                }

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "comparison_phase_a_v1",
        "plan": "docs/Adjoint_study_Phase_A.md",
        "provisional": bool(arguments.before_g1),
        "provisional_reason": (
            "gate G1-90 had not passed when this was written; adxx_etan at the ninety-day "
            "window is not yet certified ground truth"
        )
        if arguments.before_g1
        else None,
        "emulator": {
            "version": fno_report["model"]["version"],
            "checkpoint_sha256": fno_report["model"]["checkpoint_sha256"],
        },
        "reference": {
            "mitgcm_commit": mitgcm_report["mitgcm_commit"],
            "taf_version": mitgcm_report["taf_version"],
            "runs_present": mitgcm_report["runs_present"],
        },
        "shared_contract": shared,
        "operator_cutoff_cells": OPERATOR_CUTOFF_CELLS,
        "primary_metrics": comparisons,
        "error_decomposition": decomposition,
        "amplitude_against_truth": amplitude,
        "period_2_probe": period_2,
        "no_threshold": (
            "F1-F7 guard the emulator pipeline and G0-G5 guard the reference; nothing here "
            "is graded. This is the first measurement of this quantity for this model"
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if primary.get("forced") and "S_ssh_anomaly" in mitgcm:
        for index, lead in enumerate(leads):
            figure_side_by_side(
                output / f"comparison_lead_{lead:03d}.png",
                fno["S_ssh_anomaly_forced"][index],
                mitgcm["S_ssh_anomaly"][index],
                wet,
                target,
                lead,
                primary["forced"][str(lead)],
            )
        figure_metric_curves(output / "comparison_metric_curves.png", comparisons, amplitude, leads)
        figure_spectra(output / "comparison_spectra.png", comparisons, leads)

    print(f"wrote {output}")
    if primary.get("forced"):
        print(f"\n{'lead':>6} {'pattern corr':>13} {'rel L2':>9} {'amp ratio':>10} {'sign agree':>11}")
        for lead in leads:
            m = primary["forced"][str(lead)]
            print(
                f"{lead:>6} {m['pattern_correlation']:>13.5f} {m['relative_l2']:>9.4f} "
                f"{m['amplitude_ratio']:>10.4f} {m['sign_agreement']:>11.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
