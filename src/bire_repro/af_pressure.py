"""MITgcm-consistent hydrostatic pressure postprocessing for AF--FNO states.

The active tutorial uses z coordinates, the linear equation of state, no
salinity stepping, and a linear free surface.  MITgcm's ``PHIHYD`` diagnostic
is hydrostatic pressure anomaly divided by ``rhoConst`` (m2 s-2), not pressure
in Pa.  This module reproduces the finite-difference integration in
``model/src/calc_phi_hyd.F`` and the free-surface contribution in
``model/src/diags_phi_hyd.F``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .mds import read_mds


GRAVITY_M_S2 = 9.81
RHO_CONST_KG_M3 = 999.8
THERMAL_EXPANSION_PER_C = 2.0e-4
DRF_M = np.asarray(
    (50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
    dtype=np.float64,
)
T_REF_C = np.asarray(
    (30, 27, 24, 21, 18, 15, 13, 11, 9, 7, 6, 5, 4, 3, 2),
    dtype=np.float64,
)
PHIHYD_LEVELS = {
    "phihyd_surface": 0,
    "phihyd_mid": 7,
    "phihyd_bottom": 14,
}


def _vertical_distances() -> tuple[np.ndarray, np.ndarray]:
    """Distances above/below each tracer center used by MITgcm's FD form."""

    interfaces = -np.concatenate(([0.0], np.cumsum(DRF_M)))
    centers = 0.5 * (interfaces[:-1] + interfaces[1:])
    center_spacing = np.diff(centers)
    above = np.empty(DRF_M.size, dtype=np.float64)
    below = np.empty(DRF_M.size, dtype=np.float64)
    above[0] = interfaces[0] - centers[0]
    above[1:] = -0.5 * center_spacing
    below[:-1] = -0.5 * center_spacing
    below[-1] = centers[-1] - interfaces[-1]
    return above, below


def phihyd_from_theta_eta(
    theta_c: np.ndarray,
    eta_m: np.ndarray,
    wet: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct total ``PHIHYD`` from temperature and free-surface height.

    Parameters
    ----------
    theta_c
        Temperature with shape ``(..., 15, y, x)`` in degrees Celsius.
    eta_m
        Free-surface height with shape ``(..., y, x)`` in metres.
    wet
        Optional two-dimensional wet mask. Land is set to zero when supplied.

    Returns
    -------
    numpy.ndarray
        Hydrostatic pressure anomaly divided by reference density, with the
        same shape as ``theta_c`` and units m2 s-2.
    """

    theta = np.asarray(theta_c, dtype=np.float64)
    eta = np.asarray(eta_m, dtype=np.float64)
    if theta.ndim < 3 or theta.shape[-3] != DRF_M.size:
        raise ValueError("theta must have shape (..., 15, y, x)")
    if eta.shape != theta.shape[:-3] + theta.shape[-2:]:
        raise ValueError("eta must have shape (..., y, x) matching theta")
    if wet is not None and np.asarray(wet).shape != theta.shape[-2:]:
        raise ValueError("wet mask must match theta's horizontal grid")

    reference_shape = (1,) * (theta.ndim - 3) + (T_REF_C.size, 1, 1)
    density_anomaly_over_reference = -THERMAL_EXPANSION_PER_C * (
        theta - T_REF_C.reshape(reference_shape)
    )
    above, below = _vertical_distances()
    leading_shape = theta.shape[:-3]
    interface_value = np.zeros(leading_shape + theta.shape[-2:], dtype=np.float64)
    baroclinic = np.empty_like(theta)
    for level in range(DRF_M.size):
        layer = density_anomaly_over_reference[..., level, :, :]
        center_value = interface_value + above[level] * GRAVITY_M_S2 * layer
        baroclinic[..., level, :, :] = center_value
        interface_value = center_value + below[level] * GRAVITY_M_S2 * layer

    result = baroclinic + GRAVITY_M_S2 * np.expand_dims(eta, axis=-3)
    if wet is not None:
        result[..., ~np.asarray(wet, dtype=bool)] = 0.0
    return result.astype(np.float32)


def pressure_diagnostics(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    """Return Bire-facing surface/mid/bottom ``PHIHYD`` from 46-state arrays."""

    values = np.asarray(states)
    if values.ndim != 4 or values.shape[1] != 46:
        raise ValueError("states must have shape (batch, 46, y, x)")
    full = phihyd_from_theta_eta(values[:, 30:45], values[:, 45], wet)
    return {name: full[:, level].copy() for name, level in PHIHYD_LEVELS.items()}


def validate_mitgcm_dump(run_dir: str | Path, iteration: int) -> dict[str, Any]:
    """Compare reconstruction with one archived MITgcm state dump."""

    root = Path(run_dir).resolve()

    def field(prefix: str) -> np.ndarray:
        return read_mds(root / f"{prefix}.{iteration:010d}.meta")[1][0]

    theta = field("T")
    eta = field("Eta")
    truth = field("PH")
    depth = read_mds(root / "Depth.meta")[1][0]
    wet = depth > 0
    reconstructed = phihyd_from_theta_eta(theta[None], eta[None], wet)[0]
    error = reconstructed.astype(np.float64) - truth
    wet_error = error[:, wet]
    level_metrics: dict[str, Mapping[str, float | int]] = {}
    for name, level in PHIHYD_LEVELS.items():
        selected = error[level, wet]
        level_metrics[name] = {
            "level_index": level,
            "max_abs_m2_s2": float(np.max(np.abs(selected))),
            "rmse_m2_s2": float(np.sqrt(np.mean(np.square(selected)))),
            "bias_m2_s2": float(np.mean(selected)),
        }
    maximum = float(np.max(np.abs(wet_error)))
    return {
        "status": "pass" if maximum < 1.0e-3 else "fail",
        "run_directory": str(root),
        "iteration": iteration,
        "wet_cells": int(wet.sum()),
        "levels": int(theta.shape[0]),
        "formula": (
            "MITgcm finite-difference hydrostatic integration for LINEAR EOS, "
            "plus g*ETAN linear-free-surface contribution"
        ),
        "units": "m2 s-2 (pressure anomaly divided by rhoConst)",
        "pressure_pa_conversion": f"multiply PHIHYD by rhoConst={RHO_CONST_KG_M3} kg m-3",
        "global": {
            "max_abs_m2_s2": maximum,
            "rmse_m2_s2": float(np.sqrt(np.mean(np.square(wet_error)))),
            "bias_m2_s2": float(np.mean(wet_error)),
        },
        "bire_levels": level_metrics,
        "acceptance": {"max_abs_m2_s2_less_than": 1.0e-3},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AF--FNO PHIHYD postprocessing")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_mitgcm_dump(args.run_dir, args.iteration)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(output)
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
