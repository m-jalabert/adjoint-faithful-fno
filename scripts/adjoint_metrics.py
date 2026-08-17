"""Comparison metrics for adjoint sensitivity maps.

Implements section 6 of docs/fno_adjoint_plan.md.  Everything here is pure
NumPy: a sensitivity map is just a 62x62 array of numbers, and none of these
functions know or care whether it came from MITgcm's TAF adjoint or from
PyTorch's ``backward()``.  That is deliberate --- the same code scores both
sides, so a metric cannot accidentally be computed differently for each.

Vocabulary, for a reader new to the project:

``S[j, i]``
    the sensitivity map.  ``S[j, i] = dJ/d eta(j, i)``: how much the scalar
    cost ``J`` changes when the sea-surface height in cell ``(j, i)`` is raised
    by one metre.  It is dimensionless (metres of ``J`` per metre of ``eta``),
    and ``S > 0`` means "raising eta here raises the target".

``wet`` / land
    the basin is 62x62 with a one-cell land rim: 3,600 wet cells and 244 dry
    ones.  MITgcm's map is *exactly* zero on the dry cells because sea-surface
    height there is not a degree of freedom, so any non-zero value on the
    emulator's side is unambiguously spurious.

Index order is ``(j, i)`` --- row then column --- matching the zarr's spatial
axes, throughout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oceanfno.dataset import western_boundary_mask

#: The spectral loss this project already uses bins radial wavenumber into 12
#: bins after a Hann taper; the sensitivity spectra reuse that convention so the
#: two diagnostics are directly comparable.  See
#: ``oceanfno.objective.tapered_group_spectral_loss``.
SPECTRAL_BINS = 12

#: Width, in wet cells east of each row's western wall, of the boundary band the
#: project's existing diagnostics use.
WESTERN_BAND_WIDTH = 4


class AdjointMetricError(RuntimeError):
    """Raised when two maps cannot be meaningfully compared."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_map(value: Any, name: str) -> np.ndarray:
    """Coerce one input to a finite float64 2-D array."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise AdjointMetricError(f"{name} must be a two-dimensional map, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise AdjointMetricError(f"{name} contains non-finite values")
    return array


def _aligned(
    first: Any, second: Any, wet: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check that two maps and a mask describe the same grid, and return them."""

    a = _as_map(first, "first map")
    b = _as_map(second, "second map")
    mask = np.asarray(wet, dtype=bool)
    if a.shape != b.shape or a.shape != mask.shape:
        raise AdjointMetricError(
            f"maps and wet mask disagree on shape: {a.shape}, {b.shape}, {mask.shape}"
        )
    if not mask.any():
        raise AdjointMetricError("the wet mask selects no cells")
    return a, b, mask


# ---------------------------------------------------------------------------
# primary metrics  (plan section 6, "Primary")
# ---------------------------------------------------------------------------


def pattern_correlation(fno: Any, reference: Any, wet: Any) -> float:
    """Pearson correlation over wet cells: does the FNO put sensitivity in the right places?

    This is scale-free, so a model that has the right structure but the wrong
    amplitude still scores 1.  ``amplitude_ratio`` is the companion that catches
    the amplitude.
    """

    a, b, mask = _aligned(fno, reference, wet)
    x, y = a[mask], b[mask]
    if x.std() == 0.0 or y.std() == 0.0:
        raise AdjointMetricError("a constant map has no pattern to correlate")
    return float(np.corrcoef(x, y)[0, 1])


def relative_l2(fno: Any, reference: Any, wet: Any) -> float:
    """``norm(S_fno - S) / norm(S)`` over wet cells --- the overall error."""

    a, b, mask = _aligned(fno, reference, wet)
    denominator = float(np.linalg.norm(b[mask]))
    if denominator == 0.0:
        raise AdjointMetricError("the reference map is identically zero on wet cells")
    return float(np.linalg.norm(a[mask] - b[mask]) / denominator)


def amplitude_ratio(fno: Any, reference: Any, wet: Any) -> float:
    """``norm(S_fno) / norm(S)`` --- systematic over- or under-response.

    Above 1 the emulator is too sensitive, below 1 too sluggish.
    """

    a, b, mask = _aligned(fno, reference, wet)
    denominator = float(np.linalg.norm(b[mask]))
    if denominator == 0.0:
        raise AdjointMetricError("the reference map is identically zero on wet cells")
    return float(np.linalg.norm(a[mask]) / denominator)


def sign_agreement(fno: Any, reference: Any, wet: Any) -> float:
    """Fraction of wet cells where the two maps agree on the sign.

    Structural fidelity: getting the sign right everywhere means the emulator
    agrees about *which way* the basin responds, even where it disagrees about
    how much.  Cells where the reference is exactly zero are excluded, since
    zero has no sign to match.
    """

    a, b, mask = _aligned(fno, reference, wet)
    comparable = mask & (b != 0.0)
    if not comparable.any():
        raise AdjointMetricError("the reference map has no non-zero wet cells")
    return float(np.mean(np.sign(a[comparable]) == np.sign(b[comparable])))


def primary_metrics(fno: Any, reference: Any, wet: Any) -> dict[str, float]:
    """All four primary metrics in one call."""

    return {
        "pattern_correlation": pattern_correlation(fno, reference, wet),
        "relative_l2": relative_l2(fno, reference, wet),
        "amplitude_ratio": amplitude_ratio(fno, reference, wet),
        "sign_agreement": sign_agreement(fno, reference, wet),
    }


# ---------------------------------------------------------------------------
# structural metrics  (plan section 6, "Structural")
# ---------------------------------------------------------------------------


def land_leakage(sensitivity: Any, wet: Any) -> dict[str, float]:
    """How much sensitivity sits on cells where the true answer is exactly zero.

    MITgcm's adjoint is identically zero on the 244 dry cells --- sea-surface
    height on land is not a state variable, so no derivative with respect to it
    exists.  The FNO's spectral convolutions are global, so a dry-cell input
    value *can* reach the target, and this measures by how much.  It is the one
    diagnostic in the study whose correct answer is known exactly without
    running MITgcm at all.

    The ratio is against the wet-cell maximum, so it reads as "leakage is N% of
    the largest real sensitivity".
    """

    array = _as_map(sensitivity, "sensitivity")
    mask = np.asarray(wet, dtype=bool)
    if array.shape != mask.shape:
        raise AdjointMetricError("sensitivity map and wet mask disagree on shape")
    dry = ~mask
    if not dry.any():
        raise AdjointMetricError("this grid has no dry cells")
    wet_maximum = float(np.abs(array[mask]).max())
    dry_maximum = float(np.abs(array[dry]).max())
    return {
        "dry_cell_count": int(dry.sum()),
        "max_abs_dry": dry_maximum,
        "max_abs_wet": wet_maximum,
        "dry_to_wet_max_ratio": dry_maximum / wet_maximum if wet_maximum > 0.0 else float("nan"),
        "l2_dry_over_l2_wet": float(
            np.linalg.norm(array[dry]) / np.linalg.norm(array[mask])
        )
        if np.linalg.norm(array[mask]) > 0.0
        else float("nan"),
    }


def boundary_interior_split(
    sensitivity: Any, wet: Any, width: int = WESTERN_BAND_WIDTH
) -> dict[str, float]:
    """Split the map into the western boundary band and the interior.

    ``western-boundary-ratio-degrades`` records that the day-2000 anomaly
    boundary/interior ratio moved from 4.25 to 3.20 against a truth of 23.1, and
    that the acceptance gate never scores it.  If the sensitivity maps disagree
    most inside that same band, the adjoint has localised a defect the forecast
    gate is blind to --- which is the reason this metric is here rather than a
    generic regional split.
    """

    array = _as_map(sensitivity, "sensitivity")
    mask = np.asarray(wet, dtype=bool)
    if array.shape != mask.shape:
        raise AdjointMetricError("sensitivity map and wet mask disagree on shape")
    band = western_boundary_mask(mask, width) & mask
    interior = mask & ~band
    if not interior.any():
        raise AdjointMetricError("the western band covers the whole basin")

    def rms(selection: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(array[selection]))))

    boundary_rms, interior_rms = rms(band), rms(interior)
    return {
        "band_width_cells": int(width),
        "boundary_cell_count": int(band.sum()),
        "interior_cell_count": int(interior.sum()),
        "boundary_rms": boundary_rms,
        "interior_rms": interior_rms,
        "boundary_to_interior_ratio": boundary_rms / interior_rms
        if interior_rms > 0.0
        else float("nan"),
    }


def radial_decay(
    sensitivity: Any,
    wet: Any,
    target: tuple[int, int],
    *,
    maximum_radius: int | None = None,
) -> dict[str, Any]:
    """E-folding distance of ``abs(S)`` away from the target cell.

    Bins wet cells by their distance from ``p*`` in grid cells, takes the root
    mean square of ``abs(S)`` in each ring, and fits ``log(rms) = a - r / L`` by
    least squares.  ``L`` is the e-folding distance: the number of cells over
    which the influence of a perturbation falls by a factor of e.  It is the
    single number that says "how far does this model think sea-surface height
    reaches in ten days", and the two sides can disagree about it even when the
    pattern correlation is high.

    The target ring (``r = 0``) is excluded from the fit: the cost functional
    contains a delta function at ``p*``, so that one cell is not part of the
    decay the fit is trying to describe.
    """

    array = _as_map(sensitivity, "sensitivity")
    mask = np.asarray(wet, dtype=bool)
    if array.shape != mask.shape:
        raise AdjointMetricError("sensitivity map and wet mask disagree on shape")
    j0, i0 = int(target[0]), int(target[1])
    rows, columns = np.indices(array.shape)
    distance = np.sqrt((rows - j0) ** 2.0 + (columns - i0) ** 2.0)
    rings = np.rint(distance).astype(int)
    limit = int(rings[mask].max()) if maximum_radius is None else int(maximum_radius)

    radii, amplitudes = [], []
    for radius in range(1, limit + 1):
        selection = mask & (rings == radius)
        if not selection.any():
            continue
        value = float(np.sqrt(np.mean(np.square(array[selection]))))
        if value > 0.0:
            radii.append(float(radius))
            amplitudes.append(value)
    if len(radii) < 3:
        raise AdjointMetricError("too few populated rings to fit a decay length")

    radius_array = np.asarray(radii)
    amplitude_array = np.asarray(amplitudes)
    # log(rms) = intercept + slope * r, so the e-folding length is -1/slope.
    slope, intercept = np.polyfit(radius_array, np.log(amplitude_array), 1)
    predicted = intercept + slope * radius_array
    residual = np.log(amplitude_array) - predicted
    total = np.log(amplitude_array) - np.log(amplitude_array).mean()
    return {
        "target_ij": [j0, i0],
        "radius_cells": radius_array.tolist(),
        "rms_abs_sensitivity": amplitude_array.tolist(),
        "e_folding_cells": float(-1.0 / slope) if slope < 0.0 else float("inf"),
        "fit_slope_per_cell": float(slope),
        "fit_intercept_log": float(intercept),
        "fit_r_squared": float(1.0 - np.sum(residual**2) / np.sum(total**2))
        if np.sum(total**2) > 0.0
        else float("nan"),
    }


def radial_power_spectrum(
    sensitivity: Any, wet: Any, *, bins: int = SPECTRAL_BINS
) -> dict[str, Any]:
    """Absolute power per radial wavenumber bin, in the project's spectral convention.

    Mirrors ``oceanfno.objective.tapered_group_spectral_loss``: crop to the
    rectangular wet basin, remove the mean, apply a separable Hann window (so
    the basin edges do not ring across the whole spectrum), take an orthonormal
    2-D FFT, and average the squared amplitude into ``bins`` rings of radial
    wavenumber.

    ``local-branch-gamma-ablation`` records that the high-wavenumber *fraction*
    misleads --- it can fall simply because the low-wavenumber power grew --- so
    this returns absolute power per bin and never a fraction.  Compare bins
    between two maps directly.
    """

    array = _as_map(sensitivity, "sensitivity")
    mask = np.asarray(wet, dtype=bool)
    if array.shape != mask.shape:
        raise AdjointMetricError("sensitivity map and wet mask disagree on shape")
    if bins <= 1:
        raise AdjointMetricError("a radial spectrum needs at least two bins")

    rows, columns = np.where(mask)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    if not mask[y0:y1, x0:x1].all():
        raise AdjointMetricError("the spectral convention requires an exactly rectangular basin")

    field = array[y0:y1, x0:x1]
    field = field - field.mean()
    field = field * np.hanning(field.shape[0])[:, None] * np.hanning(field.shape[1])[None, :]
    power = np.abs(np.fft.rfft2(field, norm="ortho")) ** 2

    fy = np.fft.fftfreq(field.shape[0])
    fx = np.fft.rfftfreq(field.shape[1])
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    valid = radius > 0.0  # the mean was removed, so bin 0 carries no information
    indices = np.minimum(
        np.floor(radius[valid] / radius.max() * bins).astype(int), bins - 1
    )
    totals = np.bincount(indices, weights=power[valid], minlength=bins)
    counts = np.maximum(np.bincount(indices, minlength=bins), 1)
    return {
        "bins": int(bins),
        "power_per_bin": (totals / counts).tolist(),
        "bin_upper_wavenumber": (
            (np.arange(1, bins + 1) / bins) * float(radius.max())
        ).tolist(),
        "convention": "crop_to_wet_rectangle_remove_mean_hann_taper_orthonormal_rfft2",
    }


def structural_metrics(
    sensitivity: Any,
    wet: Any,
    target: tuple[int, int],
    *,
    width: int = WESTERN_BAND_WIDTH,
    bins: int = SPECTRAL_BINS,
) -> dict[str, Any]:
    """Every structural diagnostic for one map, in one call."""

    return {
        "land_leakage": land_leakage(sensitivity, wet),
        "western_band": boundary_interior_split(sensitivity, wet, width),
        "radial_decay": radial_decay(sensitivity, wet, target),
        "radial_spectrum": radial_power_spectrum(sensitivity, wet, bins=bins),
    }


def spectrum_ratio(fno: Mapping[str, Any], reference: Mapping[str, Any]) -> list[float]:
    """Per-bin power ratio between two spectra produced by the function above."""

    a = np.asarray(fno["power_per_bin"], dtype=np.float64)
    b = np.asarray(reference["power_per_bin"], dtype=np.float64)
    if a.shape != b.shape:
        raise AdjointMetricError("the two spectra use different bin counts")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0.0, a / b, np.nan).tolist()
