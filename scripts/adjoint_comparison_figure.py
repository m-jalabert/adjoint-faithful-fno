"""Shared 2-row x 3-panel MITgcm/TAF vs emulator(S_forced) figure.

Used by every comparison_posthoc_v1 figure script (ssh_anomaly, kernel,
raw_ssh, mean_conservation), so the visual convention -- shared colour scale
on the first two panels, own scale on the difference panel, target marker,
per-row metric caption -- is defined exactly once.

    row 1  fno_b_seed_20260911   [reference] [emulator S_forced] [emulator - reference]
    row 2  fno_c_seed_20260911   [reference] [emulator S_forced] [emulator - reference]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import adjoint_metrics as metrics  # noqa: E402

ROWS = (("B_20260911", "fno_b_seed_20260911"), ("C_20260911", "fno_c_seed_20260911"))


def style() -> None:
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


def bound(field: np.ndarray, wet: np.ndarray, percentile: float = 99.0) -> float:
    values = np.abs(field[wet])
    value = float(np.percentile(values, percentile))
    return value if value > 0.0 else float(values.max() or 1.0)


def _panel(
    axis, field: np.ndarray, wet: np.ndarray, target: tuple[int, int] | None, title: str, limit: float
) -> None:
    image = axis.pcolormesh(_masked(field, wet), cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
    if target is not None:
        axis.plot(target[1] + 0.5, target[0] + 0.5, "ko", ms=4, mfc="none", mew=0.9)
    axis.set_title(title, fontsize=8.5)
    axis.set_aspect("equal")
    axis.set_facecolor("0.86")
    axis.figure.colorbar(image, ax=axis, shrink=0.78)


def render_2row_grid(
    out_path: Path,
    suptitle: str,
    wet: np.ndarray,
    target_ij: tuple[int, int],
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    reference_label: str = "MITgcm / TAF",
) -> Path:
    """One PNG: 2 rows (rows[i] = (identity_label, reference_2d, emulator_2d)),
    3 panels each. Columns 1-2 share one colour scale set from row 0's
    reference field (both rows plot the same physical reference, just
    against a different model's emulator map); column 3 has its own scale
    per row."""

    style()
    shared_bound = bound(rows[0][1], wet)
    figure = plt.figure(figsize=(13.2, 9.4), constrained_layout=True)
    grid = figure.add_gridspec(4, 3, height_ratios=[0.09, 1, 0.09, 1])

    for row_index, (identity_label, reference, emulator) in enumerate(rows):
        difference = emulator - reference
        comparison = metrics.primary_metrics(emulator, reference, wet)

        caption_axis = figure.add_subplot(grid[2 * row_index, :])
        caption_axis.axis("off")
        caption_axis.text(
            0.5, 0.5,
            f"{identity_label}  --  pattern correlation {comparison['pattern_correlation']:+.4f}, "
            f"relative L2 {comparison['relative_l2']:.3f}, amplitude ratio "
            f"{comparison['amplitude_ratio']:.3f}, sign agreement {comparison['sign_agreement']:.3f}",
            ha="center", va="center", fontsize=9.5, fontweight="bold", transform=caption_axis.transAxes,
        )

        row_axes = [figure.add_subplot(grid[2 * row_index + 1, col]) for col in range(3)]
        _panel(
            row_axes[0], reference, wet, target_ij,
            f"{reference_label}\nmax|S| = {np.abs(reference[wet]).max():.3e}", shared_bound,
        )
        _panel(
            row_axes[1], emulator, wet, target_ij,
            f"emulator (S_forced)\nmax|S| = {np.abs(emulator[wet]).max():.3e}", shared_bound,
        )
        _panel(
            row_axes[2], difference, wet, target_ij,
            "emulator - MITgcm", bound(difference, wet),
        )

    figure.suptitle(suptitle, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path
