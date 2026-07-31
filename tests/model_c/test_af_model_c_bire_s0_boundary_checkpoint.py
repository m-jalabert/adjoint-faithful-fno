from __future__ import annotations

from pathlib import Path

import numpy as np

from bire_repro.af_model_c_bire_s0_boundary_checkpoint import (
    CHECKPOINT_STEPS,
    FIGURE_NAMES,
    LEAD_DAYS,
    REGION_NAMES,
    SELECTED_STEP,
    _plot_boundary_fractions,
    _plot_checkpoint_timing,
    _plot_selected_growth,
    _plot_spatial_consistency,
    _plot_transport_rmse,
    barotropic_transports,
    first_crossing,
    growth_crossing,
    region_masks,
    streamfunction_from_qx,
)


def test_protocol_and_boundary_masks() -> None:
    assert CHECKPOINT_STEPS == (
        3840,
        7680,
        11520,
        13440,
        14400,
        14880,
        15360,
    )
    assert SELECTED_STEP == 13440
    assert LEAD_DAYS == tuple(range(0, 2001, 10))
    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    masks = region_masks(wet, 4)
    assert tuple(masks) == REGION_NAMES
    assert masks["wet"].sum() == 3600
    assert masks["boundary"].sum() == 896
    assert masks["interior"].sum() == 2704
    assert np.array_equal(
        masks["wet"],
        masks["boundary"] | masks["interior"],
    )


def test_transport_and_streamfunction() -> None:
    states = np.zeros((2, 46, 62, 62), dtype=np.float32)
    states[:, :15] = 1.0
    states[:, 15:30] = 2.0
    qx, qy = barotropic_transports(states)
    assert np.allclose(qy, 2.0 * qx)
    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    psi = streamfunction_from_qx(qx, wet)
    assert psi.shape == (2, 62, 62)
    assert np.all(psi[:, ~wet] == 0.0)
    assert np.all(np.diff(psi[:, 1:61, 1:61], axis=1) < 0.0)


def test_crossing_definitions() -> None:
    leads = (0, 10, 20, 30, 40, 50)
    curve = (0.0, 1.0, 2.0, 4.1, 8.2, 9.0)
    assert first_crossing(curve, leads, 4.0) == 30
    assert first_crossing(curve, leads, 20.0) is None
    assert growth_crossing(
        curve,
        leads,
        2.0,
        baseline_day=20,
    ) == 30
    assert growth_crossing(
        curve,
        leads,
        5.0,
        baseline_day=20,
    ) is None


def test_five_plotters(tmp_path: Path) -> None:
    checkpoints = len(CHECKPOINT_STEPS)
    members = 15
    leads = len(LEAD_DAYS)
    regions = len(REGION_NAMES)
    curve = np.linspace(0.01, 10.0, leads)
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS),
        "normalized_max_abs": np.stack(
            [
                np.tile(curve * (index + 1), (members, 1))
                for index in range(checkpoints)
            ]
        ),
    }
    for field_index, field in enumerate(
        ("qx", "qy", "streamfunction"),
        start=1,
    ):
        arrays[f"rmse__{field}"] = np.empty(
            (checkpoints, members, leads, regions),
        )
        for checkpoint in range(checkpoints):
            for region in range(regions):
                arrays[f"rmse__{field}"][checkpoint, :, :, region] = (
                    field_index
                    * (checkpoint + 1)
                    * (region + 1)
                    * curve[None]
                )
        arrays[f"boundary_fraction__{field}"] = np.tile(
            np.linspace(0.25, 0.8, leads),
            (checkpoints, members, 1),
        )
        arrays[f"selected_snapshot_error__{field}"] = np.ones(
            (5, members, 62, 62),
            dtype=np.float32,
        )
    summary = {
        "checkpoints": {
            str(step): {
                "first_mean_normalized_max_crossing_days": {
                    "20": 1000 + index * 10,
                    "100": 1500 + index * 10,
                }
            }
            for index, step in enumerate(CHECKPOINT_STEPS)
        }
    }
    coordinate = np.arange(62, dtype=np.float32)
    longitude, latitude = np.meshgrid(coordinate, coordinate)
    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    _plot_transport_rmse(tmp_path, arrays)
    _plot_boundary_fractions(tmp_path, arrays)
    _plot_checkpoint_timing(tmp_path, arrays, summary)
    _plot_spatial_consistency(
        tmp_path,
        arrays,
        longitude,
        latitude,
        wet,
    )
    _plot_selected_growth(tmp_path, arrays)
    assert len(FIGURE_NAMES) == 5
    for name in FIGURE_NAMES:
        assert (tmp_path / name).stat().st_size > 0
