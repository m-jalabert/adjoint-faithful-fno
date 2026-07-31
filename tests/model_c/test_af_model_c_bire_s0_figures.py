from __future__ import annotations

from pathlib import Path

import numpy as np

from bire_repro.af_model_c_bire_s0_figures import (
    ACC_FIELDS,
    FIGURE_NAMES,
    LEAD_DAYS,
    RMSE_FIELDS,
    SHORT_LEAD_DAYS,
    ContinuousS0Truth,
    _plot_acc,
    _plot_day60_day2000,
    _plot_rmse,
    _plot_single_member,
    _plot_streamfunction_grid,
    percentile_curve,
)


def test_protocol_shapes_and_six_names() -> None:
    assert LEAD_DAYS == tuple(range(0, 2001, 10))
    assert SHORT_LEAD_DAYS == tuple(range(0, 201, 10))
    assert RMSE_FIELDS == (
        "surface_speed",
        "phihyd_surface",
        "sst",
    )
    assert ACC_FIELDS == (
        "surface_u",
        "surface_v",
        "phihyd_surface",
        "sst",
    )
    assert len(FIGURE_NAMES) == 6
    assert len(set(FIGURE_NAMES)) == 6


def test_percentile_curve() -> None:
    values = np.arange(15, dtype=np.float64)[:, None] + np.arange(3)
    summary = percentile_curve(values)
    assert np.array_equal(summary["mean"], np.asarray((7.0, 8.0, 9.0)))
    assert np.allclose(summary["p10"], (1.4, 2.4, 3.4))
    assert np.allclose(summary["p90"], (12.6, 13.6, 14.6))


def test_truth_extension_index_mapping() -> None:
    class FakeState:
        shape = (3, 7200, 46, 62, 62)

    fake = FakeState()
    truth = ContinuousS0Truth(fake, Path("/tmp"), np.ones((62, 62), bool))
    assert truth.total_records == 9360
    assert truth.extension_iteration(7200) == 3110400
    assert truth.extension_iteration(9359) == 3265848


def test_six_plotters(tmp_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    long_curve = np.linspace(0.0, 1.0, len(LEAD_DAYS))
    short_curve = long_curve[: len(SHORT_LEAD_DAYS)]
    for method_index, method in enumerate(
        ("model", "climatology", "persistence"),
        start=1,
    ):
        for field in RMSE_FIELDS:
            arrays[f"rmse__{method}__{field}"] = np.tile(
                method_index * long_curve,
                (15, 1),
            )
    for model_index, model_name in enumerate(("selected", "prior"), start=1):
        for field in ACC_FIELDS:
            arrays[f"acc__{model_name}__{field}"] = np.tile(
                1.0 - model_index * 0.1 * short_curve,
                (15, 1),
            )
    arrays["single_rmse__streamfunction"] = short_curve
    arrays["single_rmse__sst"] = 0.1 * short_curve
    shape = (62, 62)
    arrays["figure3_truth_streamfunction"] = np.ones((5, *shape))
    arrays["figure3_model_streamfunction"] = np.full((5, *shape), 0.9)
    arrays["figure7_truth_streamfunction"] = np.ones((2, *shape))
    arrays["figure7_model_streamfunction"] = np.full((2, *shape), 0.8)
    coordinate = np.arange(62, dtype=np.float32)
    longitude, latitude = np.meshgrid(coordinate, coordinate)
    wet = np.ones(shape, dtype=bool)
    _plot_streamfunction_grid(
        tmp_path,
        arrays,
        longitude,
        latitude,
        wet,
    )
    _plot_rmse(tmp_path, arrays, long=False)
    _plot_single_member(tmp_path, arrays)
    _plot_acc(tmp_path, arrays)
    _plot_day60_day2000(
        tmp_path,
        arrays,
        longitude,
        latitude,
        wet,
    )
    _plot_rmse(tmp_path, arrays, long=True)
    for name in FIGURE_NAMES:
        assert (tmp_path / name).stat().st_size > 0
