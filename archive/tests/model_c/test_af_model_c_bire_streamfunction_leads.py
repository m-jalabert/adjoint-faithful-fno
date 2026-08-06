from __future__ import annotations

import numpy as np
import pytest

from bire_repro.af_model_c_bire_streamfunction_leads import (
    FIGURE_LEADS,
    FIGURE_NAMES,
    _plot_lead,
    figure_name,
    streamfunction_metrics,
)


def test_eight_requested_figure_names() -> None:
    assert FIGURE_LEADS == (20, 30, 40, 50, 60, 70, 80, 90)
    assert len(FIGURE_NAMES) == 8
    assert figure_name(20).endswith("day020.png")
    assert figure_name(90).endswith("day090.png")
    with pytest.raises(ValueError):
        figure_name(10)


def test_streamfunction_metrics_use_wet_cells() -> None:
    truth = np.asarray([[1.0, 2.0], [3.0, 100.0]])
    prediction = np.asarray([[0.0, 2.0], [1.0, -100.0]])
    wet = np.asarray([[True, True], [True, False]])
    metrics = streamfunction_metrics(truth, prediction, wet)
    assert metrics["rmse_sv"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics["maximum_absolute_error_sv"] == pytest.approx(2.0)
    assert metrics["truth_rms_sv"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert metrics["relative_rmse"] == pytest.approx(
        np.sqrt(5.0 / 14.0)
    )


def test_sensitive_difference_plot(tmp_path) -> None:
    coordinate = np.arange(5, dtype=np.float64)
    longitude, latitude = np.meshgrid(coordinate, coordinate)
    truth = longitude - latitude
    prediction = truth + 0.01 * longitude
    wet = np.ones((5, 5), dtype=bool)
    metrics = _plot_lead(
        tmp_path,
        20,
        truth,
        prediction,
        longitude,
        latitude,
        wet,
        state_bound=4.0,
    )
    assert metrics["difference_symmetric_bound_sv"] == pytest.approx(0.04)
    assert metrics["truth_prediction_symmetric_bound_sv"] == 4.0
    assert (tmp_path / figure_name(20)).stat().st_size > 0
