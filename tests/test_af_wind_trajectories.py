from pathlib import Path

import numpy as np
import pytest

from bire_repro.af_s0 import STEPS_PER_YEAR
from bire_repro.af_wind_trajectories import EXPERIMENTS, _validate_request, scale_wind


def test_wind_branch_scales_match_project_plan() -> None:
    assert EXPERIMENTS["S1"] == {"tau0_n_m2": 0.075, "wind_scale": 0.75}
    assert EXPERIMENTS["S2"] == {"tau0_n_m2": 0.125, "wind_scale": 1.25}
    assert 100 * STEPS_PER_YEAR == 2_592_000


def test_scale_wind_preserves_big_endian_float32_layout(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    values = np.linspace(-0.1, 0.1, 62 * 62, dtype=np.float32)
    values.astype(">f4").tofile(source)
    scale_wind(source, destination, 0.75)
    actual = np.fromfile(destination, dtype=">f4")
    assert destination.stat().st_size == source.stat().st_size
    assert actual == pytest.approx(values * 0.75)


@pytest.mark.parametrize(
    ("phase", "start", "years"), (("adjust", 0, 5), ("production", 5, 10))
)
def test_only_literal_plan_segments_are_accepted(phase: str, start: int, years: int) -> None:
    _validate_request("S1", phase, start, years)
    with pytest.raises(ValueError, match="must use local start/year count"):
        _validate_request("S1", phase, start + 1, years)
