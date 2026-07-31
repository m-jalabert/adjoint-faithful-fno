from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bire_repro.af_model_c_s0_highk_damping_control import (  # noqa: E402
    ALPHAS,
    FIELDS,
    SHORT_LEADS,
    STAT_FIELDS,
    TRAINING_LEADS,
    TRAINING_SPECTRUM_LEADS,
    reflected_binomial_damping,
    select_alpha,
    training_records,
)


def test_zero_filter_is_exact_and_checkerboard_is_damped() -> None:
    wet = np.ones((8, 8), dtype=bool)
    value = torch.zeros(2, 3, 8, 8)
    value[0] = torch.arange(64).reshape(8, 8) % 2
    value[1] = 4.0
    zero = reflected_binomial_damping(value, wet, torch.zeros(2))
    filtered = reflected_binomial_damping(value, wet, torch.ones(2) * 0.2)
    assert torch.equal(zero, value)
    assert torch.equal(filtered[1], value[1])
    assert torch.std(filtered[0]) < torch.std(value[0])


def test_training_records_have_complete_split1_windows() -> None:
    split = np.zeros(7200, dtype=np.uint8)
    split[0:2520] = 1
    split[3690:6210] = 1
    records = training_records(split)
    assert records.shape == (10, 2)
    assert records[-1].tolist() == [0, 5209]


def test_selection_chooses_smallest_passing_alpha() -> None:
    member_count = 3
    modes = np.arange(1, 31, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(TRAINING_LEADS),
        "spectrum_leads": np.asarray(TRAINING_SPECTRUM_LEADS),
        "spectrum_modes": modes,
        "finite": np.ones(
            (len(ALPHAS), member_count, len(TRAINING_LEADS)),
            dtype=np.uint8,
        ),
        "normalized_max_abs": np.ones(
            (len(ALPHAS), member_count, len(TRAINING_LEADS)),
        ),
        "truth_normalized_max_abs": np.ones(
            (member_count, len(TRAINING_LEADS)),
        ),
    }
    day1000 = TRAINING_LEADS.index(1000)
    arrays["normalized_max_abs"][0, :, day1000] = 10.0
    arrays["normalized_max_abs"][1:, :, day1000] = 5.0
    for field in FIELDS:
        values = np.ones((len(ALPHAS), member_count, len(TRAINING_LEADS)))
        values[0, :, day1000] = 10.0
        values[1:, :, day1000] = 5.0
        arrays[f"rmse__{field}"] = values
        arrays[f"rmse__climatology__{field}"] = np.ones(
            (member_count, len(TRAINING_LEADS))
        )
        for lead in SHORT_LEADS:
            values[1:, :, TRAINING_LEADS.index(lead)] = 1.01
    for field in STAT_FIELDS:
        values = np.ones(
            (
                len(ALPHAS),
                member_count,
                len(TRAINING_SPECTRUM_LEADS),
                modes.size,
            )
        )
        values[0, :, 1, modes >= 10] = 10.0
        values[1:, :, 1, modes >= 10] = 2.0
        arrays[f"spectrum__{field}"] = values
        arrays[f"spectrum__truth__{field}"] = np.ones(
            (member_count, len(TRAINING_SPECTRUM_LEADS), modes.size)
        )
    selected, records = select_alpha(arrays)
    assert selected == 1
    assert records["0.02"]["passes"]
