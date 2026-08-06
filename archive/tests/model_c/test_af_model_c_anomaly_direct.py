from __future__ import annotations

import json

import numpy as np
import torch

from bire_repro.af_model_c_anomaly_direct import (
    VERSION,
    PointwiseDirectStepper,
    direct_state_unroll,
    load_anomaly_direct_contract,
    training_pointwise_normalizers,
)
from bire_repro.af_model_c_successor import ModelCSuccessorArchitecture


def test_pointwise_normalizer_uses_only_training_and_floors_quiet_tail() -> None:
    state = np.zeros((3, 6, 46, 2, 2), dtype=np.float32)
    spatial = np.asarray(((0.0, 1.0), (2.0, 0.0)), dtype=np.float32)
    for experiment in range(3):
        for time_index in range(6):
            state[experiment, time_index] = (
                (10.0 * experiment + time_index)
                * spatial[None]
                + np.arange(46, dtype=np.float32)[:, None, None]
            )
    state[:, 4:] = 1.0e6
    group = {
        "state": state,
        "wet_mask": np.asarray(((1, 1), (1, 0)), dtype=np.uint8),
    }
    result = training_pointwise_normalizers(
        group,
        np.asarray((1, 1, 1, 1, 2, 2), dtype=np.uint8),
        floor_quantile=0.25,
        absolute_floor=1.0e-6,
        chunk_days=2,
    )
    mean = result["mean"]
    raw_scale = result["raw_scale"]
    scale = result["scale"]
    floor = result["floor"]
    assert result["summary"]["training_snapshots_total"] == 12
    assert np.all(mean[:, 0, 0] == np.arange(46, dtype=np.float32))
    assert np.all(raw_scale[:, 0, 0] == 0.0)
    assert np.all(scale[:, 0, 0] == floor)
    assert np.all(scale[:, 1, 1] == 1.0)
    assert np.all(mean[:, 1, 1] == 0.0)
    assert np.allclose(
        result["summary"]["fraction_wet_cells_floored"],
        1.0 / 3.0,
    )


def test_direct_state_unroll_does_not_add_the_present_state() -> None:
    class ConstantModel(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full_like(features[:, :46], 2.0)

    features = torch.full((2, 51, 3, 3), 7.0)
    wet = torch.ones((1, 1, 3, 3))
    wet[:, :, 0, 0] = 0.0
    result = direct_state_unroll(ConstantModel(), features, wet, 3)
    assert result.shape == (2, 3, 46, 3, 3)
    assert torch.all(result[:, :, :, 1:, 1:] == 2.0)
    assert torch.all(result[:, :, :, 0, 0] == 0.0)


def test_pointwise_stepper_round_trip_and_direct_step() -> None:
    class ZeroModel(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(features[:, :46])

    wet = np.asarray(((True, True), (True, False)))
    mean = np.arange(46, dtype=np.float32)[:, None, None] * np.ones(
        (1, 2, 2),
        dtype=np.float32,
    )
    scale = np.full((46, 2, 2), 2.0, dtype=np.float32)
    mean[:, ~wet] = 0.0
    scale[:, ~wet] = 1.0
    stepper = PointwiseDirectStepper(
        model=ZeroModel(),
        device=torch.device("cpu"),
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=0.0,
        wind_scale=1.0,
    )
    physical = mean[None] + 0.5 * scale[None]
    physical[:, :, ~wet] = 0.0
    normalized = stepper.normalized_state(physical)
    assert np.array_equal(stepper.physical(normalized), physical)
    static = torch.zeros((1, 5, 2, 2))
    assert torch.count_nonzero(stepper.step(normalized, static)) == 0


def test_anomaly_direct_contract_fixes_representation(tmp_path) -> None:
    architecture = ModelCSuccessorArchitecture(
        hidden_channels=128,
        channel_mlp_expansion=4.0,
    )
    contract = {
        "version": VERSION,
        "contract_status": (
            "frozen_before_anomaly_direct_training_or_validation_metrics"
        ),
        "normalization": {
            "centering": "pooled_S0_S1_S2_split1_pointwise_time_mean",
            "scaling": "pooled_S0_S1_S2_split1_pointwise_population_std",
            "wet_cell_floor_quantile": 0.05,
            "absolute_scale_floor": 1.0e-6,
            "regime_dependent": False,
        },
        "prediction": {
            "target": "direct_normalized_future_anomaly_state",
            "residual_addition": False,
        },
        "architecture": architecture.to_dict(),
        "read_contract": {
            "training_state": True,
            "fixed_S2_validation_figure_state": True,
            "inference_state": False,
            "intermediate_wind_state": False,
            "response_state": False,
            "adjoint_state": False,
        },
        "source_hashes": {},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    loaded, resolved, digest = load_anomaly_direct_contract(
        path,
        verify_sources=False,
    )
    assert loaded["version"] == VERSION
    assert resolved == path.resolve()
    assert len(digest) == 64
