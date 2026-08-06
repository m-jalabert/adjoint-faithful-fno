from __future__ import annotations

import json

import numpy as np
import torch

from bire_repro.af_model_c import model_c_loss_config
from bire_repro.af_model_c_reduced_channel_control import (
    CONTRACT_STATUS,
    VERSION,
    ReducedRolloutDataset,
    load_contract,
)
from bire_repro.af_model_c_reduced_channels import (
    REDUCED_CHANNELS,
    ReducedChannelArchitecture,
    direct_unroll,
    reduce_full_state,
    reduced_fields,
    reduced_loss_contract_sha256,
    reduced_loss_terms,
)
from bire_repro.af_pressure import phihyd_from_theta_eta


def test_reduced_transform_channel_order_and_diagnostics() -> None:
    wet = np.ones((3, 4), dtype=bool)
    full = np.zeros((2, 46, 3, 4), dtype=np.float32)
    for channel in range(46):
        full[:, channel] = channel
    reduced = reduce_full_state(full, wet)
    pressure = phihyd_from_theta_eta(
        full[:, 30:45],
        full[:, 45],
        wet,
    )
    assert reduced.shape == (2, 10, 3, 4)
    assert np.all(reduced[:, 0] == 0.0)
    assert np.all(reduced[:, 1] == 7.0)
    assert np.all(reduced[:, 2] == 15.0)
    assert np.all(reduced[:, 3] == 22.0)
    assert np.all(reduced[:, 4] == 30.0)
    assert np.all(reduced[:, 5] == 37.0)
    assert np.array_equal(reduced[:, 6], pressure[:, 0])
    assert np.array_equal(reduced[:, 7], pressure[:, 7])
    assert np.array_equal(reduced[:, 8], pressure[:, 14])
    fields = reduced_fields(reduced, wet)
    assert np.allclose(fields["surface_speed"], 15.0)
    assert np.array_equal(fields["streamfunction"], reduced[:, 9])


def test_direct_unroll_uses_only_ten_dynamic_channels() -> None:
    class Constant(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full_like(features[:, :10], 2.0)

    features = torch.full((2, 15, 4, 4), 7.0)
    wet = torch.ones((1, 1, 4, 4))
    wet[:, :, 0, :] = 0.0
    result = direct_unroll(Constant(), features, wet, 3)
    assert result.shape == (2, 3, 10, 4, 4)
    assert torch.all(result[:, :, :, 1:] == 2.0)
    assert torch.all(result[:, :, :, 0] == 0.0)


def test_reduced_loss_is_finite_and_differentiable() -> None:
    generator = torch.Generator().manual_seed(7)
    present = torch.randn((2, 10, 8, 8), generator=generator)
    targets = torch.randn((2, 3, 10, 8, 8), generator=generator)
    predictions = torch.randn(
        (2, 3, 10, 8, 8),
        generator=generator,
        requires_grad=True,
    )
    wet = torch.ones((1, 1, 8, 8))
    boundary = torch.zeros_like(wet)
    boundary[:, :, :, :4] = 1.0
    terms = reduced_loss_terms(
        predictions,
        targets,
        present,
        wet,
        boundary,
        torch.ones(10),
        model_c_loss_config("v1"),
    )
    assert all(torch.isfinite(value) for value in terms.values())
    terms["total"].backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_contract_fixes_arm_r_delta(tmp_path) -> None:
    architecture = ReducedChannelArchitecture()
    contract = {
        "version": VERSION,
        "contract_status": CONTRACT_STATUS,
        "reduced_state": {"channels": list(REDUCED_CHANNELS)},
        "architecture": architecture.to_dict(),
        "training": {
            "seed": 20260724,
            "checkpoint_steps": [
                3840,
                7680,
                11520,
                13440,
                14400,
                14880,
                15360,
            ],
            "optimizer": {"maximum_steps": 15360},
            "loss": {
                "contract_sha256": reduced_loss_contract_sha256(
                    model_c_loss_config("v1")
                )
            },
        },
        "read_contract": {
            "training_state": True,
            "held_s0_state_after_selection": True,
            "intermediate_wind_state": False,
            "response_state": False,
            "adjoint_state": False,
        },
        "source_hashes": {},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    loaded, resolved, digest = load_contract(
        path,
        verify_sources=False,
    )
    assert loaded["architecture"]["in_channels"] == 15
    assert loaded["architecture"]["out_channels"] == 10
    assert resolved == path.resolve()
    assert len(digest) == 64


def test_reduced_rollout_dataset_is_declared_class() -> None:
    assert ReducedRolloutDataset.__name__ == "ReducedRolloutDataset"
