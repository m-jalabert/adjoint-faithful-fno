from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bire_repro import (
    af_model_c_anomaly_direct_bire_regularization_controls as controls,
)
from bire_repro import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from bire_repro.af_model_c_anomaly_direct_single_position_layernorm import (
    CONTRACT_STATUS,
    VERSION,
    SinglePositionArchitecture,
    SinglePositionLayerNormError,
    _ParentBinding,
    arm_from_contract,
    load_contract,
)

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_anomaly_direct_single_position_layernorm_v1.json"
)


def test_contract_declares_one_layernorm_arm_without_grid_embedding() -> None:
    contract, resolved, digest = load_contract(CONTRACT)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    assert contract["architecture"]["positional_embedding"] is None
    assert contract["architecture"]["in_channels"] == 51
    assert contract["architecture"]["out_channels"] == 46
    arm = arm_from_contract(contract, 0)
    assert (arm.arm_id, arm.pointwise_layer_norm, arm.channel_mlp_dropout) == (
        "layernorm",
        True,
        0.0,
    )


def test_contract_retains_the_parent_training_and_publishes_the_same_figures() -> None:
    contract, _, _ = load_contract(CONTRACT)
    training = contract["training"]
    assert training["seed"] == 20260724
    assert training["maximum_steps"] == 15360
    assert tuple(training["checkpoint_steps"]) == controls.CHECKPOINT_STEPS
    assert contract["read_contract"]["training_state"] is True
    assert contract["read_contract"]["inference_state"] is False
    figures = contract["held_evaluation_after_training_gate"]["figures"]
    assert figures == [
        "model_c_bire_figure3_streamfunction_1deg_s0_dt10.png",
        "model_c_bire_figure4_dt10_rmse_0_200_days_s0.png",
        "model_c_bire_figure5_dt10_single_member_rmse_s0.png",
        "model_c_bire_figure6_dt10_acc_0_200_days_s0.png",
        "model_c_bire_figure7_dt10_streamfunction_day060_day2000_s0.png",
        "model_c_bire_figure8_dt10_rmse_0_2000_days_s0.png",
    ]


def test_gate_corrections_keep_persistence_and_integrated_energy_vetoes() -> None:
    contract, _, _ = load_contract(CONTRACT)
    selection = contract["selection"]
    assert selection["primary_ratio_limit"] == 1.0
    assert selection["maximum_primary_relative_to_source"] == 2.0
    assert selection["integrated_energy_bounds"] == [0.8, 1.25]
    assert selection["spectral_ratio_limit"] > 1.0e6


def test_architecture_requires_the_single_position_encoding() -> None:
    architecture = SinglePositionArchitecture()
    assert architecture.positional_embedding is None
    assert architecture.in_channels == 51
    with pytest.raises(SinglePositionLayerNormError):
        SinglePositionArchitecture(positional_embedding="grid")
    with pytest.raises(SinglePositionLayerNormError):
        SinglePositionArchitecture(hidden_channels=64)


def test_rejects_a_contract_that_restores_the_grid_embedding(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text())
    contract["architecture"]["positional_embedding"] = "grid"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(SinglePositionLayerNormError):
        load_contract(path, verify_sources=False)


def test_parent_binding_is_restored() -> None:
    before = {
        name: getattr(controls, name)
        for name in (
            "ModelCSuccessorArchitecture",
            "load_contract",
            "arm_from_contract",
            "VERSION",
        )
    }
    attribution_before = attribution.ModelCSuccessorArchitecture
    with _ParentBinding():
        assert controls.ModelCSuccessorArchitecture is SinglePositionArchitecture
        assert controls.VERSION == VERSION
    for name, value in before.items():
        assert getattr(controls, name) is value
    assert attribution.ModelCSuccessorArchitecture is attribution_before


def test_checkpoint_payload_reconstructs_through_the_attribution_module() -> None:
    """Checkpoint evaluation reconstructs the architecture from the saved payload
    using the attribution module's own global.  Leaving it unbound lets training
    finish and fails only afterwards, on the first checkpoint reload."""

    payload = SinglePositionArchitecture().to_dict()
    assert payload["positional_embedding"] is None
    with pytest.raises(ValueError):
        attribution.ModelCSuccessorArchitecture(**payload)
    with _ParentBinding():
        rebuilt = attribution.ModelCSuccessorArchitecture(**payload)
        assert rebuilt.positional_embedding is None
        assert rebuilt.to_dict() == payload


def test_model_lifts_fifty_one_channels_and_keeps_the_layernorm() -> None:
    architecture = SinglePositionArchitecture()
    arm = controls.RegularizationArm("layernorm", True, 0.0)
    model = controls.build_regularized_model(architecture, arm)
    assert model.fno.positional_embedding is None
    assert isinstance(model.fno.fno_blocks.norm, torch.nn.ModuleList)
    assert len(model.fno.fno_blocks.norm) == 2 * architecture.n_layers
    lifting = model.fno.lifting
    first = next(
        module
        for module in lifting.modules()
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear))
    )
    assert first.weight.shape[1] == 51
    with torch.no_grad():
        output = model(torch.zeros(1, 51, 62, 62))
    assert output.shape == (1, 46, 62, 62)
    assert bool(torch.isfinite(output).all())
