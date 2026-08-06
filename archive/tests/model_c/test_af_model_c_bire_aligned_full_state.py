from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from bire_repro import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from bire_repro.af_data import STATIC_FEATURES
from bire_repro.af_model_c_bire_aligned_full_state import (
    CHECKPOINT_STEPS,
    CONTRACT_STATUS,
    EXTERNAL_INPUT_CHANNELS,
    FINE_TUNE_STEPS,
    LIFTING_INPUT_CHANNELS,
    MAXIMUM_STEPS,
    PRETRAIN_STEPS,
    RETAINED_STATIC_FEATURES,
    RETAINED_STATIC_INDICES,
    VERSION,
    BireAlignedArchitecture,
    BireAlignedFullStateError,
    BirePositionalEncoding,
    _GateBinding,
    arm_stages,
    bire_loss_terms,
    build_bire_aligned_model,
    load_contract,
    retained_features,
)

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_bire_aligned_full_state_v1.json"
)


def test_contract_declares_the_bire_aligned_package() -> None:
    contract, resolved, digest = load_contract(CONTRACT)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    architecture = contract["architecture"]
    assert architecture["in_channels"] == EXTERNAL_INPUT_CHANNELS == 49
    assert architecture["lifting_in_channels"] == LIFTING_INPUT_CHANNELS == 51
    assert architecture["out_channels"] == 46
    assert architecture["n_layers"] == 3
    assert architecture["local_kernel_size"] is None
    assert architecture["positional_embedding"] is None


def test_contract_declares_the_two_stage_bire_protocol() -> None:
    contract, _, _ = load_contract(CONTRACT)
    stages = arm_stages(contract)
    assert [record["stage_id"] for record in stages] == ["pretrained", "finetuned"]
    assert [record["autoregressive_steps"] for record in stages] == [1, 2]
    assert stages[0]["last_optimizer_step"] == PRETRAIN_STEPS == 3840
    assert stages[1]["last_optimizer_step"] == MAXIMUM_STEPS == 7680
    assert FINE_TUNE_STEPS == 3840
    assert tuple(contract["training"]["checkpoint_steps"]) == CHECKPOINT_STEPS


def test_contract_freezes_the_bire_optimizer_and_objective() -> None:
    contract, _, _ = load_contract(CONTRACT)
    training = contract["training"]
    assert training["optimizer"] == "adam"
    assert training["initial_learning_rate"] == 0.01
    assert training["adam_betas"] == [0.9, 0.95]
    assert training["weight_decay"] == 0.0
    assert training["batch_size"] == 8
    assert training["gradient_clipping"] is False
    assert contract["loss"]["mae_weight"] == 0.01
    # Sequence exposure, not optimizer-step count, is what is matched.
    assert 7680 * 8 == 15360 * 4


def test_contract_keeps_the_held_state_sealed_and_lists_the_same_figures() -> None:
    contract, _, _ = load_contract(CONTRACT)
    read = contract["read_contract"]
    assert read["training_state"] is True
    for sealed in (
        "validation_state",
        "inference_state",
        "held_s0_state",
        "intermediate_wind_state",
        "response_state",
        "adjoint_state",
    ):
        assert read[sealed] is False
    assert contract["held_evaluation"]["figures"] == [
        "model_c_bire_figure3_streamfunction_1deg_s0_dt10.png",
        "model_c_bire_figure4_dt10_rmse_0_200_days_s0.png",
        "model_c_bire_figure5_dt10_single_member_rmse_s0.png",
        "model_c_bire_figure6_dt10_acc_0_200_days_s0.png",
        "model_c_bire_figure7_dt10_streamfunction_day060_day2000_s0.png",
        "model_c_bire_figure8_dt10_rmse_0_2000_days_s0.png",
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("architecture", "n_layers"), 4),
        (("architecture", "local_kernel_size"), 3),
        (("architecture", "in_channels"), 51),
        (("training", "initial_learning_rate"), 0.0005),
        (("training", "batch_size"), 4),
        (("training", "optimizer"), "adamw"),
        (("loss", "mae_weight"), 0.1),
    ],
)
def test_rejects_a_contract_that_drifts_back_towards_the_incumbent(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    contract = json.loads(CONTRACT.read_text())
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    written = tmp_path / "contract.json"
    written.write_text(json.dumps(contract))
    with pytest.raises(BireAlignedFullStateError):
        load_contract(written, verify_sources=False)


def test_architecture_rejects_the_incumbent_shape() -> None:
    architecture = BireAlignedArchitecture()
    assert architecture.n_layers == 3
    assert architecture.layer_norm_count == 6
    assert architecture.local_kernel_size is None
    with pytest.raises(BireAlignedFullStateError):
        BireAlignedArchitecture(n_layers=4)
    with pytest.raises(BireAlignedFullStateError):
        BireAlignedArchitecture(local_kernel_size=3)
    with pytest.raises(BireAlignedFullStateError):
        BireAlignedArchitecture(positional_embedding="grid")
    with pytest.raises(BireAlignedFullStateError):
        BireAlignedArchitecture(channel_mlp_dropout=0.5)


def test_position_enters_exactly_once_and_the_linear_fields_are_gone() -> None:
    assert RETAINED_STATIC_FEATURES == (
        "wind_stress_x",
        "wet_mask",
        "distance_to_wall_normalized",
    )
    assert "longitude_normalized" not in RETAINED_STATIC_FEATURES
    assert "latitude_normalized" not in RETAINED_STATIC_FEATURES
    assert RETAINED_STATIC_INDICES == (0, 3, 4)
    assert EXTERNAL_INPUT_CHANNELS == 46 + 3
    assert LIFTING_INPUT_CHANNELS == EXTERNAL_INPUT_CHANNELS + 2


def test_retained_features_drops_only_the_two_coordinate_channels() -> None:
    batch = torch.arange(
        46 + len(STATIC_FEATURES), dtype=torch.float32
    ).view(1, -1, 1, 1).expand(2, -1, 62, 62)
    reduced = retained_features(batch)
    assert reduced.shape == (2, EXTERNAL_INPUT_CHANNELS, 62, 62)
    assert torch.equal(reduced[:, :46], batch[:, :46])
    assert [float(reduced[0, 46 + i, 0, 0]) for i in range(3)] == [46.0, 49.0, 50.0]
    with pytest.raises(ValueError):
        retained_features(batch[:, :49])


def test_positional_encoding_matches_the_published_construction() -> None:
    encoder = BirePositionalEncoding(62, 62)
    values = encoder(torch.zeros(3, EXTERNAL_INPUT_CHANNELS, 62, 62))
    assert values.shape == (3, LIFTING_INPUT_CHANNELS, 62, 62)
    position_x = values[0, -2]
    position_y = values[0, -1]
    for index in range(62):
        angle = math.pi / (2 * 62) * index
        expected = math.sin(angle) if index % 2 == 0 else math.cos(angle)
        # p_x varies along the last axis, p_y along the second to last.
        assert float(position_x[0, index]) == pytest.approx(expected, abs=1e-6)
        assert float(position_y[index, 0]) == pytest.approx(expected, abs=1e-6)
    assert torch.allclose(position_x[0], position_x[-1])
    assert torch.allclose(position_y[:, 0], position_y[:, -1])
    # Constants, so they must not travel in a checkpoint.
    assert encoder.state_dict() == {}


def test_model_has_three_blocks_six_layer_norms_and_no_local_branch() -> None:
    architecture = BireAlignedArchitecture()
    model = build_bire_aligned_model(architecture)
    assert model.fno.positional_embedding is None
    assert isinstance(model.fno.fno_blocks.norm, torch.nn.ModuleList)
    assert len(model.fno.fno_blocks.norm) == 6
    assert len(model.fno.fno_blocks.convs) == 3
    # Bire's pointwise residual inside each block is retained ...
    assert model.fno.fno_blocks.fno_skips is not None
    assert len(model.fno.fno_blocks.fno_skips) == 3
    # ... while the external raw-input branch is gone.
    assert not hasattr(model, "local")
    lifting = model.fno.lifting
    first = next(
        module
        for module in lifting.modules()
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear))
    )
    assert first.weight.shape[1] == LIFTING_INPUT_CHANNELS
    with torch.no_grad():
        output = model(torch.zeros(1, EXTERNAL_INPUT_CHANNELS, 62, 62))
    assert output.shape == (1, 46, 62, 62)
    assert bool(torch.isfinite(output).all())
    with pytest.raises(ValueError):
        model(torch.zeros(1, LIFTING_INPUT_CHANNELS, 62, 62))


def test_loss_is_wet_cell_mse_plus_one_percent_mae() -> None:
    wet = torch.zeros(1, 1, 4, 4)
    wet[..., :2, :] = 1.0
    prediction = torch.zeros(2, 3, 4, 4)
    target = torch.full((2, 3, 4, 4), 2.0)
    terms = bire_loss_terms(prediction, target, wet)
    # Every wet cell contributes an error of exactly 2.
    assert float(terms["mse"]) == pytest.approx(4.0)
    assert float(terms["mae"]) == pytest.approx(2.0)
    assert float(terms["total"]) == pytest.approx(4.0 + 0.01 * 2.0)
    # Dry cells contribute neither error nor denominator.
    target_dry = target.clone()
    target_dry[..., 2:, :] = 1000.0
    assert float(bire_loss_terms(prediction, target_dry, wet)["total"]) == pytest.approx(
        float(terms["total"])
    )


def test_gate_binding_installs_and_restores_the_frozen_instrument() -> None:
    before = {
        name: getattr(attribution, name)
        for name in (
            "ModelCSuccessorArchitecture",
            "build_successor",
            "PointwiseDirectStepper",
        )
    }
    payload = BireAlignedArchitecture().to_dict()
    with pytest.raises(TypeError):
        attribution.ModelCSuccessorArchitecture(**payload)
    with _GateBinding():
        rebuilt = attribution.ModelCSuccessorArchitecture(**payload)
        assert rebuilt.to_dict() == payload
        assert attribution.build_successor is build_bire_aligned_model
    for name, value in before.items():
        assert getattr(attribution, name) is value


def test_frozen_instrument_entry_points_still_exist() -> None:
    """The binding replaces names in the certified instrument; catch renames."""

    for name in ("_evaluate_seed", "training_records", "load_contract"):
        assert callable(getattr(attribution, name))
