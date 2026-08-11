"""Tests for the canonical two-in / one-out continuation of the 32x32 arm.

Four things here are worth more than the rest.

The **objective algebra** test evaluates the boxed loss by hand and compares it
with what the shared ``model_c_loss_terms`` returns under the six-step
configuration.  The arm's entire claim is that only temporal context moved; that
is only credible if the objective is still literally the incumbent's, which is
why its SHA-256 is asserted equal to the parent contract's.

The **migration** test proves the warm start is function-preserving in the one
sense that matters here: at initialization the model *ignores* ``x_{t-10}``
entirely and reproduces the one-input parent's map on ``x_t``.  If that failed,
the arm would begin somewhere other than the incumbent's measured skill and the
comparison would be meaningless.

The **unroll** test proves the pair really slides forward, so that no step after
the first sees truth --- the same exposure the one-input arm had.

The **README** test renders against a synthetic report.  The duration arm lost a
completed 1h34m job to a ``KeyError`` in its README, which is the last thing
written before promotion, so this one is exercised before the job is ever
submitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oceanfno.objective import (
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    ModelCLossConfig,
    model_c_loss_config,
)
from oceanfno.train import (
    BASELINE_OPTIMIZER_STEP,
    BATCH_SIZE,
    BOUNDARY_WEIGHT,
    CHECKPOINT_STEPS,
    CONTRACT_STATUS,
    FINE_TUNE_LOSS_CONTRACT_SHA256,
    INCREMENT_WEIGHT,
    STATIC_CHANNEL_MIGRATION,
    LEARNING_RATE,
    LOCAL_KERNEL_SIZE,
    MAXIMUM_STEPS,
    MODES,
    PARENT_VERSION,
    ROLLOUT_STEPS,
    ROLLOUT_WEIGHT,
    SPECTRAL_WEIGHT,
    TRAINING_RECORDS,
    TRAINING_STARTS_PER_REGIME,
    VERSION,
    WORST_LONG_RATIO_CEILING,
    BireProtocolRolloutFineTuneError,
    BireProtocolRolloutFineTuneLossConfig,
    ModelCNewChannelsTrainingError,
    _materialize,
    _readme,
    _resolve_contract,
    acceptance_gate,
    baseline_validation_summary,
    fine_tune_loss_config,
    fine_tune_loss_contract,
    fine_tune_loss_contract_sha256,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_v1.json"
PARENT = ROOT / "config/model_c_2in_1out_v1.json"
GREATGRANDPARENT = ROOT / "config/model_c_bire_protocol_rollout_ft_local24_y32_v1.json"
GRANDPARENT = ROOT / "config/model_c_bire_protocol_rollout_ft_y32_x32_v1.json"
ROOT_CONTRACT = ROOT / "config/model_c_bire_protocol_rollout_ft_local24_v1.json"
SBATCH = ROOT / "slurm/models/c/train_2in_1out_new_channels.sbatch"

NEW_EXTERNAL_CHANNELS = 97
NEW_LIFTING_CHANNELS = 99
PARENT_EXTERNAL_CHANNELS = 95
PARENT_LIFTING_CHANNELS = 97
PARAMETER_COUNT = 27_328_780
PARENT_PARAMETER_COUNT = 27_327_440
NEW_STATIC_CHANNELS = (
    "wind_stress_x",
    "wet_mask",
    "coriolis_parameter",
    "zonal_grid_spacing",
    "sst_relaxation_target",
)


def _parent() -> dict:
    """Resolve the 32x32 parent the way its own declaration says it resolves.

    The chain is three deep --- 32x32 inherits from Y32, Y32 from local24 ---
    so a field-by-field comparison against the raw parent bytes would compare
    this arm against fields that file does not contain.  Materializing here by
    hand, rather than calling the module's own recursive resolver, keeps the
    comparison independent of the code under test.
    """

    y32 = _materialize(
        json.loads(GREATGRANDPARENT.read_text()),
        json.loads(ROOT_CONTRACT.read_text()),
    )
    x32 = _materialize(json.loads(GRANDPARENT.read_text()), y32)
    return _materialize(json.loads(PARENT.read_text()), x32)

torch = pytest.importorskip("torch", reason="the objective algebra needs PyTorch")


def _portable_state_dict(model) -> dict:
    """Mirror the checkpoint writer without coupling this test to its helper."""

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key != "_metadata"
    }


def _tampered(mutate, directory: Path) -> Path:
    """A contract copy with one field moved.

    Written under pytest's ``tmp_path`` rather than beside the real contracts:
    ``load_contract`` is called with ``verify_sources=False`` here, so nothing
    needs the repository layout, and earlier arms' tests left hundreds of
    stray directories in ``config/``.
    """

    contract = json.loads(CONTRACT.read_text())
    mutate(contract)
    path = directory / "tampered.json"
    path.write_text(json.dumps(contract))
    return path


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------


def test_the_six_step_configuration_moves_only_depth_and_rollout_weight() -> None:
    config = fine_tune_loss_config()
    v1 = model_c_loss_config("v1")
    assert config.rollout_steps == ROLLOUT_STEPS == 6
    assert config.rollout_weight == ROLLOUT_WEIGHT == 0.50
    moved = {
        key
        for key, value in config.to_dict().items()
        if v1.to_dict()[key] != value
    }
    assert moved == {"rollout_steps", "rollout_weight"}
    assert config.increment_weight == INCREMENT_WEIGHT == 0.001
    assert config.spectral_weight == SPECTRAL_WEIGHT == 1.0e-5
    assert config.boundary_weight == BOUNDARY_WEIGHT == 0.065
    assert config.spectral_bins == 12 and config.western_boundary_width == 4


def test_the_certified_three_step_configuration_is_untouched() -> None:
    """The exception must not widen into a second free parameter."""

    v1 = model_c_loss_config("v1")
    assert v1.rollout_steps == 3 and v1.rollout_weight == 0.15
    with pytest.raises(ValueError):
        ModelCLossConfig(rollout_steps=6)
    with pytest.raises(ValueError):
        BireProtocolRolloutFineTuneLossConfig(rollout_weight=0.15)
    with pytest.raises(ValueError):
        BireProtocolRolloutFineTuneLossConfig(rollout_steps=3)


def test_the_six_step_objective_has_its_own_hash() -> None:
    assert FINE_TUNE_LOSS_CONTRACT_SHA256 != MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert FINE_TUNE_LOSS_CONTRACT_SHA256 == fine_tune_loss_contract_sha256(
        fine_tune_loss_config()
    )
    contract = fine_tune_loss_contract(fine_tune_loss_config())
    assert contract["config"]["rollout_steps"] == 6
    assert contract["derived_from_loss_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert set(contract["groups"].values()) == {0.25}
    with pytest.raises(BireProtocolRolloutFineTuneError):
        fine_tune_loss_contract(model_c_loss_config("v1"))


@pytest.mark.skipif(not PARENT.is_file(), reason="the two-input parent contract is absent")
def test_the_objective_is_literally_the_one_input_arms() -> None:
    """A moved objective would make the two runs' losses incomparable."""

    assert FINE_TUNE_LOSS_CONTRACT_SHA256 == _parent()["loss"]["contract_sha256"]


def test_the_total_is_the_declared_objective() -> None:
    """L = L_state + 0.001 L_inc + 0.50 mean_{k=2..6} + 1e-5 mean_{k=1..6} + 0.065 mean_{k=1..6}."""

    from oceanfno.objective import (
        group_increment_nrmse_terms,
        group_relative_l2_terms,
        model_c_loss_terms,
        tapered_group_spectral_loss,
    )

    torch.manual_seed(0)
    grid, channels, batch = 16, 46, 2
    config = fine_tune_loss_config()
    predictions = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    targets = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    present = torch.randn(batch, channels, grid, grid)
    wet = torch.ones(1, 1, grid, grid)
    boundary = torch.zeros(1, 1, grid, grid)
    boundary[..., :4] = 1.0
    increment_scale = torch.rand(channels) + 0.5

    terms = model_c_loss_terms(
        predictions, targets, present, wet, boundary, increment_scale, config
    )

    state = group_relative_l2_terms(predictions[:, 0], targets[:, 0], wet)["mean"]
    increment = group_increment_nrmse_terms(
        predictions[:, 0] - present, targets[:, 0] - present, wet, increment_scale
    )["mean"]
    rollout = torch.stack(
        [
            group_relative_l2_terms(predictions[:, k], targets[:, k], wet)["mean"]
            for k in range(1, ROLLOUT_STEPS)
        ]
    ).mean()
    spectral = torch.stack(
        [
            tapered_group_spectral_loss(
                predictions[:, k] - (present if k == 0 else predictions[:, k - 1]),
                targets[:, k] - (present if k == 0 else targets[:, k - 1]),
                wet,
                bins=config.spectral_bins,
            )
            for k in range(ROLLOUT_STEPS)
        ]
    ).mean()
    boundary_term = torch.stack(
        [
            group_relative_l2_terms(predictions[:, k], targets[:, k], boundary)["mean"]
            for k in range(ROLLOUT_STEPS)
        ]
    ).mean()

    # The rollout mean covers five steps, the spectral and boundary means six.
    assert len([k for k in range(1, ROLLOUT_STEPS)]) == 5
    for name, expected in (
        ("state", state),
        ("increment", increment),
        ("rollout", rollout),
        ("spectral", spectral),
        ("boundary", boundary_term),
    ):
        assert torch.allclose(terms[name], expected), name
    expected_total = (
        state
        + INCREMENT_WEIGHT * increment
        + ROLLOUT_WEIGHT * rollout
        + SPECTRAL_WEIGHT * spectral
        + BOUNDARY_WEIGHT * boundary_term
    )
    assert torch.allclose(terms["total"], expected_total)


def test_the_rollout_term_is_not_the_three_step_one() -> None:
    """A six-step tensor scored under the v1 config would be silently truncated."""

    from oceanfno.objective import model_c_loss_terms

    torch.manual_seed(1)
    grid, channels, batch = 16, 46, 2
    predictions = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    targets = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    present = torch.randn(batch, channels, grid, grid)
    wet = torch.ones(1, 1, grid, grid)
    boundary = torch.zeros(1, 1, grid, grid)
    boundary[..., :4] = 1.0
    increment_scale = torch.rand(channels) + 0.5
    with pytest.raises(ValueError):
        model_c_loss_terms(
            predictions, targets, present, wet, boundary, increment_scale,
            model_c_loss_config("v1"),
        )


# --------------------------------------------------------------------------
# The active two-input architecture and the exact one-input warm migration
# --------------------------------------------------------------------------


def test_the_active_model_reads_five_physical_statics() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireTwoInNewChannelsArchitecture,
        build_bire_two_in_new_channels_model,
    )

    architecture = BireTwoInNewChannelsArchitecture()
    assert architecture.static_channels == NEW_STATIC_CHANNELS
    assert "distance_to_wall_normalized" not in architecture.static_channels
    assert architecture.n_modes == MODES == (32, 32)
    assert architecture.input_states == 2 and architecture.input_lag_days == 10
    assert architecture.in_channels == NEW_EXTERNAL_CHANNELS == 2 * 46 + 5
    assert architecture.lifting_in_channels == NEW_LIFTING_CHANNELS == 97 + 2
    assert architecture.out_channels == 46
    assert architecture.local_kernel_size == LOCAL_KERNEL_SIZE == 3
    model = build_bire_two_in_new_channels_model(architecture)

    for index, convolution in enumerate(model.fno.fno_blocks.convs):
        # Spectral capacity is untouched by a static-channel change.
        assert tuple(convolution.n_modes) == (32, 17), index
    assert tuple(model.fno.lifting.fcs[0].weight.shape) == (256, 99, 1)
    assert tuple(model.local.weight.shape) == (46, 97, 3, 3)
    assert model.local.bias is None
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT
    # The whole cost of the change is two input columns on two tensors.
    assert PARAMETER_COUNT - PARENT_PARAMETER_COUNT == 256 * 2 + 46 * 2 * 9

def test_the_two_input_parent_is_untouched() -> None:
    """The comparator must still be the exact old-statics model."""

    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireTwoInOneOutArchitecture,
        RETAINED_STATIC_FEATURES,
        build_bire_two_in_one_out_model,
    )

    architecture = BireTwoInOneOutArchitecture()
    assert architecture.n_modes == (32, 32)
    assert architecture.in_channels == PARENT_EXTERNAL_CHANNELS
    assert architecture.lifting_in_channels == PARENT_LIFTING_CHANNELS
    assert "static_channels" not in architecture.to_dict()
    assert RETAINED_STATIC_FEATURES == (
        "wind_stress_x",
        "wet_mask",
        "distance_to_wall_normalized",
    )
    model = build_bire_two_in_one_out_model(architecture)
    assert tuple(model.fno.lifting.fcs[0].weight.shape) == (256, 97, 1)
    assert sum(p.numel() for p in model.parameters()) == PARENT_PARAMETER_COUNT

def test_static_channel_migration_drops_d_wall_and_zeroes_the_coefficients() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireTwoInNewChannelsArchitecture,
        BireTwoInOneOutArchitecture,
        build_bire_two_in_new_channels_model,
        build_bire_two_in_one_out_model,
        migrate_two_in_static_channels_state_dict,
    )

    torch.manual_seed(19)
    parent = build_bire_two_in_one_out_model(BireTwoInOneOutArchitecture()).eval()
    with torch.no_grad():
        parent.local.weight.normal_(mean=0.0, std=0.02)
        parent.fno.lifting.fcs[0].weight.normal_(mean=0.0, std=0.02)
    parent_state = _portable_state_dict(parent)
    before = {k: v.clone() for k, v in parent_state.items()}
    target = build_bire_two_in_new_channels_model(
        BireTwoInNewChannelsArchitecture()
    ).eval()
    result = migrate_two_in_static_channels_state_dict(parent_state, target)
    state, provenance = result["state_dict"], result["provenance"]

    assert provenance["source_static_channels"] == [
        "wind_stress_x", "wet_mask", "distance_to_wall_normalized",
    ]
    assert provenance["target_static_channels"] == list(NEW_STATIC_CHANNELS)
    assert provenance["spectral_capacity_unchanged"] is True
    assert provenance["temporal_context_unchanged"] is True
    # The one migration in this tree that must NOT claim exactness.
    assert provenance["function_preserving"] is False
    assert "distance_to_wall" in provenance["not_function_preserving_because"]
    assert {r["key"] for r in provenance["input_rewrites"]} == {
        "fno.lifting.fcs.0.weight", "local.weight",
    }

    widened = {"fno.lifting.fcs.0.weight", "local.weight"}
    for key, value in parent_state.items():
        if key not in widened:
            assert torch.equal(state[key], value), key
    for key in widened:
        # 94 shared columns copied: two 46-channel states, then wind and wet.
        assert torch.equal(state[key][:, :94], parent_state[key][:, :94]), key
        # the three new coefficient columns start at exact zero
        assert torch.count_nonzero(state[key][:, 94:97]).item() == 0, key
    # lifting keeps its two position columns, shifted past the new coefficients
    lifting = "fno.lifting.fcs.0.weight"
    assert torch.equal(state[lifting][:, 97:99], parent_state[lifting][:, 95:97])
    assert state[lifting].shape[1] == 99 and state["local.weight"].shape[1] == 97
    # the dropped column is genuinely gone, not smuggled in somewhere
    dropped = parent_state[lifting][:, 94:95]
    assert torch.count_nonzero(dropped).item() > 0
    assert all(torch.equal(parent_state[k], before[k]) for k in parent_state)

def test_the_warm_start_is_measurably_not_function_preserving() -> None:
    """Removing a channel the parent uses must change the map, and be measured."""

    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireTwoInNewChannelsArchitecture,
        BireTwoInOneOutArchitecture,
        build_bire_two_in_new_channels_model,
        build_bire_two_in_one_out_model,
        migrate_two_in_static_channels_state_dict,
        static_channel_perturbation,
    )

    torch.manual_seed(23)
    parent = build_bire_two_in_one_out_model(BireTwoInOneOutArchitecture()).eval()
    with torch.no_grad():
        parent.local.weight.normal_(mean=0.0, std=0.02)
    target = build_bire_two_in_new_channels_model(
        BireTwoInNewChannelsArchitecture()
    ).eval()
    migrate_two_in_static_channels_state_dict(_portable_state_dict(parent), target)

    states = torch.randn(1, 92, 62, 62)
    shared = torch.randn(1, 2, 62, 62)
    d_wall = torch.randn(1, 1, 62, 62)
    coefficients = torch.randn(1, 3, 62, 62)
    parent_features = torch.cat((states, shared, d_wall), dim=1)
    new_features = torch.cat((states, shared, coefficients), dim=1)

    # The three new columns are zero, so the new coefficients cannot yet matter.
    with torch.inference_mode():
        first = target(new_features)
        second = target(torch.cat((states, shared, coefficients * 99.0), dim=1))
    assert torch.equal(first, second)

    measured = static_channel_perturbation(
        parent, target, parent_features, new_features
    )
    assert set(measured) >= {
        "mean_abs_change", "max_abs_change",
        "mean_abs_parent_output", "relative_mean_abs_change",
    }
    assert measured["units"] == "normalized_state"
    # It is a real change, not a rounding difference.
    assert measured["mean_abs_change"] > 0.0

def test_static_channel_migration_rejects_missing_unexpected_and_wrong_shapes() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireAlignedFullStateError,
        BireTwoInNewChannelsArchitecture,
        BireTwoInOneOutArchitecture,
        build_bire_two_in_new_channels_model,
        build_bire_two_in_one_out_model,
        migrate_two_in_static_channels_state_dict,
    )

    parent = build_bire_two_in_one_out_model(BireTwoInOneOutArchitecture())
    state = _portable_state_dict(parent)
    target = build_bire_two_in_new_channels_model(BireTwoInNewChannelsArchitecture())

    missing = dict(state); missing.pop("fno.projection.fcs.1.bias")
    unexpected = dict(state); unexpected["undeclared.weight"] = torch.zeros(1)
    wrong_shape = dict(state); wrong_shape["local.weight"] = state["local.weight"][:, :40]
    undeclared = dict(state)
    undeclared["fno.fno_blocks.convs.0.weight.tensor"] = state[
        "fno.fno_blocks.convs.0.weight.tensor"
    ][..., :13]
    for tampered in (missing, unexpected, wrong_shape, undeclared):
        with pytest.raises(BireAlignedFullStateError):
            migrate_two_in_static_channels_state_dict(tampered, target)

def test_the_builders_refuse_each_others_architecture() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireAlignedFullStateError,
        BireTwoInNewChannelsArchitecture,
        BireTwoInOneOutArchitecture,
        build_bire_two_in_new_channels_model,
        build_bire_two_in_one_out_model,
    )

    with pytest.raises(BireAlignedFullStateError):
        build_bire_two_in_new_channels_model(BireTwoInOneOutArchitecture())
    with pytest.raises(BireAlignedFullStateError):
        build_bire_two_in_one_out_model(BireTwoInNewChannelsArchitecture())
    with pytest.raises(BireAlignedFullStateError):
        BireTwoInNewChannelsArchitecture(
            static_channels=("wind_stress_x", "wet_mask", "distance_to_wall_normalized")
        )

def test_the_pair_slides_forward_and_no_step_after_the_first_sees_truth() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import two_in_state_unroll

    class Recorder(torch.nn.Module):
        """Return a recognisable state and remember every pair it was handed."""

        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []
            self.call = 0

        def forward(self, features):
            assert features.shape[1] == 95
            self.seen.append((features[:, :46].clone(), features[:, 46:92].clone()))
            self.call += 1
            return torch.full_like(features[:, :46], float(self.call))

    model = Recorder()
    history = torch.full((1, 46, 4, 4), -1.0)
    present = torch.full((1, 46, 4, 4), -2.0)
    static = torch.randn(1, 3, 4, 4)
    features = torch.cat((history, present, static), dim=1)
    wet = torch.ones(1, 1, 4, 4)

    predictions = two_in_state_unroll(model, features, wet, ROLLOUT_STEPS)
    assert predictions.shape == (1, ROLLOUT_STEPS, 46, 4, 4)
    assert model.call == ROLLOUT_STEPS

    first_history, first_present = model.seen[0]
    assert torch.equal(first_history, history) and torch.equal(first_present, present)
    second_history, second_present = model.seen[1]
    # The truth present state becomes the history; the prediction is the present.
    assert torch.equal(second_history, present)
    assert torch.equal(second_present, predictions[:, 0])
    for step in range(2, ROLLOUT_STEPS):
        seen_history, seen_present = model.seen[step]
        assert torch.equal(seen_history, predictions[:, step - 2]), step
        assert torch.equal(seen_present, predictions[:, step - 1]), step

    with pytest.raises(ValueError):
        two_in_state_unroll(model, features, wet, 0)


def test_the_stepper_carries_history_across_calls() -> None:
    pytest.importorskip("neuralop")
    import numpy as np

    from oceanfno.model import BireAlignedFullStateError, BireTwoInStepper

    class Echo(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[torch.Tensor] = []

        def forward(self, features):
            self.seen.append(features[:, :46].clone())
            return features[:, 46:92] + 1.0

    wet = np.ones((4, 4), dtype=bool)
    mean = np.zeros((46, 4, 4), dtype=np.float32)
    scale = np.ones((46, 4, 4), dtype=np.float32)
    model = Echo()
    stepper = BireTwoInStepper(
        model=model, device=torch.device("cpu"), wet=wet, mean=mean, scale=scale,
        wind_mean=0.0, wind_scale=1.0,
    )
    current = torch.zeros(1, 46, 4, 4)
    static = torch.zeros(1, 3, 4, 4)

    with pytest.raises(BireAlignedFullStateError):
        stepper.step(current, static)

    history = np.full((1, 46, 4, 4), -7.0, dtype=np.float32)
    stepper.begin(history)
    first = stepper.step(current, static)
    second = stepper.step(first, static)

    assert torch.allclose(model.seen[0], torch.from_numpy(history).float())
    assert torch.equal(model.seen[1], current)
    assert torch.equal(second, first + 1.0)
    assert stepper.requires_history is True


def test_the_one_input_stepper_still_ignores_history() -> None:
    """The comparator path must be untouched by the history hook."""

    pytest.importorskip("neuralop")
    from oceanfno.model import BireAlignedStepper

    assert BireAlignedStepper.requires_history is False
    assert BireAlignedStepper.begin(None, None) is None


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the new-channel contract is absent")
def test_the_contract_moves_only_the_declared_quantities() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = _parent()
    assert contract["version"] == VERSION != parent["version"]
    assert contract["contract_status"] == CONTRACT_STATUS
    for field in ("dataset", "normalization", "training", "loss", "checkpoint_selection"):
        assert contract[field] == parent[field], field

    architecture_changes = {
        field
        for field in set(contract["architecture"]) | set(parent["architecture"])
        if contract["architecture"].get(field) != parent["architecture"].get(field)
    }
    assert architecture_changes == {
        "in_channels",
        "lifting_in_channels",
        "static_channels",
    }
    # Neither the Fourier bandwidth nor the temporal context moved with them.
    assert tuple(parent["architecture"]["n_modes"]) == MODES == (32, 32)
    assert tuple(contract["architecture"]["n_modes"]) == MODES
    assert parent["architecture"]["input_states"] == (
        contract["architecture"]["input_states"]
    ) == 2
    assert parent["architecture"]["input_lag_days"] == (
        contract["architecture"]["input_lag_days"]
    ) == 10
    assert parent["architecture"]["local_kernel_size"] == (
        contract["architecture"]["local_kernel_size"]
    ) == LOCAL_KERNEL_SIZE == 3
    assert "static_channels" not in parent["architecture"]
    assert tuple(contract["architecture"]["static_channels"]) == NEW_STATIC_CHANNELS
    assert contract["architecture"]["in_channels"] == NEW_EXTERNAL_CHANNELS
    assert contract["architecture"]["lifting_in_channels"] == NEW_LIFTING_CHANNELS


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the new-channel contract is absent")
def test_the_optimizer_matches_the_declaration() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    training = contract["training"]
    assert training["optimizer"] == "adam"
    assert float(training["initial_learning_rate"]) == LEARNING_RATE == 2.0e-5
    assert tuple(training["adam_betas"]) == (0.9, 0.95)
    assert float(training["weight_decay"]) == 0.0
    assert training["gradient_clipping"] is False
    assert int(training["batch_size"]) == BATCH_SIZE == 4
    assert int(training["seed"]) == 20260724
    assert int(training["maximum_steps"]) == MAXIMUM_STEPS == 3840
    assert tuple(training["checkpoint_steps"]) == CHECKPOINT_STEPS == (960, 1920, 2880, 3840)
    decay = round(training["maximum_steps"] * training["decay_fraction"])
    assert decay == 2880
    schedule = training["learning_rate_schedule"]
    assert schedule["steps_1_to_2880"] == 2.0e-5
    assert schedule["steps_2881_to_3840"] == pytest.approx(4.0e-6)
    assert BATCH_SIZE * ROLLOUT_STEPS == 8 * 3


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the new-channel contract is absent")
def test_initialization_is_the_two_input_checkpoint_with_zeroed_coefficients() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    initialization = contract["initialization"]
    assert initialization["version"] == PARENT_VERSION
    assert int(initialization["optimizer_step"]) == BASELINE_OPTIMIZER_STEP == 3840
    assert initialization["load_only"] == "model_state_dict"
    assert initialization["optimizer_state_loaded"] is False
    assert initialization["normalization_reused"] is True
    assert contract["training"]["from_scratch"] is False
    assert contract["training"]["load_optimizer_state"] is False
    assert initialization["checkpoint"].endswith("model_c_2in_1out_v1/selected.pt")
    assert initialization["static_channel_migration"] == STATIC_CHANNEL_MIGRATION
    assert initialization["new_coefficient_initialization"] == "zeros"
    assert initialization["dropped_channel"] == "distance_to_wall_normalized"
    # This arm must not claim an exactness it does not have.
    assert initialization["function_preserving"] is False
    assert initialization["local_branch_initialization"] == "copied_from_parent"
    assert initialization["local_branch_bias"] is False
    assert (
        contract["sources"]["initialization_checkpoint"]["path"]
        == initialization["checkpoint"]
    )
    assert (
        contract["sources"]["parent_normalization"]["path"]
        .endswith("model_c_2in_1out_train_only_normalization.npz")
    )
    # The parent is compact, so both its raw bytes and its resolved document
    # are pinned; no level of the three-deep inheritance can move unnoticed.
    record = contract["sources"]["parent_contract"]
    assert record["path"] == str(PARENT)
    assert len(record["sha256"]) == len(record["materialized_sha256"]) == 64
    assert record["sha256"] != record["materialized_sha256"]


@pytest.mark.skipif(not PARENT.is_file(), reason="the two-input parent contract is absent")
def test_the_resolver_walks_the_whole_chain_to_its_root() -> None:
    resolved = _resolve_contract(PARENT)
    assert resolved == _parent()
    # The root states everything and terminates the walk.
    root = json.loads(ROOT_CONTRACT.read_text())
    assert "inherit_parent_fields" not in root
    assert _resolve_contract(ROOT_CONTRACT) == root


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the new-channel contract is absent")
def test_the_loss_block_declares_the_six_step_objective() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = _parent()
    loss = contract["loss"]
    assert loss["contract_sha256"] == FINE_TUNE_LOSS_CONTRACT_SHA256
    assert loss["derived_from_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert int(loss["rollout_steps"]) == int(parent["loss"]["rollout_steps"]) == 6
    assert loss == parent["loss"]
    assert loss["coefficients"]["rollout"] == 0.50
    assert loss["state"] == parent["loss"]["state"]


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the new-channel contract is absent")
def test_the_output_roots_are_this_arms_own_and_do_not_collide() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    scratch = Path(contract["output"]["scratch_root"])
    project = Path(contract["output"]["project_root"])
    parent = _parent()
    assert scratch.name == project.name == "model_c_2in_1out_new_channels_v1"
    assert project != Path(parent["output"]["project_root"])
    assert scratch != Path(parent["output"]["scratch_root"])


@pytest.mark.skipif(
    not (
        Path(
            "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/"
            "model_c_2in_1out_new_channels_v1"
        ).exists()
    ),
    reason="the new-channel arm has not been trained yet",
)
def test_completed_outputs_are_pinned_and_a_rerun_would_be_refused() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    scratch = Path(contract["output"]["scratch_root"])
    project = Path(contract["output"]["project_root"])
    assert (scratch / "selected.pt").is_file()
    assert (project / "manifest.json").is_file()
    assert scratch.exists(), "the completed scratch root is the overwrite guard"



@pytest.mark.skipif(not CONTRACT.is_file(), reason="the new-channel contract is absent")
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c["architecture"].update(hidden_channels=64), id="architecture_moved"
        ),
        pytest.param(
            lambda c: c["architecture"].update(n_modes=[32, 24]),
            id="modes_moved_with_the_inputs",
        ),
        pytest.param(
            lambda c: c["architecture"].update(input_states=1),
            id="temporal_context_reverted",
        ),
        pytest.param(
            lambda c: c["architecture"].update(
                static_channels=["wind_stress_x", "wet_mask",
                                 "distance_to_wall_normalized"],
                in_channels=95, lifting_in_channels=97,
            ),
            id="static_channels_reverted",
        ),
        pytest.param(
            lambda c: c["architecture"].update(
                static_channels=["wind_stress_x", "wet_mask", "coriolis_parameter",
                                 "zonal_grid_spacing", "distance_to_wall_normalized"],
            ),
            id="d_wall_smuggled_back_in",
        ),
        pytest.param(
            lambda c: c["sources"].pop("mitgcm_sst_relaxation"),
            id="relaxation_source_dropped",
        ),
        pytest.param(
            lambda c: c["architecture"].update(input_states=3),
            id="third_time_level_added",
        ),
        pytest.param(
            lambda c: c["architecture"].update(input_lag_days=20),
            id="history_lag_moved_off_the_horizon",
        ),
        pytest.param(
            lambda c: c["architecture"].update(in_channels=141),
            id="input_width_disagrees_with_the_static_count",
        ),
        pytest.param(
            lambda c: c["architecture"].update(position_encoding="smooth_xy"),
            id="position_encoding_reintroduced",
        ),
        pytest.param(
            lambda c: c["sources"]["parent_contract"].update(
                materialized_sha256="0" * 64
            ),
            id="resolved_parent_changed",
        ),
        pytest.param(
            lambda c: c["architecture"].update(local_kernel_size=None), id="local_removed"
        ),
        pytest.param(
            lambda c: c["initialization"].update(
                static_channel_migration="copy_every_parent_column_including_d_wall"
            ),
            id="wrong_static_migration",
        ),
        pytest.param(
            lambda c: c["initialization"].update(
                new_coefficient_initialization="random"
            ),
            id="new_coefficients_not_zeroed",
        ),
        pytest.param(
            lambda c: c["initialization"].update(local_branch_initialization="zeros"),
            id="trained_local_branch_zeroed",
        ),
        pytest.param(
            lambda c: c["training"].update(load_optimizer_state=True),
            id="optimizer_state_loaded",
        ),
    ],
)
def test_a_tampered_contract_is_rejected(mutate, tmp_path) -> None:
    with pytest.raises(ModelCNewChannelsTrainingError):
        load_contract(_tampered(mutate, tmp_path), verify_sources=False)


# --------------------------------------------------------------------------
# The training set
# --------------------------------------------------------------------------


def test_two_input_starts_keep_the_whole_pair_and_target_sequence_inside_training() -> None:
    from oceanfno.dataset import (
        records_for_rollout_split,
        records_for_two_in_rollout_split,
        store_codes,
    )

    _, pair_codes = store_codes()
    records = records_for_two_in_rollout_split(
        pair_codes, 1, rollout_steps=ROLLOUT_STEPS
    )
    starts = sorted({time_index for _, time_index in records})
    assert len(records) == TRAINING_RECORDS == 17790
    assert len(starts) == TRAINING_STARTS_PER_REGIME == 5930
    # t - 10 must exist inside training, and t + 60 must still be a training day.
    assert starts[0] == 10
    assert starts[-1] == 5939
    assert starts[-1] + 10 * ROLLOUT_STEPS == 5999

    one_input = records_for_rollout_split(
        pair_codes, 1, rollout_steps=ROLLOUT_STEPS
    )
    # Starts are consecutive days, so exactly days 0--9 of each regime are lost.
    dropped = set(one_input) - set(records)
    assert dropped == {
        (experiment, day) for experiment in range(3) for day in range(10)
    }
    assert len(one_input) - len(records) == 30


def test_the_derived_static_block_is_built_from_the_simulations_own_inputs() -> None:
    """Three of the five channels are not in the store at all."""

    import numpy as np

    from oceanfno.dataset import (
        MITGCM_OMEGA,
        NEW_CHANNEL_STATIC_FEATURES,
        NewChannelStaticError,
        assert_mitgcm_coriolis_defaults,
        coriolis_parameter,
        read_mitgcm_2d,
    )
    from oceanfno.runtime import STATIC_FEATURES

    assert NEW_CHANNEL_STATIC_FEATURES == NEW_STATIC_CHANNELS
    # Only two of the five can come from the store.
    assert set(NEW_STATIC_CHANNELS) & set(STATIC_FEATURES) == {
        "wind_stress_x", "wet_mask",
    }

    # f = 2 omega sin(phi) with MITgcm's default sidereal rotation period.
    assert MITGCM_OMEGA == pytest.approx(2.0 * np.pi / 86164.0)
    equator = coriolis_parameter(np.zeros((2, 2)))
    assert np.allclose(equator, 0.0)
    pole = coriolis_parameter(np.full((1, 1), 90.0))
    assert pole[0, 0] == pytest.approx(2.0 * MITGCM_OMEGA)
    assert coriolis_parameter(np.full((1, 1), 45.0))[0, 0] > 0.0

    with pytest.raises(NewChannelStaticError):
        read_mitgcm_2d(__file__)


@pytest.mark.skipif(
    not Path(
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm/S0/spinup/"
        "years_000_010/data"
    ).is_file(),
    reason="the MITgcm run directory is not mounted",
)
def test_the_declaration_audit_refuses_a_setup_that_moved() -> None:
    from oceanfno.dataset import NewChannelStaticError, assert_mitgcm_coriolis_defaults

    run = Path(
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm/S0/spinup/"
        "years_000_010"
    )
    provenance = assert_mitgcm_coriolis_defaults(run / "data")
    assert provenance["rotation_period_seconds"] == 86164.0
    assert "thetaClimFile='SST_relax.bin'" in provenance["asserted_settings"]
    assert "rotationPeriod" in provenance["asserted_absent"]

    # An f-plane override would invalidate the derived Coriolis channel.
    tampered = Path(str(run / "data"))
    text = tampered.read_text().replace(
        "usingSphericalPolarGrid=.TRUE.,", "usingSphericalPolarGrid=.TRUE.,\n f0=1.E-4,"
    )
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        moved = Path(directory) / "data"
        moved.write_text(text)
        with pytest.raises(NewChannelStaticError):
            assert_mitgcm_coriolis_defaults(moved)

def _summary(step: int, short: float, long: float) -> dict:
    fields = ("surface_speed", "sst", "phihyd_surface")
    return {
        "optimizer_step": step,
        "short_auc_10_90": {f: short for f in fields},
        "long_ratio_to_climatology": {f: long for f in fields},
        "lead_days": [0, 10],
    }


def test_the_acceptance_gate_reads_both_conditions() -> None:
    baseline = _summary(BASELINE_OPTIMIZER_STEP, 1.0, 0.93)
    passing = acceptance_gate(_summary(3840, 0.95, 0.80), baseline)
    assert passing["short_auc_no_field_worsens_by_more_than_5_percent"]
    assert passing["worst_long_ratio_at_or_below_ceiling"]
    assert passing["validation_conditions_pass"]
    assert passing["worst_long_ratio_ceiling"] == WORST_LONG_RATIO_CEILING == 0.85

    short_breach = acceptance_gate(_summary(3840, 1.06, 0.80), baseline)
    assert not short_breach["short_auc_no_field_worsens_by_more_than_5_percent"]
    assert not short_breach["validation_conditions_pass"]

    # Exactly 5% worse is inside the tolerance; a hair more is not.
    assert acceptance_gate(_summary(3840, 1.05, 0.80), baseline)["validation_conditions_pass"]

    long_breach = acceptance_gate(_summary(3840, 0.95, 0.86), baseline)
    assert not long_breach["worst_long_ratio_at_or_below_ceiling"]
    assert not long_breach["validation_conditions_pass"]
    assert "2000_day_all_values_finite" in long_breach["deferred_to_the_figure_package"]

    # Passing the absolute 0.85 ceiling is insufficient if the temporal-context
    # arm regresses against its archived one-input parent.
    stronger_parent = _summary(BASELINE_OPTIMIZER_STEP, 1.0, 0.70)
    parent_breach = acceptance_gate(_summary(3840, 0.95, 0.80), stronger_parent)
    assert parent_breach["worst_long_ratio_at_or_below_ceiling"]
    assert not parent_breach["worst_long_ratio_no_worse_than_parent"]
    assert not parent_breach["validation_conditions_pass"]


def test_the_baseline_summary_must_be_the_parent_step_3840_one() -> None:
    report = {
        "validation_summaries": [
            _summary(1920, 1.0, 1.0),
            _summary(BASELINE_OPTIMIZER_STEP, 0.9, 0.93),
        ]
    }
    assert baseline_validation_summary(report)["optimizer_step"] == BASELINE_OPTIMIZER_STEP
    with pytest.raises(BireProtocolRolloutFineTuneError):
        baseline_validation_summary({"validation_summaries": [_summary(1920, 1.0, 1.0)]})


def _report() -> dict:
    summaries = [
        _summary(step, 0.9 + index * 0.01, 0.95 - index * 0.05)
        for index, step in enumerate(CHECKPOINT_STEPS)
    ]
    baseline = _summary(BASELINE_OPTIMIZER_STEP, 1.0, 0.932)
    selected = summaries[-1]
    return {
        "content_sha256": "a" * 64,
        "parameter_count": PARAMETER_COUNT,
        "optimizer": {"decay_step": 2880},
        "static_channels": {"channels": list(NEW_STATIC_CHANNELS)},
        "initialization": {
            "static_channel_perturbation": {
                "mean_abs_change": 0.00988,
                "mean_abs_parent_output": 0.48719,
                "relative_mean_abs_change": 0.02028,
            }
        },
        "counts": {
            "training_rollout_records": TRAINING_RECORDS,
            "training_starts_per_regime": TRAINING_STARTS_PER_REGIME,
            "earliest_training_start": 10,
            "latest_training_start": 5939,
            "validation_records": 102,
        },
        "validation_summaries": summaries,
        "baseline_validation_summary": baseline,
        "checkpoint_comparison_to_baseline": {
            str(int(s["optimizer_step"])): acceptance_gate(s, baseline) for s in summaries
        },
        "selection_decision": {
            "selected_optimizer_step": int(selected["optimizer_step"]),
            "branch": "primary_rule",
        },
        "acceptance_gate": acceptance_gate(selected, baseline),
    }


def test_the_readme_renders_before_the_job_is_ever_submitted() -> None:
    """Regression guard: the duration arm lost a finished job to a README KeyError."""

    text = _readme(_report())
    assert "Physical static channels" in text
    assert f"{BASELINE_OPTIMIZER_STEP:,}" in text
    assert "Selected step 3,840" in text and "primary_rule" in text
    for step in CHECKPOINT_STEPS:
        assert f"{step:,}" in text
    flat = " ".join(text.split())
    assert "f(phi), dx(phi), theta_clim(x, y)" in flat
    assert "distance to wall" in flat
    assert "95 to 97" in flat and "97 to 99" in flat
    # The README must state the one thing this arm cannot claim.
    assert "not function-preserving" in flat
    assert "2.03%" in flat
    for channel in NEW_STATIC_CHANNELS:
        assert channel in text
    for stale in ("zonal half-spectrum", "from scratch", "Adams--Bashforth"):
        assert stale not in text


def test_the_readme_rejects_a_report_missing_the_gate() -> None:
    report = _report()
    del report["acceptance_gate"]
    with pytest.raises(KeyError):
        _readme(report)


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------


def test_artifact_names_are_distinct_and_name_this_arm() -> None:
    """Two arms wrote the same checkpoint filename with different weights once."""

    from oceanfno import train as arm

    names = [getattr(arm, n) for n in ("REPORT_NAME", "ARRAYS_NAME", "FIGURE_NAME",
                                       "NORMALIZATION_NAME", "DIVERGENCE_NAME",
                                       "CHECKPOINT_STEM")]
    assert len(set(names)) == len(names)
    assert all("2in_1out_new_channels" in n for n in names)
    checkpoints = {f"{arm.CHECKPOINT_STEM}_{s:05d}.pt" for s in CHECKPOINT_STEPS}
    assert len(checkpoints) == len(CHECKPOINT_STEPS)



@pytest.mark.skipif(not SBATCH.is_file(), reason="launcher absent")
def test_launcher_invokes_its_own_module_and_contract() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "oceanfno." in line
    }
    assert invoked == {"oceanfno.train"}
    assert "model_c_2in_1out_new_channels_v1.json" in text


def test_the_package_carries_no_module_rebinding_machinery() -> None:
    """The consolidated tree calls its collaborators directly.

    The previous layout reused a parent arm's training loop by swapping module
    globals in and out around the call. That is the mechanism this refactor
    removed, and its absence is what makes the call graph readable.
    """

    from oceanfno import anomaly, figures, train

    for module in (train, figures, anomaly):
        assert not hasattr(module, "_ParentBinding")
        assert not hasattr(module, "_SuiteBinding")
        assert not hasattr(module, "PARENT_BINDINGS")
        assert not hasattr(module, "_y32_runner")
        assert not hasattr(module, "_y32_figures_runner")
        assert not hasattr(module, "_y32_anomaly_runner")
