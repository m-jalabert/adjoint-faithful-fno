"""Tests for the production training contract, architecture and objective.

Runnable without a GPU and without the trajectory store: every test that needs
the cluster's data reads the contract with ``verify_sources=False``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oceanfno.barotropic_transport import barotropic_transport_relative_l2
from oceanfno.continuity import (
    ContinuityContext,
    continuity_relative_l2,
    depth_integrated_transport,
)
from oceanfno.dataset import (
    STATIC_FEATURES,
    TRAIN_CODE,
    records_for_rollout_split,
    store_codes,
    validation_records,
    validation_starts,
    verify,
)
from oceanfno.model import (
    EXPECTED_PARAMETER_COUNT,
    LIFTING_INPUT_CHANNELS,
    ProductionArchitecture,
    build_model,
    parameter_count,
    state_unroll,
)
from oceanfno.objective import (
    LOSS_CONTRACT_SHA256,
    ProductionLossConfig,
    production_loss_config,
    production_loss_terms,
)
from oceanfno.perturbation_growth import (
    GROWTH_RATE_CEILING,
    initial_direction,
    measure_growth_rate,
    single_call_amplification,
)
from oceanfno.spectral_norm import (
    apply_mode_spectral_norm,
    materialized_state_dict,
    mode_sigma_summary,
)
from oceanfno.validation import select_by_validation
from oceanfno.pressure_gradient import (
    DRF_M,
    PressureGradientContext,
    phihyd_from_normalized_state,
    pressure_gradient_relative_l2,
)
from oceanfno.runtime import AUDIT_TERMS, _json_sha256, json_safe, torch
import oceanfno.train as train

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_production_1in_1out_spectralnorm_v1.json"
SBATCH = ROOT / "slurm/models/c/train_production_1in_1out_spectralnorm_v1.sbatch"

requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is optional")


# ---------------------------------------------------------------------------
# the input/output contract
# ---------------------------------------------------------------------------


def test_the_model_reads_one_time_level_and_five_statics() -> None:
    architecture = ProductionArchitecture()
    assert architecture.input_states == 1
    assert architecture.in_channels == 51  # 46 state + 5 static
    assert architecture.positional_channels == 2
    assert architecture.lifting_in_channels == LIFTING_INPUT_CHANNELS == 53
    assert architecture.out_channels == 46
    assert architecture.static_channels == STATIC_FEATURES
    assert architecture.static_channels == (
        "wind_stress_x",
        "wet_mask",
        "coriolis_parameter",
        "zonal_grid_spacing",
        "sst_relaxation_target",
    )


def test_architecture_is_the_32x32_three_block_width_128_operator() -> None:
    architecture = ProductionArchitecture()
    assert architecture.n_modes == (32, 32)
    assert architecture.hidden_channels == 128
    assert architecture.n_layers == 3
    assert architecture.layer_norm_count == 6
    assert architecture.channel_mlp_expansion == 4.0
    assert architecture.channel_mlp_dropout == 0.0
    assert architecture.domain_padding == 0.1
    assert architecture.local_kernel_size == 3
    assert architecture.local_bias is False
    assert architecture.factorization is None


def test_a_history_channel_is_rejected() -> None:
    with pytest.raises(RuntimeError):
        ProductionArchitecture(input_states=2)
    with pytest.raises(RuntimeError):
        ProductionArchitecture(in_channels=97, lifting_in_channels=99)


@requires_torch
def test_parameter_count_is_exactly_the_declared_value() -> None:
    model = build_model(ProductionArchitecture())
    assert parameter_count(model) == EXPECTED_PARAMETER_COUNT == 27_297_960


@requires_torch
def test_the_local_branch_starts_at_exact_zeros_and_has_no_bias() -> None:
    model = build_model(ProductionArchitecture())
    assert model.local.bias is None
    assert float(model.local.weight.abs().max()) == 0.0


@requires_torch
def test_the_forward_map_is_51_channels_in_and_46_out() -> None:
    model = build_model(ProductionArchitecture())
    assert model(torch.zeros((2, 51, 62, 62))).shape == (2, 46, 62, 62)
    with pytest.raises(ValueError):
        model(torch.zeros((2, 97, 62, 62)))


@requires_torch
def test_the_rollout_is_self_generated_after_the_first_call() -> None:
    """Only the first call sees the supplied state; later calls see predictions."""

    seen: list[Any] = []

    class Recorder:
        architecture = ProductionArchitecture()

        def __call__(self, features: Any) -> Any:
            seen.append(features[:, :46].clone())
            return features[:, :46] + 1.0

    features = torch.zeros((1, 51, 8, 8))
    wet = torch.ones((1, 1, 8, 8))
    predictions = state_unroll(Recorder(), features, wet, 6)
    assert predictions.shape == (1, 6, 46, 8, 8)
    assert float(seen[0].abs().max()) == 0.0
    for step in range(1, 6):
        assert torch.equal(seen[step], predictions[:, step - 1])


# ---------------------------------------------------------------------------
# the split and the training set
# ---------------------------------------------------------------------------


def test_one_input_recovers_every_training_start() -> None:
    _, pair_codes = store_codes()
    records = records_for_rollout_split(pair_codes, TRAIN_CODE, rollout_steps=6)
    assert len(records) == train.TRAINING_RECORDS == 17_820
    assert len(records) // 3 == train.TRAINING_STARTS_PER_REGIME == 5940
    # Day 0 is usable precisely because no t-10 history state is required.
    assert min(time for _, time in records) == 0
    assert max(time for _, time in records) == 5939


def test_the_bire_split_is_unchanged_and_inference_is_nested() -> None:
    summary = verify()
    assert summary["train"] == [0, 5999]
    assert summary["validation"] == [6000, 7199]
    assert summary["inference"] == [6200, 7200]
    assert summary["inference_nested_in_validation"] is True
    assert validation_starts().size == 34
    assert validation_records().shape == (102, 2)


def test_the_training_budget_matches_the_declared_exposure() -> None:
    assert train.MAXIMUM_STEPS == 7680
    assert train.BATCH_SIZE == 8
    assert train.ROLLOUT_STEPS == 6
    assert train.STATE_TRANSITIONS == 368_640
    assert train.MICROBATCH_SIZE * train.GRADIENT_ACCUMULATION_STEPS == train.BATCH_SIZE
    # Equal-size microbatches are what make accumulation exact, and 5,940 starts
    # per regime divide by the microbatch without remainder.
    assert train.TRAINING_STARTS_PER_REGIME % train.MICROBATCH_SIZE == 0


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def test_contract_loads_without_cluster_source_verification() -> None:
    contract, resolved, digest = train.load_contract(CONTRACT, verify_sources=False)
    assert contract["version"] == train.VERSION == "model_c_production_1in_1out_spectralnorm_v1"
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64


def test_the_contract_declares_no_lineage_at_all() -> None:
    contract = json.loads(CONTRACT.read_text())
    initialization = contract["initialization"]
    assert initialization["from_scratch"] is True
    assert initialization["parent_checkpoint"] is None
    assert initialization["load_model_state"] is False
    assert initialization["load_optimizer_state"] is False
    assert initialization["normalization_reused"] is False
    assert "checkpoint" not in initialization
    assert initialization["local_branch_initialization"] == "zeros"
    for forbidden in (
        "parent_checkpoint",
        "parent_normalization",
        "parent_report",
        "initialization_checkpoint",
    ):
        assert forbidden not in contract["sources"]
    # The retired lineage vocabulary must not survive as contract *keys*. The
    # `lineage` value states in prose that none of it applies, which is why this
    # walks keys rather than the raw text.
    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    banned = {
        "parent",
        "grandparent",
        "migration",
        "function_preserving",
        "strict_load",
        "strict_same_shape_load",
        "pretrained_checkpoint",
        "fine_tune",
        "inherited_normalization",
        "comparator_model",
        "inherit_parent_fields",
    }
    assert not banned & keys(contract)


def test_a_contract_that_names_a_parent_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text())
    contract["initialization"]["checkpoint"] = "/somewhere/selected.pt"
    target = tmp_path / "with_parent.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(train.TrainingContractError):
        train.load_contract(target, verify_sources=False)


def test_the_contract_declares_the_frozen_schedule() -> None:
    training = json.loads(CONTRACT.read_text())["training"]
    assert training["optimizer"] == "adam"
    assert training["initial_learning_rate"] == 5e-4
    assert training["adam_betas"] == [0.9, 0.95]
    assert training["weight_decay"] == 0.0
    assert training["batch_size"] == 8
    assert training["rollout_steps"] == 6
    assert training["maximum_steps"] == 7680
    assert training["checkpoint_steps"] == [1920, 3840, 5760, 7680]
    assert training["decay_fraction"] == 0.75
    assert training["decay_factor"] == 0.2
    assert training["seed"] == 20260724
    assert training["gradient_clipping"] is False
    assert training["dropout"] == 0.0
    assert training["from_scratch"] is True
    # 75 % of 7,680 is 5,760, so the decayed rate covers steps 5,761-7,680.
    assert training["maximum_steps"] * training["decay_fraction"] == 5760
    assert training["initial_learning_rate"] * training["decay_factor"] == 1e-4


def test_the_contract_recomputes_its_own_normalization() -> None:
    normalization = json.loads(CONTRACT.read_text())["normalization"]
    assert normalization["recomputed_from_training_days_only"] is True
    assert normalization["reused_from_a_previous_run"] is False
    assert normalization["train_days"] == [0, 6000]
    assert normalization["pointwise_shape"] == [46, 62, 62]


def test_every_loss_term_is_active_from_step_one() -> None:
    loss = json.loads(CONTRACT.read_text())["loss"]
    config = production_loss_config()
    assert loss["contract_sha256"] == LOSS_CONTRACT_SHA256
    assert loss["all_terms_active_from_step_1"] is True
    assert loss["staged_fine_tuning"] is False
    assert loss["increment_weight"] == config.increment_weight == 0.001
    assert loss["rollout_weight"] == config.rollout_weight == 0.50
    assert loss["spectral_weight"] == config.spectral_weight == 1e-5
    assert loss["boundary_weight"] == config.boundary_weight == 0.065
    assert loss["pressure_gradient_weight"] == 0.05
    assert loss["continuity_weight"] == 0.05
    assert loss["barotropic_transport_weight"] == 0.05
    assert loss["contraction_penalty"] is False
    assert "contraction_weight" not in loss


def test_the_objective_configuration_is_frozen() -> None:
    assert production_loss_config().rollout_steps == 6
    with pytest.raises(ValueError):
        ProductionLossConfig(rollout_weight=0.15)
    with pytest.raises(ValueError):
        ProductionLossConfig(continuity_weight=0.0)
    with pytest.raises(ValueError):
        ProductionLossConfig(barotropic_transport_weight=0.2)


def test_the_selection_rule_is_declared() -> None:
    selection = json.loads(CONTRACT.read_text())["checkpoint_selection"]
    assert selection["rollout_days"] == 360
    assert selection["short_auc_window_days"] == [10, 90]
    assert selection["short_auc_tolerance"] == 1.05
    assert selection["long_auc_window_days"] == [90, 360]
    assert selection["worst_long_ratio_ceiling"] == 0.85
    assert selection["primary_fields"] == ["surface_speed", "sst", "phihyd_surface"]
    assert selection["growth_rate_ceiling"] == GROWTH_RATE_CEILING == 1.0
    assert selection["growth_rate_calls"] == 200


def test_the_contract_records_the_single_operator_change() -> None:
    contract = json.loads(CONTRACT.read_text())
    record = contract["constrained_relative_to"]
    assert record["version"] == train.REFERENCE_VERSION
    assert record["measured_growth_rate_per_call"] == train.REFERENCE_GROWTH_RATE
    assert record["weight_lineage"] == "none; this run is from scratch"
    assert record["changed"] == [
        "per_mode_spectral_normalization_of_the_three_fourier_convolutions",
        "contraction_penalty_removed_from_the_objective",
    ]
    for name in ("reduce_modes_32_to_24", "remove_the_local_3x3_branch", "tanh_stabilizer"):
        assert name in record["rejected_alternatives"]
    # Architecture and schedule are byte-identical to every preceding arm.
    for ancestor in (
        "model_c_production_1in_1out_v1",
        "model_c_production_1in_1out_v2",
        "model_c_production_1in_1out_v3",
    ):
        old = json.loads((ROOT / f"archive/config/{ancestor}.json").read_text())
        assert contract["architecture"] == old["architecture"], ancestor
        assert contract["training"] == old["training"], ancestor


def test_the_contract_declares_the_spectral_normalization() -> None:
    spectral = json.loads(CONTRACT.read_text())["spectral_normalization"]
    assert spectral["applied"] is True
    assert spectral["form"] == "R_k <- R_k * min(1, 1 / sigma_max(R_k))"
    assert spectral["applies_to"] == "spectral_convolutions_only"
    assert spectral["one_sided"] is True
    assert spectral["adds_parameters"] == 0
    assert spectral["checkpoints_materialized"] is True
    # 32 x 17 = 544 modes per block, three blocks.
    assert spectral["modes_per_block"] * spectral["blocks"] == spectral["matrices_total"] == 1632
    assert spectral["does_not_impose"] == "a global bound on the emulator Jacobian"


# ---------------------------------------------------------------------------
# the spectral normalization itself
# ---------------------------------------------------------------------------


def _exact_sigma(tensor: Any) -> Any:
    """True sigma_max per mode, by SVD -- the check for the power iteration."""

    matrices = tensor.reshape(tensor.shape[0], tensor.shape[1], -1).permute(2, 0, 1)
    return torch.linalg.matrix_norm(matrices, ord=2)


@requires_torch
def test_spectral_norm_caps_every_mode_and_adds_no_parameters() -> None:
    model = build_model(ProductionArchitecture())
    before = parameter_count(model)
    info = apply_mode_spectral_norm(model)  # full warmup: the cap is the claim
    assert parameter_count(model) == before == EXPECTED_PARAMETER_COUNT
    assert info["matrices_total"] == 1632
    for conv in model.fno.fno_blocks.convs:
        with torch.no_grad():
            capped = _exact_sigma(conv.weight.tensor.detach())
        assert float(capped.max()) <= 1.0 + 1.0e-3


@requires_torch
def test_spectral_norm_leaves_already_contracting_modes_untouched() -> None:
    """One-sided: min(1, 1/sigma) is exactly the identity where sigma <= 1."""

    model = build_model(ProductionArchitecture())
    conv = model.fno.fno_blocks.convs[0]
    with torch.no_grad():  # shrink half the modes well below one
        conv.weight.tensor.data[..., :8, :] *= 0.1
    reference = conv.weight.tensor.data.clone()
    apply_mode_spectral_norm(model, warmup_iterations=200)
    conv = model.fno.fno_blocks.convs[0]
    with torch.no_grad():
        free, capped = conv.weight.original.detach(), conv.weight.tensor.detach()
    below = _exact_sigma(free) <= 1.0
    assert int(below.sum()) > 0
    def flat(tensor: Any) -> Any:
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1).permute(2, 0, 1)

    assert torch.allclose(flat(reference)[below], flat(capped)[below], atol=1.0e-7)


@requires_torch
def test_spectral_norm_is_differentiable_to_the_free_parameter() -> None:
    model = build_model(ProductionArchitecture())
    apply_mode_spectral_norm(model, warmup_iterations=4)
    model.train()
    model(torch.zeros((1, 51, 62, 62))).square().mean().backward()
    for conv in model.fno.fno_blocks.convs:
        assert conv.weight.original.grad is not None
        assert bool(torch.isfinite(conv.weight.original.grad).all())
    assert model.local.weight.grad is not None


@requires_torch
def test_materialized_checkpoint_loads_into_a_plain_model_unchanged() -> None:
    """The published operator must be exactly the normalized one."""

    model = build_model(ProductionArchitecture())
    apply_mode_spectral_norm(model, warmup_iterations=4)
    state = materialized_state_dict(model)
    assert not any(".weight.original" in k or ".weight.norm." in k for k in state)
    plain = build_model(ProductionArchitecture())
    incompatible = plain.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.eval()
    plain.eval()
    features = torch.randn((2, 51, 62, 62), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.allclose(model(features), plain(features), atol=1.0e-5)
    # A short warmup under-estimates sigma, so the cap is loose here by design;
    # the tight check lives in test_spectral_norm_caps_every_mode_and_adds_no_parameters.
    with torch.no_grad():
        published = _exact_sigma(
            dict(plain.named_parameters())["fno.fno_blocks.convs.0.weight.tensor"]
        )
    assert float(published.max()) <= 1.5


@requires_torch
def test_mode_sigma_summary_reports_the_unnormalized_spectrum() -> None:
    model = build_model(ProductionArchitecture())
    apply_mode_spectral_norm(model, warmup_iterations=8)
    summary = mode_sigma_summary(model)
    assert len(summary["per_block"]) == 3
    for block in summary["per_block"]:
        assert block["modes"] == 544
        # neuralop's initialization puts every mode above one, so the cap is
        # active everywhere at step zero.
        assert block["sigma_max"] > 1.0
        assert 0.0 <= block["fraction_above_one"] <= 1.0


@requires_torch
def test_the_amplification_probe_carries_no_gradient() -> None:
    """g is diagnostic in this arm; it must not enter the graph."""

    wet = torch.ones((1, 1, 8, 8))
    state = torch.randn((2, 46, 8, 8), generator=torch.Generator().manual_seed(0))
    static = torch.zeros((2, 5, 8, 8))
    direction = initial_direction((1, 46, 8, 8), wet, torch.device("cpu"), seed=0)

    class Scaled:
        def __call__(self, features: Any) -> Any:
            return 1.5 * features[:, :46]

    baseline = Scaled()(torch.cat((state, static), dim=1))
    growth, advanced = single_call_amplification(
        Scaled(), state, static, wet, baseline, direction
    )
    assert float(growth) == pytest.approx(1.5, rel=1.0e-4)
    assert not growth.requires_grad
    assert not advanced.requires_grad
    assert float(advanced.norm()) == pytest.approx(1.0, rel=1.0e-5)

def test_outputs_are_written_under_the_production_roots() -> None:
    output = json.loads(CONTRACT.read_text())["output"]
    assert output["project_root"].endswith("outputs/af_fno/C/" + train.VERSION)
    assert output["scratch_root"].endswith("af_fno/models/C/" + train.VERSION)


def test_slurm_uses_the_production_training_entrypoint() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.train preflight" in text
    assert "-m oceanfno.train run" in text
    assert "model_c_production_1in_1out_spectralnorm_v1.json" in text


# ---------------------------------------------------------------------------
# the objective, term by term
# ---------------------------------------------------------------------------


def _context(n: int = 8) -> PressureGradientContext:
    mean = np.zeros((46, n, n), dtype=np.float32)
    scale = np.ones_like(mean)
    dx = np.full((n, n), 1.0e5, dtype=np.float32)
    wet = np.ones((n, n), dtype=bool)
    return PressureGradientContext(mean, scale, dx, wet)


def _continuity_context(n: int = 8) -> ContinuityContext:
    mean = np.zeros((46, n, n), dtype=np.float32)
    scale = np.ones_like(mean)
    dx = np.full((n, n), 1.0e5, dtype=np.float32)
    wet = np.ones((n, n), dtype=bool)
    return ContinuityContext(mean, scale, dx, wet)


@requires_torch
def test_phihyd_shape_and_identity_loss() -> None:
    context = _context()
    state = torch.zeros((2, 3, 46, 8, 8), dtype=torch.float32)
    phi = phihyd_from_normalized_state(state, context)
    assert phi.shape == (2, 3, 15, 8, 8)
    assert float(pressure_gradient_relative_l2(state, state, context)) == 0.0


@requires_torch
def test_uniform_eta_offset_is_dynamically_irrelevant() -> None:
    context = _context()
    truth = torch.zeros((1, 2, 46, 8, 8), dtype=torch.float32)
    prediction = truth.clone()
    prediction[:, :, 45] += 0.25
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert float(value) == pytest.approx(0.0, abs=1.0e-7)


@requires_torch
def test_pressure_slope_is_seen_and_backpropagates() -> None:
    context = _context()
    truth = torch.zeros((1, 2, 46, 8, 8), dtype=torch.float32)
    ramp = torch.linspace(0.0, 0.1, 8, dtype=torch.float32)[None, :].expand(8, 8)
    truth[:, :, 45] = ramp
    prediction = torch.zeros_like(truth, requires_grad=True)
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0
    value.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all().item())
    assert float(prediction.grad[:, :, 45].abs().sum()) > 0.0


@requires_torch
def test_temperature_gradient_contributes_to_pressure_force() -> None:
    context = _context()
    truth = torch.zeros((1, 1, 46, 8, 8), dtype=torch.float32)
    ramp = torch.linspace(0.0, 2.0, 8, dtype=torch.float32)[None, :].expand(8, 8)
    truth[:, :, 30] = ramp
    prediction = torch.zeros_like(truth)
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0


@requires_torch
def test_depth_integral_uses_the_mitgcm_layer_thicknesses() -> None:
    context = _continuity_context()
    state = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    state[:, 0:15] = 1.0
    transport_x, transport_y = depth_integrated_transport(state, context)
    assert transport_x.shape == (1, 8, 8)
    assert float(transport_x.mean()) == pytest.approx(float(sum(DRF_M)), rel=1.0e-6)
    assert float(transport_y.abs().max()) == 0.0


@requires_torch
def test_continuity_identity_loss_is_exactly_zero() -> None:
    context = _continuity_context()
    state = torch.zeros((2, 6, 46, 8, 8), dtype=torch.float32)
    present = torch.zeros((2, 46, 8, 8), dtype=torch.float32)
    assert float(continuity_relative_l2(state, state, present, context)) == 0.0


@requires_torch
def test_missing_surface_height_tendency_costs_one_of_six_calls() -> None:
    # Truth raises the surface uniformly on the first step only, so exactly one
    # of the six residuals is non-zero and a zero prediction misses all of it.
    context = _continuity_context()
    truth = torch.zeros((1, 6, 46, 8, 8), dtype=torch.float32)
    truth[:, :, 45] = 0.5
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = torch.zeros_like(truth)
    value = continuity_relative_l2(prediction, truth, present, context)
    assert float(value) == pytest.approx(1.0 / 6.0, rel=1.0e-5)


@requires_torch
def test_uniform_transport_offset_is_divergence_free() -> None:
    context = _continuity_context()
    truth = torch.zeros((1, 6, 46, 8, 8), dtype=torch.float32)
    truth[:, :, 45] = 0.5
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = truth.clone()
    prediction[:, :, 0:15] += 0.25
    value = continuity_relative_l2(prediction, truth, present, context)
    assert float(value) == pytest.approx(0.0, abs=1.0e-8)


@requires_torch
def test_continuity_backpropagates_to_velocity_and_surface_height() -> None:
    context = _continuity_context()
    truth = torch.zeros((1, 6, 46, 8, 8), dtype=torch.float32)
    truth[:, :, 45] = 0.5
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = torch.zeros_like(truth, requires_grad=True)
    value = continuity_relative_l2(prediction, truth, present, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0
    value.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all().item())
    assert float(prediction.grad[:, :, 45].abs().sum()) > 0.0
    assert float(prediction.grad[:, :, 0:30].abs().sum()) > 0.0


def _ramped_truth(amplitude: float, meridional: bool = False) -> Any:
    """Truth whose transport grows by a fixed increment on every rollout step."""

    truth = torch.zeros((1, 6, 46, 8, 8), dtype=torch.float32)
    for step in range(6):
        truth[:, step, 0:15] = (step + 1) * amplitude
        if meridional:
            truth[:, step, 15:30] = (step + 1) * amplitude
    return truth


@requires_torch
def test_barotropic_identity_loss_is_exactly_zero() -> None:
    context = _continuity_context()
    state = torch.zeros((2, 6, 46, 8, 8), dtype=torch.float32)
    present = torch.zeros((2, 46, 8, 8), dtype=torch.float32)
    assert float(barotropic_transport_relative_l2(state, state, present, context)) == 0.0


@requires_torch
def test_barotropic_term_scores_the_tendency_not_the_absolute_transport() -> None:
    # A constant transport offset is invisible to every predicted-to-predicted
    # increment, so only the first call -- measured from the observed present
    # state -- can see it. That pins the arithmetic exactly.
    context = _continuity_context()
    amplitude, offset = 0.01, 0.001
    truth = _ramped_truth(amplitude)
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = truth.clone()
    prediction[:, :, 0:15] += offset
    value = barotropic_transport_relative_l2(prediction, truth, present, context)
    expected = 0.5 * (offset / amplitude) ** 2 / 6.0
    assert float(value) == pytest.approx(expected, rel=1.0e-4)


@requires_torch
def test_barotropic_term_weights_zonal_and_meridional_equally() -> None:
    context = _continuity_context()
    amplitude, error = 0.01, 0.002
    truth = _ramped_truth(amplitude, meridional=True)
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)

    zonal = truth.clone()
    zonal[:, :, 0:15] += error
    meridional = truth.clone()
    meridional[:, :, 15:30] += error
    assert float(
        barotropic_transport_relative_l2(zonal, truth, present, context)
    ) == pytest.approx(
        float(barotropic_transport_relative_l2(meridional, truth, present, context))
    )


@requires_torch
def test_barotropic_term_touches_only_the_velocity_channels() -> None:
    context = _continuity_context()
    truth = _ramped_truth(0.01)
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = torch.zeros_like(truth, requires_grad=True)
    value = barotropic_transport_relative_l2(prediction, truth, present, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0
    value.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all().item())
    assert float(prediction.grad[:, :, 0:15].abs().sum()) > 0.0
    # Temperature and the free surface are not part of the transport integral.
    assert float(prediction.grad[:, :, 30:46].abs().sum()) == 0.0


@requires_torch
def test_the_total_is_the_declared_weighted_sum_and_audits_every_term() -> None:
    config = production_loss_config()
    generator = torch.Generator().manual_seed(0)
    predictions = torch.randn((2, 6, 46, 8, 8), generator=generator)
    targets = torch.randn((2, 6, 46, 8, 8), generator=generator)
    present = torch.randn((2, 46, 8, 8), generator=generator)
    wet = torch.ones((1, 1, 8, 8))
    boundary = torch.zeros((1, 1, 8, 8))
    boundary[..., :4] = 1.0
    increment_scale = torch.ones(46)
    auxiliary = {
        "pressure_gradient": pressure_gradient_relative_l2(
            predictions, targets, _context()
        ),
        "continuity": continuity_relative_l2(
            predictions, targets, present, _continuity_context()
        ),
        "barotropic_transport": barotropic_transport_relative_l2(
            predictions, targets, present, _continuity_context()
        ),
    }
    terms = production_loss_terms(
        predictions, targets, present, wet, boundary, increment_scale, config, auxiliary
    )
    assert set(AUDIT_TERMS).issubset(terms)
    expected = (
        terms["state"]
        + config.increment_weight * terms["increment"]
        + config.rollout_weight * terms["rollout"]
        + config.spectral_weight * terms["spectral"]
        + config.boundary_weight * terms["boundary"]
        + 0.05 * terms["pressure_gradient"]
        + 0.05 * terms["continuity"]
        + 0.05 * terms["barotropic_transport"]
    )
    assert float(terms["total"]) == pytest.approx(float(expected), rel=1.0e-6)


@requires_torch
def test_the_objective_is_zero_when_the_prediction_is_the_truth() -> None:
    config = production_loss_config()
    generator = torch.Generator().manual_seed(1)
    targets = torch.randn((1, 6, 46, 8, 8), generator=generator)
    present = torch.randn((1, 46, 8, 8), generator=generator)
    wet = torch.ones((1, 1, 8, 8))
    boundary = torch.ones((1, 1, 8, 8))
    auxiliary = {
        "pressure_gradient": pressure_gradient_relative_l2(targets, targets, _context()),
        "continuity": continuity_relative_l2(
            targets, targets, present, _continuity_context()
        ),
        "barotropic_transport": barotropic_transport_relative_l2(
            targets, targets, present, _continuity_context()
        ),
    }
    terms = production_loss_terms(
        targets, targets, present, wet, boundary, torch.ones(46), config, auxiliary
    )
    assert float(terms["total"]) == pytest.approx(0.0, abs=1.0e-6)


# ---------------------------------------------------------------------------
# the v2 stability term
# ---------------------------------------------------------------------------


class _Scaled:
    """A linear operator with a known, exact amplification factor."""

    def __init__(self, gain: float) -> None:
        self.gain = float(gain)

    def __call__(self, features: Any) -> Any:
        return self.gain * features[:, :46]
@requires_torch
def test_growth_rate_diagnostic_recovers_a_known_rate() -> None:
    """A pure gain operator has growth rate exactly its gain, per call."""

    n = 8
    wet = torch.ones((1, 1, n, n))
    static = torch.zeros((1, 5, n, n))
    state = torch.randn((1, 46, n, n), generator=torch.Generator().manual_seed(2))
    for gain in (0.98, 1.0, 1.02):
        record = measure_growth_rate(
            _Scaled(gain), state, static, wet, calls=120, fit_from=20
        )
        assert record["finite"]
        assert record["growth_rate_per_call"] == pytest.approx(gain, rel=1.0e-4)


def _summary(step: int, short: float, long_ratio: float, growth: float) -> dict[str, Any]:
    return {
        "optimizer_step": step,
        "short_auc_10_90": {f: short for f in ("surface_speed", "sst", "phihyd_surface")},
        "long_ratio_to_climatology": {
            f: long_ratio for f in ("surface_speed", "sst", "phihyd_surface")
        },
        "perturbation_growth": {"worst_growth_rate_per_call": growth},
    }


def test_selection_rejects_the_best_rmse_checkpoint_when_it_is_unstable() -> None:
    """This is exactly the v1 failure: best RMSE, growth rate above one."""

    summaries = [
        _summary(1920, 1.00, 0.60, 1.0162),
        _summary(3840, 1.00, 0.50, 1.0227),
        _summary(5760, 1.00, 0.45, 0.9990),   # slightly worse RMSE, but stable
        _summary(7680, 1.00, 0.30, 1.0168),   # best RMSE, unstable -> ineligible
    ]
    decision = select_by_validation(summaries)
    assert decision["branch"] == "primary_rule"
    assert decision["stable_steps"] == [5760]
    assert decision["selected_optimizer_step"] == 5760
    assert decision["selected_growth_rate_per_call"] == pytest.approx(0.9990)


def test_selection_falls_back_explicitly_when_nothing_is_stable() -> None:
    summaries = [
        _summary(1920, 1.00, 0.60, 1.0162),
        _summary(7680, 1.00, 0.30, 1.0168),
    ]
    decision = select_by_validation(summaries)
    assert decision["branch"] == (
        "declared_fallback_no_checkpoint_met_the_growth_rate_ceiling"
    )
    assert decision["stable_steps"] == []
    # The least unstable one, not the best-RMSE one.
    assert decision["selected_optimizer_step"] == 1920
def test_the_fallback_pool_is_short_feasible_only() -> None:
    """v2 case: the least unstable checkpoint failed the short guard.

    Step 5,760 had the lowest growth rate (1.0036) but short AUCs 1.58x/1.55x/
    2.57x the best, so it was never eligible and 7,680 was correctly returned.
    """

    summaries = [
        _summary(5760, 1.60, 0.45, 1.0036),   # least unstable, fails short guard
        _summary(7680, 1.00, 0.30, 1.01126),  # only short-feasible one
    ]
    decision = select_by_validation(summaries)
    assert decision["short_feasible_steps"] == [7680]
    assert decision["stable_steps"] == []
    assert decision["branch"] == (
        "declared_fallback_no_checkpoint_met_the_growth_rate_ceiling"
    )
    assert decision["selected_optimizer_step"] == 7680


# ---------------------------------------------------------------------------
# report serialization: a completed 3-hour run must never be lost to a float
# ---------------------------------------------------------------------------


def test_non_finite_diagnostics_never_break_a_report() -> None:
    """v3 job 338524 lost 2 h 46 m of finished work to exactly this."""

    import json
    import math

    payload = {
        "doubling_days": math.inf,
        "nested": {"ratio": -math.inf, "rate": math.nan},
        "series": [1.0, math.inf, 3.0],
        "fine": 2.5,
        "text": "unchanged",
        "count": 7,
    }
    clean = json_safe(payload)
    assert clean["doubling_days"] is None
    assert clean["nested"]["ratio"] is None and clean["nested"]["rate"] is None
    assert clean["series"] == [1.0, None, 3.0]
    assert clean["fine"] == 2.5 and clean["text"] == "unchanged" and clean["count"] == 7
    # Both the hash and the written report must survive it.
    assert len(_json_sha256(payload)) == 64
    assert json.dumps(clean)


@requires_torch
def test_a_non_growing_operator_reports_no_doubling_time() -> None:
    """slope <= 0 must give None, not infinity, or the report cannot be written."""

    n = 8
    wet = torch.ones((1, 1, n, n))
    static = torch.zeros((1, 5, n, n))
    state = torch.randn((1, 46, n, n), generator=torch.Generator().manual_seed(3))
    record = measure_growth_rate(_Scaled(0.99), state, static, wet, calls=120, fit_from=20)
    assert record["growth_rate_per_call"] == pytest.approx(0.99, rel=1.0e-4)
    assert record["doubling_days"] is None
    import json as _json
    assert _json.dumps(json_safe(record))


def test_a_failed_growth_measurement_is_treated_as_unstable() -> None:
    """None means unknown, which must never be selected as 'least unstable'."""

    summaries = [
        _summary(3840, 1.00, 0.40, 0.999),
        _summary(7680, 1.00, 0.30, 1.02),
    ]
    summaries[1]["perturbation_growth"]["worst_growth_rate_per_call"] = None
    decision = select_by_validation(summaries)
    assert decision["selected_optimizer_step"] == 3840
    assert decision["stable_steps"] == [3840]
