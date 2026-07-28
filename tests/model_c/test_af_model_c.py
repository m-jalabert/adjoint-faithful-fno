"""Contracts and numerical checks for forward-optimized Model C."""

from pathlib import Path

import numpy as np
import pytest

from bire_repro.af_model_c import (
    GROUP_SLICES,
    ModelCArchitecture,
    ModelCLossConfig,
    ModelCLossV2Config,
    group_increment_nrmse_terms,
    group_relative_l2_terms,
    loss_config_from_contract,
    loss_contract,
    loss_contract_sha256,
    model_c_architecture,
    model_c_loss_config,
    model_c_loss_terms,
    tapered_group_spectral_loss,
)
from bire_repro.diagnostics.af_model_c_diagnostics import (
    assert_training_pairs_have_training_targets,
    balanced_training_records,
    integrated_autocorrelation_time,
    retained_energy_fraction,
    wet_rectangle_bounds,
)
from bire_repro.diagnostics.af_model_c_calibrate import (
    ModelCCalibrationConfig,
    gradient_l2_norm,
    propose_loss_weights,
)
from bire_repro.af_model_c_overfit import (
    AUDIT_TERMS,
    ModelCOverfitConfig,
    materialize_rollouts,
    overfit_acceptance,
)
from bire_repro.diagnostics.af_model_c_objective_audit import (
    gradient_inner_product,
    summarize_learning_history,
)
from bire_repro.af_model_c_validation import (
    ModelCValidationError,
    checkpoint_steps,
    chronology_records,
    final_seed_gate_acceptance,
    load_search_contract,
    resolve_final_seed,
    selection_key,
    verify_split_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_model_c_architecture_search_is_bounded() -> None:
    assert model_c_architecture().n_modes == (16, 16)
    assert ModelCArchitecture(n_modes=(24, 24), hidden_channels=64).hidden_channels == 64
    assert ModelCArchitecture(n_layers=6, domain_padding=0.20).n_layers == 6
    with pytest.raises(ValueError, match="modes"):
        ModelCArchitecture(n_modes=(30, 30))
    with pytest.raises(ValueError, match="hidden"):
        ModelCArchitecture(hidden_channels=48)
    assert ModelCLossConfig().to_dict() == {
        "rollout_steps": 3,
        "increment_weight": 0.001,
        "rollout_weight": 0.15,
        "spectral_weight": 0.00001,
        "boundary_weight": 0.065,
        "spectral_bins": 12,
        "western_boundary_width": 4,
    }
    assert len(loss_contract_sha256(ModelCLossConfig())) == 64
    assert (
        loss_contract_sha256(ModelCLossConfig())
        == "19000a1426ea928db7799c82a73ce071a874911eb7e1df50bd276582ec30b5f9"
    )
    assert model_c_loss_config("v2") == ModelCLossV2Config()
    assert loss_contract(ModelCLossV2Config())["version"] == "v2"
    assert ModelCLossV2Config().increment_weight == 0.0025
    with pytest.raises(ValueError, match="changes only increment"):
        ModelCLossV2Config(rollout_weight=0.2)
    with pytest.raises(ValueError, match="unsupported Model C loss contract"):
        loss_config_from_contract(
            {"version": "v3", "config": ModelCLossConfig().to_dict()}
        )
    assert ModelCCalibrationConfig().warmup_epochs == 20


def test_group_state_loss_does_not_let_channel_multiplicity_choose_the_fit() -> None:
    torch = pytest.importorskip("torch")
    target = torch.ones((1, 46, 4, 4))
    wet = torch.ones((1, 1, 4, 4))
    u_error = target.clone()
    u_error[:, GROUP_SLICES["u"]] += 1.0
    ssh_error = target.clone()
    ssh_error[:, GROUP_SLICES["ssh"]] += 1.0
    u_terms = group_relative_l2_terms(u_error, target, wet)
    ssh_terms = group_relative_l2_terms(ssh_error, target, wet)
    assert float(u_terms["u"]) == pytest.approx(1.0)
    assert float(ssh_terms["ssh"]) == pytest.approx(1.0)
    assert float(u_terms["mean"]) == pytest.approx(float(ssh_terms["mean"]))


def test_increment_nrmse_makes_persistence_one_by_construction() -> None:
    torch = pytest.importorskip("torch")
    target_increment = torch.ones((2, 46, 4, 4))
    persistence_increment = torch.zeros_like(target_increment)
    wet = torch.ones((1, 1, 4, 4))
    terms = group_increment_nrmse_terms(
        persistence_increment,
        target_increment,
        wet,
        torch.ones(46),
    )
    for name in (*GROUP_SLICES, "mean"):
        assert float(terms[name]) == pytest.approx(1.0)


def test_tapered_increment_spectrum_is_zero_for_identical_fields() -> None:
    torch = pytest.importorskip("torch")
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    target = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    target[:] = torch.sin(2 * torch.pi * x / 8)
    different = target.clone()
    different[:, 45] = torch.sin(2 * torch.pi * 3 * x / 8)
    wet = torch.ones((1, 1, 8, 8))
    assert float(tapered_group_spectral_loss(target, target, wet, bins=4)) == pytest.approx(
        0.0
    )
    assert float(tapered_group_spectral_loss(different, target, wet, bins=4)) > 0.01


def test_complete_model_c_loss_reports_group_audit_terms() -> None:
    torch = pytest.importorskip("torch")
    present = torch.zeros((1, 46, 8, 8))
    targets = torch.ones((1, 3, 46, 8, 8))
    predictions = targets.clone()
    wet = torch.ones((1, 1, 8, 8))
    terms = model_c_loss_terms(
        predictions,
        targets,
        present,
        wet,
        wet,
        torch.ones(46),
        ModelCLossConfig(),
    )
    assert set(("total", "state_ssh", "state_temperature", "increment_ssh")) <= set(terms)
    assert float(terms["total"]) == pytest.approx(0.0)


def test_training_only_diagnostic_helpers() -> None:
    codes = np.asarray([1] * 100 + [0] * 10 + [2] * 10, dtype=np.uint8)
    records = balanced_training_records(codes, sample_count=96, seed=7)
    assert len(records) == 96
    assert {experiment for experiment, _ in records} == {0, 1, 2}
    assert all(codes[index] == 1 for _, index in records)
    assert integrated_autocorrelation_time([1.0, -1.0] * 10) == pytest.approx(1.0)
    assert integrated_autocorrelation_time([2.0] * 12) == pytest.approx(12.0)


def test_training_pair_gate_checks_target_snapshots_not_future_pair_starts() -> None:
    pair_codes = np.asarray([1, 1, 1, 0, 0], dtype=np.uint8)
    snapshot_codes = np.asarray([1, 1, 1, 1, 0], dtype=np.uint8)
    assert_training_pairs_have_training_targets(
        pair_codes,
        snapshot_codes,
        horizon_days=1,
    )
    snapshot_codes[3] = 0
    with pytest.raises(ValueError, match="crosses"):
        assert_training_pairs_have_training_targets(
            pair_codes,
            snapshot_codes,
            horizon_days=1,
        )


def test_loss_weight_proposal_targets_relative_gradient_contributions() -> None:
    weights = propose_loss_weights(
        {
            "state": 2.0,
            "increment": 4.0,
            "rollout": 1.0,
            "spectral": 0.5,
            "boundary": 10.0,
        }
    )
    assert weights == pytest.approx(
        {
            "increment": 0.25,
            "rollout": 1.0,
            "spectral": 1.0,
            "boundary": 0.05,
        }
    )
    with pytest.raises(ValueError, match="state gradient"):
        propose_loss_weights({"state": 0.0})


def test_gradient_norm_accumulates_without_float32_overflow() -> None:
    torch = pytest.importorskip("torch")
    real_gradient = torch.tensor([2.0e20, 2.0e20], dtype=torch.float32)
    complex_gradient = torch.tensor([2.0e20 + 3.0e20j], dtype=torch.complex64)
    norm = gradient_l2_norm(
        (real_gradient, complex_gradient, None),
        torch.device("cpu"),
    )
    assert torch.isfinite(norm)
    assert float(norm) == pytest.approx(21.0**0.5 * 1.0e20, rel=1.0e-6)


def test_late_gradient_inner_product_preserves_complex_components() -> None:
    torch = pytest.importorskip("torch")
    left = (torch.tensor([1.0 + 2.0j], dtype=torch.complex128), torch.tensor([3.0]))
    right = (torch.tensor([2.0 - 1.0j], dtype=torch.complex128), torch.tensor([4.0]))
    value = gradient_inner_product(left, right, torch.device("cpu"))
    assert float(value) == pytest.approx(12.0)


def test_complete_learning_history_is_rechecked_at_every_evaluation() -> None:
    config = ModelCOverfitConfig(epochs=10, evaluation_interval=5, learning_rates=(5e-4,))
    initial = {name: 1.0 for name in AUDIT_TERMS}

    def metrics(increment_ssh: float, total: float) -> dict[str, float]:
        result = {
            "total": total,
            "state": 0.02,
            "increment": 0.4,
            "rollout": 0.03,
            "spectral": 0.2,
            "boundary": 0.04,
        }
        for group in GROUP_SLICES:
            result[f"state_{group}"] = 0.03
            result[f"increment_{group}"] = 0.5
        result["increment_ssh"] = increment_ssh
        return result

    epoch_five = metrics(1.2, 0.08)
    epoch_ten = metrics(0.8, 0.05)
    report = {
        "config": {
            "sample_count": config.sample_count,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "evaluation_interval": config.evaluation_interval,
            "learning_rates": list(config.learning_rates),
            "adam_betas": list(config.adam_betas),
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "minimum_total_reduction_fraction": config.minimum_total_reduction_fraction,
            "minimum_spectral_reduction_fraction": (
                config.minimum_spectral_reduction_fraction
            ),
            "maximum_state_group": config.maximum_state_group,
            "maximum_increment_group_ratio_to_persistence": (
                config.maximum_increment_group_ratio_to_persistence
            ),
            "maximum_rollout": config.maximum_rollout,
            "maximum_boundary": config.maximum_boundary,
        },
        "loss_contract": {"config": ModelCLossConfig().to_dict()},
        "persistence_increment_nrmse": {name: 1.0 for name in GROUP_SLICES},
        "save_reload_three_step_bitwise_exact": True,
        "attempts": [
            {
                "learning_rate": 5e-4,
                "initial": initial,
                "best_epoch": 10,
                "best": epoch_ten,
                "acceptance": overfit_acceptance(
                    initial,
                    epoch_ten,
                    {name: 1.0 for name in GROUP_SLICES},
                    config,
                ),
                "history": [
                    {
                        "epoch": epoch,
                        "training": epoch_five,
                        **(
                            {"evaluation": epoch_five}
                            if epoch == 5
                            else {"evaluation": epoch_ten}
                            if epoch == 10
                            else {}
                        ),
                    }
                    for epoch in range(1, 11)
                ],
            }
        ],
    }
    audit = summarize_learning_history(report)
    attempt = audit["attempts"][0]
    assert attempt["evaluation_count"] == 2
    assert attempt["criterion_summary"]["increment_group_ratio_to_persistence"] == {
        "pass_count": 1,
        "fail_count": 1,
        "first_pass_epoch": 10,
        "last_fail_epoch": 5,
    }
    assert attempt["any_epoch_accepted"]
    assert attempt["best_balanced_increment"]["epoch"] == 10


def test_model_c_lr_decay_is_explicit_and_bounded() -> None:
    config = ModelCOverfitConfig(
        epochs=320,
        learning_rate_decay_epoch=240,
        learning_rate_decay_factor=0.2,
    )
    assert config.learning_rate_decay_epoch == 240
    with pytest.raises(ValueError, match="requires a decay epoch"):
        ModelCOverfitConfig(learning_rate_decay_factor=0.2)
    with pytest.raises(ValueError, match="inside training"):
        ModelCOverfitConfig(
            learning_rate_decay_epoch=320,
            learning_rate_decay_factor=0.2,
        )


def test_model_c_validation_search_was_frozen_before_metrics() -> None:
    contract, path, digest = load_search_contract(
        PROJECT_ROOT / "config/model_c_validation_search_v1.json"
    )
    assert path.is_file()
    assert digest == "9e1d44299ae6cb36acbb3fc4ad158fb133cbb2d474f004cd2ef5f0e66dc4c6f6"
    assert len(contract["candidate_architectures"]) == 10
    assert checkpoint_steps(contract, 7680) == (
        1920,
        3840,
        5760,
        6720,
        7200,
        7440,
        7560,
        7680,
    )
    assert contract["final_seed_gate"]["seeds"] == [20260723, 20260724, 20260725]
    assert contract["sealed_data"]["inference_read"] is False


def test_model_c_chronology_and_split_checks_never_touch_inference() -> None:
    pair_codes = np.zeros(500, dtype=np.uint8)
    snapshot_codes = np.zeros(500, dtype=np.uint8)
    pair_codes[:200] = 1
    snapshot_codes[:210] = 1
    pair_codes[220:450] = 2
    snapshot_codes[220:460] = 2
    pair_codes[470:] = 3
    snapshot_codes[470:] = 3
    training = chronology_records(pair_codes, 0.25)
    assert {experiment for experiment, _ in training} == {0, 1, 2}
    validation = tuple((experiment, 220) for experiment in range(3))
    long_records = tuple((experiment, 220) for experiment in range(3))
    verify_split_records(
        pair_codes,
        snapshot_codes,
        training,
        validation,
        long_records,
    )
    broken = pair_codes.copy()
    broken[220 + 17 * 10] = 3
    with pytest.raises(ModelCValidationError, match="long validation"):
        verify_split_records(
            broken,
            snapshot_codes,
            training,
            validation,
            long_records,
        )


def test_model_c_validation_rank_is_lexicographic() -> None:
    eligible = {
        "all_groups_beat_persistence": True,
        "worst_group_ratio": 0.99,
    }
    ineligible = {
        "all_groups_beat_persistence": False,
        "worst_group_ratio": 0.90,
    }
    physics = {"physics_score": 0.8}
    assert selection_key(eligible, physics, 100, "a") < selection_key(
        ineligible, {"physics_score": 0.1}, 10, "b"
    )
    assert selection_key(eligible, physics, 100, "a") < selection_key(
        {**eligible, "worst_group_ratio": 0.995},
        {"physics_score": 0.1},
        10,
        "b",
    )


def test_model_c_final_seed_gate_requires_every_group_and_reload() -> None:
    seeds = (20260723, 20260724, 20260725)

    def report(seed: int, *, ssh: float = 0.99, reload: bool = True) -> dict:
        return {
            "training_seed": seed,
            "selected_checkpoint": {
                "validation_ten_day": {
                    "ratio_to_persistence": {
                        "u": 0.2,
                        "v": 0.4,
                        "temperature": 0.8,
                        "ssh": ssh,
                    }
                }
            },
            "save_reload_three_step_bitwise_exact": reload,
        }

    accepted = final_seed_gate_acceptance(
        [report(seed) for seed in seeds],
        seeds,
    )
    assert accepted["accepted"]
    assert accepted["every_seed_every_group_below_persistence"]
    assert accepted["every_seed_three_step_reload_bitwise_exact"]

    slow_field_failure = final_seed_gate_acceptance(
        [report(seeds[0]), report(seeds[1], ssh=1.0), report(seeds[2])],
        seeds,
    )
    assert not slow_field_failure["accepted"]
    assert not slow_field_failure["per_seed"][1]["group_pass"]["ssh"]

    reload_failure = final_seed_gate_acceptance(
        [report(seeds[0]), report(seeds[1], reload=False), report(seeds[2])],
        seeds,
    )
    assert not reload_failure["accepted"]
    assert not reload_failure["every_seed_three_step_reload_bitwise_exact"]


def test_model_c_final_seed_resolution_is_frozen() -> None:
    contract = PROJECT_ROOT / "config/model_c_validation_search_v1.json"
    assert [
        resolve_final_seed(contract, array_index=index)
        for index in range(3)
    ] == [20260723, 20260724, 20260725]
    with pytest.raises(ValueError, match="outside the 3 final seeds"):
        resolve_final_seed(contract, array_index=3)


def test_model_c_overfit_gate_requires_every_physical_group() -> None:
    config = ModelCOverfitConfig()
    initial = {name: 1.0 for name in ("total", "state", "increment", "rollout", "spectral", "boundary")}
    best = {
        "total": 0.05,
        "state": 0.03,
        "increment": 0.5,
        "rollout": 0.05,
        "spectral": 0.2,
        "boundary": 0.08,
    }
    for group in GROUP_SLICES:
        initial[f"state_{group}"] = 1.0
        initial[f"increment_{group}"] = 2.0
        best[f"state_{group}"] = 0.04
        best[f"increment_{group}"] = 0.8
    persistence = {name: 1.0 for name in GROUP_SLICES}
    assert overfit_acceptance(initial, best, persistence, config)["accepted"]
    best["increment_ssh"] = 1.01
    rejected = overfit_acceptance(initial, best, persistence, config)
    assert not rejected["accepted"]
    assert not rejected["criteria"]["increment_group_ratio_to_persistence"]


def test_overfit_materialization_preserves_example_order_and_values() -> None:
    torch = pytest.importorskip("torch")

    class TinyDataset:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> tuple[object, object]:
            return torch.full((2,), index), torch.full((2, 2), index + 10)

    source = TinyDataset()
    cached = materialize_rollouts(source)  # type: ignore[arg-type]
    assert len(cached) == len(source)
    for index in range(len(source)):
        expected = source[index]
        actual = cached[index]
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_spectral_mode_semantics_and_wet_rectangle() -> None:
    field = np.zeros((1, 60, 60), dtype=np.float64)
    y, x = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
    field[0] = np.sin(2 * np.pi * x / 60) + np.cos(2 * np.pi * y / 60)
    low = retained_energy_fraction(field, (12, 12))
    high = retained_energy_fraction(field, (24, 24))
    assert 0 < low <= high <= 1

    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    assert wet_rectangle_bounds(wet) == (1, 61, 1, 61)
    wet[10, 10] = False
    with pytest.raises(ValueError, match="exact wet rectangle"):
        wet_rectangle_bounds(wet)
