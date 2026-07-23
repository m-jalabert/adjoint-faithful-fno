"""Contracts and numerical checks for forward-optimized Model C."""

import numpy as np
import pytest

from bire_repro.af_model_c import (
    GROUP_SLICES,
    ModelCArchitecture,
    ModelCLossConfig,
    group_increment_nrmse_terms,
    group_relative_l2_terms,
    loss_contract_sha256,
    model_c_architecture,
    model_c_loss_terms,
    tapered_group_spectral_loss,
)
from bire_repro.af_model_c_diagnostics import (
    assert_training_pairs_have_training_targets,
    balanced_training_records,
    integrated_autocorrelation_time,
    retained_energy_fraction,
    wet_rectangle_bounds,
)
from bire_repro.af_model_c_calibrate import (
    ModelCCalibrationConfig,
    gradient_l2_norm,
    propose_loss_weights,
)
from bire_repro.af_model_c_overfit import (
    ModelCOverfitConfig,
    materialize_rollouts,
    overfit_acceptance,
)


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
