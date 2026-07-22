"""Contract and numerical checks for forward-loss Model B."""

import json

import numpy as np
import pytest

from bire_repro.af_model_a import model_a_architecture
from bire_repro.af_model_b import (
    ModelBDevelopmentConfig,
    ModelBFinalConfig,
    ModelBLossConfig,
    ModelBOverfitConfig,
    binned_spectral_loss,
    loss_config_for_profile,
    loss_contract_sha256,
    model_b_architecture,
    records_for_rollout_split,
    rollout_start_indices,
    run_final,
    western_boundary_mask,
)


def test_model_b_architecture_is_exactly_model_a() -> None:
    assert model_b_architecture().to_dict() == model_a_architecture().to_dict()
    assert ModelBOverfitConfig().seed == ModelBDevelopmentConfig().seed == 20260721
    assert ModelBFinalConfig().seed == 20260721


def test_model_b_loss_profiles_add_terms_incrementally() -> None:
    rollout = loss_config_for_profile("rollout")
    spectral = loss_config_for_profile("rollout_spectral")
    full = loss_config_for_profile("full")
    assert rollout.rollout_weight == spectral.rollout_weight == full.rollout_weight == 0.5
    assert rollout.spectral_weight == rollout.boundary_weight == 0.0
    assert spectral.spectral_weight == full.spectral_weight == 0.05
    assert spectral.boundary_weight == 0.0
    assert full.boundary_weight == 0.25
    assert full.rollout_steps == 3
    assert full.spectral_bins == 12
    assert full.western_boundary_width == 4
    assert len(loss_contract_sha256(full)) == 64


def test_model_b_rejects_incomplete_or_nonpositive_loss_contracts() -> None:
    with pytest.raises(ValueError, match="three-step"):
        ModelBLossConfig(rollout_steps=2)
    with pytest.raises(ValueError, match="positive rollout"):
        ModelBLossConfig(rollout_weight=0.0)
    with pytest.raises(ValueError, match="frozen Model B"):
        ModelBFinalConfig(epochs=0)


def test_model_b_final_must_match_complete_development_minimum(tmp_path) -> None:
    report = tmp_path / "development.json"
    report.write_text(
        json.dumps(
            {"loss_contract_sha256": loss_contract_sha256(ModelBLossConfig()), "best_epoch": 7}
        )
    )
    with pytest.raises(ValueError, match="epoch must equal"):
        run_final(
            tmp_path / "dataset.zarr",
            tmp_path / "final",
            development_report_path=report,
            config=ModelBFinalConfig(),
            device_name="cpu",
        )


def test_rollout_starts_cannot_cross_chronological_split_boundaries() -> None:
    codes = np.asarray([1] * 8 + [0] * 2 + [2] * 8, dtype=np.uint8)
    assert rollout_start_indices(codes, 1, horizon_days=2, rollout_steps=3) == (0, 1, 2, 3)
    assert rollout_start_indices(codes, 2, horizon_days=2, rollout_steps=3) == (10, 11)
    records = records_for_rollout_split(
        codes, 1, experiment_count=2, horizon_days=2, rollout_steps=3
    )
    assert records == ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3))


def test_western_boundary_mask_follows_first_wet_cell_in_each_row() -> None:
    wet = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    expected = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(western_boundary_mask(wet, width=2), expected)


def test_binned_spectral_loss_is_zero_for_identical_fields_and_detects_change() -> None:
    torch = pytest.importorskip("torch")
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    target = torch.sin(2 * torch.pi * x / 8)[None, None].to(torch.float32)
    different = torch.sin(2 * torch.pi * 3 * x / 8)[None, None].to(torch.float32)
    wet = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    assert float(binned_spectral_loss(target, target, wet, bins=4)) == pytest.approx(0.0)
    assert float(binned_spectral_loss(different, target, wet, bins=4)) > 0.1
