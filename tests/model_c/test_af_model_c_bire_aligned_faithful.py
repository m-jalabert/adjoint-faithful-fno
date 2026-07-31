from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro.af_model_c_bire_aligned_faithful import (
    COSINE_ETA_MIN,
    COSINE_T_MAX,
    CONTRACT_STATUS,
    EPOCH_BOUNDARIES,
    EPOCH_STEPS,
    FROZEN_TRAINING_FIELDS,
    LEARNING_RATE,
    MAE_WEIGHT,
    PARENT_MAE_WEIGHT,
    VALIDATION_FRACTION,
    VALIDATION_SPLIT_SEED,
    VERSION,
    BireAlignedFaithfulError,
    autoregressive_steps,
    epoch_of_step,
    load_contract,
    split_validation_records,
    stage_of_step,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_faithful_v1.json"
PARENT = ROOT / "config/model_c_bire_aligned_full_state_lr5e4_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_aligned_faithful.sbatch"


def test_contract_declares_the_three_corrections() -> None:
    contract, resolved, digest = load_contract(CONTRACT)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    assert contract["loss"]["mae_weight"] == MAE_WEIGHT == 0.05
    schedule = contract["learning_rate_schedule"]
    assert schedule["kind"] == "cosine_annealing"
    assert schedule["t_max"] == COSINE_T_MAX == 3
    assert schedule["eta_min"] == COSINE_ETA_MIN == 1.0e-5
    selection = contract["checkpoint_selection"]
    assert selection["rule"] == "lowest_validation_loss_within_each_stage"
    assert selection["validation_fraction"] == VALIDATION_FRACTION == 0.1
    assert selection["validation_split_seed"] == VALIDATION_SPLIT_SEED


def test_only_the_three_declared_quantities_moved_from_the_parent() -> None:
    contract, _, _ = load_contract(CONTRACT)
    parent = json.loads(PARENT.read_text())
    assert contract["architecture"] == parent["architecture"]
    assert contract["stages"] == parent["stages"]
    assert contract["selection"] == parent["selection"]
    for field in FROZEN_TRAINING_FIELDS:
        assert contract["training"][field] == parent["training"][field], field
    assert contract["training"]["initial_learning_rate"] == LEARNING_RATE == 5.0e-4
    assert parent["loss"]["mae_weight"] == PARENT_MAE_WEIGHT == 0.01


def test_dropout_is_deliberately_left_at_zero() -> None:
    """The repo default of 0.5 is a regulariser, not a faithfulness fix."""

    contract, _, _ = load_contract(CONTRACT)
    assert contract["architecture"]["channel_mlp_dropout"] == 0.0
    excluded = contract["scientific_motivation"]["excluded_deliberately"]
    assert "channel_mlp_dropout" in excluded


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("loss", "mae_weight"), 0.01),
        (("learning_rate_schedule", "kind"), "step_decay"),
        (("learning_rate_schedule", "t_max"), 5),
        (("checkpoint_selection", "rule"), "fixed_optimizer_steps"),
        (("checkpoint_selection", "validation_fraction"), 0.2),
        (("training", "initial_learning_rate"), 0.01),
        (("training", "batch_size"), 4),
        (("architecture", "channel_mlp_dropout"), 0.5),
        (("architecture", "n_layers"), 4),
    ],
)
def test_rejects_drift_in_any_declared_or_frozen_quantity(
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
    with pytest.raises(BireAlignedFaithfulError):
        load_contract(written, verify_sources=False)


def test_epoch_and_stage_mapping_covers_the_budget() -> None:
    assert EPOCH_BOUNDARIES == (1920, 3840, 5760, 7680)
    assert EPOCH_STEPS == 1920
    assert [epoch_of_step(s) for s in (1, 1920, 1921, 3840, 3841, 7680)] == [
        1,
        1,
        2,
        2,
        3,
        4,
    ]
    assert stage_of_step(3840) == "pretrained"
    assert stage_of_step(3841) == "finetuned"
    assert autoregressive_steps("pretrained") == 1
    assert autoregressive_steps("finetuned") == 2
    # Stage boundary must fall exactly on an epoch boundary, or a checkpoint
    # would straddle two objectives.
    assert 3840 in EPOCH_BOUNDARIES
    with pytest.raises(ValueError):
        epoch_of_step(0)
    with pytest.raises(ValueError):
        epoch_of_step(7681)


def test_validation_holdout_is_disjoint_seeded_and_the_right_size() -> None:
    records = tuple((e, t) for e in range(3) for t in range(500))
    train, valid = split_validation_records(records)
    assert len(valid) == 150
    assert len(train) == 1350
    assert set(train).isdisjoint(set(valid))
    assert set(train) | set(valid) == set(records)
    again = split_validation_records(records)
    assert again[1] == valid
    other = split_validation_records(records, seed=VALIDATION_SPLIT_SEED + 1)
    assert other[1] != valid
    # Training order is preserved so the chunk-aware sampler still reads
    # contiguous Zarr blocks.
    assert list(train) == [r for r in records if r in set(train)]


def test_validation_holdout_rejects_degenerate_fractions() -> None:
    records = tuple((0, t) for t in range(100))
    for fraction in (0.0, 1.0, 1.5):
        with pytest.raises(BireAlignedFaithfulError):
            split_validation_records(records, fraction=fraction)


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    text = SBATCH.read_text()
    assert text.count("bire_repro.af_model_c_bire_aligned_faithful") == 2
    assert "bire_repro.af_model_c_bire_aligned_lr_control" not in text
    assert "bire_repro.af_model_c_bire_aligned_full_state" not in text
    assert CONTRACT.name in text
