"""Contract tests for the loss-only pressure-gradient fine-tune."""
from __future__ import annotations

import json
from pathlib import Path

from oceanfno.model import BireTwoInNewChannelsArchitecture
from oceanfno.train_pressure_gradient import (
    DEFAULT_PRESSURE_WEIGHT,
    PARENT_VERSION,
    VERSION,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_v1.json"
PARENT = ROOT / "config/model_c_2in_1out_new_channels_v1.json"
SBATCH = ROOT / "slurm/models/c/train_2in_1out_new_channels_pressure_gradient.sbatch"


def test_pressure_gradient_successor_has_no_architecture_change() -> None:
    child = json.loads(CONTRACT.read_text())
    parent = json.loads(PARENT.read_text())
    assert child["version"] == VERSION
    assert child["initialization"]["version"] == PARENT_VERSION
    assert child["architecture"] == parent["architecture"]
    assert BireTwoInNewChannelsArchitecture(**child["architecture"]).to_dict() == child["architecture"]
    assert child["initialization"]["strict_same_shape_load"] is True
    assert child["initialization"]["function_preserving"] is True
    assert child["initialization"]["optimizer_state_loaded"] is False


def test_only_declared_scientific_change_is_pressure_loss() -> None:
    child = json.loads(CONTRACT.read_text())
    assert child["loss"]["pressure_gradient_weight"] == DEFAULT_PRESSURE_WEIGHT
    assert child["loss"]["pressure_gradient"]["new_model_output_channels"] == 0
    assert child["training"]["rollout_steps"] == 6
    assert child["training"]["maximum_steps"] == 3840
    assert child["training"]["checkpoint_steps"] == [960, 1920, 2880, 3840]
    assert child["training"]["load_optimizer_state"] is False
    assert child["training"]["from_scratch"] is False


def test_contract_loads_without_cluster_source_verification() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert contract["version"] == VERSION
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64


def test_slurm_calls_pressure_gradient_runner() -> None:
    text = SBATCH.read_text()
    assert "oceanfno.train_pressure_gradient" in text
    assert "model_c_2in_1out_new_channels_pressure_gradient_v1.json" in text
