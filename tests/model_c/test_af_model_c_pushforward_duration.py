from __future__ import annotations

from pathlib import Path

from bire_repro.af_model_c_pushforward_duration import (
    load_duration_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "model_c_pushforward_duration_v1.json"


def test_duration_contract_is_replay_verified_and_training_only() -> None:
    contract, path, digest = load_duration_contract(CONTRACT)
    assert path == CONTRACT
    assert len(digest) == 64
    assert contract["replay"]["required_exact_step"] == 1920
    assert contract["replay"]["same_objective"] is True
    assert contract["extension"]["absolute_decay_step"] == 1440
    assert contract["extension"]["maximum_total_steps"] == 5760
    assert contract["extension"]["checkpoint_steps"][-1] == 5760
    assert contract["read_contract"]["validation_state_read"] is False
    assert contract["read_contract"]["inference_read"] is False
