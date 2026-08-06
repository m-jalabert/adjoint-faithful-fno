"""Contract and pure-function tests for the trajectory-v2 Model C successor."""

from pathlib import Path

import numpy as np

from bire_repro.af_model_c_successor import (
    _contiguous_runs,
    apply_training_gate,
    architecture_from_candidate,
    load_successor_contract,
    resolve_candidate_id,
    training_increment_scale,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "model_c_successor_training_v1.json"


def test_successor_contract_preserves_bounded_phase_order() -> None:
    contract, path, digest = load_successor_contract(CONTRACT)
    assert path == CONTRACT
    assert len(digest) == 64
    assert resolve_candidate_id(CONTRACT, phase=1, array_index=0) == (
        "v2_control_w64_mlp05"
    )
    assert resolve_candidate_id(CONTRACT, phase=1, array_index=1) == (
        "v2_mix_w64_mlp4"
    )
    assert resolve_candidate_id(CONTRACT, phase=2, array_index=0) == (
        "v2_bireprop_w128_mlp4"
    )
    control = architecture_from_candidate(contract, "v2_control_w64_mlp05")
    mixing = architecture_from_candidate(contract, "v2_mix_w64_mlp4")
    bire = architecture_from_candidate(contract, "v2_bireprop_w128_mlp4")
    assert control.hidden_channels == mixing.hidden_channels == 64
    assert control.channel_mlp_expansion == 0.5
    assert mixing.channel_mlp_expansion == 4.0
    assert bire.hidden_channels == 128
    assert bire.lifting_channel_ratio * bire.hidden_channels == 256
    assert bire.projection_channel_ratio * bire.hidden_channels == 256


def test_increment_scale_handles_two_disconnected_training_blocks() -> None:
    state = np.zeros((3, 50, 46, 2, 2), dtype=np.float32)
    state[:] = np.arange(50, dtype=np.float32)[None, :, None, None, None]
    pair_codes = np.zeros(50, dtype=np.uint8)
    pair_codes[0:15] = 1
    pair_codes[25:40] = 1
    group = {
        "state": state,
        "state_scale": np.ones(46, dtype=np.float32),
        "wet_mask": np.ones((2, 2), dtype=np.uint8),
    }
    assert _contiguous_runs(np.flatnonzero(pair_codes == 1)) == (
        (0, 15),
        (25, 40),
    )
    result = training_increment_scale(group, pair_codes, chunk_days=4)
    assert result.shape == (46,)
    assert np.allclose(result, 10.0)


def test_training_gate_requires_regime_skill_stability_and_reload() -> None:
    one_step = {"every_regime_and_group_beats_persistence": True}
    lead_metrics = {
        lead: {
            group: {"amplitude_ratio": 1.0}
            for group in ("u", "v", "temperature", "ssh")
        }
        for lead in ("30", "90", "180")
    }
    stability = {"finite": True, "lead_metrics": lead_metrics}
    gate = {
        "long_rollout_amplitude_ratio_bounds": [0.5, 2.0],
        "long_rollout_gate_leads_days": [90, 180],
    }
    passed = apply_training_gate(
        one_step, stability, gate, reload_exact=True
    )
    assert passed["passed"]
    assert passed["status"] == "eligible_for_fresh_v2_validation"

    lead_metrics["180"]["ssh"]["amplitude_ratio"] = 2.1
    rejected = apply_training_gate(
        one_step, stability, gate, reload_exact=True
    )
    assert not rejected["passed"]
    assert not rejected["long_rollout_amplitude_passed"]
