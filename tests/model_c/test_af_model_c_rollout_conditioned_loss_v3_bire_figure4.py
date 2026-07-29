from __future__ import annotations

import json

import pytest

from bire_repro.analysis.af_model_c_rollout_conditioned_loss_v3_bire_figure4 import (
    EXPECTED_STARTS,
    FIELDS,
    LEAD_DAYS,
    SELECTED_STEP,
    VERSION,
    load_current_figure4_contract,
)


def _contract() -> dict[str, object]:
    return {
        "version": VERSION,
        "contract_status": "frozen_before_current_model_100_to_200_day_validation_metrics",
        "protocol": {
            "lead_days": list(LEAD_DAYS),
            "fields": list(FIELDS),
            "member_count": 15,
            "start_draw_order": list(EXPECTED_STARTS),
            "selected_loss_v3_step": SELECTED_STEP,
            "apply_loss_v3_projection_every_call": True,
        },
        "read_contract": {
            "training_state_for_climatology": True,
            "fresh_validation_state": True,
            "inference_state": False,
            "intermediate_wind_state": False,
            "response_state": False,
            "adjoint_state": False,
        },
        "source_hashes": {},
    }


def test_load_current_figure4_contract(tmp_path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_contract()))
    loaded, resolved, digest = load_current_figure4_contract(
        path,
        verify_sources=False,
    )
    assert loaded["version"] == VERSION
    assert resolved == path.resolve()
    assert len(digest) == 64


def test_contract_rejects_unprojected_rollout(tmp_path) -> None:
    contract = _contract()
    contract["protocol"]["apply_loss_v3_projection_every_call"] = False
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="protocol changed"):
        load_current_figure4_contract(path, verify_sources=False)
