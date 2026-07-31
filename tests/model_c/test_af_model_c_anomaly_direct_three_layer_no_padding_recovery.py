from __future__ import annotations

from bire_repro import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from bire_repro.af_model_c_anomaly_direct_three_layer_no_padding import (
    ThreeLayerNoPaddingArchitecture,
)
from bire_repro.af_model_c_anomaly_direct_three_layer_no_padding_recovery import (
    _patched_attribution_architecture,
    _validate_checkpoint_payload,
)


def _architecture() -> dict:
    return ThreeLayerNoPaddingArchitecture().to_dict()


def test_recovery_patch_accepts_control_architecture() -> None:
    original = attribution.ModelCSuccessorArchitecture
    with _patched_attribution_architecture():
        restored = attribution.ModelCSuccessorArchitecture(**_architecture())
        assert restored.n_layers == 3
        assert restored.domain_padding is None
    assert attribution.ModelCSuccessorArchitecture is original


def test_recovery_payload_requires_completed_frozen_checkpoint() -> None:
    source = {
        "architecture": _architecture(),
        "training": {"base_loss_contract_sha256": "loss"},
    }
    payload = {
        "version": "model_c_anomaly_direct_three_layer_no_padding_v1",
        "optimizer_step": 15360,
        "fine_tune_step": 15360,
        "architecture": _architecture(),
        "contract": "/tmp/source.json",
        "contract_sha256": "source",
        "base_loss_contract_sha256": "loss",
        "arm": {
            "arm_id": "three_layer_no_padding",
            "pointwise_layer_norm": False,
            "channel_mlp_dropout": 0.0,
        },
        "training_history_record": {"optimizer_step": 15360},
        "model_state_dict": {"weight": object()},
    }
    _validate_checkpoint_payload(
        payload,
        step=15360,
        source_contract=source,
        source_contract_path=__import__("pathlib").Path("/tmp/source.json"),
        source_contract_sha="source",
    )
