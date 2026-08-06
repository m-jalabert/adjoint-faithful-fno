from __future__ import annotations

import pytest

from bire_repro.analysis.af_model_c_anomaly_direct_replication_summary import (
    EXPECTED_LEADS,
    curve_auc,
    seed_summary,
)


def test_curve_auc_constant_and_lead_contract() -> None:
    assert curve_auc({lead: 2.5 for lead in EXPECTED_LEADS}) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="leads changed"):
        curve_auc({10: 1.0, 20: 1.0})


def test_seed_summary_applies_pressure_persistence_as_non_veto() -> None:
    report = {
        "selection_decision": {
            "passed": True,
            "selected_fine_tune_step": 13440,
        },
        "save_reload_nine_step_bitwise_exact": True,
        "selected_checkpoint": "checkpoint.pt",
        "selected_checkpoint_sha256": "checkpoint-sha",
        "report_content_sha256": "report-sha",
        "arrays_sha256": "arrays-sha",
        "inference_state_opened": False,
        "selected_training_summary": {
            "checkpoint_gate": {
                "normalized_land_max_abs": 0.0,
                "worst_primary_rmse_auc_ratio": 0.4,
                "worst_slow_field_lead_ratio": 0.5,
            }
        },
        "validation_figure": {
            "metrics": {
                field: {
                    "model": {
                        "day200_mean": 0.5 if field == "phihyd_surface" else 0.1,
                        "maximum_mean": 0.5,
                        "maximum_p90": 0.6,
                        "mean_curve_exceeds_requested_axis": False,
                        "p90_curve_exceeds_requested_axis": False,
                    },
                    "persistence": {
                        "day200_mean": 0.4 if field == "phihyd_surface" else 0.2
                    },
                    "climatology": {"day200_mean": 0.8},
                }
                for field in ("surface_speed", "sst", "phihyd_surface")
            }
        },
    }
    auc = {
        field: {"model": 0.2, "persistence": 0.4, "climatology": 0.5}
        for field in ("surface_speed", "sst", "phihyd_surface")
    }
    result = seed_summary(20260724, report, auc)
    assert result["passed"] is True
    assert (
        result["fixed_S2_day200_rmse"]["phihyd_surface"]["model"]
        > result["fixed_S2_day200_rmse"]["phihyd_surface"]["persistence"]
    )
