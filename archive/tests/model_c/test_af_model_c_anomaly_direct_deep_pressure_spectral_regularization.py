from __future__ import annotations

import numpy as np
import torch

from bire_repro.af_model_c_anomaly_direct_deep_pressure_spectral_regularization import (
    deep_pressure_high_mode_loss,
    select_candidate,
    torch_phihyd_from_normalized_state,
)
from bire_repro.af_pressure import phihyd_from_theta_eta


def test_torch_phihyd_matches_numpy_reference() -> None:
    generator = np.random.default_rng(20260729)
    state = generator.normal(size=(2, 46, 6, 5)).astype(np.float32)
    mean = np.zeros((46, 6, 5), dtype=np.float32)
    scale = np.ones_like(mean)
    observed = torch_phihyd_from_normalized_state(
        torch.from_numpy(state),
        torch.from_numpy(mean),
        torch.from_numpy(scale),
    )
    expected = phihyd_from_theta_eta(state[:, 30:45], state[:, 45])
    assert np.allclose(observed.detach().numpy(), expected, rtol=2.0e-6, atol=2.0e-5)


def test_deep_pressure_tail_loss_is_zero_for_exact_target_and_detects_tail() -> None:
    target = torch.zeros((1, 3, 46, 16, 16), dtype=torch.float32)
    prediction = target.clone()
    mean = torch.zeros((46, 16, 16), dtype=torch.float32)
    scale = torch.ones_like(mean)
    wet = torch.ones((1, 1, 16, 16), dtype=torch.float32)
    exact = deep_pressure_high_mode_loss(
        prediction,
        target,
        mean,
        scale,
        wet,
        tail_start_mode=4,
        tail_end_mode=8,
    )
    assert float(exact) == 0.0
    checkerboard = torch.as_tensor(
        np.indices((16, 16)).sum(axis=0) % 2,
        dtype=torch.float32,
    )
    prediction[:, :, 45] = 0.01 * (2.0 * checkerboard - 1.0)
    perturbed = deep_pressure_high_mode_loss(
        prediction,
        target,
        mean,
        scale,
        wet,
        tail_start_mode=4,
        tail_end_mode=8,
    )
    assert torch.isfinite(perturbed)
    assert float(perturbed) > 0.0


def test_candidate_selection_prefers_best_passing_spectrum() -> None:
    source = {
        "fine_tune_step": 0,
        "optimizer_step": 13440,
        "gate": {"pass": False},
    }
    earlier = {
        "fine_tune_step": 480,
        "optimizer_step": 13920,
        "worst_mid_bottom_modewise_ratio_all_leads": 3.5,
        "worst_primary_relative_to_source": 1.01,
        "gate": {"pass": True},
    }
    later = {
        "fine_tune_step": 960,
        "optimizer_step": 14400,
        "worst_mid_bottom_modewise_ratio_all_leads": 2.5,
        "worst_primary_relative_to_source": 1.04,
        "gate": {"pass": True},
    }
    decision = select_candidate((source, earlier, later))
    assert decision["candidate_training_gate_passed"] is True
    assert decision["selected_fine_tune_step"] == 960


def test_candidate_selection_retains_source_when_none_pass() -> None:
    decision = select_candidate(
        (
            {
                "fine_tune_step": 0,
                "optimizer_step": 13440,
                "gate": {"pass": False},
            },
            {
                "fine_tune_step": 240,
                "optimizer_step": 13680,
                "gate": {"pass": False},
            },
        )
    )
    assert decision["candidate_training_gate_passed"] is False
    assert decision["selected_fine_tune_step"] == 0
