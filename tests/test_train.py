"""Tests for the canonical retained continuity training pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oceanfno.continuity import (
    ContinuityContext,
    continuity_relative_l2,
    depth_integrated_transport,
)
from oceanfno.model import BireTwoInNewChannelsArchitecture
from oceanfno.pressure_gradient import (
    DRF_M,
    PressureGradientContext,
    phihyd_from_normalized_state,
    pressure_gradient_relative_l2,
)
from oceanfno.runtime import torch
import oceanfno.train as train

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_continuity_v1.json"
PARENT = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_v1.json"
GRANDPARENT = ROOT / "config/model_c_2in_1out_new_channels_v1.json"
SBATCH = ROOT / "slurm/models/c/train_2in_1out_new_channels_pressure_gradient_continuity.sbatch"


def test_continuity_successor_has_no_architecture_change() -> None:
    child = json.loads(CONTRACT.read_text())
    parent = json.loads(PARENT.read_text())
    assert child["version"] == train.VERSION
    assert child["initialization"]["version"] == train.PARENT_VERSION
    assert child["architecture"] == parent["architecture"]
    assert BireTwoInNewChannelsArchitecture(**child["architecture"]).to_dict() == child["architecture"]
    assert child["initialization"]["strict_same_shape_load"] is True
    assert child["initialization"]["function_preserving"] is True
    assert child["initialization"]["optimizer_state_loaded"] is False


def test_architecture_is_unchanged_along_the_whole_fine_tune_chain() -> None:
    child = json.loads(CONTRACT.read_text())
    grandparent = json.loads(GRANDPARENT.read_text())
    assert child["architecture"] == grandparent["architecture"]


def test_only_declared_scientific_change_is_continuity_loss() -> None:
    child = json.loads(CONTRACT.read_text())
    parent = json.loads(PARENT.read_text())
    # The pressure-gradient term is retained verbatim at the parent's weight.
    assert child["loss"]["pressure_gradient_weight"] == train.DEFAULT_PRESSURE_WEIGHT
    assert child["loss"]["pressure_gradient_weight"] == parent["loss"]["pressure_gradient_weight"]
    assert child["loss"]["continuity_weight"] == train.DEFAULT_CONTINUITY_WEIGHT
    assert child["loss"]["continuity"]["new_model_output_channels"] == 0
    assert child["loss"]["continuity"]["reference"] == "truth_referenced_not_driven_to_zero"
    assert "continuity" not in parent["loss"]
    assert child["training"] == parent["training"]
    assert child["training"]["rollout_steps"] == 6
    assert child["training"]["maximum_steps"] == 3840
    assert child["training"]["checkpoint_steps"] == [960, 1920, 2880, 3840]
    assert child["training"]["load_optimizer_state"] is False
    assert child["training"]["from_scratch"] is False


def test_contract_loads_without_cluster_source_verification() -> None:
    contract, resolved, digest = train.load_contract(CONTRACT, verify_sources=False)
    assert contract["version"] == train.VERSION
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64


def test_contract_initializes_from_the_selected_pressure_gradient_checkpoint() -> None:
    child = json.loads(CONTRACT.read_text())
    checkpoint = child["sources"]["parent_checkpoint"]
    assert child["initialization"]["checkpoint"] == checkpoint["path"]
    assert child["initialization"]["checkpoint_sha256"] == checkpoint["sha256"]
    assert child["sources"]["initialization_checkpoint"] == checkpoint
    assert "model_c_2in_1out_new_channels_pressure_gradient_v1" in checkpoint["path"]


def test_slurm_uses_canonical_training_entrypoint() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.train" in text
    assert "oceanfno.train_pressure_gradient" not in text
    assert "model_c_2in_1out_new_channels_pressure_gradient_continuity_v1.json" in text


pytestmark_pressure = pytest.mark.skipif(torch is None, reason="PyTorch is optional")


def _context(n: int = 8) -> PressureGradientContext:
    mean = np.zeros((46, n, n), dtype=np.float32)
    scale = np.ones_like(mean)
    dx = np.full((n, n), 1.0e5, dtype=np.float32)
    wet = np.ones((n, n), dtype=bool)
    return PressureGradientContext(mean, scale, dx, wet)


def _continuity_context(n: int = 8) -> ContinuityContext:
    mean = np.zeros((46, n, n), dtype=np.float32)
    scale = np.ones_like(mean)
    dx = np.full((n, n), 1.0e5, dtype=np.float32)
    wet = np.ones((n, n), dtype=bool)
    return ContinuityContext(mean, scale, dx, wet)


@pytestmark_pressure
def test_phihyd_shape_and_identity_loss() -> None:
    context = _context()
    state = torch.zeros((2, 3, 46, 8, 8), dtype=torch.float32)
    phi = phihyd_from_normalized_state(state, context)
    assert phi.shape == (2, 3, 15, 8, 8)
    assert float(pressure_gradient_relative_l2(state, state, context)) == 0.0


@pytestmark_pressure
def test_uniform_eta_offset_is_dynamically_irrelevant() -> None:
    context = _context()
    truth = torch.zeros((1, 2, 46, 8, 8), dtype=torch.float32)
    prediction = truth.clone()
    prediction[:, :, 45] += 0.25
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert float(value) == pytest.approx(0.0, abs=1.0e-7)


@pytestmark_pressure
def test_pressure_slope_is_seen_and_backpropagates() -> None:
    context = _context()
    truth = torch.zeros((1, 2, 46, 8, 8), dtype=torch.float32)
    ramp = torch.linspace(0.0, 0.1, 8, dtype=torch.float32)[None, :].expand(8, 8)
    truth[:, :, 45] = ramp
    prediction = torch.zeros_like(truth, requires_grad=True)
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0
    value.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all().item())
    assert float(prediction.grad[:, :, 45].abs().sum()) > 0.0


@pytestmark_pressure
def test_temperature_gradient_contributes_to_pressure_force() -> None:
    context = _context()
    truth = torch.zeros((1, 1, 46, 8, 8), dtype=torch.float32)
    ramp = torch.linspace(0.0, 2.0, 8, dtype=torch.float32)[None, :].expand(8, 8)
    truth[:, :, 30] = ramp
    prediction = torch.zeros_like(truth)
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0


@pytestmark_pressure
def test_depth_integral_uses_the_mitgcm_layer_thicknesses() -> None:
    context = _continuity_context()
    state = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    state[:, 0:15] = 1.0
    transport_x, transport_y = depth_integrated_transport(state, context)
    assert transport_x.shape == (1, 8, 8)
    assert float(transport_x.mean()) == pytest.approx(float(sum(DRF_M)), rel=1.0e-6)
    assert float(transport_y.abs().max()) == 0.0


@pytestmark_pressure
def test_continuity_identity_loss_is_exactly_zero() -> None:
    context = _continuity_context()
    state = torch.zeros((2, 6, 46, 8, 8), dtype=torch.float32)
    present = torch.zeros((2, 46, 8, 8), dtype=torch.float32)
    assert float(continuity_relative_l2(state, state, present, context)) == 0.0


@pytestmark_pressure
def test_missing_surface_height_tendency_costs_one_of_six_calls() -> None:
    # Truth raises the surface uniformly on the first step only, so exactly one
    # of the six residuals is non-zero and a zero prediction misses all of it.
    context = _continuity_context()
    truth = torch.zeros((1, 6, 46, 8, 8), dtype=torch.float32)
    truth[:, :, 45] = 0.5
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = torch.zeros_like(truth)
    value = continuity_relative_l2(prediction, truth, present, context)
    assert float(value) == pytest.approx(1.0 / 6.0, rel=1.0e-5)


@pytestmark_pressure
def test_uniform_transport_offset_is_divergence_free() -> None:
    context = _continuity_context()
    truth = torch.zeros((1, 6, 46, 8, 8), dtype=torch.float32)
    truth[:, :, 45] = 0.5
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = truth.clone()
    prediction[:, :, 0:15] += 0.25
    value = continuity_relative_l2(prediction, truth, present, context)
    assert float(value) == pytest.approx(0.0, abs=1.0e-8)


@pytestmark_pressure
def test_continuity_backpropagates_to_velocity_and_surface_height() -> None:
    context = _continuity_context()
    truth = torch.zeros((1, 6, 46, 8, 8), dtype=torch.float32)
    truth[:, :, 45] = 0.5
    present = torch.zeros((1, 46, 8, 8), dtype=torch.float32)
    prediction = torch.zeros_like(truth, requires_grad=True)
    value = continuity_relative_l2(prediction, truth, present, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0
    value.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all().item())
    assert float(prediction.grad[:, :, 45].abs().sum()) > 0.0
    assert float(prediction.grad[:, :, 0:30].abs().sum()) > 0.0
