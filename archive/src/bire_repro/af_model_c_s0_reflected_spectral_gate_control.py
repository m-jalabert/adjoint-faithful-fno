"""Zero-retraining reflected spectral-gate control for selected Model C.

The completed binomial-filter control reduced the long-horizon high-k tail
but also damaged useful short-range scales.  This control keeps reflected
radial modes k<=8 exactly unchanged, transitions over 8<k<10, and attenuates
only modes k>=10 after each direct-state call.  Strength is selected only on
the same fixed training-only S0 trajectories; fixed inference remains sealed
unless a nonzero strength passes every prospective gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import af_model_c_s0_highk_damping_control as base

try:
    import torch
except (ImportError, OSError):  # pragma: no cover
    torch = None  # type: ignore[assignment]


VERSION = "model_c_s0_reflected_spectral_gate_control_v1"
CONTRACT_STATUS = "frozen_zero_retraining_after_job304753"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LOW_MODE_EDGE = 8.0
HIGH_MODE_EDGE = 10.0
FIGURES = (
    "model_c_s0_reflected_spectral_training_selection.png",
    "model_c_s0_reflected_spectral_inference_log_rmse.png",
    "model_c_s0_reflected_spectral_inference_normalized_envelope.png",
    "model_c_s0_reflected_spectral_inference_day2000_spectra.png",
)
ARRAYS = "model_c_s0_reflected_spectral_gate_arrays.npz"
REPORT = "model_c_s0_reflected_spectral_gate_report.json"
SUMMARY = "model_c_s0_reflected_spectral_gate_summary.json"
CSV = "model_c_s0_reflected_spectral_gate_curves.csv"
README = "README.md"
MANIFEST = "manifest.json"
FILTER_NAME = (
    "reflected_even_extension_radial_spectral_gate_k8_to_k10_"
    "on_normalized_anomaly"
)


def reflected_spectral_gate(
    value: Any,
    wet: np.ndarray,
    alpha_by_member: Any,
) -> Any:
    """Attenuate only reflected radial modes above the frozen transition."""

    if torch is None:
        raise RuntimeError("reflected spectral gate requires PyTorch")
    rows, columns = np.where(wet)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    cropped = value[..., y0:y1, x0:x1]

    reflected_y = torch.cat(
        (cropped, torch.flip(cropped[..., 1:-1, :], dims=(-2,))),
        dim=-2,
    )
    reflected = torch.cat(
        (reflected_y, torch.flip(reflected_y[..., 1:-1], dims=(-1,))),
        dim=-1,
    )
    height, width = reflected.shape[-2:]
    ky = torch.fft.fftfreq(height, device=value.device, dtype=value.dtype) * height
    kx = torch.fft.rfftfreq(width, device=value.device, dtype=value.dtype) * width
    radial = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    transition = torch.clamp(
        (radial - LOW_MODE_EDGE) / (HIGH_MODE_EDGE - LOW_MODE_EDGE),
        min=0.0,
        max=1.0,
    )
    high_gate = 0.5 - 0.5 * torch.cos(torch.pi * transition)

    coefficients = torch.fft.rfft2(reflected, dim=(-2, -1))
    high_component = torch.fft.irfft2(
        coefficients * high_gate[None, None],
        s=(height, width),
        dim=(-2, -1),
    )[..., : cropped.shape[-2], : cropped.shape[-1]]
    alpha = alpha_by_member.to(device=value.device, dtype=value.dtype)
    filtered = cropped - alpha[:, None, None, None] * high_component

    result = value.clone()
    result[..., y0:y1, x0:x1] = filtered
    wet_tensor = torch.from_numpy(wet.astype(np.float32))[None, None].to(
        device=value.device,
        dtype=value.dtype,
    )
    return result * wet_tensor


def _verify_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or base.file_sha256(path) != record["sha256"]:
        raise base.HighKDampingError(f"immutable artifact changed: {label}")
    return path


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Validate the frozen scale-selective contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or tuple(protocol.get("alphas", ())) != ALPHAS
        or tuple(protocol.get("training_times", ())) != base.TRAINING_TIMES
        or tuple(protocol.get("training_leads", ())) != (0, 1000)
        or tuple(protocol.get("training_spectrum_leads", ()))
        != base.TRAINING_SPECTRUM_LEADS
        or tuple(protocol.get("short_leads", ())) != base.SHORT_LEADS
        or tuple(protocol.get("inference_leads", ())) != (0, 2000)
        or tuple(protocol.get("inference_spectrum_leads", ()))
        != base.INFERENCE_SPECTRUM_LEADS
        or tuple(protocol.get("fields", ())) != base.FIELDS
        or tuple(protocol.get("statistical_fields", ())) != base.STAT_FIELDS
        or tuple(protocol.get("spectral_transition", ()))
        != (LOW_MODE_EDGE, HIGH_MODE_EDGE)
        or protocol.get("filter") != FILTER_NAME
        or protocol.get("selection")
        != "smallest_nonzero_alpha_passing_all_gates"
        or protocol.get("training_gate") != base.TRAINING_GATE
        or protocol.get("inference_characterization")
        != base.INFERENCE_CHARACTERIZATION
        or bool(protocol.get("retraining", True))
    ):
        raise base.HighKDampingError("reflected spectral-gate contract changed")
    for label, record in contract["artifacts"].items():
        _verify_file(record, label)
    root = resolved.parents[1]
    for relative, expected in contract["source_hashes"].items():
        source = root / relative
        if not source.is_file() or base.file_sha256(source) != expected:
            raise base.HighKDampingError(f"source changed: {relative}")
    for key in ("scratch", "project"):
        output = Path(contract["output"][key]).resolve()
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
    return contract, resolved, base.file_sha256(resolved)


def configure_base() -> None:
    """Bind the shared transactional runner to this frozen intervention."""

    base.VERSION = VERSION
    base.CONTRACT_STATUS = CONTRACT_STATUS
    base.ALPHAS = ALPHAS
    base.FIGURES = FIGURES
    base.ARRAYS = ARRAYS
    base.REPORT = REPORT
    base.SUMMARY = SUMMARY
    base.CSV = CSV
    base.README = README
    base.MANIFEST = MANIFEST
    base.reflected_binomial_damping = reflected_spectral_gate
    base.load_contract = load_contract


def preflight(contract_path: str | Path) -> dict[str, Any]:
    configure_base()
    result = base.preflight(contract_path)
    result["filter"] = FILTER_NAME
    result["low_modes_exactly_unchanged"] = f"k<={LOW_MODE_EDGE:g}"
    return result


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    configure_base()
    return base.run(contract_path, device_name=device_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--contract", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--contract", type=Path, required=True)
    execute.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = (
        preflight(args.contract)
        if args.command == "preflight"
        else run(args.contract, device_name=args.device)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
