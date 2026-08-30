"""Contract adapter: the frozen S0 figure package for the B/C study arms.

Plan section 19 step 2 runs "the established final S0 figure package" on the
same 15 starts for every arm, and section 23.1 asks for this as an *adapter*
that reuses the numerical helpers of the frozen module rather than a second
copy of them.

The reason it cannot simply be ``figures.py`` with a different contract is
identity, in three places: ``figures.VERSION`` (the figure package's own
name), ``figures.TRAINING_VERSION`` (imported from ``train.py``, the parent's
training identity), and the ``GATE_NAME`` derived from the latter. All three
are module constants bound to the parent.

The reason it cannot be ``figures.py`` *modified* to accept a parameter is
provenance: the parent's own frozen figure contract pins
``src/oceanfno/figures.py`` in its ``source_hashes`` and re-verifies that hash
on every load, so editing that file would retire the A package's ability to
re-verify itself. Section 19 step 6 requires the existing A/ft90 reports be
preserved, so ``figures.py`` stays byte-identical and this module supplies
only what is identity-bound:

  * ``IDENTITIES`` -- the allow-list of (figure version -> training version)
    pairs. It lives in code, not in the contract, so a contract cannot
    authorize its own identity;
  * ``_training_provenance`` and ``_stepper`` -- the same checks as the frozen
    module's, with the expected training version supplied rather than
    hard-coded;
  * ``load_contract`` / ``finalize`` / ``run`` / ``preflight`` / ``publish``
    -- orchestration only.

Everything numerical -- ``evaluate_regime``, the climatology, the static
block, every plot, the summary and the acceptance gate -- is imported from
``figures``/``plots`` and executed unchanged, so an arm's figures are produced
by exactly the code that produced the parent's.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import plots
from . import figures
from .figures import (
    DAY2000_STD_RATIO_RANGE,
    FigureContractError,
    MAXIMUM_NORMALIZED_MAGNITUDE,
    MEMBER_COUNT,
    MINIMUM_STREAMFUNCTION_SV,
    PENDING,
    PENDING_PATHS,
    REGIME_INDEX,
    REGIMES,
    START_SEED,
    TAU0_N_M2,
    _EXPECTED_BASELINES,
    _EXPECTED_OUTPUTS,
    _EXPECTED_TRUTH,
    _REQUIRED_ARTIFACTS,
    _integer,
    _read,
    declared_inference_starts,
    evaluate_regime,
    long_rollout_gate,
)
from .dataset import (
    DATASET_VERSION,
    INFERENCE_RANGE,
    INFERENCE_START_RANGE,
    MAXIMUM_INFERENCE_ROLLOUT_DAYS,
    STATIC_FEATURES,
    TRAIN_RANGE,
    VALIDATION_RANGE,
)
from .model import ProductionArchitecture, ProductionStepper, build_model
from .runtime import _device, _file_sha256, _json_sha256, json_safe, torch
from .validation import train_only_climatology
from .train import (
    CHECKPOINT_STEPS,
    LOSS_CONTRACT_SHA256,
    ROLLOUT_STEPS,
    physical_static_block,
)

#: The study runner publishes plain artifact names (``report.json``,
#: ``normalization.npz``, ``selected.pt``) where the parent runner prefixes
#: each with its own model version. Both are checked the same way; only the
#: expected names differ, so they are named here rather than imported from
#: ``train.py``.
TRAINING_REPORT_NAME = "report.json"
SELECTED_NORMALIZATION_NAME = "normalization.npz"
SELECTED_CHECKPOINT_NAME = "selected.pt"

CONTRACT_STATUS = figures.CONTRACT_STATUS

SEEDS = (20260724, 20260911, 20260912)

#: figure package version -> the training identity it is allowed to bind to.
#: In code deliberately: a contract must match an entry here, so it cannot
#: declare itself into existence.
IDENTITIES: dict[str, str] = {
    f"{arm}_seed_{seed}_s0_figures_v1": arm
    for arm in (
        "model_c_adjoint_faithful_nominal_control_v1",
        "model_c_adjoint_faithful_response_v1",
    )
    for seed in SEEDS
}

_REQUIRED_SOURCE_HASHES = frozenset(
    figures._REQUIRED_SOURCE_HASHES | {"src/oceanfno/figures_response.py", "src/oceanfno/train_response.py"}
)


def gate_name(training_version: str, seed: int) -> str:
    return f"{training_version}_seed_{seed}_acceptance_gate.json"


def identity_of(contract: Mapping[str, Any]) -> tuple[str, str, int]:
    """(figure version, training version, seed) for a declared contract."""

    version = str(contract.get("version", ""))
    training_version = IDENTITIES.get(version)
    if training_version is None:
        raise FigureContractError(f"{version!r} is not a declared B/C figure identity")
    seed = _integer(_read(contract, ("selected_model", "seed")))
    if seed not in SEEDS:
        raise FigureContractError(f"seed {seed} is not one of the frozen paired seeds {SEEDS}")
    if version != f"{training_version}_seed_{seed}_s0_figures_v1":
        raise FigureContractError("the figure version and the declared seed disagree")
    return version, training_version, seed


def unfilled_fields(contract: Mapping[str, Any]) -> list[str]:
    return figures.unfilled_fields(contract)


def _training_provenance(contract: Mapping[str, Any], training_version: str, seed: int) -> None:
    """``figures._training_provenance`` with the arm's identity supplied.

    Every condition is the frozen module's, with two additions the study
    contract makes meaningful: the training report must name the same seed,
    and the training contract must be the arm's own.
    """

    selected = contract["selected_model"]
    artifacts = contract["artifacts"]
    report_path = Path(str(artifacts["selected_report"].get("path", ""))).resolve()
    if report_path.name != TRAINING_REPORT_NAME or not report_path.is_file():
        raise FigureContractError("the selected report is not a completed training report")
    if _file_sha256(report_path) != artifacts["selected_report"].get("sha256"):
        raise FigureContractError("the selected training report hash changed")
    report = json.loads(report_path.read_text())
    published = report.get("published_checkpoint", {})
    checkpoint = artifacts["selected_checkpoint"]
    normalization = artifacts["selected_normalization"]
    if (
        report.get("status") != "complete"
        or report.get("version") != training_version
        or _integer(report.get("seed")) != seed
        or report.get("architecture") != selected.get("architecture")
        or report.get("loss_contract_sha256") != LOSS_CONTRACT_SHA256
        or report.get("initialization", {}).get("from_scratch") is not True
        or _integer(published.get("optimizer_step")) != _integer(selected.get("optimizer_step"))
        or published.get("checkpoint") != checkpoint.get("path")
        or published.get("checkpoint_sha256") != checkpoint.get("sha256")
        or published.get("normalization") != normalization.get("path")
        or published.get("normalization_sha256") != normalization.get("sha256")
        or Path(str(normalization.get("path", ""))).name != SELECTED_NORMALIZATION_NAME
        or Path(str(checkpoint.get("path", ""))).name != SELECTED_CHECKPOINT_NAME
    ):
        raise FigureContractError("the selected model disagrees with its training report")
    training_path = Path(str(selected.get("training_contract", ""))).resolve()
    if not training_path.is_file():
        raise FigureContractError("the selected training contract is absent")
    training = json.loads(training_path.read_text())
    if (
        training.get("version") != training_version
        or training.get("architecture") != selected.get("architecture")
        or training.get("initialization", {}).get("from_scratch") is not True
    ):
        raise FigureContractError("the training provenance changed")


def _stepper(
    contract: Mapping[str, Any], device: Any, wet: np.ndarray, statics: np.ndarray, training_version: str
) -> ProductionStepper:
    """``figures._stepper`` with the arm's identity supplied."""

    if torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("figure evaluation requires PyTorch")
    declared = contract["selected_model"]
    architecture = ProductionArchitecture(**declared["architecture"])
    payload = torch.load(
        Path(contract["artifacts"]["selected_checkpoint"]["path"]), map_location=device, weights_only=False
    )
    if (
        payload.get("version") != training_version
        or payload.get("architecture") != declared["architecture"]
        or _integer(payload.get("optimizer_step")) != _integer(declared.get("optimizer_step"))
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("loss_contract_sha256") != declared["loss_contract_sha256"]
        or _integer(payload.get("rollout_steps")) != _integer(declared.get("rollout_steps"))
        or payload.get("from_scratch") is not True
    ):
        raise FigureContractError(
            "the selected checkpoint's identity, architecture, dataset or objective changed"
        )
    try:
        model = build_model(architecture).to(device)
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise FigureContractError("the selected state dictionary changed") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FigureContractError("the selected checkpoint did not load strictly")
    model.eval()
    with np.load(Path(contract["artifacts"]["selected_normalization"]["path"])) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    return ProductionStepper(model=model, device=device, wet=wet, mean=mean, scale=scale, statics=statics)


def load_contract(path: str | Path, *, verify_sources: bool = True) -> tuple[dict[str, Any], Path, str]:
    """The frozen module's audit, with the arm identity resolved from
    ``IDENTITIES`` instead of a hard-coded parent constant."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    pending = unfilled_fields(contract)
    if pending:
        raise FigureContractError(
            "the figure contract still carries post-training fields: "
            + ", ".join(pending)
            + " -- run `python -m oceanfno.figures_response finalize` first"
        )

    version, training_version, seed = identity_of(contract)
    protocol = contract.get("protocol", {})
    selected = contract.get("selected_model", {})
    output = contract.get("output", {})
    dataset = contract.get("dataset", {})
    expected_starts = tuple(int(value) for value in declared_inference_starts())
    protocol_ok = (
        _integer(protocol.get("member_count")) == MEMBER_COUNT
        and _integer(protocol.get("start_seed")) == START_SEED
        and tuple(protocol.get("start_draw_order", ())) == expected_starts
        and tuple(protocol.get("regimes", ())) == REGIMES
        and protocol.get("primary_regime") == "S0"
        and tuple(protocol.get("figure_names", ())) == tuple(plots.FIGURE_NAMES)
        and tuple(protocol.get("figure3_lead_days", ())) == tuple(plots.FIGURE_3_LEADS)
        and tuple(protocol.get("figure7_lead_days", ())) == tuple(plots.FIGURE_7_LEADS)
        and tuple(protocol.get("rmse_fields", ())) == tuple(plots.RMSE_FIELDS)
        and tuple(protocol.get("acc_fields", ())) == tuple(plots.ACC_FIELDS)
        and tuple(protocol.get("inference_set", ())) == INFERENCE_RANGE
        and tuple(protocol.get("start_window", ())) == INFERENCE_START_RANGE
        and _integer(protocol.get("maximum_lead_days")) == MAXIMUM_INFERENCE_ROLLOUT_DAYS
        and _integer(protocol.get("prediction_interval_days")) == 10
        and protocol.get("short_lead_days") == "0_to_200_inclusive_by_10"
        and protocol.get("long_lead_days") == "0_to_2000_inclusive_by_10"
        and protocol.get("comparator_model") is None
        and protocol.get("nesting")
        == "nested_validation_inference_protocol_no_independent_third_test_split"
        and tuple(protocol.get("static_channels", ())) == STATIC_FEATURES
    )
    models_ok = (
        selected.get("version") == training_version
        and _integer(selected.get("optimizer_step")) in CHECKPOINT_STEPS
        and _integer(selected.get("rollout_steps")) == ROLLOUT_STEPS
        and selected.get("loss_contract_sha256") == LOSS_CONTRACT_SHA256
        and selected.get("architecture") == ProductionArchitecture().to_dict()
        and selected.get("from_scratch") is True
    )
    output_ok = (
        str(output.get("project_root", "")).endswith(version)
        and str(output.get("scratch_root", "")).endswith(version)
        and output.get("overwrite") is False
        and output.get("one_folder_per_regime") is True
        and tuple(output.get("required", ())) == _EXPECTED_OUTPUTS
    )
    if (
        contract.get("contract_status") != CONTRACT_STATUS
        or "comparator_model" in contract
        or "figure6" in contract
        or dataset.get("version") != DATASET_VERSION
        or tuple(dataset.get("train", ())) != TRAIN_RANGE
        or tuple(dataset.get("validation", ())) != VALIDATION_RANGE
        or tuple(dataset.get("inference", ())) != INFERENCE_RANGE
        or dataset.get("tau0_n_m2") != TAU0_N_M2
        or contract.get("baselines") != _EXPECTED_BASELINES
        or contract.get("truth") != _EXPECTED_TRUTH
        or not set(_REQUIRED_ARTIFACTS).issubset(contract.get("artifacts", {}))
        or not protocol_ok
        or not models_ok
        or not output_ok
    ):
        raise FigureContractError("the B/C S0 figure contract changed")
    try:
        ProductionArchitecture(**selected["architecture"])
        _training_provenance(contract, training_version, seed)
    except FigureContractError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise FigureContractError("the selected figure provenance changed") from error
    if verify_sources:
        hashes = contract.get("source_hashes", {})
        if not _REQUIRED_SOURCE_HASHES.issubset(hashes):
            raise FigureContractError("the figure source declaration is incomplete")
        for label, specification in contract.get("artifacts", {}).items():
            plots._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in hashes.items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise FigureContractError(f"a pinned figure source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def finalize(contract_path: str | Path) -> dict[str, Any]:
    """Fill the declared-pending fields from the arm's own training report.

    Idempotent and non-destructive in exactly the frozen module's sense: a
    field already equal to what the report says is left alone, and one that
    disagrees is refused rather than overwritten.
    """

    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    _, training_version, seed = identity_of(contract)
    report_path = Path(str(_read(contract, ("artifacts", "selected_report", "path")) or "")).resolve()
    if report_path.name != TRAINING_REPORT_NAME or not report_path.is_file():
        raise FigureContractError("the selected training report is absent")
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or report.get("version") != training_version:
        raise FigureContractError("the training report is not this arm's completed report")
    published = report.get("published_checkpoint", {})
    values = {
        ("selected_model", "optimizer_step"): int(published["optimizer_step"]),
        ("artifacts", "selected_checkpoint", "sha256"): published["checkpoint_sha256"],
        ("artifacts", "selected_normalization", "sha256"): published["normalization_sha256"],
        ("artifacts", "selected_report", "sha256"): _file_sha256(report_path),
    }
    changed = []
    for path in PENDING_PATHS:
        node: Any = contract
        for key in path[:-1]:
            node = node[key]
        current = node[path[-1]]
        wanted = values[path]
        if current == wanted:
            continue
        if current != PENDING:
            raise FigureContractError(
                f"{'.'.join(path)} is already {current!r}, not {PENDING!r}; refusing to overwrite"
            )
        node[path[-1]] = wanted
        changed.append(".".join(path))
    if changed:
        resolved.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "status": "finalized",
        "version": contract["version"],
        "training_version": training_version,
        "seed": seed,
        "filled": changed,
        "contract": str(resolved),
        "contract_sha256": _file_sha256(resolved),
    }


def acceptance_gate(contract: Mapping[str, Any], regime: str = "S0") -> dict[str, Any]:
    """``figures.acceptance_gate`` with this package's identity in the header.

    The numerics are the frozen module's ``long_rollout_gate``, unchanged.
    """

    version, training_version, seed = identity_of(contract)
    output = Path(contract["output"]["project_root"]).resolve() / regime
    with np.load(output / plots.ARRAYS_NAME) as stored:
        arrays = {name: stored[name] for name in stored.files}
    summary = json.loads((output / plots.SUMMARY_NAME).read_text())
    training = json.loads(Path(contract["artifacts"]["selected_report"]["path"]).read_text())
    validation = training["acceptance_gate"]
    long_half = long_rollout_gate(arrays, summary)
    gate = {
        "version": version,
        "training_version": training_version,
        "seed": seed,
        "regime": regime,
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "selected_checkpoint_sha256": training["published_checkpoint"]["checkpoint_sha256"],
        "validation_half": validation,
        "long_rollout_half": long_half,
        "measurable_conditions_pass": bool(
            validation["validation_conditions_pass"] and long_half["long_rollout_conditions_pass"]
        ),
        "remaining_by_inspection": (
            "western boundary sharp, gyre structure identifiable, no basin-wide "
            "smoothing -- figures 3 and 7"
        ),
        "decision_note": (
            "these measurable diagnostics and visual inspection describe this arm's held "
            "performance; this package promotes nothing"
        ),
    }
    gate = json_safe(gate)
    gate["content_sha256"] = _json_sha256(gate)
    return gate


def _readme(regime: str, report: Mapping[str, Any]) -> str:
    return (
        f"# {report['version']}\n\n"
        f"The frozen S0 figure package for `{report['training_version']}` seed "
        f"{report['seed']} on the 15-member {regime} inference protocol, produced by the "
        "unchanged numerics of `src/oceanfno/figures.py` through the study adapter "
        "`src/oceanfno/figures_response.py`.\n\n"
        f"Selected checkpoint: optimizer step {report['selected_optimizer_step']}.\n"
        f"Starts: {report['start_draw_order']}.\n\n"
        "The 2,000-day acceptance gate is written beside this folder as "
        f"`{report['acceptance_gate_artifact']}`.\n"
    )


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Publish the six figures for one B/C arm and seed."""

    if torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("figure evaluation requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    version, training_version, seed = identity_of(contract)
    device = _device(device_name)
    dataset = Path(contract["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    starts = declared_inference_starts()
    climatology_state, climatology_derived, climatology_days = train_only_climatology(state, wet)
    if climatology_days != TRAIN_RANGE[1] - TRAIN_RANGE[0]:
        raise FigureContractError("the train-only climatology did not cover 0--5999")

    with np.load(Path(contract["artifacts"]["selected_normalization"]["path"])) as stored:
        point_mean = np.asarray(stored["pointwise_mean"], dtype=np.float32)
        point_scale = np.asarray(stored["pointwise_scale"], dtype=np.float32)
    statics, static_provenance = physical_static_block(contract["artifacts"], group, point_mean, point_scale)
    stepper = _stepper(contract, device, wet, statics, training_version)

    published: dict[str, Any] = {}
    plots._style()
    for regime in REGIMES:
        regime_index = REGIME_INDEX[regime]
        output = Path(contract["output"]["project_root"]).resolve() / regime
        scratch = Path(contract["output"]["scratch_root"]).resolve() / regime
        for path in (output, scratch):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        arrays = evaluate_regime(
            stepper, state, static, regime_index, starts, climatology_state, climatology_derived, wet
        )
        arrays["longitude_deg"] = longitude
        arrays["latitude_deg"] = latitude
        arrays["wet_mask"] = wet.astype(np.uint8)
        summary = plots._summary(arrays)
        output_tmp = output.with_name(output.name + ".tmp")
        scratch_tmp = scratch.with_name(scratch.name + ".tmp")
        output_tmp.parent.mkdir(parents=True, exist_ok=True)
        scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
        output_tmp.mkdir()
        scratch_tmp.mkdir()
        try:
            scratch_arrays = scratch_tmp / plots.ARRAYS_NAME
            np.savez_compressed(scratch_arrays, **arrays)
            plots._plot_streamfunction_grid(output_tmp, arrays, longitude, latitude, wet)
            plots._plot_rmse(output_tmp, arrays, long=False)
            plots._plot_single_member(output_tmp, arrays)
            plots._plot_acc(output_tmp, arrays)
            plots._plot_day60_day2000(output_tmp, arrays, longitude, latitude, wet)
            plots._plot_rmse(output_tmp, arrays, long=True)
            plots._write_csv(output_tmp / plots.CSV_NAME, arrays)
            (output_tmp / plots.SUMMARY_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            report = {
                "version": version,
                "training_version": training_version,
                "seed": seed,
                "status": "complete",
                "regime": regime,
                "tau0_n_m2": contract["dataset"]["tau0_n_m2"][regime],
                "role": "primary",
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset": str(dataset),
                "dataset_version": DATASET_VERSION,
                "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
                "comparator_model": None,
                "start_draw_order": starts.astype(int).tolist(),
                "inference_start_range": list(INFERENCE_START_RANGE),
                "summary": summary,
                "arrays": str(scratch / plots.ARRAYS_NAME),
                "arrays_sha256": _file_sha256(scratch_arrays),
                "figures": list(plots.FIGURE_NAMES),
                "static_channels": list(STATIC_FEATURES),
                "static_channel_provenance": static_provenance,
                "acceptance_gate_artifact": gate_name(training_version, seed),
                "numerics_source": "src/oceanfno/figures.py (unchanged); adapter src/oceanfno/figures_response.py",
                "elapsed_seconds": time.monotonic() - started,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            report = json_safe(report)
            report["report_content_sha256"] = _json_sha256(report)
            (scratch_tmp / plots.REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            shutil.copy2(scratch_tmp / plots.REPORT_NAME, output_tmp / plots.REPORT_NAME)
            shutil.copy2(scratch_arrays, output_tmp / plots.ARRAYS_NAME)
            (output_tmp / plots.README_NAME).write_text(_readme(regime, report))
            manifest = {
                "version": version,
                "regime": regime,
                "contract_sha256": contract_sha,
                "report_content_sha256": report["report_content_sha256"],
                "artifacts": {
                    path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
                    for path in sorted(output_tmp.iterdir())
                    if path.is_file()
                },
            }
            manifest["manifest_content_sha256"] = _json_sha256(manifest)
            (output_tmp / plots.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            scratch_tmp.replace(scratch)
            output_tmp.replace(output)
        except Exception:
            shutil.rmtree(scratch_tmp, ignore_errors=True)
            shutil.rmtree(output_tmp, ignore_errors=True)
            raise
        published[regime] = report
    return published


def preflight(contract_path: str | Path) -> dict[str, Any]:
    contract, resolved, digest = load_contract(contract_path)
    version, training_version, seed = identity_of(contract)
    starts = declared_inference_starts()
    return {
        "status": "pass",
        "version": version,
        "training_version": training_version,
        "seed": seed,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset_version": DATASET_VERSION,
        "regimes": list(REGIMES),
        "primary_regime": "S0",
        "member_count": MEMBER_COUNT,
        "start_draw_order": starts.astype(int).tolist(),
        "inference_start_range": list(INFERENCE_START_RANGE),
        "maximum_lead_days": max(figures.LEAD_DAYS),
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "comparator_model": None,
        "static_channels": list(STATIC_FEATURES),
        "selected_rollout_steps": ROLLOUT_STEPS,
        "selected_loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "acceptance_gate_artifact": str(
            Path(contract["output"]["project_root"]).resolve() / gate_name(training_version, seed)
        ),
        "continuation_required": False,
    }


def publish(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    published = dict(run(contract_path, device_name=device_name))
    contract, _, _ = load_contract(contract_path, verify_sources=False)
    _, training_version, seed = identity_of(contract)
    root = Path(contract["output"]["project_root"]).resolve()
    gate = acceptance_gate(contract, "S0")
    (root / gate_name(training_version, seed)).write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    published["acceptance_gate"] = gate
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("finalize", "preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        result: Any = finalize(args.contract)
    elif args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = publish(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
