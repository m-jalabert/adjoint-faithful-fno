"""Compare complete frozen forward-evaluation packages on one common protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .af_forward_complete import BIRE_FIELDS, FIELD_LABELS


COLORS = {"A0": "#2F75B5", "A": "#218A8D", "B": "#A86600", "C": "#7B4AB5"}
LEADS_TO_REPORT = (10, 90, 180, 360)


def _load(label: str, path: Path) -> tuple[str, dict[str, Any], Mapping[str, np.ndarray]]:
    metrics_path = path / "forward_complete_metrics.json"
    arrays_path = path / "forward_complete_arrays.npz"
    return label, json.loads(metrics_path.read_text()), np.load(arrays_path)


def _check_common_protocol(
    packages: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    reference = packages[0][1]["protocol"]
    starts = reference["ensemble_starts"]
    for label, metrics, arrays in packages:
        if metrics["protocol"].get("evaluation_contract_version") != reference.get(
            "evaluation_contract_version"
        ):
            raise ValueError(f"{label} does not use the common evaluation contract")
        if metrics["protocol"].get("bire_diagnostics") != reference.get("bire_diagnostics"):
            raise ValueError(f"{label} does not use the common Bire diagnostic registry")
        if metrics["protocol"]["ensemble_starts"] != starts:
            raise ValueError(f"{label} does not use the common ensemble starts")
        if metrics["protocol"]["rollout_days"] != reference["rollout_days"]:
            raise ValueError(f"{label} does not use the common rollout duration")
        if not np.array_equal(arrays["lead_days"], packages[0][2]["lead_days"]):
            raise ValueError(f"{label} does not use the common lead array")


def _summary(
    packages: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"models": {}, "reported_leads_days": list(LEADS_TO_REPORT)}
    for label, metrics, _ in packages:
        curves = metrics["rollout"]["curves"]
        model: dict[str, Any] = {
            "forward_gate": metrics["forward_gate"],
            "bire_diagnostics": {},
        }
        for name in BIRE_FIELDS:
            entries = {}
            for lead in LEADS_TO_REPORT:
                index = lead // 10 - 1
                model_rmse = curves[name]["model"]["rmse"][index]["mean"]
                persistence = curves[name]["persistence"]["rmse"][index]["mean"]
                climatology = curves[name]["climatology"]["rmse"][index]["mean"]
                entries[str(lead)] = {
                    "rmse": model_rmse,
                    "rmse_over_persistence": model_rmse / persistence,
                    "rmse_over_climatology": model_rmse / climatology,
                    "acc": curves[name]["model"]["acc"][index]["mean"],
                }
            model["bire_diagnostics"][name] = entries
        result["models"][label] = model
    return result


def _plot_skill(
    output: Path,
    packages: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leads = np.asarray(packages[0][2]["lead_days"])
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, constrained_layout=True)
    for axis, name in zip(axes.flat, BIRE_FIELDS):
        for label, metrics, _ in packages:
            curve = metrics["rollout"]["curves"][name]
            model = np.asarray([entry["mean"] for entry in curve["model"]["rmse"]])
            persistence = np.asarray([entry["mean"] for entry in curve["persistence"]["rmse"]])
            climatology = np.asarray([entry["mean"] for entry in curve["climatology"]["rmse"]])
            axis.plot(
                leads,
                model / np.minimum(persistence, climatology),
                color=COLORS.get(label),
                label=label,
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(FIELD_LABELS[name])
        axis.set_ylabel("RMSE / stronger baseline")
        axis.grid(alpha=0.3)
    for axis in axes.flat[len(BIRE_FIELDS) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Lead (model days)")
    axes[0, 0].legend()
    figure.suptitle("Frozen common-protocol forward comparison")
    figure.savefig(output / "forward_model_comparison_rmse.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, constrained_layout=True)
    reference = packages[0][1]["rollout"]["curves"]
    for axis, name in zip(axes.flat, BIRE_FIELDS):
        for label, metrics, _ in packages:
            values = [entry["mean"] for entry in metrics["rollout"]["curves"][name]["model"]["acc"]]
            axis.plot(leads, values, color=COLORS.get(label), label=label)
        for method, style in (("persistence", "--"), ("climatology", ":")):
            values = [entry["mean"] for entry in reference[name][method]["acc"]]
            axis.plot(leads, values, color="#333333", linestyle=style, label=method)
        axis.axhline(0.0, color="0.6", linewidth=0.7)
        axis.set_ylim(-1.0, 1.02)
        axis.set_title(FIELD_LABELS[name])
        axis.set_ylabel("Anomaly correlation")
        axis.grid(alpha=0.3)
    for axis in axes.flat[len(BIRE_FIELDS) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Lead (model days)")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Frozen common-protocol ACC comparison")
    figure.savefig(output / "forward_model_comparison_acc.png", dpi=180)
    plt.close(figure)


def _plot_gate(
    output: Path,
    packages: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    criteria = []
    for _, metrics, _ in packages:
        for name in metrics["forward_gate"]["criteria"]:
            if name not in criteria:
                criteria.append(name)
    values = np.full((len(criteria), len(packages)), np.nan)
    for column, (_, metrics, _) in enumerate(packages):
        source = metrics["forward_gate"]["criteria"]
        for row, name in enumerate(criteria):
            if name in source:
                values[row, column] = 1.0 if source[name] else 0.0
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    masked = np.ma.masked_invalid(values)
    axis.imshow(masked, cmap=ListedColormap(("#B13A3A", "#2E7D32")), vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(packages)), [label for label, _, _ in packages])
    axis.set_yticks(
        np.arange(len(criteria)),
        [name.replace("_", " ") for name in criteria],
        fontsize=8,
    )
    axis.set_title("Predeclared forward-gate scorecard (switch response is provisional)")
    figure.savefig(output / "forward_model_comparison_gate.png", dpi=180)
    plt.close(figure)


def compare(
    specifications: Sequence[tuple[str, str | Path]], output_dir: str | Path
) -> dict[str, Any]:
    if len(specifications) < 2:
        raise ValueError("comparison requires at least two complete model packages")
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite comparison output: {output}")
    packages = [_load(label, Path(path).resolve()) for label, path in specifications]
    _check_common_protocol(packages)
    summary = _summary(packages)
    temporary.mkdir(parents=True)
    (temporary / "forward_model_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot_skill(temporary, packages)
    _plot_gate(temporary, packages)
    temporary.rename(output)
    for _, _, arrays in packages:
        close = getattr(arrays, "close", None)
        if close is not None:
            close()
    return summary


def _specification(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("model packages must be LABEL=PATH")
    return label, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare complete frozen forward packages")
    parser.add_argument("--package", action="append", type=_specification, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.package, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
