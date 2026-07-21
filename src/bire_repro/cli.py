"""One command-line entry point for simulation, reduction, training, and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, experiment, load_config


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _experiment_id(config: dict[str, Any], selector: str) -> int:
    return int(experiment(config, selector)["id"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="af-fno",
        description="AF-FNO and adapted Bire A0 data/model utilities",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="locked TOML configuration")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="convert, normalize, validate, and seal data")
    data_commands = data.add_subparsers(dest="action", required=True)
    initialize = data_commands.add_parser("init")
    initialize.add_argument("--store", type=Path)
    convert = data_commands.add_parser("convert")
    convert.add_argument("--experiment", required=True)
    convert.add_argument("--raw-dir", type=Path)
    convert.add_argument("--store", type=Path)
    convert.add_argument("--allow-partial", action="store_true")
    stats = data_commands.add_parser("stats")
    stats.add_argument("--store", type=Path)
    stats.add_argument("--output", type=Path)
    validate = data_commands.add_parser("validate")
    validate.add_argument("--store", type=Path)
    validate.add_argument("--allow-incomplete", action="store_true")
    archive = data_commands.add_parser("archive")
    archive.add_argument("--store", type=Path)
    archive.add_argument("--manifest", type=Path)
    cleanup = data_commands.add_parser("cleanup")
    cleanup.add_argument("--manifest", type=Path, required=True)
    cleanup.add_argument(
        "--execute",
        action="store_true",
        help="unlink only sealed raw diagnostic files; preserve provenance and pickups",
    )

    fno = commands.add_parser("fno", help="train and roll out paper-faithful FNOs")
    fno_commands = fno.add_subparsers(dest="action", required=True)
    train = fno_commands.add_parser("train")
    train.add_argument("--lag-days", required=True, choices=("5", "10", "30", "all"))
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--device", choices=("cpu", "cuda"))
    rollout = fno_commands.add_parser("rollout")
    rollout.add_argument("--lag-days", required=True, type=int, choices=(5, 10, 30))
    rollout.add_argument("--experiment", default="3")
    rollout.add_argument("--checkpoint", type=Path)
    rollout.add_argument("--stage", choices=("pretrain", "finetune"), default="finetune")
    rollout.add_argument("--initial-index", action="append", type=int)
    rollout.add_argument("--horizon-days", type=int)
    rollout.add_argument("--resolution", default="full", choices=("full", "low", "coarse", "2deg"))
    rollout.add_argument("--output", type=Path)
    rollout.add_argument("--device", choices=("cpu", "cuda"))

    plots = commands.add_parser("plots", help="generate paper Figures 2 through 11")
    selection = plots.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--figure", action="append", type=int, choices=range(2, 12))
    plots.add_argument("--store", type=Path)
    plots.add_argument("--rollout-root", type=Path)
    plots.add_argument("--output-dir", type=Path)

    report = commands.add_parser("report", help="write HTML and JSON audit reports")
    report.add_argument("--output-dir", type=Path)
    return parser


def _run_data(config: dict[str, Any], args: argparse.Namespace) -> Any:
    from . import data

    if args.action == "init":
        return {"store": str(data.initialize_store(config, args.store))}
    if args.action == "convert":
        path = data.convert_experiment(
            config,
            args.experiment,
            raw_dir=args.raw_dir,
            store_path=args.store,
            allow_partial=args.allow_partial,
        )
        return {"store": str(path), "experiment": args.experiment}
    if args.action == "stats":
        return {"stats": str(data.compute_stats(config, args.store, args.output))}
    if args.action == "validate":
        result = data.validate_store(config, args.store)
        if not result["valid"] and not args.allow_incomplete:
            _json(result)
            raise SystemExit(1)
        return result
    if args.action == "archive":
        return {"manifest": str(data.archive_manifest(config, args.store, args.manifest))}
    if args.action == "cleanup":
        targets = data.cleanup_raw(config, args.manifest, execute=args.execute)
        return {
            "executed": bool(args.execute),
            "targets": [str(path) for path in targets],
            "recoverable": False if args.execute else None,
        }
    raise AssertionError(args.action)


def _run_fno(config: dict[str, Any], args: argparse.Namespace) -> Any:
    if args.action == "train":
        from .training import train_all_lags, train_from_config

        if args.lag_days == "all":
            return train_all_lags(config, output_root=args.output_dir, device=args.device)
        return train_from_config(
            config, lag_days=int(args.lag_days), output_dir=args.output_dir, device=args.device
        )
    if args.action == "rollout":
        from .rollout import rollout_from_config

        return rollout_from_config(
            config,
            lag_days=args.lag_days,
            experiment_id=_experiment_id(config, args.experiment),
            checkpoint_path=args.checkpoint,
            stage=args.stage,
            initial_indices=args.initial_index,
            horizon_days=args.horizon_days,
            resolution=args.resolution,
            output_path=args.output,
            device=args.device,
        )
    raise AssertionError(args.action)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "data":
            result = _run_data(config, args)
        elif args.command == "fno":
            result = _run_fno(config, args)
        elif args.command == "plots":
            from .plots import generate

            numbers = range(2, 12) if args.all else args.figure
            result = {
                "figures": [
                    str(path)
                    for path in generate(
                        config,
                        numbers,
                        store_path=args.store,
                        rollout_root=args.rollout_root,
                        output_dir=args.output_dir,
                    )
                ]
            }
        elif args.command == "report":
            from .report import generate_report

            html_path, json_path = generate_report(config, args.output_dir)
            result = {"html": str(html_path), "json": str(json_path)}
        else:  # pragma: no cover
            raise AssertionError(args.command)
        _json(result)
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        parser.exit(2, f"repro: error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
