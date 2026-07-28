"""Typed access and validation for the single-source reproduction configuration."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "bire_a0_reference.toml"


class ConfigError(ValueError):
    """Raised when a reproduction configuration violates a locked invariant."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    grid = config["grid"]
    data = config["data"]
    fno = config["fno"]
    experiments = config["experiments"]

    if len(grid["del_r_m"]) != grid["nr"]:
        raise ConfigError("grid.del_r_m length must equal grid.nr")
    if sum(grid["del_r_m"]) != 1800.0:
        raise ConfigError("locked 15-layer reconstruction must sum to 1800 m")
    if grid["nx"] != 248 or grid["ny"] != 248:
        raise ConfigError("paper-faithful grid must remain 248 x 248")
    if len(experiments) != 5 or [item["id"] for item in experiments] != [1, 2, 3, 4, 5]:
        raise ConfigError("Table 1 must contain experiments 1 through 5 in order")
    if [item["tau0_n_m2"] for item in experiments] != [0.075, 0.0875, 0.1, 0.1125, 0.125]:
        raise ConfigError("Table 1 wind stresses changed")
    if len(data["channels"]) != 11 or len(data["target_channels"]) != 10:
        raise ConfigError("the state/input must contain 10/11 channels")
    if fno["input_channels"] != 11 or fno["output_channels"] != 10:
        raise ConfigError("FNO channel counts do not match the data schema")
    if data["inference_start"] < data["validation_start"]:
        raise ConfigError("inference should be the final 1000-day overlap of validation")


def experiment(config: Mapping[str, Any], selector: int | str) -> Mapping[str, Any]:
    """Resolve an experiment by integer ID or slug."""
    text = str(selector)
    for item in config["experiments"]:
        if text in {str(item["id"]), item["slug"]}:
            return item
    valid = ", ".join(f"{x['id']}:{x['slug']}" for x in config["experiments"])
    raise ConfigError(f"unknown experiment {selector!r}; choose one of {valid}")


def canonical_json(config: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in config.items() if not key.startswith("_")}
    return json.dumps(filtered, sort_keys=True, separators=(",", ":"))


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()
