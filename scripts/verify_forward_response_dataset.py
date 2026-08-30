"""Execution step 10 (continued), Gate D3 verifier.

Implements config/forward_response_schema_v1.json's
``x-verifier-only-cross-array-gates`` -- the checks the schema itself says a
plain JSON-Schema pass cannot make ("json_schema_success_without_this_verifier_is_insufficient").
No ``jsonschema`` package is installed in this environment (checked directly)
and this project already avoids adding dependencies purely for schema/format
validation (section 13's own reasoning for not adding a Parquet engine), so
this module includes a small, dependency-free validator for the JSON-Schema
2020-12 subset the frozen ``$defs`` actually use (``type``, ``enum``,
``const``, ``required``, ``properties``/``additionalProperties``, numeric
bounds, array ``minItems``/``maxItems``/``prefixItems``/``uniqueItems``,
``allOf``/``oneOf``/``if``-``then``, and same-document ``$ref``), then applies
it to every row of every extracted table before running the array-level and
cross-table checks that no per-row schema can express.

Never reads the sealed blind manifest or the blind evaluator store; verifies
their absence from the development artifacts instead (the one blind-isolation
check this module *can* make without opening blind data).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from build_forward_response_inventory import load_json_strict, sha256_file  # noqa: E402
from extract_forward_response_dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHEMA_PATH,
    ROLES,
    _array_sha256,
    _chunk_shape,
)

DEFAULT_DATASET_CONTRACT_PATH = PROJECT_ROOT / "config" / "forward_response_dataset_v3.json"
_COMPRESSOR_REPR = "Blosc(cname='zstd', clevel=3, shuffle=BITSHUFFLE, blocksize=0)"

#: Reviewed, documented exceptions to Gate D3's Q_lin/Q_SNR criterion for
#: TRAIN-role directions only (plan document, 2026-08-27 amendment). Gate
#: D3's own text -- "every failed validation case becomes development data
#: and the successor must create new response-validation and blind
#: inventories" -- names validation and blind specifically; a failed TRAIN
#: case carries no such provenance constraint (the same reasoning already
#: used for SSH_ALPHA_OVERRIDES vs. the validation/blind centre repairs).
#: Reviewed with the researcher 2026-08-27: root-caused to two distinct real
#: mechanisms rather than a defect --
#: (a) genuine northern-region (the ten wet rows against the basin's solid
#:     northern wall) Theta nonlinearity at alpha=0.005, concentrated in
#:     shallow levels (1-9) and absent below level 10; region-wide failure
#:     rate 22.2% (8/36) versus 0-8.3% elsewhere, and weakly anti-correlated
#:     with local-sigma percentile (-0.11), so this is not primarily a
#:     pooled-normalizer artifact;
#: (b) a handful of eastern/WBC/interior Theta directions the pilot's own
#:     6-location sample (one location per region at most) never had a
#:     chance to catch, the same "pilot sample didn't happen to probe this
#:     location" shape already established for the SSH peak-cap and SSH-SNR
#:     gaps elsewhere in this study.
#: The 7 validation-role directions found alongside these were NOT treated
#: this way -- they were given fresh, individually MITgcm-verified centres
#: instead (scripts/repair_gate_d3_validation_centres.py), per Gate D3's
#: explicit validation/blind provenance rule.
GATE_D3_TRAIN_EXCEPTIONS: dict[tuple[str, int, str, int], dict[str, Any]] = {
    ("S0", 360, "Theta", 9): {
        "leads": (10,),
        "region": "eastern",
        "reason": "Q_lin fails at lead 10 only; pilot sample never probed this eastern location.",
    },
    ("S0", 1440, "Theta", 9): {
        "leads": (10,),
        "region": "northern",
        "reason": "Northern-region Theta nonlinearity at lead 10.",
    },
    ("S0", 1800, "Theta", 11): {
        "leads": (10, 20, 30),
        "region": "northern",
        "reason": "Northern-region Theta nonlinearity, leads 10-30.",
    },
    ("S0", 3960, "Theta", 9): {
        "leads": (10,),
        "region": "northern",
        "reason": "Northern-region Theta nonlinearity at lead 10.",
    },
    ("S1", 360, "Theta", 8): {
        "leads": (10,),
        "region": "eastern",
        "reason": "Q_lin fails at lead 10 only; pilot sample never probed this eastern location.",
    },
    ("S1", 5040, "Theta", 9): {
        "leads": (40, 50, 60),
        "region": "eastern",
        "reason": "Late-lead Q_SNR decay toward the floor (40-60d only), same shape as the "
        "already-accepted GATE_D2_EXCEPTIONS S1/day720/V case.",
    },
    ("S2", 0, "Theta", 9): {
        "leads": (10, 20, 30, 40, 50, 60),
        "region": "northern",
        "reason": "Northern-region Theta nonlinearity, all six leads (worst case, Q_lin up to 0.41).",
    },
    ("S2", 1440, "Theta", 11): {
        "leads": (10,),
        "region": "northern",
        "reason": "Northern-region Theta nonlinearity at lead 10.",
    },
    ("S2", 1440, "Theta", 9): {
        "leads": (10,),
        "region": "eastern",
        "reason": "Q_lin fails at lead 10 only; pilot sample never probed this eastern location.",
    },
    ("S2", 1800, "Theta", 9): {
        "leads": (10, 20, 30),
        "region": "WBC",
        "reason": "Q_lin fails leads 10-30; pilot's WBC Theta sample was at a different anchor.",
    },
    ("S2", 2520, "Theta", 10): {
        "leads": (10, 20, 30, 40, 50, 60),
        "region": "interior",
        "reason": "Q_lin fails all six leads; pilot's interior Theta sample was at a different anchor.",
    },
    ("S2", 3240, "Theta", 10): {
        "leads": (10,),
        "region": "northern",
        "reason": "Northern-region Theta nonlinearity at lead 10.",
    },
}


def _parse_regime_day_family_slot(direction_id: str) -> tuple[str, int, str, int]:
    parts = direction_id.split(":")
    return parts[2], int(parts[3][1:]), parts[4], int(parts[7][1:])


class VerificationError(RuntimeError):
    """Raised when the curated dataset fails a Gate D3 check."""


# ---------------------------------------------------------------------------
# A small, dependency-free JSON-Schema (2020-12 subset) validator.


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise VerificationError(f"unsupported $ref outside this document: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _validate(instance: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if "$ref" in schema:
        _validate(instance, _resolve_ref(schema["$ref"], root), root, path, errors)
        return
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
        return
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']}")
        return
    type_spec = schema.get("type")
    if type_spec is not None:
        types = [type_spec] if isinstance(type_spec, str) else type_spec
        if not any(_matches_type(instance, t) for t in types):
            errors.append(f"{path}: {instance!r} does not match type {types}")
            return
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        for key, op in (("minimum", "<"), ("exclusiveMinimum", "<="), ("maximum", ">"), ("exclusiveMaximum", ">=")):
            if key in schema:
                bound = schema[key]
                violated = {
                    "minimum": instance < bound,
                    "exclusiveMinimum": instance <= bound,
                    "maximum": instance > bound,
                    "exclusiveMaximum": instance >= bound,
                }[key]
                if violated:
                    errors.append(f"{path}: {instance!r} violates {key}={bound}")
    if isinstance(instance, str) and "minLength" in schema and len(instance) < schema["minLength"]:
        errors.append(f"{path}: string shorter than minLength={schema['minLength']}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: {len(instance)} items < minItems={schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: {len(instance)} items > maxItems={schema['maxItems']}")
        if schema.get("uniqueItems") and len(set(canonical_repr(v) for v in instance)) != len(instance):
            errors.append(f"{path}: items are not unique")
        prefix_items = schema.get("prefixItems")
        if prefix_items:
            for index, item_schema in enumerate(prefix_items):
                if index < len(instance):
                    _validate(instance[index], item_schema, root, f"{path}[{index}]", errors)
        items_schema = schema.get("items")
        if items_schema is not None and items_schema is not False:
            start = len(prefix_items) if prefix_items else 0
            for index in range(start, len(instance)):
                _validate(instance[index], items_schema, root, f"{path}[{index}]", errors)
        elif items_schema is False and prefix_items and len(instance) > len(prefix_items):
            errors.append(f"{path}: extra items beyond prefixItems, and items=false")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key}: required field missing")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(f"{path}: fewer than minProperties={schema['minProperties']} properties")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], root, f"{path}.{key}", errors)
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(value, schema["additionalProperties"], root, f"{path}.{key}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: unexpected property, additionalProperties=false")
    for sub in schema.get("allOf", []):
        _validate(instance, sub, root, path, errors)
    if "oneOf" in schema:
        matched = 0
        for sub in schema["oneOf"]:
            probe: list[str] = []
            _validate(instance, sub, root, path, probe)
            if not probe:
                matched += 1
        if matched != 1:
            errors.append(f"{path}: matched {matched} of oneOf branches (need exactly 1)")
    if "if" in schema:
        probe: list[str] = []
        _validate(instance, schema["if"], root, path, probe)
        branch = schema.get("then") if not probe else schema.get("else")
        if branch is not None:
            _validate(instance, branch, root, path, errors)


def canonical_repr(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _matches_type(instance: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(instance, dict)
    if type_name == "array":
        return isinstance(instance, list)
    if type_name == "string":
        return isinstance(instance, str)
    if type_name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if type_name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if type_name == "boolean":
        return isinstance(instance, bool)
    if type_name == "null":
        return instance is None
    raise VerificationError(f"unsupported schema type {type_name!r}")


def validate_rows(rows: Sequence[Mapping[str, Any]], def_name: str, schema: Mapping[str, Any], label: str) -> list[str]:
    node = schema["$defs"][def_name]
    errors: list[str] = []
    for index, row in enumerate(rows):
        _validate(row, node, schema, f"{label}[{index}]", errors)
    return errors


# ---------------------------------------------------------------------------
# Table loading.


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Checks.


def check_row_schemas(role: str, anchor_table, direction_table, solver_control_table, schema, findings):
    findings.extend(f"[{role}] anchor: {e}" for e in validate_rows(anchor_table, "anchor", schema, f"{role}.anchor_table"))
    findings.extend(f"[{role}] direction: {e}" for e in validate_rows(direction_table, "direction", schema, f"{role}.direction_table"))
    if solver_control_table is not None:
        findings.extend(
            f"[{role}] solver_control: {e}"
            for e in validate_rows(solver_control_table, "solver_control", schema, f"{role}.solver_control_table")
        )


def check_bijective_row_mappings(role: str, group, anchor_table, direction_table, findings):
    n_anchor = len(anchor_table)
    short_dirs = [d for d in direction_table if d["array_group"] == "short"]
    long_dirs = [d for d in direction_table if d["array_group"] == "long"]
    for label, dirs, array_prefix in (("short", short_dirs, "short"), ("long", long_dirs, "long")):
        expected_rows = {d["array_row"] for d in dirs}
        if expected_rows != set(range(len(dirs))):
            findings.append(f"[{role}] {label} array_row is not a bijection onto 0..{len(dirs)-1}")
        stored_anchor_rows = np.asarray(group[array_prefix]["anchor_row"])
        stored_direction_rows = np.asarray(group[array_prefix]["direction_row"])
        if len(stored_anchor_rows) != len(dirs):
            findings.append(f"[{role}] {label}/anchor_row length {len(stored_anchor_rows)} != {len(dirs)} directions")
        if np.any(stored_anchor_rows < 0) or np.any(stored_anchor_rows >= max(n_anchor, 1)):
            findings.append(f"[{role}] {label}/anchor_row has an out-of-bounds anchor index")
        for array_row, direction_row in enumerate(stored_direction_rows):
            if int(direction_row) < 0 or int(direction_row) >= len(direction_table):
                findings.append(f"[{role}] {label}/direction_row[{array_row}] out of bounds")
                continue
            entry = direction_table[int(direction_row)]
            if entry["array_group"] != label or entry["array_row"] != array_row:
                findings.append(
                    f"[{role}] {label}/direction_row[{array_row}] -> direction_table[{direction_row}] "
                    f"does not point back consistently"
                )
    ids = [row["anchor_id"] for row in anchor_table]
    if len(set(ids)) != len(ids):
        findings.append(f"[{role}] anchor_id is not unique within the role")
    dir_ids = [row["direction_id"] for row in direction_table]
    if len(set(dir_ids)) != len(dir_ids):
        findings.append(f"[{role}] direction_id is not unique within the role")


def check_array_contract(role: str, group, schema, role_shape, findings):
    contract = schema["$defs"]["array_contract"]["properties"]
    dims = {"A": role_shape["A"], "A_short": role_shape["A_short"], "A_long": role_shape["A_long"],
            "Q_short": role_shape["Q_short"], "Q_long": role_shape["Q_long"], "L": role_shape["L"],
            "12": 12, "2": 2, "9": 9, "6": 6}
    for array_name, spec in contract.items():
        is_control = array_name.startswith("controls/")
        if is_control and role != "pilot":
            continue  # controls/* only exist for the pilot role (not in array_contract's required list)
        const = spec["const"]  # e.g. "(A,46,62,62) <f4"
        shape_str, dtype_str = const.rsplit(" ", 1)
        dim_names = shape_str.strip("()").split(",")
        expected_shape = tuple(dims[name.strip()] if name.strip() in dims else int(name.strip()) for name in dim_names if name.strip())
        if "/" in array_name:
            sub_group_name, leaf = array_name.split("/", 1)
            container = group[sub_group_name] if sub_group_name in group else None
        else:
            sub_group_name, leaf, container = None, array_name, group
        if container is None or leaf not in container:
            findings.append(f"[{role}] missing array {array_name}")
            continue
        dataset = container[leaf]
        if tuple(dataset.shape) != expected_shape:
            findings.append(f"[{role}] {array_name} shape {dataset.shape} != expected {expected_shape}")
        if dataset.dtype.str != dtype_str:
            findings.append(f"[{role}] {array_name} dtype {dataset.dtype.str} != expected {dtype_str}")
        expected_chunks = _chunk_shape(array_name, expected_shape)
        if tuple(dataset.chunks) != expected_chunks:
            findings.append(f"[{role}] {array_name} chunks {dataset.chunks} != expected {expected_chunks}")
        if array_name != "lead_days" and repr(dataset.compressor) != _COMPRESSOR_REPR:
            findings.append(f"[{role}] {array_name} compressor {dataset.compressor!r} != expected")


def check_hashes(role: str, group, direction_table, findings, *, sample_limit: int | None = None):
    short_response = group["short"]["response_p64"]
    long_response = group["long"]["response_p64"]
    rows = direction_table if sample_limit is None else direction_table[:sample_limit]
    for entry in rows:
        array = short_response if entry["array_group"] == "short" else long_response
        response = np.asarray(array[entry["array_row"]])
        leads = response.shape[1]
        expected_count = 2 * leads
        if len(entry["response_hashes"]) != expected_count:
            findings.append(
                f"[{role}] {entry['direction_id']}: response_hashes has {len(entry['response_hashes'])}, "
                f"expected {expected_count}"
            )
            continue
        index = 0
        mismatches = 0
        for sign_index in (0, 1):
            for lead_index in range(leads):
                actual = _array_sha256(response[sign_index, lead_index], "<f8")
                if actual != entry["response_hashes"][index]:
                    mismatches += 1
                index += 1
        if mismatches:
            findings.append(f"[{role}] {entry['direction_id']}: {mismatches} response_hashes mismatch")


def check_sparse_edits(role: str, direction_table, findings):
    for entry in direction_table:
        family = entry["input_family"]
        for label, edits in (("sparse_edits_minus", entry["sparse_edits_minus"]), ("sparse_edits_plus", entry["sparse_edits_plus"])):
            for edit in edits:
                if family == "SSH" and edit["record_index0"] != 105:
                    findings.append(f"[{role}] {entry['direction_id']} {label}: SSH edit not at record 105")
                if family != "SSH":
                    base = {"U": 0, "V": 15, "Theta": 30}[family]
                    levels = entry["one_based_levels"] or [1]
                    valid_records = {base + level - 1 for level in levels}
                    if edit["record_index0"] not in valid_records:
                        findings.append(
                            f"[{role}] {entry['direction_id']} {label}: record {edit['record_index0']} "
                            f"not implied by levels {levels}"
                        )
        minus_support = {(e["j_index0"], e["i_index0"], e["record_index0"]) for e in entry["sparse_edits_minus"]}
        plus_support = {(e["j_index0"], e["i_index0"], e["record_index0"]) for e in entry["sparse_edits_plus"]}
        if minus_support != plus_support:
            findings.append(f"[{role}] {entry['direction_id']}: minus/plus sparse edit support differ")
        if len(entry["sparse_edits_minus"]) != len(entry["sparse_edits_plus"]):
            continue
        for minus, plus in zip(entry["sparse_edits_minus"], entry["sparse_edits_plus"]):
            if (minus["j_index0"], minus["i_index0"], minus["record_index0"]) != (
                plus["j_index0"], plus["i_index0"], plus["record_index0"]
            ):
                findings.append(f"[{role}] {entry['direction_id']}: minus/plus edit order differs")
                break
            if abs(minus["value"] + plus["value"]) > 1e-9 * max(1.0, abs(plus["value"])):
                findings.append(f"[{role}] {entry['direction_id']}: minus/plus edit values are not exact sign reversals")


def check_vertical_weights(role: str, direction_table, findings):
    import math

    for entry in direction_table:
        weights = entry["vertical_weights"]
        levels = entry["one_based_levels"]
        if len(weights) != len(levels):
            findings.append(f"[{role}] {entry['direction_id']}: weight count != level count")
            continue
        if weights:
            norm = math.sqrt(sum(w * w for w in weights))
            if abs(norm - 1.0) > 1e-9:
                findings.append(f"[{role}] {entry['direction_id']}: vertical weights not unit-L2 ({norm})")


def check_p32_realization(role: str, direction_table, findings):
    for entry in direction_table:
        alpha = entry["alpha"]
        for value in entry["p32_realized_standardized_rms"]:
            if abs(value - alpha) > 0.01 * alpha:
                findings.append(f"[{role}] {entry['direction_id']}: realized P32 RMS {value} not within 1% of alpha {alpha}")
        if entry["p32_antisymmetry_relative_error"] > 0.01:
            findings.append(f"[{role}] {entry['direction_id']}: antisymmetry error exceeds 1%")
        if entry["centred_support_count"] == 0:
            findings.append(f"[{role}] {entry['direction_id']}: zero centre weight")


def check_qlin_qsnr(role: str, group, direction_table, combined_floor_gb_by_lead, wet, sigma, findings):
    """Gate D3's own criterion, never checked before this step: train/validation
    directions were only ever checked for magnitude/antisymmetry/SSH cap
    (section 10 governs *amplitude selection*, run once per family on the
    pilot); nothing previously verified Q_lin/Q_SNR for the 888 production
    directions themselves. Reuses the pilot's frozen combined GB floor
    (Gate D2) rather than the section-14.2 per-group floor this same step
    also freezes -- Gate D3's text names "the final combined numerical
    floor", i.e. the one floor already frozen before this step, not the new
    per-output-group one this step derives from it."""

    if role not in ("train", "validation"):
        return
    GROUP_SLICES = {"U": slice(0, 15), "V": slice(15, 30), "Theta": slice(30, 45), "SSH": slice(45, 46)}

    def gb_norm(z):
        return float(np.sqrt(np.mean([np.mean(z[sl][:, wet] ** 2) for sl in GROUP_SLICES.values()])))

    short_response = group["short"]["response_p64"]
    long_response = group["long"]["response_p64"]
    failures = 0
    for entry in direction_table:
        array = short_response if entry["array_group"] == "short" else long_response
        response = np.asarray(array[entry["array_row"]])
        leads = (10,) if entry["array_group"] == "short" else (10, 20, 30, 40, 50, 60)
        for lead_index, lead in enumerate(leads):
            # Stored arrays are the raw, unoriented difference P64[perturbed]-P64[nominal]
            # in physical units (section 13: "the minus record is normally negative").
            # Section 10.2's R^s is that raw delta, standardized by sigma exactly like
            # every other norm/floor in this study, then divided by its own sign, i.e.
            # R^- = -delta_minus/sigma, R^+ = +delta_plus/sigma -- both the sigma
            # standardization and the sign orientation are required before this is
            # comparable to combined_floor_gb_by_lead, which is itself sigma-normalized
            # (analyze_amplitude_pilot_controls.py divides by sigma throughout).
            r_minus = -response[0, lead_index] / sigma
            r_plus = response[1, lead_index] / sigma
            q_lin = gb_norm(r_plus - r_minus) / max(1e-300, 0.5 * (gb_norm(r_plus) + gb_norm(r_minus)))
            floor = combined_floor_gb_by_lead[str(lead)]
            q_snr = 0.5 * (gb_norm(r_plus) + gb_norm(r_minus)) / floor
            if q_lin > 0.05 or q_snr < 20:
                exception = None
                if role == "train":
                    exception = GATE_D3_TRAIN_EXCEPTIONS.get(_parse_regime_day_family_slot(entry["direction_id"]))
                    if exception is not None and lead not in exception["leads"]:
                        exception = None
                if exception is not None:
                    continue
                failures += 1
                findings.append(
                    f"[{role}] {entry['direction_id']} lead={lead}: Q_lin={q_lin:.4f} Q_snr={q_snr:.2f} "
                    "fails Gate D3's train/validation criterion"
                )
    return failures


def check_blind_isolation(store_attrs: Mapping[str, Any], output_root: Path, findings):
    roles = store_attrs.get("roles", [])
    if "blind_test" in roles:
        findings.append("development store declares a blind_test role")
    for path in output_root.glob("blind*"):
        findings.append(f"development output root contains a blind-looking artifact: {path}")


# ---------------------------------------------------------------------------
# Orchestration.


def verify(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    *,
    hash_sample_limit: int | None = None,
) -> list[str]:
    schema = load_json_strict(schema_path)
    store = zarr.open_consolidated(str(dataset_path), mode="r")
    findings: list[str] = []
    check_blind_isolation(dict(store.attrs), output_root, findings)

    final_selection = load_json_strict(output_root / "amplitude_pilot_final_selection_v1.json")
    combined_floor = final_selection["combined_floor_gb_by_lead"]
    contract, _roots, grid, sigma = _load_grid_and_normalizer()

    role_shapes = {}
    for role in ROLES:
        anchor_table = _load_jsonl(output_root / f"{role}_anchor_table.jsonl")
        direction_table = _load_jsonl(output_root / f"{role}_direction_table.jsonl")
        solver_control_path = output_root / f"{role}_solver_control_table.jsonl"
        solver_control_table = _load_jsonl(solver_control_path) if solver_control_path.is_file() else None
        group = store[role]

        check_row_schemas(role, anchor_table, direction_table, solver_control_table, schema, findings)
        check_bijective_row_mappings(role, group, anchor_table, direction_table, findings)
        role_shape = {
            "A": len(anchor_table),
            "A_short": int(np.asarray(group["anchors"]["nominal_short_anchor_row"]).shape[0]),
            "A_long": int(np.asarray(group["anchors"]["nominal_long_anchor_row"]).shape[0]),
            "Q_short": int(np.asarray(group["short"]["anchor_row"]).shape[0]),
            "Q_long": int(np.asarray(group["long"]["anchor_row"]).shape[0]),
            "L": int(np.asarray(group["lead_days"]).shape[0]),
        }
        role_shapes[role] = role_shape
        check_array_contract(role, group, schema, role_shape, findings)
        check_hashes(role, group, direction_table, findings, sample_limit=hash_sample_limit)
        check_sparse_edits(role, direction_table, findings)
        check_vertical_weights(role, direction_table, findings)
        check_p32_realization(role, direction_table, findings)
        check_qlin_qsnr(role, group, direction_table, combined_floor, grid.wet, sigma, findings)

        if role == "pilot":
            if solver_control_table is None:
                findings.append("pilot role is missing solver_control_table.jsonl")
            else:
                n_short = sum(1 for d in direction_table if d["array_group"] == "short")
                n_long = sum(1 for d in direction_table if d["array_group"] == "long")
                if n_short != 36 or n_long != 35:
                    findings.append(f"pilot direction_table has {n_short} short / {n_long} long, expected 36/35")
                if len(solver_control_table) != 60:
                    findings.append(f"pilot solver_control_table has {len(solver_control_table)} rows, expected 60")

    for nan_check_role in ROLES:
        group = store[nan_check_role]
        for prefix in ("short", "long"):
            arr = np.asarray(group[prefix]["response_p64"])
            if arr.size and (np.isnan(arr).any() or np.isinf(arr).any()):
                findings.append(f"[{nan_check_role}] {prefix}/response_p64 contains NaN/Inf")

    return findings


def _load_grid_and_normalizer():
    import build_amplitude_pilot as pilot

    contract, roots, grid = pilot._load_sources(pilot.DEFAULT_DATASET_CONTRACT)
    pilot_contract = load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)
    return contract, roots, grid, sigma


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hash-sample-limit", type=int, default=None)
    args = parser.parse_args(argv)
    findings = verify(args.dataset_path, args.output_root, hash_sample_limit=args.hash_sample_limit)
    if findings:
        print(f"Gate D3: FAIL -- {len(findings)} findings")
        for finding in findings[:200]:
            print(" -", finding)
        return 1
    print("Gate D3: PASS -- no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
