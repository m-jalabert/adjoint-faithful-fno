"""Emulator-side S_forced for the raw-SSH (point_only) objective, all 3 targets.

Non-confirmatory, same status as fno_adjoint_posthoc_v1.py, which this
mirrors: does not modify fno_adjoint_ft90.py, builds a second SharedContract
via dataclasses.replace for each target, and calls the same forced_chain the
frozen western comparison uses. The only difference is the objective
(point_only instead of ssh_anomaly/ssh_anomaly_kernel) and that western is
now included -- point_only was never computed for western either, frozen or
otherwise, so this is new for all three targets.

Weight fields are read from each target's just-extracted
mitgcm_s0_adjoint_raw_ssh_<target>_v1.npz (w_point_only), which
extract_mitgcm_adjoint_raw_ssh_v1.py already verified by sha256 against what
MITgcm staged as costWeight.bin -- so both sides read the identical field by
construction, same as gate F6.

    python scripts/fno_adjoint_raw_ssh_v1.py --identity B_20260911
    python scripts/fno_adjoint_raw_ssh_v1.py --identity C_20260911
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fno_adjoint_ft90 as runner  # noqa: E402
from fno_adjoint_model import IDENTITIES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEAD_DAYS = (10, 30, 90)
TARGETS = ("western", "interior", "eastern")


def _raw_ssh_weight(target: str) -> tuple[np.ndarray, tuple[int, int]]:
    npz_path = (
        PROJECT_ROOT
        / f"outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/{target}/mitgcm_s0_adjoint_raw_ssh_{target}_v1.npz"
    )
    with np.load(npz_path) as store:
        return np.asarray(store["w_point_only"], dtype=np.float64), tuple(int(v) for v in store["target_ij"])


def run_identity(identity_key: str, force: bool) -> None:
    identity = IDENTITIES[identity_key]
    project_root = PROJECT_ROOT

    plan = json.loads((project_root / "config" / f"{runner.PLAN_CONTRACT}.json").read_text())
    provenance = runner.load_model_provenance(project_root, identity)
    dataset_path = Path(provenance["contract"]["sources"]["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    base_contract = runner.load_shared_contract(project_root, group, plan)

    print(f"[{identity_key}] loading {provenance['checkpoint'].name}")
    model = runner.load_frozen_model(provenance["checkpoint"], double=True, identity=identity)
    precision = runner.verify_double_precision_spectrum(model)
    if not precision["passed"]:
        raise runner.FnoAdjointError(f"[{identity_key}] spectral buffer is not complex128: {precision}")

    with np.load(provenance["normalization"]) as stored:
        normalizers = {
            "mean": np.asarray(stored["pointwise_mean"], dtype=np.float64),
            "scale": np.asarray(stored["pointwise_scale"], dtype=np.float64),
        }
    sources = provenance["contract"]["sources"]
    statics, _ = runner.static_block(
        group,
        zonal_spacing_path=runner._verify(sources["mitgcm_zonal_spacing"], "zonal spacing"),
        sst_relax_path=runner._verify(sources["mitgcm_sst_relaxation"], "SST relaxation target"),
        data_path=runner._verify(sources["mitgcm_declaration"], "MITgcm declaration"),
        pointwise_mean=normalizers["mean"].astype(np.float32),
        pointwise_scale=normalizers["scale"].astype(np.float32),
    )
    operator = runner.build_operator(model, normalizers, statics, base_contract.wet)

    max_calls = runner.calls_for_lead(max(LEAD_DAYS))
    days = [runner.SOURCE_DAY + runner.HORIZON_DAYS * k for k in range(max_calls + 1)]
    truth_numpy = runner._truth_states(group, days)
    truth = [torch.from_numpy(truth_numpy[day]).to(operator.dtype) for day in days]

    for target in TARGETS:
        weight_field, target_ij = _raw_ssh_weight(target)
        if np.any(weight_field[~base_contract.wet] != 0.0):
            raise runner.FnoAdjointError(f"{target}/point_only weight is nonzero on land")
        weight_tensor = torch.from_numpy(np.ascontiguousarray(weight_field, dtype=np.float64)).to(operator.dtype)
        print(f"[{identity_key}/{target}] raw-SSH target_ij={target_ij}")

        maps: dict[int, np.ndarray] = {}
        for lead in LEAD_DAYS:
            calls = runner.calls_for_lead(lead)
            forced = runner.forced_chain(operator, truth, {"point_only": weight_tensor}, calls)
            maps[lead] = forced["maps"]["point_only"]

        out_dir = project_root / f"outputs/af_fno/adjoint/fno_adjoint_posthoc_v1/{identity_key}/{target}"
        npz_path = out_dir / "raw_ssh.npz"
        if npz_path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite {npz_path}; pass --force")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            S_forced_point_only=np.stack([maps[lead] for lead in LEAD_DAYS]),
            lead_days=np.asarray(LEAD_DAYS, dtype=np.int64),
            target_ij=np.asarray(target_ij, dtype=np.int64),
            wet_mask=base_contract.wet,
        )
        report = {
            "version": "fno_adjoint_raw_ssh_v1",
            "status": "non_confirmatory_posthoc_diagnostic",
            "identity": identity_key,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "target_name": target,
            "target_ij_j_i": list(target_ij),
            "lead_days": list(LEAD_DAYS),
            "chain": "forced",
            "objective": "point_only",
            "double_precision_spectrum": precision,
            "npz_path": str(npz_path),
        }
        (out_dir / "raw_ssh_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {npz_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, choices=sorted(IDENTITIES))
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    run_identity(arguments.identity, arguments.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
