"""Emulator-side S_forced at the post-hoc interior/eastern targets.

**Not part of the frozen adjoint-faithful contract** -- see
scripts/make_cost_weight_posthoc_v1.py for why. This is a post-hoc "why did
it fail" diagnostic for B_20260911/C_20260911 against the two new MITgcm/TAF
targets in outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/, not a rerun
of, or an addition to, Gate A1.

Deliberately does NOT modify fno_adjoint_ft90.py, which is the trusted,
already-executed machinery behind the frozen western-target result. Instead
it imports that module's already-modular pieces (model loading, the
FrozenOperator, forced_chain) and calls them with a second SharedContract,
built by dataclasses.replace() off the validated western one with only
``target`` and the two target-dependent weight fields swapped -- everything
target-independent (wet mask, rA, grid, mean_only weight, longitude/latitude)
is reused unchanged from the frozen contract.

The weight arrays used here are read from the packaged post-hoc MITgcm npz
(outputs/.../mitgcm_s0_adjoint_posthoc_v1/<target>/*.npz's w_ssh_anomaly /
w_ssh_anomaly_kernel), which extract_mitgcm_adjoint_posthoc_v1.py already
verified by sha256 to be the exact bytes MITgcm staged as costWeight.bin --
so both sides are reading the identical field by construction, the same
guarantee gate F6 gives the frozen western target.

Only forced_chain is run (the truth-forced local-Jacobian map the plan calls
the primary endpoint) -- not free_chain, and not gates F1-F7, which validate
the model/operator generically and already passed at the western target.

    python scripts/fno_adjoint_posthoc_v1.py --identity B_20260911 --target interior
    python scripts/fno_adjoint_posthoc_v1.py --identity C_20260911 --target eastern
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

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
TARGETS = ("interior", "eastern")


def _posthoc_weight_arrays(target: str) -> dict[str, np.ndarray]:
    npz_path = (
        PROJECT_ROOT
        / f"outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/{target}/mitgcm_s0_adjoint_posthoc_{target}_v1.npz"
    )
    with np.load(npz_path) as store:
        return {
            "ssh_anomaly": np.asarray(store["w_ssh_anomaly"], dtype=np.float64),
            "ssh_anomaly_kernel": np.asarray(store["w_ssh_anomaly_kernel"], dtype=np.float64),
            "target_ij": tuple(int(v) for v in store["target_ij"]),
        }


def build_target_contract(base: "runner.SharedContract", target: str, dtype: torch.dtype) -> tuple["runner.SharedContract", dict[str, torch.Tensor]]:
    posthoc = _posthoc_weight_arrays(target)
    weights = dict(base.weights)
    digests = dict(base.weight_digests)
    for name in ("ssh_anomaly", "ssh_anomaly_kernel"):
        field = posthoc[name]
        if np.any(field[~base.wet] != 0.0):
            raise runner.FnoAdjointError(f"{target}/{name} weight is nonzero on land")
        weights[name] = field
        digests[name] = "posthoc:" + target  # not a frozen-contract digest; see module docstring

    contract = dataclasses.replace(base, target=posthoc["target_ij"], weights=weights, weight_digests=digests)
    torch_weights = {
        name: torch.from_numpy(np.ascontiguousarray(weights[name], dtype=np.float64)).to(dtype)
        for name in ("ssh_anomaly", "ssh_anomaly_kernel")
    }
    return contract, torch_weights


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
        contract, weights = build_target_contract(base_contract, target, operator.dtype)
        print(f"[{identity_key}/{target}] target_ij={contract.target}")

        maps: dict[str, dict[int, np.ndarray]] = {"ssh_anomaly": {}, "ssh_anomaly_kernel": {}}
        for lead in LEAD_DAYS:
            calls = runner.calls_for_lead(lead)
            forced = runner.forced_chain(operator, truth, weights, calls)
            for name in maps:
                maps[name][lead] = forced["maps"][name]

        out_dir = project_root / f"outputs/af_fno/adjoint/fno_adjoint_posthoc_v1/{identity_key}/{target}"
        if out_dir.exists() and not force and any(out_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite {out_dir}; pass --force")
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "s_forced.npz"
        np.savez(
            npz_path,
            S_forced_ssh_anomaly=np.stack([maps["ssh_anomaly"][lead] for lead in LEAD_DAYS]),
            S_forced_ssh_anomaly_kernel=np.stack([maps["ssh_anomaly_kernel"][lead] for lead in LEAD_DAYS]),
            lead_days=np.asarray(LEAD_DAYS, dtype=np.int64),
            target_ij=np.asarray(contract.target, dtype=np.int64),
            wet_mask=base_contract.wet,
        )
        report = {
            "version": "fno_adjoint_posthoc_v1",
            "status": "non_confirmatory_posthoc_diagnostic",
            "identity": identity_key,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "target_name": target,
            "target_ij_j_i": list(contract.target),
            "lead_days": list(LEAD_DAYS),
            "chain": "forced",
            "double_precision_spectrum": precision,
            "npz_path": str(npz_path),
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
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
