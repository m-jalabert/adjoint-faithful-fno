"""Generate the six B/C S0 figure contracts (execution step 15, plan section 19).

Each is the parent's own frozen figure contract with only the identity
changed: version, selected model/seed, the three selected artifacts, the two
output roots, and the source-hash block. Protocol, dataset, baselines and
truth are copied byte-for-byte from the parent contract, so every arm is
evaluated on the identical 15 starts, leads, fields and baselines -- which is
what makes the arms comparable at all.

Post-training fields (optimizer step and the three artifact hashes) are left
at PENDING; ``oceanfno.figures_response finalize`` fills them from each run's
own training report, exactly as the parent package did.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oceanfno.figures_response import IDENTITIES, PENDING, _REQUIRED_SOURCE_HASHES  # noqa: E402

PARENT = PROJECT_ROOT / "config" / "model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1.json"
SCRATCH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C")
PROJECT_OUT = PROJECT_ROOT / "outputs" / "af_fno" / "C"


def build(figure_version: str, training_version: str, seed: int) -> Path:
    parent = json.loads(PARENT.read_text())
    contract = json.loads(json.dumps(parent))  # deep copy

    contract["version"] = figure_version
    contract["purpose"] = (
        f"the frozen S0 evaluation suite for {training_version} seed {seed}, on the identical "
        "15-start protocol, leads, fields and baselines as the parent package, so the arms are "
        "directly comparable"
    )
    selected = contract["selected_model"]
    selected["version"] = training_version
    selected["seed"] = seed
    selected["training_contract"] = str(PROJECT_ROOT / "config" / f"{training_version}.json")
    selected["optimizer_step"] = PENDING

    run_scratch = SCRATCH / training_version / f"seed_{seed}"
    run_project = PROJECT_OUT / training_version / f"seed_{seed}"
    contract["artifacts"]["selected_checkpoint"] = {"path": str(run_scratch / "selected.pt"), "sha256": PENDING}
    contract["artifacts"]["selected_normalization"] = {"path": str(run_scratch / "normalization.npz"), "sha256": PENDING}
    contract["artifacts"]["selected_report"] = {"path": str(run_project / "report.json"), "sha256": PENDING}

    contract["output"]["project_root"] = str(PROJECT_OUT / figure_version)
    contract["output"]["scratch_root"] = str(SCRATCH / figure_version)

    contract["source_hashes"] = {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in sorted(_REQUIRED_SOURCE_HASHES)
    }

    out = PROJECT_ROOT / "config" / f"{figure_version}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite an existing figure contract: {out}")
    out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return out


def main() -> int:
    for figure_version, training_version in sorted(IDENTITIES.items()):
        seed = int(figure_version.split("_seed_")[1].split("_")[0])
        print("wrote", build(figure_version, training_version, seed).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
