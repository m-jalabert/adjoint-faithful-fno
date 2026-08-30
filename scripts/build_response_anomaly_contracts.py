"""Generate the six B/C S0 anomaly contracts (execution step 15, plan section 19 step 4).

Each is the parent's own frozen anomaly contract with only the identity
changed: version, the four sealed figure-package artifacts, the two output
roots, and the source-hash block. Protocol, reference and dataset are copied
byte-for-byte, so every arm's anomaly is taken about the *same* MITgcm
training-mean field and reported with the same diagnostics -- which is what
makes the arms comparable.

The four sealed hashes are left at PENDING; ``oceanfno.anomaly_response
finalize`` fills them from the arm's own published figure package, exactly as
the parent package did, and refuses if that package is not internally
consistent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oceanfno.anomaly_response import IDENTITIES, PENDING, _REQUIRED_SOURCE_HASHES  # noqa: E402
from oceanfno import plots  # noqa: E402

PARENT = PROJECT_ROOT / "config" / "model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1.json"
SCRATCH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C")
PROJECT_OUT = PROJECT_ROOT / "outputs" / "af_fno" / "C"


def build(anomaly_version: str, figure_version: str) -> Path:
    contract = json.loads(json.dumps(json.loads(PARENT.read_text())))
    seed = int(figure_version.split("_seed_")[1].split("_")[0])
    training_version = figure_version.replace(f"_seed_{seed}_s0_figures_v1", "")

    contract["version"] = anomaly_version
    contract["purpose"] = (
        f"streamfunction-anomaly companions to figures 3 and 7 for {training_version} seed "
        f"{seed}, about the identical MITgcm S0 training-mean field the parent package used"
    )
    figures_root = PROJECT_OUT / figure_version / "S0"
    contract["artifacts"]["figure_package_contract"] = {
        "path": str(PROJECT_ROOT / "config" / f"{figure_version}.json"), "sha256": PENDING
    }
    contract["artifacts"]["figure_package_report"] = {
        "path": str(figures_root / plots.REPORT_NAME), "sha256": PENDING
    }
    contract["artifacts"]["figure_package_arrays"] = {
        "path": str(figures_root / plots.ARRAYS_NAME), "sha256": PENDING
    }
    contract["artifacts"]["figure_package_manifest"] = {
        "path": str(figures_root / plots.MANIFEST_NAME), "sha256": PENDING
    }

    contract["output"]["project_root"] = str(PROJECT_OUT / anomaly_version)
    contract["output"]["scratch_root"] = str(SCRATCH / anomaly_version)

    contract["source_hashes"] = {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in sorted(_REQUIRED_SOURCE_HASHES)
    }

    out = PROJECT_ROOT / "config" / f"{anomaly_version}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite an existing anomaly contract: {out}")
    out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return out


def main() -> int:
    for anomaly_version, figure_version in sorted(IDENTITIES.items()):
        print("wrote", build(anomaly_version, figure_version).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
