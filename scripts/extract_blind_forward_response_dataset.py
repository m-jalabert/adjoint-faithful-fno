"""Execution step 16: extract the blind forward-response store.

Writes the ``blind_test`` role into its **own** zarr
(``forward_response_blind_v1.zarr``), not into the development store. Two
reasons: the development store's hash is already pinned by the C study
contract and must not change, and section 17's blind package is a separate
sealed artifact with its own provenance.

Everything numerical is `extract_forward_response_dataset` unchanged -- the
same P32 projection, the same antisymmetry checks, the same schema. Only the
inventory, the report root and the output path are supplied, all explicitly:
that module's defaults still name development paths only, so no default call
can reach blind data.

The blind role's long directions carry 90-day horizons, and
`_long_leads_for_role` already maps ``blind_test`` to leads 10..90 -- the same
lead set the amplitude pilot used -- so `S_resp^90` has the endpoints it needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import extract_forward_response_dataset as extract_module  # noqa: E402
import stage_blind_forward_response_run as blind  # noqa: E402

BLIND_DATASET_PATH = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/forward_response_blind_v1.zarr"
)
BLIND_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_blind_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=BLIND_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=BLIND_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    precondition = blind.assert_precondition()
    rows = blind.load_blind_rows()
    print(f"[blind-extract] precondition satisfied (lambda={precondition['selected_lambda_resp']})", flush=True)
    print(f"[blind-extract] {len(rows)} blind_test rows; report root {blind.BLIND_REPORT_ROOT}", flush=True)

    result = extract_module.extract(
        roles=["blind_test"],
        inventory_path=blind.BLIND_INVENTORY,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        report_root=blind.BLIND_REPORT_ROOT,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "arrays"}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
