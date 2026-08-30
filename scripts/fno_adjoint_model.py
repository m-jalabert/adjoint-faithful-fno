"""Execution step 17: run the trusted FNO derivative machinery for A, B and C.

Plan section 18.2 requires the *same* FNO-side machinery that produced the
Phase-A ft90 result to be run for frozen parent A, all paired B and C
replicates, and ft90 as context -- and plan section 23.1 asks for this as a
"contract-parameterized adapter of the trusted one-input ft90 runner,
retaining its validated complex128 fix".

That is exactly what this is. Every gate, every objective, the operator
construction, and in particular the complex128 spectral-buffer promotion that
`neuralop-spectral-buffer-complex64` exists to remember, are imported from
`fno_adjoint_ft90` and executed unchanged. This module contributes only the
identity registry below: which checkpoint, which report path, which expected
hashes, and where the result is written.

The registry is hard-coded rather than read from the reports at run time, for
the same reason the ft90 runner hard-codes its own: a different checkpoint is
a different operator, and every number downstream would then be about
something else. The values were read once from the frozen reports and are
asserted against them on every run.

Gate A0 (plan section 22) requires the FNO finite-difference, forward/reverse
identity, dtype, mask, and checkpoint/normalizer hash gates to pass for
parent, B and C before any adjoint comparison. Those are gates F1-F5, F-sigma
and F7 inside the runner; this module just points them at each model.

These checks occur only after model freeze and cannot affect model decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import fno_adjoint_ft90 as runner  # noqa: E402
from fno_adjoint_ft90 import ModelIdentity  # noqa: E402

#: label -> identity. ft90 keeps its own published result and its own module
#: default, so it is present here only for completeness of the section-18.2
#: model list.
IDENTITIES: dict[str, ModelIdentity] = {
    "A": ModelIdentity(
        contract="model_c_production_1in_1out_spectralnorm_v1",
        report_relative="model_c_production_1in_1out_spectralnorm_v1/model_c_production_1in_1out_spectralnorm_v1_report.json",
        checkpoint_sha256="e75951681b1abbecb44ff35ac8319b38504fad9995e8e08fbc33c86bb8c2798a",
        normalization_sha256="fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf",
        optimizer_step=7680,
        output_relative="outputs/af_fno/adjoint/fno_a_s0_adjoint_v1",
        label="frozen parent A",
        seed=None,
    ),
    "B_20260724": ModelIdentity(
        contract="model_c_adjoint_faithful_nominal_control_v1",
        report_relative="model_c_adjoint_faithful_nominal_control_v1/seed_20260724/report.json",
        checkpoint_sha256="83ee7e71e4982a653f85e0073ebce24655f738e518954a39c47be0979fedeba0",
        normalization_sha256="fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf",
        optimizer_step=7680,
        output_relative="outputs/af_fno/adjoint/fno_b_seed_20260724_s0_adjoint_v1",
        label="arm B seed 20260724",
        seed=20260724,
    ),
    "B_20260911": ModelIdentity(
        contract="model_c_adjoint_faithful_nominal_control_v1",
        report_relative="model_c_adjoint_faithful_nominal_control_v1/seed_20260911/report.json",
        checkpoint_sha256="127fa557160df6b635849e1687d49ebf54175a47c62781af4e5a0eb8d2f8046e",
        normalization_sha256="fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf",
        optimizer_step=7680,
        output_relative="outputs/af_fno/adjoint/fno_b_seed_20260911_s0_adjoint_v1",
        label="arm B seed 20260911",
        seed=20260911,
    ),
    "B_20260912": ModelIdentity(
        contract="model_c_adjoint_faithful_nominal_control_v1",
        report_relative="model_c_adjoint_faithful_nominal_control_v1/seed_20260912/report.json",
        checkpoint_sha256="f20bf96a667d1d120d7af6de7d47bf8dacb020ed182b5aebd4d37c93d8dfbecd",
        normalization_sha256="fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf",
        optimizer_step=7680,
        output_relative="outputs/af_fno/adjoint/fno_b_seed_20260912_s0_adjoint_v1",
        label="arm B seed 20260912",
        seed=20260912,
    ),
    "C_20260724": ModelIdentity(
        contract="model_c_adjoint_faithful_response_v1",
        report_relative="model_c_adjoint_faithful_response_v1/seed_20260724/report.json",
        checkpoint_sha256="cfa3c0bb9dcd4d67e164e7fab4a39ba79f2a44d7ff7d86b85df8629a7759323e",
        normalization_sha256="fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf",
        optimizer_step=7680,
        output_relative="outputs/af_fno/adjoint/fno_c_seed_20260724_s0_adjoint_v1",
        label="arm C seed 20260724",
        seed=20260724,
    ),
    "C_20260911": ModelIdentity(
        contract="model_c_adjoint_faithful_response_v1",
        report_relative="model_c_adjoint_faithful_response_v1/seed_20260911/report.json",
        checkpoint_sha256="0033a6dbfde5db3c26a2b44e5e756370f1da0fff3084020f539b943c94bf90a5",
        normalization_sha256="fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf",
        optimizer_step=7680,
        output_relative="outputs/af_fno/adjoint/fno_c_seed_20260911_s0_adjoint_v1",
        label="arm C seed 20260911",
        seed=20260911,
    ),
    "C_20260912": ModelIdentity(
        contract="model_c_adjoint_faithful_response_v1",
        report_relative="model_c_adjoint_faithful_response_v1/seed_20260912/report.json",
        checkpoint_sha256="6fda1636a853ef1410079a7b106011699df1031f679a9a338217bf860a5099e1",
        normalization_sha256="fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf",
        optimizer_step=7680,
        output_relative="outputs/af_fno/adjoint/fno_c_seed_20260912_s0_adjoint_v1",
        label="arm C seed 20260912",
        seed=20260912,
    ),
    "ft90": runner.FT90_IDENTITY,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(IDENTITIES), required=True)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="two finite-difference epsilons instead of four")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    identity = IDENTITIES[args.model]
    epsilons = runner.DEFAULT_FD_EPSILONS[:2] if args.quick else runner.DEFAULT_FD_EPSILONS
    report = runner.run(
        Path(args.project_root),
        force=args.force,
        epsilons=epsilons,
        threads=args.threads,
        identity=identity,
    )
    print(json.dumps({"model": args.model, "identity": identity.contract,
                      "seed": identity.seed, "output": identity.output_relative,
                      "status": report.get("status", "complete")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
