"""Execution step 15, plan section 19 step 7: freeze all ordinary forward outputs.

Hashes every artifact of the twelve B/C packages produced in this step (six
S0 figure packages, six anomaly packages), together with the frozen A and ft90
packages they are compared against and the six training reports they derive
from, into one write-once manifest.

Section 19 step 8: "Only after both packages and their hashes are in the
freeze manifest may the MITgcm/TAF adjoint evaluator be enabled." This
manifest is the ordinary-forward half of that precondition; the section-17
blind forward-response package (execution step 16) is the other half, and this
manifest records that it is still outstanding so the precondition cannot be
mistaken for satisfied.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oceanfno.anomaly_response import IDENTITIES as ANOMALY_IDENTITIES  # noqa: E402
from oceanfno.figures_response import IDENTITIES as FIGURE_IDENTITIES  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1" / "step15_forward_freeze"
CONTEXT_PACKAGES = (
    "model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1",
    "model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1",
    "model_c_production_1in_1out_spectralnorm_ft90_v1_s0_figures_v1",
    "model_c_production_1in_1out_spectralnorm_ft90_v1_s0_anomaly_v1",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(version: str) -> dict:
    root = PROJECT_ROOT / "outputs" / "af_fno" / "C" / version
    if not root.is_dir():
        raise SystemExit(f"package missing: {root}")
    return {
        "root": str(root),
        "artifacts": {
            str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": _sha(p)}
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    }


def main() -> int:
    training = {}
    for arm, base in (
        ("B", "model_c_adjoint_faithful_nominal_control_v1"),
        ("C", "model_c_adjoint_faithful_response_v1"),
    ):
        for seed in (20260724, 20260911, 20260912):
            report = PROJECT_ROOT / "outputs" / "af_fno" / "C" / base / f"seed_{seed}" / "report.json"
            payload = json.loads(report.read_text())
            training[f"{arm}_{seed}"] = {
                "report": str(report),
                "report_sha256": _sha(report),
                "report_content_sha256": payload["content_sha256"],
                "selected_optimizer_step": payload["published_checkpoint"]["optimizer_step"],
                "checkpoint_sha256": payload["published_checkpoint"]["checkpoint_sha256"],
                "normalization_sha256": payload["published_checkpoint"]["normalization_sha256"],
            }

    manifest = {
        "step": 15,
        "plan_section": "19 (forward figure and anomaly evaluation after training), step 7",
        "purpose": "freeze all ordinary forward outputs for arms A, ft90, B and C",
        "training_runs": training,
        "study_packages": {v: _package(v) for v in sorted(FIGURE_IDENTITIES) + sorted(ANOMALY_IDENTITIES)},
        "context_packages_unchanged": {v: _package(v) for v in CONTEXT_PACKAGES},
        "context_note": (
            "section 19 step 6: the existing A and ft90 reports are preserved, not regenerated. "
            "They are hashed here as they stand. src/oceanfno/figures.py and src/oceanfno/anomaly.py "
            "were deliberately not modified -- both are pinned by those packages' own contracts -- "
            "so A and ft90 remain able to re-verify themselves."
        ),
        "contracts": {
            v: {"path": str(PROJECT_ROOT / "config" / f"{v}.json"), "sha256": _sha(PROJECT_ROOT / "config" / f"{v}.json")}
            for v in sorted(FIGURE_IDENTITIES) + sorted(ANOMALY_IDENTITIES)
        },
        "adapters": {
            r: _sha(PROJECT_ROOT / r)
            for r in (
                "src/oceanfno/figures_response.py",
                "src/oceanfno/anomaly_response.py",
                "src/oceanfno/figures.py",
                "src/oceanfno/anomaly.py",
            )
        },
        "section_19_step_8_precondition": {
            "ordinary_forward_frozen": True,
            "blind_forward_response_frozen": False,
            "adjoint_evaluator_may_be_enabled": False,
            "note": (
                "both packages and their hashes must be in the freeze manifest before the "
                "MITgcm/TAF adjoint evaluator is enabled. The section-17 blind forward-response "
                "package is execution step 16 and has not been run."
            ),
        },
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(manifest, indent=2, sort_keys=True).encode()
    ).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "step15_forward_freeze_manifest.json"
    if path.exists():
        raise SystemExit(f"step 15 forward outputs are already frozen: {path}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(len(p["artifacts"]) for p in manifest["study_packages"].values())
    context = sum(len(p["artifacts"]) for p in manifest["context_packages_unchanged"].values())
    print(f"froze {len(manifest['study_packages'])} study packages ({total} artifacts), "
          f"{len(CONTEXT_PACKAGES)} context packages ({context} artifacts)")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
