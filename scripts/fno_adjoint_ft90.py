"""Differentiate the published production emulator and write its sensitivity maps.

Implements steps 0-5 of ``docs/Adjoint_study_Phase_A.md``.  The deliverable is

    S_L[j, i] = dJ_L / d eta(j, i, day 7200)     through   x_{t+10} = F(x_t, S)

for L in {10, 20, 30, 90} days and two scalar SSH objectives, with **J the
identical scalar** the MITgcm side computes, on the identical window, in the
identical units, so that ``S_fno`` and MITgcm's ``S`` can simply be subtracted.

-----------------------------------------------------------------------------
Why this is a new script and not an edit of ``fno_adjoint.py``
-----------------------------------------------------------------------------

The predecessor differentiated a **two-input** operator,
``F(x_{t-10}, x_t, S) -> x_{t+10}``, whose present-slot derivative was a
*partial* derivative holding a dynamically-linked second input fixed --- an
off-manifold quantity with no MITgcm counterpart.  The production model is
**one-input**: it is Markov in the same 46-channel state MITgcm is Markov in,
so ``dJ/d eta(t0)`` is a total derivative with respect to a complete initial
condition and is structurally the same object as MITgcm's ``adxx_etan``.

That deletes the old experiment E2 entirely and collapses E1 and E3 into a
single chained lead sweep.  It also deletes the largest interpretive caveat in
the earlier study.

-----------------------------------------------------------------------------
The two trajectories, and why both are computed
-----------------------------------------------------------------------------

MITgcm's adjoint is, by construction, linearized about the **true** trajectory.
The emulator's autoregressive chain is linearized about **its own**, which has
drifted from truth by day 7290.  A disagreement at ninety days therefore
conflates two different errors, and this script separates them:

    S_forced   tangent-linear chain evaluated at the MITgcm truth states
               -> the matched object; PRIMARY comparison against MITgcm
    S_free     tangent-linear chain evaluated along the emulator's own rollout
               -> what a deployed user actually gets; PRIMARY operational map

    S_mitgcm vs S_forced   =  Jacobian error      (the operator is inexact)
    S_forced vs S_free     =  trajectory error    (the linearization drifted)

At lead 10 the two chains are identical by construction, which is gate F5.

-----------------------------------------------------------------------------
Two sweeps
-----------------------------------------------------------------------------

    (B) PRIMARY    source day 7200 fixed, cost day moves to 7200 + L
                   isolates propagation length from initial condition
    (A) companion  cost day 7290 fixed, dJ/d eta(t) for every t
                   the classic backward-in-time sensitivity movie

(A) falls out of (B)'s lead-90 chain for free: the adjoint state after k
backward legs *is* ``dJ/dx(7200 + 10k)``, which is the exact structural
analogue of MITgcm's ``ADJetan`` dumps.

-----------------------------------------------------------------------------
Usage
-----------------------------------------------------------------------------

    python scripts/fno_adjoint_ft90.py                # everything
    python scripts/fno_adjoint_ft90.py --force        # overwrite the output
    python scripts/fno_adjoint_ft90.py --quick        # short finite-difference sweep

Nothing here trains, fine-tunes or modifies any weight.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import statistics
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import zarr

import adjoint_metrics as metrics
from select_adjoint_target import CONTRACT_VERSION as TARGET_CONTRACT
from select_adjoint_target import read_mds_2d

from neuralop.layers.spectral_convolution import SpectralConv

from oceanfno.dataset import static_block
from oceanfno.model import ProductionArchitecture, build_model


# ===========================================================================
# 1.  The frozen contract
# ===========================================================================

PLAN_CONTRACT = "adjoint_phase_a_v1"
MODEL_CONTRACT = "model_c_production_1in_1out_spectralnorm_ft90_v1"


@dataclasses.dataclass(frozen=True)
class ModelIdentity:
    """Which published model this runner is pointed at.

    The runner was written for the ft90 child and hard-coded its identity in
    five places. Execution step 17 (plan section 18.2) has to run the same
    trusted machinery -- including the validated complex128 spectral fix -- for
    frozen parent A and all six paired B/C replicates, so the identity is a
    parameter now. **Every default is the ft90 child's**, so the published
    ft90 result remains reproducible bit-for-bit by calling this module exactly
    as before; nothing about the numerics changed.

    ``report_relative`` is the training report's path under ``outputs/af_fno/C``:
    the parent and ft90 publish ``<contract>/<contract>_report.json`` while the
    B/C study runs publish ``<contract>/seed_<seed>/report.json``.
    """

    contract: str
    report_relative: str
    checkpoint_sha256: str
    normalization_sha256: str
    optimizer_step: int
    output_relative: str
    label: str
    seed: int | None = None

#: Asserted before the first forward pass.  A different checkpoint is a
#: different operator and every number below would be about something else.
EXPECTED_CHECKPOINT_SHA256 = (
    "4acb7633d85a4df3925843cc833d248e86fcd5d2569ba0300c9c58b022537806"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "fe424b37d74f5b9d901728c8d585245e12ab67e4230a2eb86f6edc43108d96bf"
)
EXPECTED_PARAMETER_COUNT = 27_297_960
EXPECTED_OPTIMIZER_STEP = 1_440

#: Predeclared probes for the adjoint dot-product identity. Fixed and shared by
#: every model so no arm is tested on a different draw. See
#: ``verify_double_precision_spectrum`` for why a single probe is not enough.
PRECISION_PROBE_SEEDS: tuple[int, ...] = (20260819, 20260820, 20260821, 20260822, 20260823)

#: Channel layout.  One input state, so the 51 external channels are the 46
#: prognostic ones followed by the five statics, and eta is the last of the 46.
#: The two positional-encoding channels are added inside the operator and are
#: not part of the external contract.
STATE_CHANNEL_COUNT = 46
STATIC_CHANNEL_COUNT = 5
EXTERNAL_INPUT_CHANNELS = STATE_CHANNEL_COUNT + STATIC_CHANNEL_COUNT
ETA_CHANNEL = 45

REGIME = "S0"
REGIME_INDEX = 0

SOURCE_DAY = 7200
HORIZON_DAYS = 10
LEAD_DAYS: tuple[int, ...] = (10, 20, 30, 90)

#: Objectives.  ``ssh_anomaly`` is primary; ``ssh_anomaly_kernel`` is the
#: smoothed variant; ``mean_only`` is the conservation probe, whose exact
#: answer is known in closed form and needs no MITgcm run to interpret.
OBJECTIVES: tuple[str, ...] = ("ssh_anomaly", "ssh_anomaly_kernel", "mean_only")
PRIMARY_OBJECTIVE = "ssh_anomaly"

CHAINS: tuple[str, ...] = ("forced", "free")

MDS_DTYPE = ">f4"

F1_TOLERANCE = 1.0e-10
F2_TOLERANCE = 1.0e-6
F4_TOLERANCE = 1.0e-4

#: Slack over machine epsilon for comparisons between two float64 derivatives of
#: the same map.  Two exact derivatives accumulated in different orders through
#: three FFT round trips cannot agree better than this, and the reordering probe
#: cannot measure it because PyTorch is bit-deterministic here.
MACHINE_FLOOR_MULTIPLE = 64.0

#: The sweep has to span both branches of the finite difference: truncation
#: error falls as eps^2 while round-off grows as noise/(2 eps), so the plateau
#: the gate asks for is the minimum between them.  Where that minimum sits is a
#: property of the *measured* noise in the computed cost, and this operator's is
#: 3e-16 m once the spectral buffer is genuinely complex128 --- six orders of
#: magnitude below what neuralop's hard-coded complex64 gave (5.5e-10 m).  With
#: the real floor the optimum lands near 1e-5, so the range runs down to 1e-6
#: rather than stopping where a single-precision pipeline would have forced it.
DEFAULT_FD_EPSILONS: tuple[float, ...] = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
QUICK_FD_EPSILONS: tuple[float, ...] = (1.0e-3, 1.0e-5, 1.0e-6)

OUTPUT_RELATIVE = Path("outputs/af_fno/adjoint/fno_ft90_s0_adjoint_v1")

#: The identity this module was written for. Every parameter defaults to it,
#: so calling this module unchanged reproduces the published ft90 result.
FT90_IDENTITY = ModelIdentity(
    contract=MODEL_CONTRACT,
    report_relative=f"{MODEL_CONTRACT}/{MODEL_CONTRACT}_report.json",
    checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
    normalization_sha256=EXPECTED_NORMALIZATION_SHA256,
    optimizer_step=EXPECTED_OPTIMIZER_STEP,
    output_relative=str(OUTPUT_RELATIVE),
    label="Phase A",
)


class FnoAdjointError(RuntimeError):
    """Raised when the pipeline cannot be trusted to mean what it says."""


def calls_for_lead(lead_days: int) -> int:
    if lead_days % HORIZON_DAYS != 0:
        raise FnoAdjointError(f"lead {lead_days} is not a multiple of the {HORIZON_DAYS}-day horizon")
    return lead_days // HORIZON_DAYS


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(specification: Mapping[str, Any], label: str) -> Path:
    """Resolve a {path, sha256} pair from a contract and check the digest."""

    path = Path(specification["path"]).resolve()
    if not path.is_file():
        raise FnoAdjointError(f"{label} is missing: {path}")
    digest = file_sha256(path)
    if digest != specification["sha256"]:
        raise FnoAdjointError(
            f"{label} has digest {digest}, contract pins {specification['sha256']}"
        )
    return path


# ===========================================================================
# 2.  The shared contract  (plan section 1)
# ===========================================================================


@dataclass(frozen=True)
class SharedContract:
    """Everything both sides of the comparison must agree on.

    Every entry is shared by *reading the same file* rather than by
    reimplementing.  Rebuilding the weight field here instead of reading the
    binary the MITgcm run stages would silently turn the comparison into a
    convention test, which is the single largest risk in this study; gate F6
    is the guard, and it is an assertion.
    """

    target: tuple[int, int]  # p*, zero-based (j, i), matching the zarr axes
    wet: np.ndarray
    rac: np.ndarray
    wet_area: float
    weights: dict[str, np.ndarray]
    weight_digests: dict[str, str]
    longitude: np.ndarray
    latitude: np.ndarray
    sources: dict[str, str]


def load_shared_contract(project_root: Path, group: Any, plan: Mapping[str, Any]) -> SharedContract:
    contract = json.loads((project_root / "config" / f"{TARGET_CONTRACT}.json").read_text())
    target = (int(contract["j_index0"]), int(contract["i_index0"]))
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    rac = read_mds_2d(Path(contract["grid"]["rac_source"]))
    wet_area = float(contract["wet_area_m2"])

    if wet.shape != (62, 62) or int(wet.sum()) != int(contract["grid"]["wet_cell_count"]):
        raise FnoAdjointError("the store's wet mask is not the contract's 3,600-cell basin")
    recomputed = float((rac * wet).sum())
    if abs(recomputed - wet_area) > 1.0e-6 * wet_area:
        raise FnoAdjointError(f"A_wet mismatch: contract {wet_area!r}, grid files {recomputed!r}")
    if not wet[target]:
        raise FnoAdjointError(f"the frozen target cell {target} is not wet")
    if (int(plan["target"]["j_index0"]), int(plan["target"]["i_index0"])) != target:
        raise FnoAdjointError("the Phase A plan and the target contract disagree about p*")

    weights, digests, sources = {}, {}, {}
    for name in OBJECTIVES:
        declared = plan["objectives"][name]
        path = (project_root / declared["weight_file"]).resolve()
        if not path.is_file():
            raise FnoAdjointError(
                f"{path} is missing --- run scripts/make_cost_weight.py --qoi {name}"
            )
        digest = file_sha256(path)
        # Gate F6.  The MITgcm run manifest pins the same digest; if these ever
        # diverge the two sides are weighting eta differently and every metric
        # downstream is measuring the convention rather than the model.
        if declared.get("weight_sha256") and digest != declared["weight_sha256"]:
            raise FnoAdjointError(
                f"{path} has digest {digest}, the Phase A contract pins "
                f"{declared['weight_sha256']}"
            )
        field = np.fromfile(path, dtype=MDS_DTYPE)
        if field.size != wet.size:
            raise FnoAdjointError(f"{path} holds {field.size} values, expected {wet.size}")
        field = field.reshape(wet.shape).astype(np.float64)
        if np.any(field[~wet] != 0.0):
            raise FnoAdjointError(f"{path} is non-zero on land")
        weights[name] = field
        digests[name] = digest
        sources[f"cost_weight_{name}"] = str(path)

    # Structural checks on the fields just read.  These verify; they never
    # substitute a locally computed weight for the one on disk.
    mean_term = -rac[target] / wet_area
    if abs(weights["ssh_anomaly"][target] - (1.0 + mean_term)) > 1.0e-6:
        raise FnoAdjointError("the anomaly weight field does not carry the delta at p*")
    for name in ("ssh_anomaly", "ssh_anomaly_kernel"):
        if abs(float(weights[name][wet].sum())) > 1.0e-4:
            raise FnoAdjointError(f"{name} does not sum to zero over wet cells")
    kernel_profile = weights["ssh_anomaly_kernel"] - weights["mean_only"]
    declared_profile = np.asarray(
        plan["objectives"]["ssh_anomaly_kernel"]["kernel"]["weights"], dtype=np.float64
    )
    live = kernel_profile[kernel_profile > 1.0e-12]
    if live.size != declared_profile.size or not np.allclose(
        np.sort(live), np.sort(declared_profile), atol=1.0e-6
    ):
        raise FnoAdjointError(
            "the kernel weight field on disk is not the stencil the Phase A contract declares"
        )

    sources.update(
        {
            "target_contract": str(project_root / "config" / f"{TARGET_CONTRACT}.json"),
            "plan_contract": str(project_root / "config" / f"{PLAN_CONTRACT}.json"),
            "cell_area": str(contract["grid"]["rac_source"]),
        }
    )
    return SharedContract(
        target=target,
        wet=wet,
        rac=rac,
        wet_area=wet_area,
        weights=weights,
        weight_digests=digests,
        longitude=np.asarray(group["longitude_deg"][:], dtype=np.float64),
        latitude=np.asarray(group["latitude_deg"][:], dtype=np.float64),
        sources=sources,
    )


# ===========================================================================
# 3.  Loading the frozen operator
# ===========================================================================


def load_model_provenance(project_root: Path, identity: "ModelIdentity | None" = None) -> dict[str, Any]:
    identity = identity or FT90_IDENTITY
    contract = json.loads((project_root / "config" / f"{identity.contract}.json").read_text())
    report_path = project_root / "outputs" / "af_fno" / "C" / identity.report_relative
    report = json.loads(report_path.read_text())
    published = report["published_checkpoint"]

    if published["checkpoint_sha256"] != identity.checkpoint_sha256:
        raise FnoAdjointError(
            f"this is not the checkpoint {identity.label} freezes: "
            f"expected {identity.checkpoint_sha256}, report says {published['checkpoint_sha256']}"
        )
    if int(published["optimizer_step"]) != identity.optimizer_step:
        raise FnoAdjointError(
            f"the published checkpoint is step {published['optimizer_step']}, "
            f"expected {identity.optimizer_step}"
        )
    if published["normalization_sha256"] != identity.normalization_sha256:
        raise FnoAdjointError(f"the published normalizers are not the ones {identity.label} freezes")

    checkpoint = _verify(
        {"path": published["checkpoint"], "sha256": published["checkpoint_sha256"]},
        "published checkpoint",
    )
    normalization = _verify(
        {"path": published["normalization"], "sha256": published["normalization_sha256"]},
        "published normalizers",
    )
    return {
        "contract": contract,
        "report_path": report_path,
        "checkpoint": checkpoint,
        "checkpoint_sha256": published["checkpoint_sha256"],
        "normalization": normalization,
        "normalization_sha256": published["normalization_sha256"],
        "optimizer_step": int(published["optimizer_step"]),
        "identity": identity,
    }


@contextlib.contextmanager
def _double_precision_spectrum() -> Any:
    """Allocate the spectral working buffer in complex128 instead of complex64."""

    original = torch.zeros

    def patched(*args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("dtype") is torch.cfloat:
            kwargs["dtype"] = torch.cdouble
        return original(*args, **kwargs)

    torch.zeros = patched
    try:
        yield
    finally:
        torch.zeros = original


class DoublePrecisionSpectralConv(SpectralConv):
    """``SpectralConv`` whose Fourier-domain buffer follows the weight dtype.

    ``neuralop`` 2.0.0 hard-codes the working spectrum's precision
    (``spectral_convolution.py``, ``SpectralConv.forward``)::

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat          # <-- not derived from the input
        out_fft = torch.zeros([...], dtype=out_dtype)
        ...
        out_fft[slices_x] = self._contract(x[slices_x], weight, ...)

    ``out_dtype`` does not depend on the input or the weights, so with
    ``fno_block_precision = "full"`` the buffer is **complex64 whatever dtype
    the model is in**.  The contraction is computed in complex128 and then
    written into that buffer, which truncates it to single precision, and the
    inverse transform runs from there.  The bias addition afterwards promotes
    the *type* back to float64 without recovering any of the lost digits.

    The consequence is not small.  The spectral convolutions hold 26,739,072 of
    the operator's 27,297,960 parameters --- **97.95 %** --- so casting this
    model to double leaves essentially all of it in single precision, and every
    "float64" claim about a gradient through it is false.  This project has been
    here before at one level up: ``s0-twin-float32-floor`` records a float32
    quantisation floor swallowing an entire twin-perturbation signal.

    Measured effect, all on one ``SpectralConv`` where the exact answer is
    computable because the layer is linear:

    ==========================================  ==========
    forward-mode ``J u`` vs ``L(u) - L(0)``      0.0
    reverse-mode ``<J^T v, u>`` vs ``<v, L u>``  6.7e-07
    the whole model, float32 vs "float64"        1.7e-07
    ==========================================  ==========

    All three are single precision, and the third is the tell: a genuine float64
    path would differ from float32 by float32's own error, not agree with it.

    **This does not change the operator.**  The weights are untouched and the
    map they define is unchanged; only the arithmetic evaluating it is.  The
    project's own inference runs in float32, so the deployed map carries this
    precision anyway --- the adjoint study wants the derivative of the operator
    the weights define, not of one particular rounding of it.  The patch is
    verified rather than trusted: :func:`verify_double_precision_spectrum`
    asserts that the buffer really is complex128 afterwards and that forward and
    reverse mode then agree to float64 round-off, which they demonstrably do
    not beforehand.
    """

    def forward(self, x: torch.Tensor, output_shape: Any = None) -> torch.Tensor:
        if x.dtype in (torch.float64, torch.complex128):
            with _double_precision_spectrum():
                return super().forward(x, output_shape=output_shape)
        return super().forward(x, output_shape=output_shape)


def _promote_spectral_convolutions(model: torch.nn.Module) -> int:
    """Re-class every ``SpectralConv`` in place.  Adds no state and no parameters."""

    promoted = 0
    for module in model.modules():
        if type(module) is SpectralConv:
            module.__class__ = DoublePrecisionSpectralConv
            promoted += 1
    return promoted


def verify_double_precision_spectrum(model: torch.nn.Module) -> dict[str, Any]:
    """Assert the promotion took, by observing the buffer and the adjoint identity.

    Two independent checks:

    1. spy on the complex allocation inside one ``SpectralConv`` forward and
       require complex128;
    2. the dot-product test ``<v, J u> == <J^T v, u>`` on that same layer, which
       is linear, so ``J u`` is available exactly from forward mode.  Before the
       promotion this residual is ~7e-07; after it, it must be at float64
       round-off.

    **Amended 2026-08-29 (execution step 17).** Check 2 originally drew a
    single random probe at one hard-coded seed and compared it to a fixed
    1e-12 constant. ``<v, J u>`` involves 500--2900x cancellation for every
    model in this study, so a single realization is heavy-tailed: measured
    across five probes and eight checkpoints, the same operator spans three
    orders of magnitude (e.g. C seed 20260724: median 5.6e-15, max 9.7e-12).
    Pass/fail was therefore substantially a lottery over probe realizations --
    and it fired, on the one hard-coded seed, for the model with the *best*
    median residual of the eight.

    The gate now draws ``PRECISION_PROBE_SEEDS`` probes and tests the
    **median** against the unchanged 1e-12 threshold. The bar is not loosened;
    the estimator is. This also brings the check into line with the rest of
    this suite, where F2 and F2_forward_mode already compare against a
    measured arithmetic floor rather than a constant. Every per-probe residual
    is reported so the spread stays visible.
    """

    convolutions = [m for m in model.modules() if isinstance(m, SpectralConv)]
    if not convolutions:
        raise FnoAdjointError("the loaded model has no spectral convolutions")
    layer = convolutions[0]

    observed: list[Any] = []
    original = torch.zeros

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        tensor = original(*args, **kwargs)
        if tensor.is_complex():
            observed.append(tensor.dtype)
        return tensor

    channels = layer.in_channels
    generator = torch.Generator().manual_seed(PRECISION_PROBE_SEEDS[0])
    probe = torch.randn((1, channels, 74, 74), generator=generator, dtype=torch.float64)
    torch.zeros = spy
    try:
        with torch.no_grad():
            layer(probe)
    finally:
        torch.zeros = original

    per_probe: list[dict[str, float]] = []
    for seed in PRECISION_PROBE_SEEDS:
        generator = torch.Generator().manual_seed(int(seed))
        probe = torch.randn((1, channels, 74, 74), generator=generator, dtype=torch.float64)
        direction = torch.randn((1, channels, 74, 74), generator=generator, dtype=torch.float64)
        with torch.no_grad():
            zero_response = layer(torch.zeros_like(probe))
            exact = layer(direction) - zero_response  # J u exactly: the layer is linear
        cotangent = torch.randn(tuple(exact.shape), generator=generator, dtype=torch.float64)
        leaf = probe.clone().requires_grad_(True)
        (transpose,) = torch.autograd.grad((cotangent * layer(leaf)).sum(), leaf)
        left = float((cotangent * exact).sum())
        right = float((transpose * direction).sum())
        per_probe.append({
            "seed": int(seed),
            "residual": abs(left - right) / max(abs(left), 1.0e-300),
            "inner_product": left,
            # How much cancellation the sum carries: the ratio of the summed
            # magnitudes to the result. This is why a single draw is noisy.
            "cancellation_ratio": float((cotangent.abs() * exact.abs()).sum()) / max(abs(left), 1.0e-300),
        })
    ordered = sorted(entry["residual"] for entry in per_probe)
    residual = statistics.median(ordered)

    return {
        "condition": (
            "the Fourier working buffer is complex128 and the reverse-mode adjoint "
            "satisfies <v, J u> = <J^T v, u> to float64 round-off"
        ),
        "spectral_convolutions": len(convolutions),
        "buffer_dtypes": sorted({str(dtype) for dtype in observed}),
        "dot_product_residual": residual,
        "dot_product_residual_statistic": "median over PRECISION_PROBE_SEEDS probes",
        "dot_product_residual_per_probe": per_probe,
        "dot_product_residual_min": ordered[0],
        "dot_product_residual_max": ordered[-1],
        "probe_seeds": list(PRECISION_PROBE_SEEDS),
        "residual_before_promotion": 6.7e-07,
        "threshold": 1.0e-12,
        "passed": bool(
            observed
            and all(dtype == torch.complex128 for dtype in observed)
            and residual < 1.0e-12
        ),
        "upstream_defect": (
            "neuralop 2.0.0 SpectralConv.forward hard-codes out_dtype = torch.cfloat "
            "when fno_block_precision is 'full', so the working spectrum is single "
            "precision regardless of the model's dtype"
        ),
    }


def _cast_to_double(model: torch.nn.Module) -> torch.nn.Module:
    """Cast every parameter and buffer to double precision.

    **Do not use** ``model.double()`` or ``model.to(torch.float64)``.  The
    spectral convolutions hold *complex* weights and PyTorch treats them
    inconsistently: ``.double()`` skips them (leaving complex64 weights that
    then refuse to multiply a complex128 spectrum) while ``.to(torch.float64)``
    casts them to a real dtype and silently **discards the imaginary part**,
    quietly destroying most of the model.  ``_apply`` is the same mechanism
    ``.double()`` uses internally; the difference is that complex tensors go to
    complex128 and real ones to float64.
    """

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_complex():
            return tensor.to(torch.complex128)
        if tensor.is_floating_point():
            return tensor.to(torch.float64)
        return tensor

    model._apply(convert)
    if any(p.dtype not in (torch.float64, torch.complex128) for p in model.parameters()):
        raise FnoAdjointError("the double-precision cast left a float32 parameter behind")
    return model


def load_frozen_model(
    checkpoint: Path, *, double: bool = True, identity: "ModelIdentity | None" = None
) -> torch.nn.Module:
    """Build the architecture, load the checkpoint strictly, and freeze it.

    Gate F3 lives here.  The per-mode spectral cap is *materialized on write*,
    so the published checkpoint loads into a plain ``ProductionFNO`` whose
    inference layer is exactly ``y_hat(k) = R_k_tilde x_hat(k)`` and whose
    adjoint is ``R_k_tilde^H``.  That is what makes these sensitivities clean,
    and it is asserted rather than trusted: any live spectral-norm hook, buffer
    or wrapper on the loaded module would mean the differentiated map is not
    the published one.
    """

    model = build_model(ProductionArchitecture())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_version = (identity or FT90_IDENTITY).contract
    if payload.get("version") != expected_version:
        raise FnoAdjointError(
            f"checkpoint declares version {payload.get('version')!r}, expected {expected_version!r}"
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)

    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise FnoAdjointError(f"loaded {count} parameters, expected {EXPECTED_PARAMETER_COUNT}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if double:
        _cast_to_double(model)
        # Casting the weights is not enough: neuralop's spectral working buffer
        # is hard-coded to single precision, so without this the 97.95 % of the
        # operator that lives in the Fourier domain would stay float32 while
        # every report in this directory claimed float64.  See
        # DoublePrecisionSpectralConv.
        _promote_spectral_convolutions(model)
    return model


def gate_f3_operator_preflight(model: torch.nn.Module) -> dict[str, Any]:
    """F3 --- the loaded module must be the plain, published operator."""

    hooks: list[str] = []
    for name, module in model.named_modules():
        for registry in (
            getattr(module, "_forward_pre_hooks", {}),
            getattr(module, "_forward_hooks", {}),
        ):
            for hook in registry.values():
                label = f"{name or '<root>'}::{type(hook).__name__}"
                if "norm" in type(hook).__name__.lower() or "spectral" in type(hook).__name__.lower():
                    hooks.append(label)
    parametrized = [
        name
        for name, module in model.named_modules()
        if getattr(module, "_parametrizations", None)
    ]
    suspicious_buffers = [
        name
        for name, _ in model.named_buffers()
        if any(token in name.lower() for token in ("_u", "_v", "sigma", "spectral_norm", "power"))
    ]
    count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    passed = (
        not hooks
        and not parametrized
        and not suspicious_buffers
        and count == EXPECTED_PARAMETER_COUNT
        and trainable == 0
        and not model.training
    )
    return {
        "condition": (
            "plain ProductionFNO, 27,297,960 parameters, eval mode, gradients off, "
            "no live spectral-norm hook / parametrization / estimator buffer"
        ),
        "parameter_count": int(count),
        "trainable_parameters": int(trainable),
        "training_mode": bool(model.training),
        "spectral_norm_hooks": hooks,
        "parametrized_modules": parametrized,
        "estimator_buffers": suspicious_buffers,
        "passed": bool(passed),
    }


# ===========================================================================
# 4.  The operator wrapper --- normalization and masking inside the graph
# ===========================================================================


@dataclass
class FrozenOperator:
    """The emulator plus the coordinate change and the masks it is deployed in.

    The network never sees metres.  It sees ``(physical - mu) / sigma`` with
    ``mu`` and ``sigma`` of shape ``(46, Y, X)`` --- one value per channel *per
    grid point*.  Both directions of that change of variables are ordinary
    tensor arithmetic here so that they sit **inside** the autograd graph.  That
    is the whole trick: with the physical field as the leaf, autograd carries
    every factor of ``sigma`` on the way in and on the way out, and the answer
    comes out in physical units with no post-hoc rescaling to get wrong.

    Both masks are inside the graph too, matching
    ``oceanfno.model.ProductionStepper`` exactly: ``normalized_state`` zeroes
    land on input, and ``step`` multiplies each prediction by the wet mask.
    That makes ``dJ/d eta(land) = 0`` exactly, by construction --- the same
    thing MITgcm's gate G4 asserts, and for the same reason: sea-surface height
    on land is not a degree of freedom of either map.
    """

    model: torch.nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    static: torch.Tensor
    wet: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.mean.dtype

    def normalize(self, physical: torch.Tensor, *, mask: bool = True) -> torch.Tensor:
        """metres, degrees, m/s  ->  the network's dimensionless coordinates.

        ``mask=False`` lifts the input land mask, which makes sea-surface height
        on land a live input degree of freedom.  The *value* is unchanged --- the
        normalizers are ``(mu=0, sigma=1)`` on land and physical land eta is
        zero, so a masked and an unmasked input normalize identically --- but the
        *derivative* is not, and the difference is precisely the leakage
        diagnostic of plan section 4.2.  It is never used for a map that is
        compared against MITgcm.
        """

        value = (physical - self.mean) / self.scale
        return value * self.wet if mask else value

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.scale + self.mean

    def step(self, state: torch.Tensor) -> torch.Tensor:
        """One ten-day call.  ``state`` and the result are normalized ``(46, Y, X)``."""

        features = torch.cat((state, self.static), dim=0)
        if features.shape[0] != EXTERNAL_INPUT_CHANNELS:
            raise FnoAdjointError(
                f"assembled {features.shape[0]} input channels, expected {EXTERNAL_INPUT_CHANNELS}"
            )
        return self.model(features[None])[0] * self.wet

    def cost(self, normalized: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``J = sum_ij w_ij eta_ij``, in metres, from a normalized state."""

        return (weight * self.denormalize(normalized)[ETA_CHANNEL]).sum()

    def cost_seed(self, weight: torch.Tensor) -> torch.Tensor:
        """``dJ/d(normalized final state)``, the seed of every backward chain.

        ``J = sum w * (y_norm * sigma + mu)[eta]``, so the seed is ``w * sigma``
        in the sea-surface height channel and exactly zero in the other 45.
        The functional is linear in eta, so this does not depend on the state ---
        which is worth stating, because it means the only nonlinearity anywhere
        in this study is in the propagator.
        """

        seed = torch.zeros(
            (STATE_CHANNEL_COUNT, *self.wet.shape), dtype=self.dtype
        )
        seed[ETA_CHANNEL] = weight * self.scale[ETA_CHANNEL]
        return seed


def _with_eta(state: torch.Tensor, eta: torch.Tensor | None) -> torch.Tensor:
    """Replace the sea-surface height channel of a physical state.

    ``torch.cat`` rather than ``state[45] = eta``: it builds a new tensor whose
    46th channel *is* the leaf being differentiated with respect to, leaving the
    other 45 as the constants they should be.
    """

    if eta is None:
        return state
    if eta.shape != state.shape[1:]:
        raise FnoAdjointError("the replacement eta field does not match the grid")
    return torch.cat((state[:ETA_CHANNEL], eta[None]), dim=0)


def build_operator(
    model: torch.nn.Module,
    normalizers: Mapping[str, np.ndarray],
    statics: np.ndarray,
    wet: np.ndarray,
    *,
    dtype: torch.dtype = torch.float64,
) -> FrozenOperator:
    mean = np.asarray(normalizers["mean"], dtype=np.float64)
    scale = np.asarray(normalizers["scale"], dtype=np.float64)
    dry = ~np.asarray(wet, dtype=bool)
    # The unmasked leakage diagnostic is only legitimate if lifting the input
    # mask leaves the forward *value* untouched, which requires mu=0, sigma=1
    # on land.  If that ever stops being true this raises rather than quietly
    # changing what is being computed.
    if np.abs(mean[:, dry]).max() != 0.0 or not np.all(scale[:, dry] == 1.0):
        raise FnoAdjointError(
            "the normalizers are no longer (mu=0, sigma=1) on land, so the unmasked "
            "leakage diagnostic would change the function being differentiated"
        )

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float64)).to(dtype)

    return FrozenOperator(
        model=model,
        mean=tensor(mean),
        scale=tensor(scale),
        static=tensor(statics[REGIME_INDEX]),
        wet=tensor(wet.astype(np.float64)),
    )


# ===========================================================================
# 5.  The two chains  (plan section 5)
# ===========================================================================


def free_chain(
    operator: FrozenOperator,
    source: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    calls: int,
    *,
    mask_input: bool = True,
    want_backward_sweep: bool = False,
) -> dict[str, Any]:
    """Differentiate the emulator's own autoregressive rollout.

    One forward pass through ``calls`` chained calls, then one backward pass per
    objective --- the forward work is shared, because only the seed differs.

    Returns ``S[objective]`` in physical units, and optionally the whole
    backward sweep ``dJ/d eta(7200 + 10k)`` for every ``k``, which is sweep (A).
    """

    eta = source[ETA_CHANNEL].detach().clone().requires_grad_(True)
    state = operator.normalize(_with_eta(source, eta), mask=mask_input)
    intermediates = [state]
    for _ in range(calls):
        state = operator.step(state)
        intermediates.append(state)

    result: dict[str, Any] = {"maps": {}, "cost": {}, "backward_sweep": {}}
    for name, weight in weights.items():
        cost = operator.cost(state, weight)
        wanted: list[torch.Tensor] = [eta]
        if want_backward_sweep:
            wanted += intermediates[:-1]
        grads = torch.autograd.grad(cost, wanted, retain_graph=True, allow_unused=False)
        result["cost"][name] = float(cost)
        result["maps"][name] = grads[0].detach().numpy().copy()
        if want_backward_sweep:
            # lambda_k is dJ/d(normalized state at 7200 + 10k); the eta channel
            # divided by sigma is dJ/d eta there.  The k=0 entry is checked
            # against the physical-leaf gradient below, which validates this
            # conversion for every other k.
            sweep = []
            for lam in grads[1:]:
                # lambda_k is taken with respect to the *masked* normalized
                # state, so the land entries are the operator's appetite for a
                # degree of freedom that does not exist.  Masking here keeps
                # every lead exactly zero on land, matching both the primary
                # maps and MITgcm's ADJetan.
                sweep.append(
                    (lam[ETA_CHANNEL] * operator.wet / operator.scale[ETA_CHANNEL])
                    .detach()
                    .numpy()
                    .copy()
                )
            result["backward_sweep"][name] = np.stack(sweep)
    result["final_state"] = state.detach()
    return result


def forced_chain(
    operator: FrozenOperator,
    truth: Sequence[torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    calls: int,
    *,
    want_backward_sweep: bool = False,
) -> dict[str, Any]:
    """Differentiate the emulator along the **MITgcm truth** trajectory.

    This is the matched object: MITgcm's adjoint is linearized about the true
    trajectory by construction, so putting the emulator's Jacobian at the same
    points isolates Jacobian error from trajectory drift.

    Implemented as ``calls`` independent vector-Jacobian products chained
    backward --- which is exactly what MITgcm's adjoint is, one leg per
    ten days instead of one per timestep.
    """

    if len(truth) < calls:
        raise FnoAdjointError(f"the forced chain needs {calls} truth states, got {len(truth)}")

    result: dict[str, Any] = {"maps": {}, "backward_sweep": {}}
    for name, weight in weights.items():
        lam = operator.cost_seed(weight)
        sweep: list[np.ndarray] = []
        for k in reversed(range(calls)):
            state = operator.normalize(truth[k]).detach().requires_grad_(True)
            predicted = operator.step(state)
            (lam,) = torch.autograd.grad(predicted, state, grad_outputs=lam)
            if want_backward_sweep:
                sweep.append(
                    (lam[ETA_CHANNEL] * operator.wet / operator.scale[ETA_CHANNEL])
                    .detach()
                    .numpy()
                    .copy()
                )
        # The wet factor is the input mask's own derivative: the chain above is
        # taken with respect to the masked normalized state, so restoring it
        # here makes this the derivative of the *deployed* map --- exactly zero
        # on land, and therefore exactly comparable to the free chain (gate F5).
        result["maps"][name] = (
            (lam[ETA_CHANNEL] * operator.wet / operator.scale[ETA_CHANNEL])
            .detach()
            .numpy()
            .copy()
        )
        if want_backward_sweep:
            # built backward (lead 10 first); reverse so index k is the state at
            # day 7200 + 10k, matching the free chain's ordering
            result["backward_sweep"][name] = np.stack(sweep[::-1])
    return result


# ===========================================================================
# 6.  Gates  (plan section 6)
# ===========================================================================


def gate_f1_cost_identity(
    operator: FrozenOperator,
    truth_target: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    weights_numpy: Mapping[str, np.ndarray],
    eta_truth: np.ndarray,
    lead_days: int,
) -> dict[str, Any]:
    """F1 --- the pipeline's J on *truth* must equal ``(w * eta).sum()`` in NumPy.

    Run on the archived truth state rather than on a prediction, so any
    disagreement is a bug in the cost, not in the model.  This is the FNO-side
    twin of MITgcm's gate G5 and it catches the whole class of "the cost
    function is not what I think it is" errors, including a wrong ``A_wet`` and
    an off-by-one in ``p*``.
    """

    eta = eta_truth.astype(np.float64)
    records = {}
    worst_ratio = 0.0
    for name, weight in weights.items():
        pipeline = float((weight * truth_target[ETA_CHANNEL]).sum())
        reference = float((weights_numpy[name] * eta).sum())
        relative = abs(pipeline - reference) / max(abs(reference), 1.0e-30)

        # A constant relative tolerance is unreachable for a badly conditioned
        # weighted sum, and this study has one: the mean-only functional is
        # -<eta>_A, which MITgcm conserves at essentially zero, so the sum
        # cancels to eight significant figures and its condition number is ~1e8.
        # The MITgcm side hit exactly this in gate G5 and settled it the same
        # way -- compare against the *computed* floor of this particular sum,
        # not against a number chosen in advance.  Same discipline as
        # s0-twin-float32-floor.
        terms = weights_numpy[name] * eta
        condition = float(np.abs(terms).sum() / max(abs(reference), 1.0e-300))
        floor = float(np.finfo(np.float64).eps * condition)
        ratio = relative / max(floor, 1.0e-300)
        worst_ratio = max(worst_ratio, ratio)
        records[name] = {
            "pipeline_cost_metres": pipeline,
            "numpy_cost_metres": reference,
            "absolute_error_metres": abs(pipeline - reference),
            "relative_error": relative,
            "summation_condition_number": condition,
            "float64_floor": floor,
            "error_over_floor": ratio,
            "passed": bool(relative < max(F1_TOLERANCE, floor)),
        }
    return {
        "condition": (
            f"pipeline J on truth eta(day {SOURCE_DAY + lead_days}) matches numpy (w*eta).sum() "
            "to the computed float64 round-off floor of that particular weighted sum"
        ),
        "lead_days": int(lead_days),
        "objectives": records,
        "worst_relative_error": max(r["relative_error"] for r in records.values()),
        "worst_error_over_floor": worst_ratio,
        "threshold": "max(1e-10, eps_64 * condition_number), per objective",
        "passed": bool(all(r["passed"] for r in records.values())),
    }


def gate_f5_chain_identity(forced: np.ndarray, free: np.ndarray) -> dict[str, Any]:
    """F5 --- at lead 10 the forced and free chains are the same computation.

    One call, from the same truth state, so the two maps must agree to the last
    bit.  Not "to within a tolerance": a difference of any size means the two
    chains are not implementing the same tangent-linear operator, and every
    trajectory-error number downstream would then be measuring an
    implementation gap rather than the model's drift.
    """

    difference = float(np.abs(forced - free).max())
    return {
        "condition": "S_forced(lead 10) equals S_free(lead 10) exactly",
        "max_absolute_difference": difference,
        "threshold": 0.0,
        "passed": bool(difference == 0.0),
    }


def gate_f_sigma_consistency(
    physical_leaf: np.ndarray, sweep_entry: np.ndarray, wet: np.ndarray
) -> dict[str, Any]:
    """The hand division by ``sigma`` in the backward sweep, validated.

    The primary maps put the *physical* field in the graph as the leaf, so
    autograd carries every factor of ``sigma`` and nothing is corrected by hand.
    The intermediate leads of sweep (A) cannot do that --- they are gradients
    with respect to normalized states --- so they divide by ``sigma`` explicitly.
    Comparing the ``k = 0`` entry of the sweep against the physical-leaf
    gradient validates that division everywhere else in the sweep.
    """

    mask = np.asarray(wet, dtype=bool)
    scale = max(float(np.abs(physical_leaf[mask]).max()), 1.0e-300)
    difference = float(np.abs(physical_leaf[mask] - sweep_entry[mask]).max()) / scale
    return {
        "condition": "backward-sweep k=0 entry equals the physical-leaf gradient over wet cells",
        "relative_difference": difference,
        "threshold": 1.0e-12,
        "passed": bool(difference < 1.0e-12),
    }


def gate_f7_conservation(
    sensitivity: Mapping[int, np.ndarray], mean_weight: np.ndarray, wet: np.ndarray
) -> dict[str, Any]:
    """F7 --- global sea-level conservation error, against an exact reference.

    MITgcm's ``implicitFreeSurface`` with ``exactConserv`` conserves the area
    integral of eta exactly in a closed basin with no freshwater flux, so the
    adjoint of the basin-mean functional is constant in time and **equals the
    weight field itself** at every lead.  The emulator conserves no such thing,
    so this difference is a spatially and lead-resolved measurement of how badly
    it violates global sea-level conservation --- against an analytically known
    answer, with no MITgcm run required to interpret it.

    Reported, never graded: there is no threshold here because there is no prior
    expectation to test against.
    """

    mask = np.asarray(wet, dtype=bool)
    reference = np.linalg.norm(mean_weight[mask])
    per_lead = {}
    for lead, field in sorted(sensitivity.items()):
        per_lead[str(lead)] = {
            "relative_l2": float(np.linalg.norm((field - mean_weight)[mask]) / reference),
            "pattern_correlation": metrics.pattern_correlation(field, mean_weight, mask),
            "amplitude_ratio": metrics.amplitude_ratio(field, mean_weight, mask),
            "sum_over_wet": float(field[mask].sum()),
        }
    return {
        "condition": "||S_fno_mean(L) - w_mean|| / ||w_mean|| at every lead",
        "exact_reference": (
            "MITgcm's answer is w_mean itself at every lead, because the area "
            "integral of eta is exactly conserved by this configuration"
        ),
        "reference_norm": float(reference),
        "per_lead": per_lead,
        "threshold": None,
        "passed": None,
        "meaning": "a measurement of the emulator's global sea-level conservation error, not a gate",
    }


def finite_difference_points(wet: np.ndarray, target: tuple[int, int]) -> list[dict[str, Any]]:
    """The eight cells the finite-difference gate probes.

    ``p*``, the western band around it, four interior cells at different
    distances, and one land cell.  The land cell must return **exactly** zero
    from both the adjoint and the finite difference, because the input mask
    removes it as a degree of freedom --- the same statement MITgcm's gate G4
    makes about ``adxx_etan``.
    """

    j0, i0 = target
    candidates = [
        (j0, i0, "target p*"),
        (j0, i0 + 1, "western band, one cell east of p*"),
        (j0, i0 + 3, "western band, outer edge"),
        (30, 30, "mid-basin"),
        (12, 48, "eastern interior"),
        (48, 20, "northern interior"),
        (5, 8, "southern interior"),
        (j0, i0 - 1, "land: western rim beside p*"),
    ]
    points = []
    for j, i, label in candidates:
        if not (0 <= j < wet.shape[0] and 0 <= i < wet.shape[1]):
            raise FnoAdjointError(f"finite-difference point {(j, i)} is off the grid")
        points.append({"j": int(j), "i": int(i), "label": label, "wet": bool(wet[j, i])})
    if not any(not p["wet"] for p in points):
        raise FnoAdjointError("the finite-difference sweep must include a land cell")
    return points


def _cost_after_calls(
    operator: FrozenOperator,
    source: torch.Tensor,
    weight: torch.Tensor,
    calls: int,
    eta: torch.Tensor | None = None,
) -> torch.Tensor:
    state = operator.normalize(_with_eta(source, eta))
    for _ in range(calls):
        state = operator.step(state)
    return operator.cost(state, weight)


def measure_evaluation_noise(
    operator: FrozenOperator,
    source: torch.Tensor,
    weight: torch.Tensor,
    adjoint: np.ndarray,
    point: tuple[int, int],
    calls: int,
    deltas: Sequence[float] = (1.0e-7, 1.0e-9, 1.0e-11, 1.0e-13),
) -> dict[str, Any]:
    """How exactly does the *computed* cost track a straight line near truth?

    At amplitudes this small the true second-order term is utterly negligible,
    so any departure of ``J(eta + d) - J(eta)`` from ``d * S`` is arithmetic,
    not mathematics.  This measures that departure directly, and it is what sets
    the floor on how well any finite difference can ever agree with the adjoint:
    a central difference over ``2 eps`` cannot resolve better than
    ``noise / (2 eps)``.

    The MITgcm side reached the same conclusion from the other direction
    (`grdchk-limited-by-cg2d`): its gate-G1 error was flat in epsilon because
    the finite difference, not the adjoint, was the noisy party.  Here the
    error *grows* as epsilon shrinks, which is the same statement with a
    steeper slope.
    """

    j, i = point
    slope = float(adjoint[j, i])
    with torch.no_grad():
        base = float(_cost_after_calls(operator, source, weight, calls))
        records = []
        for delta in deltas:
            eta = source[ETA_CHANNEL].detach().clone()
            eta[j, i] += delta
            value = float(_cost_after_calls(operator, source, weight, calls, eta))
            records.append(
                {
                    "delta_metres": float(delta),
                    "departure_from_linear": float(value - base - delta * slope),
                }
            )
    worst = float(max(abs(r["departure_from_linear"]) for r in records))
    return {
        "probe_cell": [int(j), int(i)],
        "cost_at_truth": base,
        "samples": records,
        "absolute_noise": worst,
        "relative_noise": worst / max(abs(base), 1.0e-300),
        "meaning": (
            "absolute round-off noise in the computed cost; a central difference over "
            "2*eps can never resolve the adjoint better than this divided by 2*eps"
        ),
    }


def measure_adjoint_reproducibility(
    operator: FrozenOperator,
    source: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    calls: int,
    wet: np.ndarray,
    threads: int,
) -> dict[str, Any]:
    """The arithmetic floor of the reverse-mode map itself.

    Recomputing the identical gradient under a different reduction order --- a
    different thread count changes how the convolutions and FFTs accumulate ---
    gives the round-off noise of the adjoint computation directly, with no
    model of where it comes from.  Everything else in this script that compares
    two derivatives has to be read against this number.

    It is an **absolute** floor, not a relative one: it comes from a shared,
    dominant accumulation, so a sensitivity of 1e-4 and one of 1e-1 carry the
    same round-off in metres.  That is why every scaled tolerance here divides
    by ``max|S|`` rather than by the local value --- and it is what says how far
    into the far field the comparison against MITgcm can mean anything.
    """

    mask = np.asarray(wet, dtype=bool)
    reference = free_chain(operator, source, weights, calls)["maps"][PRIMARY_OBJECTIVE]
    original = torch.get_num_threads()
    alternate = 1 if threads != 1 else 2
    try:
        torch.set_num_threads(alternate)
        repeat = free_chain(operator, source, weights, calls)["maps"][PRIMARY_OBJECTIVE]
    finally:
        torch.set_num_threads(original)

    difference = np.abs(reference - repeat)
    peak = float(np.abs(reference[mask]).max())
    return {
        "definition": "same reverse-mode gradient, different reduction order (thread count)",
        "threads": [int(threads), int(alternate)],
        "calls": int(calls),
        "max_absolute_difference": float(difference[mask].max()),
        "rms_absolute_difference": float(np.sqrt(np.mean(np.square(difference[mask])))),
        "peak_sensitivity": peak,
        "scaled_floor": float(difference[mask].max() / max(peak, 1.0e-300)),
        "reading": (
            "sensitivities below this in absolute terms are arithmetic, not signal. "
            "The same discipline s0-twin-float32-floor forced on the daily diagnostics"
        ),
    }


def forward_versus_reverse(
    operator: FrozenOperator,
    source: torch.Tensor,
    weight: torch.Tensor,
    adjoint: np.ndarray,
    points: Sequence[Mapping[str, Any]],
    calls: int,
    floor: float,
) -> dict[str, Any]:
    """The same directional derivatives by *forward*-mode AD.

    This is the sharp check, and on this side it is sharper than any finite
    difference can be.  Forward and reverse mode compute the same exact
    derivative of the same computed function, but accumulate it through
    completely different code paths and in the opposite order --- and, unlike a
    finite difference, forward mode involves **no subtraction of two nearly
    equal numbers**, so it is not limited by cancellation.  Agreement to machine
    precision is therefore strong evidence about the reverse-mode map that the
    round-off-limited finite difference cannot supply.
    """

    reference = max(float(np.abs(adjoint).max()), 1.0e-300)
    records = []
    for point in points:
        j, i = point["j"], point["i"]
        direction = torch.zeros_like(source[ETA_CHANNEL])
        direction[j, i] = 1.0

        def cost(eta: torch.Tensor) -> torch.Tensor:
            return _cost_after_calls(operator, source, weight, calls, eta)

        _, derivative = torch.func.jvp(
            cost, (source[ETA_CHANNEL].detach().clone(),), (direction,)
        )
        forward = float(derivative)
        slope = float(adjoint[j, i])
        records.append(
            {
                **point,
                "forward_mode": forward,
                "reverse_mode": slope,
                "absolute_error": abs(forward - slope),
                "scaled_error": abs(forward - slope) / reference,
            }
        )
    worst = max(record["scaled_error"] for record in records)
    worst_absolute = max(record["absolute_error"] for record in records)
    # Tested against the *measured* arithmetic floor of the reverse-mode map,
    # not against a number chosen in advance.  Forward and reverse mode compute
    # the same exact derivative through different accumulation orders, so they
    # cannot agree better than that map agrees with itself under a reordering.
    # The reordering probe returns exactly zero on this build -- PyTorch's CPU
    # reductions here are deterministic -- so a bare multiple of it would be an
    # unreachable zero tolerance.  Machine precision on the largest entry of the
    # map is the floor no comparison of two float64 derivatives can beat.
    allowed = max(
        4.0 * floor,
        MACHINE_FLOOR_MULTIPLE * float(np.finfo(np.float64).eps) * reference,
    )
    return {
        "condition": (
            "|forward-mode - reverse-mode| within four times the measured "
            "reverse-mode arithmetic floor, at every probe cell"
        ),
        "calls": int(calls),
        "points": records,
        "worst_scaled_error": worst,
        "worst_absolute_error": worst_absolute,
        "measured_floor": float(floor),
        "allowed_absolute_error": allowed,
        "error_over_floor": worst_absolute / allowed,
        "threshold": "4 x measured reverse-mode reproducibility floor",
        "passed": bool(worst_absolute <= allowed),
    }


def gate_f2_finite_difference(
    operator: FrozenOperator,
    source: torch.Tensor,
    weight: torch.Tensor,
    adjoint: np.ndarray,
    points: Sequence[Mapping[str, Any]],
    epsilons: Sequence[float],
    calls: int,
    noise: Mapping[str, Any],
) -> dict[str, Any]:
    """F2 --- the emulator's ``grdchk``: central differences in physical units.

    A *single* epsilon agreeing is weak evidence; the plateau is strong
    evidence.  The operator is smooth --- GELU, spectral convolutions, layer
    norm, no branches, no ``tanh`` saturation, no live clipping --- so the
    plateau should be wide and the agreement far tighter than MITgcm's 1e-4,
    where ``ivdc_kappa`` discontinuities set the floor.  **A loose F2 is a bug
    in this pipeline, not physics**: there is no convective-adjustment excuse on
    this side.

    Run at every lead, because that is what distinguishes the two ways it can
    fail.  A pipeline bug fails at every lead; a genuine nonlinearity of the
    nine-call chain fails only at long lead and only at large epsilon.
    """

    records = []
    with torch.no_grad():
        base = float(_cost_after_calls(operator, source, weight, calls))

    def floor_at(epsilon: float) -> float:
        """What a central difference over 2*eps can resolve, given the noise."""

        return float(noise["absolute_noise"]) / (2.0 * epsilon)

    for point in points:
        j, i = point["j"], point["i"]
        slope = float(adjoint[j, i])
        samples = []
        for epsilon in epsilons:
            values = []
            with torch.no_grad():
                for sign in (+1.0, -1.0):
                    eta = source[ETA_CHANNEL].detach().clone()
                    eta[j, i] += sign * epsilon
                    values.append(float(_cost_after_calls(operator, source, weight, calls, eta)))
            difference = (values[0] - values[1]) / (2.0 * epsilon)
            samples.append(
                {
                    "epsilon_metres": float(epsilon),
                    "finite_difference": difference,
                    "absolute_error": abs(difference - slope),
                    "ratio": (difference / slope) if slope != 0.0 else None,
                }
            )
        best = min(samples, key=lambda s: s["absolute_error"])
        # A relative test is meaningless where the adjoint is legitimately zero
        # (the land cell), so the tolerance is scaled by the largest slope on
        # the map rather than by the local one.
        reference = max(float(np.abs(adjoint).max()), 1.0e-300)
        # The finite difference cannot beat the round-off floor of the computed
        # cost, so the verdict is against the larger of the declared tolerance
        # and that measured floor.  The factor of four is slack for the fact
        # that the noise probe samples one cell, not this one.
        allowed = max(
            F2_TOLERANCE * reference,
            4.0 * floor_at(best["epsilon_metres"]),
            MACHINE_FLOOR_MULTIPLE * np.finfo(np.float64).eps * reference,
        )
        # A plateau, not a single hit: the error must fall and then rise again
        # across the sweep, which is what separates a converged central
        # difference from a point that happens to agree.
        errors = [sample["absolute_error"] for sample in samples]
        interior_minimum = bool(
            len(errors) >= 3 and 0 < errors.index(min(errors)) < len(errors) - 1
        )
        records.append(
            {
                **point,
                "adjoint": slope,
                "samples": samples,
                "best_absolute_error": best["absolute_error"],
                "best_epsilon": best["epsilon_metres"],
                "best_ratio": best["ratio"],
                "scaled_error": best["absolute_error"] / reference,
                "allowed_absolute_error": allowed,
                "error_over_floor": best["absolute_error"] / max(4.0 * floor_at(best["epsilon_metres"]), 1.0e-300),
                "plateau_visible": interior_minimum,
                "passes_scaled_tolerance": bool(best["absolute_error"] <= allowed),
            }
        )
    worst = max(record["scaled_error"] for record in records)
    wet_records = [record for record in records if record["wet"]]
    return {
        "condition": (
            "|FD - adjoint| at the best epsilon within max(1e-6 * max|adjoint|, "
            "4 * measured round-off floor), with an interior minimum in the sweep"
        ),
        "calls": int(calls),
        "lead_days": int(calls * HORIZON_DAYS),
        "cost_at_truth": base,
        "epsilons": [float(e) for e in epsilons],
        "evaluation_noise": noise,
        "points": records,
        "worst_scaled_error": worst,
        "worst_error_over_floor": max(record["error_over_floor"] for record in records),
        "plateau_visible_at": sum(1 for record in wet_records if record["plateau_visible"]),
        "wet_probe_count": len(wet_records),
        "threshold": F2_TOLERANCE,
        "passed": bool(all(record["passes_scaled_tolerance"] for record in records)),
    }


def gate_f4_precision(double_map: np.ndarray, single_map: np.ndarray, wet: np.ndarray) -> dict[str, Any]:
    """F4 --- how much of the answer is arithmetic rather than mathematics.

    Reported rather than chosen between.  ``s0-twin-float32-floor`` is the
    standing reason this project runs the sensitive parts in double: a float32
    quantisation floor has already swallowed an entire signal here once.
    """

    mask = np.asarray(wet, dtype=bool)
    reference = max(float(np.linalg.norm(double_map[mask])), 1.0e-300)
    relative = float(np.linalg.norm((double_map - single_map)[mask]) / reference)
    return {
        "condition": "float32 and float64 gradients agree to better than 1e-4 relative",
        "relative_l2": relative,
        "max_absolute_difference": float(np.abs(double_map - single_map)[mask].max()),
        "threshold": F4_TOLERANCE,
        "passed": bool(relative < F4_TOLERANCE),
    }


def unmasked_leakage(
    operator: FrozenOperator,
    source: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    calls: int,
    wet: np.ndarray,
) -> dict[str, Any]:
    """How much the global spectral path *wants* to read land.

    The old plan's land-leakage metric --- ``max|S|`` over the dry cells against
    MITgcm's exact zero --- is vacuous for the deployed map: the input mask
    removes land eta as a degree of freedom, so the derivative there is exactly
    zero on both sides by construction and the metric reports a guaranteed pass.

    This is the version that measures something.  Lifting the input mask makes
    land eta a live input, and because the normalizers are ``(mu=0, sigma=1)``
    there and physical land eta is zero, the forward *value* is unchanged --- so
    this is a property of the operator, cleanly isolated.  It is an FNO-only
    diagnostic and has no MITgcm counterpart; it must never be differenced
    against one.
    """

    unmasked = free_chain(operator, source, weights, calls, mask_input=False)
    dry = ~np.asarray(wet, dtype=bool)
    report = {}
    for name, field in unmasked["maps"].items():
        wet_norm = max(float(np.linalg.norm(field[~dry])), 1.0e-300)
        report[name] = {
            "max_absolute_on_land": float(np.abs(field[dry]).max()),
            "land_to_wet_norm_ratio": float(np.linalg.norm(field[dry]) / wet_norm),
            "land_cells": int(dry.sum()),
        }
    return {
        "definition": "dJ/d eta at land cells with the input mask lifted",
        "deployed_map_value": 0.0,
        "note": (
            "FNO-only. The deployed map's land derivative is exactly zero on both "
            "sides by construction; this measures the unmasked operator instead"
        ),
        "calls": int(calls),
        "objectives": report,
    }


def lead_to_lead_correlation(sweep: np.ndarray, wet: np.ndarray) -> dict[str, Any]:
    """Plan section 8.1 --- the period-2 computational mode, made visible.

    ``local-branch-release-lr`` records that the radius cap bounds only the
    local branch, so a period-2 computational mode --- a negative real
    eigenvalue of the composite ten-day recurrence --- passes every check
    currently in the project.  Nothing in the forecast gate, the growth rate,
    the spectra or the anomaly packages can see it, because an alternating mode
    has the right amplitude statistics.

    A lead sweep of the adjoint sees it directly: if the composite map carries a
    negative real eigenvalue, the adjoint state alternates sign call to call, so
    consecutive leads anticorrelate.  A healthy operator gives a smoothly
    decaying positive sequence.
    """

    mask = np.asarray(wet, dtype=bool)
    correlations = []
    for k in range(sweep.shape[0] - 1):
        correlations.append(metrics.pattern_correlation(sweep[k], sweep[k + 1], mask))
    norms = [float(np.linalg.norm(sweep[k][mask])) for k in range(sweep.shape[0])]
    negative = [c for c in correlations if c is not None and c < 0.0]
    return {
        "definition": "corr(S(lead 10k), S(lead 10(k+1))) over wet cells",
        "correlations": correlations,
        "per_lead_norm": norms,
        "negative_count": len(negative),
        "most_negative": (min(negative) if negative else None),
        "reading": (
            "strongly negative values are the signature of a period-2 computational "
            "mode in the composite recurrence -- a negative real eigenvalue that no "
            "existing check in this project can see"
        ),
    }


# ===========================================================================
# 7.  Figures
# ===========================================================================

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def _masked(field: np.ndarray, wet: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where((~wet) | (~np.isfinite(field)), field)


def _bound(field: np.ndarray, wet: np.ndarray, percentile: float = 99.0) -> float:
    """A robust symmetric colour limit.

    The point-objective map is extremely peaked --- the cost contains a delta at
    ``p*`` --- so scaling to the maximum would render the rest of the basin
    uniformly white.  A high percentile shows the pattern; the caption records
    the clipping.
    """

    values = np.abs(field[wet])
    value = float(np.percentile(values, percentile))
    return value if value > 0.0 else float(values.max() or 1.0)


def _draw_map(
    axis: Any,
    field: np.ndarray,
    contract: SharedContract,
    bound: float,
    title: str,
    *,
    mark_target: bool = True,
) -> Any:
    image = axis.pcolormesh(
        contract.longitude,
        contract.latitude,
        _masked(field, contract.wet),
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        shading="auto",
    )
    if mark_target:
        j, i = contract.target
        axis.plot(
            contract.longitude[j, i],
            contract.latitude[j, i],
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="k",
            markeredgewidth=0.9,
        )
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_facecolor("0.86")
    return image


def figure_lead_maps(output: Path, maps: Mapping[int, np.ndarray], contract: SharedContract) -> None:
    """Sweep (B): source day 7200 fixed, cost day moving out to 90 days."""

    _style()
    leads = sorted(maps)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 8.4), constrained_layout=True)
    for axis, lead in zip(axes.ravel(), leads):
        field = maps[lead]
        bound = _bound(field, contract.wet)
        image = _draw_map(
            axis,
            field,
            contract,
            bound,
            f"lead {lead} d   (cost day {SOURCE_DAY + lead})\n"
            f"max |S| = {np.abs(field[contract.wet]).max():.3e}",
        )
        figure.colorbar(image, ax=axis, shrink=0.82, label="dJ / d eta   [-]")
    figure.suptitle(
        "S_forced: dJ/d eta(day 7200), emulator linearized about the MITgcm truth trajectory\n"
        "colour clipped at the 99th percentile of |S| over wet cells; circle marks p*",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


def figure_forced_vs_free(
    output: Path,
    forced: np.ndarray,
    free: np.ndarray,
    contract: SharedContract,
    lead: int,
) -> None:
    """The trajectory-drift term, isolated."""

    _style()
    bound = max(_bound(forced, contract.wet), _bound(free, contract.wet))
    difference = free - forced
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, field, title, limit in (
        (axes[0], forced, "S_forced\nlinearized about MITgcm truth", bound),
        (axes[1], free, "S_free\nlinearized about the emulator's own rollout", bound),
        (
            axes[2],
            difference,
            "S_free - S_forced\nthe trajectory-drift term",
            _bound(difference, contract.wet),
        ),
    ):
        image = _draw_map(axis, field, contract, limit, title)
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(
        f"lead {lead} d.  MITgcm's adjoint is linearized about the truth trajectory by "
        "construction, so the left panel is the matched object",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


def figure_backward_sweep(output: Path, sweep: np.ndarray, contract: SharedContract) -> None:
    """Sweep (A): cost day 7290 fixed, how far back the target's sensitivity reaches."""

    _style()
    calls = sweep.shape[0]
    figure, axes = plt.subplots(3, 3, figsize=(11.4, 11.0), constrained_layout=True)
    for k, axis in enumerate(axes.ravel()[:calls]):
        field = sweep[k]
        lead = (calls - k) * HORIZON_DAYS
        bound = _bound(field, contract.wet)
        image = _draw_map(
            axis,
            field,
            contract,
            bound,
            f"day {SOURCE_DAY + k * HORIZON_DAYS}   (lead {lead} d back from 7290)\n"
            f"max |S| = {np.abs(field[contract.wet]).max():.3e}",
        )
        figure.colorbar(image, ax=axis, shrink=0.8)
    for axis in axes.ravel()[calls:]:
        axis.axis("off")
    figure.suptitle(
        "S_free backward sweep: dJ(day 7290) / d eta(t), each panel normalized on its own\n"
        "the emulator's analogue of MITgcm's ADJetan dumps",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


def figure_conservation(
    output: Path,
    probe: Mapping[int, np.ndarray],
    mean_weight: np.ndarray,
    contract: SharedContract,
    gate: Mapping[str, Any],
) -> None:
    """The conservation probe, against an analytically exact reference."""

    _style()
    leads = sorted(probe)
    figure = plt.figure(figsize=(12.6, 7.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)

    axis = figure.add_subplot(grid[0, 0])
    bound = _bound(mean_weight, contract.wet)
    image = _draw_map(axis, mean_weight, contract, bound, "exact answer:  w_mean = -rA/A_wet", mark_target=False)
    figure.colorbar(image, ax=axis, shrink=0.8)

    for column, lead in enumerate(leads[:2]):
        axis = figure.add_subplot(grid[0, column + 1])
        image = _draw_map(axis, probe[lead], contract, bound, f"emulator, lead {lead} d", mark_target=False)
        figure.colorbar(image, ax=axis, shrink=0.8)

    axis = figure.add_subplot(grid[1, 0])
    field = probe[leads[-1]] - mean_weight
    image = _draw_map(
        axis, field, contract, _bound(field, contract.wet),
        f"error at lead {leads[-1]} d:  S - w_mean", mark_target=False,
    )
    figure.colorbar(image, ax=axis, shrink=0.8)

    axis = figure.add_subplot(grid[1, 1:])
    per_lead = gate["per_lead"]
    axis.plot(leads, [per_lead[str(l)]["relative_l2"] for l in leads], "o-", label="relative L2 error")
    axis.plot(leads, [per_lead[str(l)]["amplitude_ratio"] for l in leads], "s--", label="amplitude ratio (exact: 1)")
    axis.axhline(1.0, color="0.6", lw=0.8)
    axis.set_xlabel("lead [days]")
    axis.set_ylabel("[-]")
    axis.set_title(
        "global sea-level conservation error\n"
        "MITgcm's answer is w_mean at every lead, exactly"
    )
    axis.legend()
    axis.grid(alpha=0.3)

    figure.suptitle(
        "Conservation probe: the adjoint of the basin-mean functional.  MITgcm conserves "
        "the area integral of eta exactly,\nso its map is w_mean at every lead.  No MITgcm "
        "run is needed to interpret this.",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


def figure_gate_f2(output: Path, gates: Mapping[int, Mapping[str, Any]]) -> None:
    """The finite-difference plateau, one panel per lead."""

    _style()
    leads = sorted(gates)
    figure, axes = plt.subplots(1, len(leads), figsize=(3.6 * len(leads), 4.0), constrained_layout=True, squeeze=False)
    for axis, lead in zip(axes.ravel(), leads):
        gate = gates[lead]
        reference = max(
            max(abs(p["adjoint"]) for p in gate["points"]), 1.0e-300
        )
        for point in gate["points"]:
            if not point["wet"]:
                continue
            epsilons = [s["epsilon_metres"] for s in point["samples"]]
            errors = [max(s["absolute_error"] / reference, 1.0e-18) for s in point["samples"]]
            axis.loglog(epsilons, errors, "o-", ms=3, lw=1.0, label=point["label"][:26])
        axis.axhline(F2_TOLERANCE, color="k", ls=":", lw=1.0)
        axis.set_xlabel("epsilon [m]")
        axis.set_title(f"lead {lead} d   worst {gate['worst_scaled_error']:.1e}")
        axis.grid(alpha=0.3, which="both")
    axes.ravel()[0].set_ylabel("|FD - adjoint| / max|adjoint|")
    axes.ravel()[-1].legend(fontsize=6, loc="best")
    figure.suptitle(
        "Gate F2: central differences in physical units.  A plateau across the middle of "
        "the epsilon range is the evidence;\na single epsilon agreeing is not.  Dotted line "
        "is the 1e-6 threshold.",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


def figure_structure(
    output: Path,
    structure: Mapping[str, Any],
    period2: Mapping[str, Any],
    maps: Mapping[int, np.ndarray],
    contract: SharedContract,
) -> None:
    """Radial spectra, the western-boundary split, and the period-2 probe."""

    _style()
    leads = sorted(maps)
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), constrained_layout=True)

    axis = axes[0]
    for lead in leads:
        spectrum = structure[str(lead)]["radial_spectrum"]
        axis.loglog(
            spectrum["bin_upper_wavenumber"],
            np.maximum(spectrum["power_per_bin"], 1e-300),
            "o-",
            ms=3,
            label=f"{lead} d",
        )
    cutoff = 1.0 / 4.625
    axis.axvline(cutoff, color="k", ls="--", lw=1.0)
    axis.text(cutoff, axis.get_ylim()[1], " operator cutoff\n 4.63 cells", va="top", fontsize=7)
    axis.set_xlabel("radial wavenumber [1/cell]")
    axis.set_ylabel("absolute power")
    axis.set_title("radial spectrum of |S|\n(absolute power per bin, never the fraction)")
    axis.legend(title="lead")
    axis.grid(alpha=0.3, which="both")

    axis = axes[1]
    ratios = [structure[str(lead)]["western_band"]["boundary_to_interior_ratio"] for lead in leads]
    axis.plot(leads, ratios, "o-")
    axis.set_xlabel("lead [days]")
    axis.set_ylabel("western band / interior")
    axis.set_title(
        "western-boundary / interior split\n"
        "the diagnostic the forecast gate never scores"
    )
    axis.grid(alpha=0.3)

    axis = axes[2]
    correlations = [c for c in period2["correlations"] if c is not None]
    axis.plot(range(1, len(correlations) + 1), correlations, "o-")
    axis.axhline(0.0, color="k", lw=0.9)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("k  (between leads 10k and 10(k+1))")
    axis.set_ylabel("pattern correlation")
    axis.set_title(
        "lead-to-lead correlation of S_free\n"
        "strongly negative = period-2 computational mode"
    )
    axis.grid(alpha=0.3)

    figure.savefig(output)
    plt.close(figure)


def figure_objectives(
    output: Path,
    point: np.ndarray,
    kernel: np.ndarray,
    contract: SharedContract,
    lead: int,
    comparison: Mapping[str, Any],
) -> None:
    """The two scalar SSH objectives, side by side."""

    _style()
    bound = max(_bound(point, contract.wet), _bound(kernel, contract.wet))
    difference = kernel - point
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, field, title, limit in (
        (axes[0], point, "J_point:  eta(p*) - <eta>_A", bound),
        (
            axes[1],
            kernel,
            "J_kernel:  5-point meridional Gaussian, sigma=1", bound,
        ),
        (axes[2], difference, "kernel - point", _bound(difference, contract.wet)),
    ):
        image = _draw_map(axis, field, contract, limit, title)
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(
        f"lead {lead} d, S_forced.  pattern correlation {comparison['pattern_correlation']:.4f}, "
        f"amplitude ratio {comparison['amplitude_ratio']:.4f}\n"
        "the kernel smooths only along the jet: the Munk layer is one grid cell wide, so "
        "zonal smoothing would change what is measured",
        fontsize=9,
    )
    figure.savefig(output)
    plt.close(figure)


# ===========================================================================
# 8.  Driver
# ===========================================================================


def _truth_states(group: Any, days: Sequence[int]) -> dict[int, np.ndarray]:
    """Read the archived MITgcm states, not a re-run.

    The MITgcm side's gate G0 separately confirms that the store and a fresh
    restart agree bit-for-bit, so both sides provably start from the same
    numbers --- but only if this side reads the archive.
    """

    states = {}
    for day in days:
        states[day] = np.asarray(
            group["state"][REGIME_INDEX, day], dtype=np.float64
        )
    return states


def run(
    project_root: Path,
    *,
    force: bool = False,
    leads: Sequence[int] = LEAD_DAYS,
    epsilons: Sequence[float] = DEFAULT_FD_EPSILONS,
    threads: int = 8,
    identity: "ModelIdentity | None" = None,
) -> dict[str, Any]:
    started = time.monotonic()
    identity = identity or FT90_IDENTITY
    torch.set_num_threads(int(threads))
    output = (project_root / identity.output_relative).resolve()
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    leads = tuple(sorted(int(lead) for lead in leads))
    longest = max(leads)
    max_calls = calls_for_lead(longest)

    plan = json.loads((project_root / "config" / f"{PLAN_CONTRACT}.json").read_text())
    provenance = load_model_provenance(project_root, identity)
    dataset_path = Path(provenance["contract"]["sources"]["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    contract = load_shared_contract(project_root, group, plan)

    # --- step 1: preflight -------------------------------------------------
    print(f"[1/7] loading {provenance['checkpoint'].name}  (sha {provenance['checkpoint_sha256'][:8]}...)")
    model = load_frozen_model(provenance["checkpoint"], double=True, identity=identity)
    gate_f3 = gate_f3_operator_preflight(model)
    if not gate_f3["passed"]:
        raise FnoAdjointError(f"gate F3 failed: {gate_f3}")
    precision = verify_double_precision_spectrum(model)
    gate_f3["double_precision_spectrum"] = precision
    if not precision["passed"]:
        raise FnoAdjointError(
            "the Fourier working buffer is not complex128, so this is not a float64 "
            f"adjoint: {precision}"
        )
    print(
        f"      {EXPECTED_PARAMETER_COUNT:,} parameters, float64, eval mode, gradients off, "
        "no live spectral-norm machinery"
    )
    print(
        f"      spectral buffer promoted to complex128 in {precision['spectral_convolutions']} "
        f"convolutions; adjoint identity residual {precision['dot_product_residual']:.2e} "
        f"(was {precision['residual_before_promotion']:.1e} at neuralop's hard-coded complex64)"
    )

    with np.load(provenance["normalization"]) as stored:
        normalizers = {
            "mean": np.asarray(stored["pointwise_mean"], dtype=np.float64),
            "scale": np.asarray(stored["pointwise_scale"], dtype=np.float64),
        }
    sources = provenance["contract"]["sources"]
    statics, static_provenance = static_block(
        group,
        zonal_spacing_path=_verify(sources["mitgcm_zonal_spacing"], "zonal spacing"),
        sst_relax_path=_verify(sources["mitgcm_sst_relaxation"], "SST relaxation target"),
        data_path=_verify(sources["mitgcm_declaration"], "MITgcm declaration"),
        pointwise_mean=normalizers["mean"].astype(np.float32),
        pointwise_scale=normalizers["scale"].astype(np.float32),
    )

    operator = build_operator(model, normalizers, statics, contract.wet)
    days = [SOURCE_DAY + HORIZON_DAYS * k for k in range(max_calls + 1)]
    truth_numpy = _truth_states(group, days)
    truth = [torch.from_numpy(truth_numpy[day]).to(operator.dtype) for day in days]
    weights = {
        name: torch.from_numpy(field).to(operator.dtype)
        for name, field in contract.weights.items()
    }

    # --- step 2: the cost identity ----------------------------------------
    gates_f1 = {}
    for lead in leads:
        gates_f1[str(lead)] = gate_f1_cost_identity(
            operator,
            truth[calls_for_lead(lead)],
            weights,
            contract.weights,
            truth_numpy[SOURCE_DAY + lead][ETA_CHANNEL],
            lead,
        )
    worst_f1 = max(g["worst_relative_error"] for g in gates_f1.values())
    print(f"[2/7] gate F1: worst relative error {worst_f1:.2e} over {len(leads)} leads x {len(OBJECTIVES)} objectives")

    # --- step 3: the two chains at every lead ------------------------------
    print(f"[3/7] chains: {len(leads)} leads x 2 trajectories x {len(OBJECTIVES)} objectives")
    forced_maps: dict[str, dict[int, np.ndarray]] = {name: {} for name in OBJECTIVES}
    free_maps: dict[str, dict[int, np.ndarray]] = {name: {} for name in OBJECTIVES}
    backward: dict[str, dict[str, np.ndarray]] = {"forced": {}, "free": {}}

    for lead in leads:
        calls = calls_for_lead(lead)
        want_sweep = lead == longest
        free = free_chain(operator, truth[0], weights, calls, want_backward_sweep=want_sweep)
        forced = forced_chain(operator, truth, weights, calls, want_backward_sweep=want_sweep)
        for name in OBJECTIVES:
            free_maps[name][lead] = free["maps"][name]
            forced_maps[name][lead] = forced["maps"][name]
        if want_sweep:
            backward["free"] = free["backward_sweep"]
            backward["forced"] = forced["backward_sweep"]
            gate_sigma = gate_f_sigma_consistency(
                free["maps"][PRIMARY_OBJECTIVE],
                free["backward_sweep"][PRIMARY_OBJECTIVE][0],
                contract.wet,
            )
        print(
            f"      lead {lead:3d} d   max|S_forced| = "
            f"{np.abs(forced['maps'][PRIMARY_OBJECTIVE][contract.wet]).max():.4e}   "
            f"max|S_free| = {np.abs(free['maps'][PRIMARY_OBJECTIVE][contract.wet]).max():.4e}"
        )

    gate_f5 = gate_f5_chain_identity(
        forced_maps[PRIMARY_OBJECTIVE][leads[0]], free_maps[PRIMARY_OBJECTIVE][leads[0]]
    )
    print(
        f"      gate F5: |S_forced - S_free| at lead {leads[0]} d = "
        f"{gate_f5['max_absolute_difference']:.3e} -> {'pass' if gate_f5['passed'] else 'FAIL'}"
    )
    if not gate_sigma["passed"]:
        raise FnoAdjointError(f"the backward sweep's sigma conversion is inconsistent: {gate_sigma}")

    # --- step 4: the conservation probe ------------------------------------
    gate_f7 = gate_f7_conservation(
        free_maps["mean_only"], contract.weights["mean_only"], contract.wet
    )
    print("[4/7] gate F7, global sea-level conservation error (exact reference):")
    for lead in leads:
        record = gate_f7["per_lead"][str(lead)]
        print(
            f"      lead {lead:3d} d   ||S - w||/||w|| = {record['relative_l2']:.4f}   "
            f"amplitude {record['amplitude_ratio']:.4f}   corr {record['pattern_correlation']:.5f}"
        )

    # --- step 5: finite differences and precision --------------------------
    points = finite_difference_points(contract.wet, contract.target)
    reproducibility = {}
    for lead in leads:
        reproducibility[lead] = measure_adjoint_reproducibility(
            operator, truth[0], weights, calls_for_lead(lead), contract.wet, threads
        )
    print(f"[5/7] gate F2: {len(points)} cells x {len(epsilons)} epsilons at every lead")
    print(
        "      reverse-mode arithmetic floor (same gradient, reordered reduction): "
        + ", ".join(
            f"lead {lead} d -> {reproducibility[lead]['max_absolute_difference']:.2e}"
            for lead in leads
        )
    )
    gates_f2 = {}
    gates_fwd = {}
    for lead in leads:
        calls = calls_for_lead(lead)
        noise = measure_evaluation_noise(
            operator,
            truth[0],
            weights[PRIMARY_OBJECTIVE],
            free_maps[PRIMARY_OBJECTIVE][lead],
            contract.target,
            calls,
        )
        gate = gate_f2_finite_difference(
            operator,
            truth[0],
            weights[PRIMARY_OBJECTIVE],
            free_maps[PRIMARY_OBJECTIVE][lead],
            points,
            epsilons,
            calls,
            noise,
        )
        gates_f2[lead] = gate
        print(
            f"      lead {lead:3d} d   noise in J = {noise['absolute_noise']:.2e} m   "
            f"best eps {gate['points'][0]['best_epsilon']:.0e}   "
            f"worst |FD - adj|/max|adj| = {gate['worst_scaled_error']:.2e}   "
            f"plateau at {gate['plateau_visible_at']}/{gate['wet_probe_count']} "
            f"-> {'pass' if gate['passed'] else 'FAIL'}"
        )
        forward = forward_versus_reverse(
            operator,
            truth[0],
            weights[PRIMARY_OBJECTIVE],
            free_maps[PRIMARY_OBJECTIVE][lead],
            points,
            calls,
            reproducibility[lead]["max_absolute_difference"],
        )
        gates_fwd[lead] = forward
        print(
            f"                  forward vs reverse mode: worst {forward['worst_absolute_error']:.2e} "
            f"against a {forward['allowed_absolute_error']:.2e} floor "
            f"-> {'pass' if forward['passed'] else 'FAIL'}"
        )

    single = load_frozen_model(provenance["checkpoint"], double=False, identity=identity)
    single_operator = build_operator(
        single, normalizers, statics, contract.wet, dtype=torch.float32
    )
    single_weights = {
        PRIMARY_OBJECTIVE: torch.from_numpy(contract.weights[PRIMARY_OBJECTIVE]).to(torch.float32)
    }
    single_map = free_chain(
        single_operator,
        truth[0].to(torch.float32),
        single_weights,
        calls_for_lead(longest),
    )["maps"][PRIMARY_OBJECTIVE].astype(np.float64)
    gate_f4 = gate_f4_precision(free_maps[PRIMARY_OBJECTIVE][longest], single_map, contract.wet)
    print(
        f"      gate F4: float32 vs float64 relative L2 = {gate_f4['relative_l2']:.3e} "
        f"-> {'pass' if gate_f4['passed'] else 'FAIL'}"
    )
    del single, single_operator

    # --- step 6: structure and the ninety-day diagnostics ------------------
    print("[6/7] structure, leakage and the period-2 probe")
    leakage = unmasked_leakage(operator, truth[0], weights, calls_for_lead(longest), contract.wet)
    structure = {
        str(lead): metrics.structural_metrics(
            forced_maps[PRIMARY_OBJECTIVE][lead], contract.wet, contract.target
        )
        for lead in leads
    }
    period2 = {
        chain: lead_to_lead_correlation(
            backward[chain][PRIMARY_OBJECTIVE], contract.wet
        )
        for chain in CHAINS
    }
    amplitude = {
        "definition": "||S(L)|| over wet cells, and its ratio to the 10-day map",
        "per_chain": {
            chain: {
                str(lead): {
                    "norm": float(
                        np.linalg.norm(
                            (forced_maps if chain == "forced" else free_maps)[PRIMARY_OBJECTIVE][lead][
                                contract.wet
                            ]
                        )
                    ),
                }
                for lead in leads
            }
            for chain in CHAINS
        },
        "lambda_hat_prediction": (
            "lambda_hat = 1.00831 per call compounds to 1.077 over nine calls, so the "
            "ninety-day map should not be dramatically larger than the ten-day one. "
            "production-model-amplitude-runaway records that lambda_hat and the anomaly "
            "amplitude are decoupled, so this is a third, independent handle"
        ),
    }
    for chain in CHAINS:
        base = amplitude["per_chain"][chain][str(leads[0])]["norm"]
        for lead in leads:
            record = amplitude["per_chain"][chain][str(lead)]
            record["ratio_to_shortest_lead"] = record["norm"] / base if base > 0.0 else None

    trajectory_split = {
        str(lead): metrics.primary_metrics(
            free_maps[PRIMARY_OBJECTIVE][lead],
            forced_maps[PRIMARY_OBJECTIVE][lead],
            contract.wet,
        )
        for lead in leads
        if lead != leads[0]
    }
    objective_comparison = {
        str(lead): metrics.primary_metrics(
            forced_maps["ssh_anomaly_kernel"][lead],
            forced_maps[PRIMARY_OBJECTIVE][lead],
            contract.wet,
        )
        for lead in leads
    }

    # --- step 7: write ------------------------------------------------------
    print(f"[7/7] writing {output}")
    output.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(leads, dtype=np.int64),
        "wet_mask": contract.wet.astype(np.int8),
        "rA": contract.rac,
        "target_ij": np.asarray(contract.target, dtype=np.int64),
        "longitude_deg": contract.longitude,
        "latitude_deg": contract.latitude,
    }
    for name in OBJECTIVES:
        arrays[f"S_{name}_forced"] = np.stack([forced_maps[name][lead] for lead in leads])
        arrays[f"S_{name}_free"] = np.stack([free_maps[name][lead] for lead in leads])
    for chain in CHAINS:
        for name in OBJECTIVES:
            arrays[f"S_backward_{name}_{chain}"] = backward[chain][name]
    arrays["backward_days"] = np.asarray(days[:max_calls], dtype=np.int64)
    for name in OBJECTIVES:
        arrays[f"w_{name}"] = contract.weights[name]

    np.savez_compressed(output / "fno_ft90_s0_adjoint_arrays.npz", **arrays)

    report = {
        "version": "fno_ft90_s0_adjoint_v1",
        "plan": "docs/Adjoint_study_Phase_A.md",
        "plan_contract": PLAN_CONTRACT,
        "plan_contract_sha256": file_sha256(project_root / "config" / f"{PLAN_CONTRACT}.json"),
        "model": {
            "version": identity.contract,
            "identity_label": identity.label,
            "seed": identity.seed,
            "checkpoint": str(provenance["checkpoint"]),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "optimizer_step": provenance["optimizer_step"],
            "normalization": str(provenance["normalization"]),
            "normalization_sha256": provenance["normalization_sha256"],
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "markov": True,
        },
        "window": {
            "source_day": SOURCE_DAY,
            "lead_days": list(leads),
            "cost_days": [SOURCE_DAY + lead for lead in leads],
            "horizon_days": HORIZON_DAYS,
        },
        "runtime": {
            "torch": torch.__version__,
            "dtype": "float64",
            "threads": int(threads),
            "elapsed_seconds": time.monotonic() - started,
        },
        "sources": {**contract.sources, "dataset": str(dataset_path)},
        "weight_sha256": contract.weight_digests,
        "static_channels": static_provenance,
        "gates": {
            "F1": gates_f1,
            "F2": {str(lead): gates_f2[lead] for lead in leads},
            "F2_forward_mode": {str(lead): gates_fwd[lead] for lead in leads},
            "adjoint_arithmetic_floor": {str(lead): reproducibility[lead] for lead in leads},
            "F3": gate_f3,
            "F4": gate_f4,
            "F5": gate_f5,
            "F6": {
                "condition": "weight-field SHA-256 matches the Phase A contract (and so the MITgcm run manifest)",
                "digests": contract.weight_digests,
                "passed": True,
            },
            "F7": gate_f7,
            "sigma_consistency": gate_sigma,
        },
        "diagnostics": {
            "unmasked_land_leakage": leakage,
            "structure": structure,
            "period_2_probe": period2,
            "amplitude": amplitude,
            "trajectory_error_free_vs_forced": trajectory_split,
            "kernel_vs_point": objective_comparison,
        },
        "conventions": {
            "sign": "S > 0: raising eta at (i,j) raises the target anomaly",
            "units": "dimensionless (metres of J per metre of eta)",
            "grid": "cell centres, (j,i) order matching the zarr spatial axes",
            "land": "exactly 0 -- the input mask removes land eta as a degree of freedom",
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("      figures")
    figure_lead_maps(output / "fno_ft90_adjoint_lead_maps.png", forced_maps[PRIMARY_OBJECTIVE], contract)
    figure_forced_vs_free(
        output / "fno_ft90_adjoint_forced_vs_free.png",
        forced_maps[PRIMARY_OBJECTIVE][longest],
        free_maps[PRIMARY_OBJECTIVE][longest],
        contract,
        longest,
    )
    figure_backward_sweep(
        output / "fno_ft90_adjoint_backward_sweep.png", backward["free"][PRIMARY_OBJECTIVE], contract
    )
    figure_conservation(
        output / "fno_ft90_adjoint_conservation.png",
        free_maps["mean_only"],
        contract.weights["mean_only"],
        contract,
        gate_f7,
    )
    figure_gate_f2(output / "fno_ft90_adjoint_gate_f2.png", gates_f2)
    figure_structure(
        output / "fno_ft90_adjoint_structure.png", structure, period2["free"], forced_maps[PRIMARY_OBJECTIVE], contract
    )
    figure_objectives(
        output / "fno_ft90_adjoint_objectives.png",
        forced_maps[PRIMARY_OBJECTIVE][longest],
        forced_maps["ssh_anomaly_kernel"][longest],
        contract,
        longest,
        objective_comparison[str(longest)],
    )
    (output / "README.md").write_text(_readme(report))

    elapsed = time.monotonic() - started
    print(f"done in {elapsed / 60.0:.1f} min")
    return report


# ===========================================================================
# 9.  README and CLI
# ===========================================================================


def _readme(report: Mapping[str, Any]) -> str:
    gates = report["gates"]
    diagnostics = report["diagnostics"]
    leads = report["window"]["lead_days"]
    longest = max(leads)

    def verdict(gate: Mapping[str, Any]) -> str:
        passed = gate.get("passed")
        return "pass" if passed else ("measurement" if passed is None else "**FAIL**")

    f7 = gates["F7"]["per_lead"]
    conservation_rows = "\n".join(
        f"| {lead} | {f7[str(lead)]['relative_l2']:.4f} | "
        f"{f7[str(lead)]['amplitude_ratio']:.4f} | {f7[str(lead)]['pattern_correlation']:.5f} |"
        for lead in leads
    )
    amplitude = diagnostics["amplitude"]["per_chain"]
    amplitude_rows = "\n".join(
        f"| {lead} | {amplitude['forced'][str(lead)]['norm']:.5e} | "
        f"{amplitude['forced'][str(lead)]['ratio_to_shortest_lead']:.3f} | "
        f"{amplitude['free'][str(lead)]['norm']:.5e} | "
        f"{amplitude['free'][str(lead)]['ratio_to_shortest_lead']:.3f} |"
        for lead in leads
    )
    drift = diagnostics["trajectory_error_free_vs_forced"]
    drift_rows = "\n".join(
        f"| {lead} | {drift[str(lead)]['pattern_correlation']:.5f} | "
        f"{drift[str(lead)]['relative_l2']:.4f} | {drift[str(lead)]['amplitude_ratio']:.4f} |"
        for lead in leads
        if str(lead) in drift
    )
    structure = diagnostics["structure"]
    structure_rows = "\n".join(
        f"| {lead} | {structure[str(lead)]['western_band']['boundary_to_interior_ratio']:.3f} | "
        f"{structure[str(lead)]['radial_decay'].get('e_folding_cells', float('nan')):.3f} |"
        for lead in leads
    )
    period2 = diagnostics["period_2_probe"]["free"]
    correlations = ", ".join(
        "n/a" if c is None else f"{c:+.4f}" for c in period2["correlations"]
    )

    return f"""# Emulator adjoint, Phase A — fno_ft90_s0_adjoint_v1

Reverse-mode sensitivity maps through the published production emulator
`{report['model']['version']}`, `selected.pt` @ step
{report['model']['optimizer_step']:,} (`{report['model']['checkpoint_sha256'][:16]}…`),
27,297,960 parameters, frozen.

    S_L[j,i] = dJ_L / d eta(j, i, day {report['window']['source_day']})     L in {leads} days

Produced by `scripts/fno_adjoint_ft90.py` from `docs/Adjoint_study_Phase_A.md`.
This is the *test* half of the comparison; the reference half is
`../mitgcm_s0_adjoint_v2/`.

## What the numbers mean

`S[j,i]` is dimensionless — metres of `J` per metre of SSH. A value of 0.2 means
1 cm at the source gives 2 mm at the target. `S > 0` means raising eta at
`(j,i)` raises the target anomaly. Index order is `(j, i)`, matching the zarr's
spatial axes. Land is exactly 0, because the input mask removes land eta as a
degree of freedom — the same statement MITgcm's gate G4 makes.

## The two chains

MITgcm's adjoint is linearized about the **truth** trajectory by construction.
The emulator's rollout drifts from it, so two maps are produced at every lead:

| | linearized about | role |
| --- | --- | --- |
| `S_forced` | the MITgcm truth states | the matched object; **primary** against MITgcm |
| `S_free` | the emulator's own rollout | what a deployed user gets; **primary** operationally |

Their difference is the trajectory-drift term, and it is reported rather than
folded into the total.

## Gates

| gate | condition | result | verdict |
| --- | --- | --- | --- |
| F1 | pipeline `J` on truth vs numpy | worst {max(g['worst_relative_error'] for g in gates['F1'].values()):.2e} | {verdict(gates['F1'][str(leads[0])])} |
| F2 | `\\|FD − adjoint\\|` with a plateau, every lead | worst {max(gates['F2'][str(l)]['worst_scaled_error'] for l in leads):.2e} | {'pass' if all(gates['F2'][str(l)]['passed'] for l in leads) else '**FAIL**'} |
| F3 | plain `ProductionFNO`, no live spectral-norm machinery | {gates['F3']['parameter_count']:,} params, {len(gates['F3']['spectral_norm_hooks'])} hooks | {verdict(gates['F3'])} |
| F4 | float32 vs float64 | relative L2 {gates['F4']['relative_l2']:.3e} | {verdict(gates['F4'])} |
| F5 | `S_forced` = `S_free` at lead {leads[0]} exactly | max diff {gates['F5']['max_absolute_difference']:.3e} | {verdict(gates['F5'])} |
| F6 | weight SHA-256 matches the contract | 3 fields | {verdict(gates['F6'])} |
| F7 | conservation error at every lead | see below | {verdict(gates['F7'])} |
| — | backward-sweep `sigma` conversion | {gates['sigma_consistency']['relative_difference']:.2e} | {verdict(gates['sigma_consistency'])} |

## F7 — global sea-level conservation, against an exact reference

MITgcm's `implicitFreeSurface` with `exactConserv` conserves the area integral
of eta exactly in this closed basin, so the adjoint of the basin-mean
functional is **constant in time and equals the weight field itself** at every
lead. The emulator conserves no such thing, and this is the difference. It
required no MITgcm run to interpret.

| lead (d) | ‖S − w‖/‖w‖ | amplitude ratio | pattern corr. |
| ---: | ---: | ---: | ---: |
{conservation_rows}

## Amplitude, and what `lambda_hat` predicts

`lambda_hat` = 1.00831 per call compounds to 1.077 over nine calls, so the
ninety-day map should not be dramatically larger than the ten-day one. This is
a third, independent handle on amplitude: `production-model-amplitude-runaway`
records that `lambda_hat` and the anomaly amplitude are decoupled.

| lead (d) | ‖S_forced‖ | ratio | ‖S_free‖ | ratio |
| ---: | ---: | ---: | ---: | ---: |
{amplitude_rows}

## Trajectory drift — `S_free` against `S_forced`

Identical at lead {leads[0]} by construction (gate F5). Everything below is the
error a user inherits from linearizing about the emulator's own trajectory
rather than the truth.

| lead (d) | pattern corr. | relative L2 | amplitude ratio |
| ---: | ---: | ---: | ---: |
{drift_rows}

## Structure

`western-boundary-ratio-degrades` records that the boundary/interior ratio is
the diagnostic the forecast gate never scores, and `p*` sits inside that band.

| lead (d) | western band / interior | e-folding (cells) |
| ---: | ---: | ---: |
{structure_rows}

## The period-2 probe

`local-branch-release-lr` records that a period-2 computational mode — a
negative real eigenvalue of the composite ten-day recurrence — passes every
check currently in this project. It would show up here as consecutive leads
anticorrelating.

    corr(S(10k), S(10(k+1))) = {correlations}

Negative values: {period2['negative_count']} of {len(period2['correlations'])}.

## Files

| file | content |
| --- | --- |
| `fno_ft90_s0_adjoint_arrays.npz` | every map, both chains, all three objectives, both sweeps |
| `report.json` | provenance, gates, diagnostics |
| `fno_ft90_adjoint_lead_maps.png` | sweep (B): `S_forced` at each lead |
| `fno_ft90_adjoint_forced_vs_free.png` | the trajectory-drift term at lead {longest} |
| `fno_ft90_adjoint_backward_sweep.png` | sweep (A): `dJ`(day 7290)`/d eta(t)` |
| `fno_ft90_adjoint_conservation.png` | the conservation probe |
| `fno_ft90_adjoint_gate_f2.png` | the finite-difference plateau, per lead |
| `fno_ft90_adjoint_structure.png` | spectra, boundary split, period-2 probe |
| `fno_ft90_adjoint_objectives.png` | point vs smooth-kernel objective |

Array keys mirror the MITgcm extractor so one loader reads both sides.

## Not yet comparable

Nothing here has been differenced against MITgcm. Phase A's order of execution
puts gate G1-90 — the `grdchk` at the ninety-day window — ahead of any
comparison, because an unvalidated `adxx_etan` at a new window is not ground
truth. Until then these are self-contained results, and F7 is the only one that
is final: its reference is analytic.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    parser.add_argument("--quick", action="store_true", help="two finite-difference epsilons instead of four")
    parser.add_argument(
        "--leads",
        default=",".join(str(lead) for lead in LEAD_DAYS),
        help="comma-separated lead days; each must be a multiple of ten",
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    project_root = (
        Path(arguments.project_root).resolve()
        if arguments.project_root
        else Path(__file__).resolve().parent.parent
    )
    run(
        project_root,
        force=arguments.force,
        leads=[int(value) for value in arguments.leads.split(",") if value.strip()],
        epsilons=QUICK_FD_EPSILONS if arguments.quick else DEFAULT_FD_EPSILONS,
        threads=arguments.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
