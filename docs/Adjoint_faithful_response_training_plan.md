# Adjoint-faithful FNO training from forward perturbation responses

**Prospective study plan — 2026-08-22 — not yet approved for execution**

This document is deliberately prospective. It specifies the data, controls,
loss, selection rule, blind tests, and stop/go gates for a response-aware FNO,
but it does **not** authorize or launch an MITgcm run or an FNO training run.

Evidence labels used throughout:

- **Verified** means checked in the current repository, current scratch products,
  or current generated reports.
- **Inferred** means a scientific interpretation of verified evidence.
- **Proposed** means a choice to freeze before the new study begins.
- **Unresolved** means the repository does not currently establish the detail;
  the specified gate must resolve it without using any adjoint result.

The three required existing documents remain unchanged:
`docs/model_c_spectralnorm_ft90_handbook.md`,
`docs/Adjoint_study_Phase_A.md`, and
`docs/mitgcm_adjoint_ground_truth_plan.md`.

---

## 1. Executive summary

**Verified.** The production parent
`model_c_production_1in_1out_spectralnorm_v1` is a strong one-input,
one-output, 10-day Markov emulator. It predicts the full 46-channel state
`[U(15), V(15), Theta(15), SSH(1)]`, uses five physical static inputs, and was
trained from random initialization with a six-call/60-day autonomous rollout.
Its architecture has 32 by 32 Fourier modes, width 128, three FNO blocks, a
zero-initialized bias-free local 3 by 3 branch, and per-mode spectral
normalization with `rho=1`. Its selected checkpoint is step 7,680 with SHA-256
`e75951681b1a...`.

**Verified.** The later child
`model_c_production_1in_1out_spectralnorm_ft90_v1` has the same architecture,
state, normalizer, loss coefficients, and spectral-normalization machinery. It
was initialized from the parent and fine-tuned for 1,920 updates using a
nine-call/90-day rollout. It improved most reported forward metrics, but it is
not the baseline or weight source for this study.

**Verified.** Phase A established a severe adjoint failure for the **ft90
child**, not yet for the parent. Against MITgcm/TAF SSH adjoints that passed
the scalar error, cross-run, land, and conservation gates, the child's
truth-forced adjoint has pattern
correlation near zero at 10, 20, 30, and 90 days and amplitude ratios of about
38, 16, 9.7, and 6.6 for the point objective. The mismatch is overwhelmingly
Jacobian error rather than trajectory drift. The parent has not yet been run
through the current one-input Phase-A adjoint package, so its quantitative
adjoint error is unresolved.

**Proposed.** The primary experiment will preserve the parent model design and
60-day training schedule while adding one new term: a forward perturbation-
response loss. The response-aware model will start from random initialization.
No parent or ft90 checkpoint will be loaded. Its matched nominal-only control
will use the same random seeds, optimizer budget, native chronological split,
and parent loss.

The core finite-difference identity is

$$
\frac{M_k(x+\epsilon v)-M_k(x-\epsilon v)}{2\epsilon}
  = D M_k(x)v + O(\epsilon^2),
$$

with the analogous expression for the FNO. Matching these forward responses
constrains Jacobian-vector products without providing an adjoint label. For any
scalar-objective cotangent $w$,

$$
\left\langle
  (D F_k(x)^T-D M_k(x)^T)w,\,v
\right\rangle
=
\left\langle
  w,\,(D F_k(x)-D M_k(x))v
\right\rangle.
$$

Thus better forward JVPs imply better adjoint **projections onto the sampled
direction span**. They do not guarantee the full Jacobian in unsampled
directions; the final blind adjoint suite is needed to answer that question.

**Proposed data scale.** The production response dataset will contain, per
regime, 224 training directions at 14 annual pickup anchors, 72 validation
directions at three later anchors, and 72 blind-test directions at three still
later anchors. Every direction is run with both signs. Most runs stop at 10
days; a predeclared subset continues to 90 days and writes endpoints every 10
days. Including the three-amplitude pilot, paired nominal branches, and all
repeat/solver controls, the exact total is **65,520 model-days = 182.0
model-years**. This is forward MITgcm only.

**Non-negotiable firewall.** No MITgcm adjoint, TAF output, existing adjoint
metric, new FNO adjoint map, or analytic adjoint probe will enter amplitude
selection, response-loss-weight selection, early stopping, checkpoint
selection, or architecture selection. Historical summary numbers are already
known, but no new-model adjoint evaluation or development-pipeline access is
permitted until model identity, selected checkpoints, amplitudes, response
weight, and all forward reports are frozen and hashed.

---

## 2. Current parent/child forward evidence

### 2.1 Implemented production map

**Verified from** `config/model_c_production_1in_1out_spectralnorm_v1.json`,
`src/oceanfno/model.py`, `src/oceanfno/dataset.py`,
`src/oceanfno/objective.py`, `src/oceanfno/spectral_norm.py`, and the current
parent report:

| Item | Implemented value |
| --- | --- |
| Dynamic input/output | `U_01:15`, `V_01:15`, `Theta_01:15`, `Eta`; 46 channels |
| Static input | wind stress, wet mask, Coriolis parameter, MITgcm `DXF`, SST relaxation target; 5 channels |
| Position encoding | 2 Bire sine/cosine channels added inside the model |
| Map | direct state, `x(t+10 d) = F_theta(x(t), S)`, not a residual |
| Grid | 62 by 62 by 15, with 3,600 wet tracer cells |
| Fourier operator | 32 by 32 modes, width 128, 3 blocks |
| Lifting/projection | ratio 2, hence width 256 |
| Channel MLP | expansion 4, no dropout |
| Local path | bias-free 3 by 3 convolution, zero initialized |
| Parameters | 27,297,960 |
| Spectral normalization | 1,632 per-mode complex matrices, `rho=1`, 400-iteration warm start, **2** power iterations per forward |
| Inference checkpoint | materialized spectral weights; no live clipping or power iteration |

The parent objective implemented in `src/oceanfno/objective.py` is

$$
\begin{aligned}
L_{\rm parent}={}&L_{\rm state}
+0.001L_{\rm increment}
+0.50L_{\rm rollout}
+10^{-5}L_{\rm spectral}\\
&+0.065L_{\rm boundary}
+0.05L_{\rm pressure}
+0.05L_{\rm continuity}
+0.05L_{\rm transport}.
\end{aligned}
$$

All terms are dimensionless. The five state-space terms are explicitly group-
balanced over U, V, Theta, and SSH; pressure-gradient, continuity, and
barotropic-transport use their own implemented level/component balances. The
three physics terms are truth-referenced. No physics or adjoint channel is
added to the output.

### 2.2 Parent and ft90 optimization

**Verified.**

| Item | Parent | ft90 child |
| --- | ---: | ---: |
| Initialization | random | parent selected checkpoint |
| Autonomous rollout | 6 calls / 60 d | 9 calls / 90 d |
| Adam learning rate | `5e-4` through step 5,760, then `1e-4` | constant `5e-5` |
| Updates | 7,680 | 1,920 |
| Effective batch | 8 = 4 x 2 accumulation | 8 = 2 x 4 accumulation |
| State transitions | 368,640 | 138,240 |
| Selected step | 7,680 | 1,440 |
| Measured V100 wall time | 3.215 h | 1.671 h |
| Checkpoint SHA-256 | `e75951681b1a...` | `4acb7633d85a...` |

The child normalizer is byte-identical to the parent's, SHA-256
`fe424b37d74f...`. This is correct for a continuation experiment, but it is not
a reason to reuse the checkpoint or normalizer in a new from-scratch study.

### 2.3 Forward metrics

**Verified selected pooled-validation metrics:**

| Metric | Parent | ft90 child |
| --- | ---: | ---: |
| 10-90 d RMSE-AUC, surface speed | 0.080820 | 0.071042 |
| 10-90 d RMSE-AUC, SST | 0.821037 | 0.735979 |
| 10-90 d RMSE-AUC, surface PHIHYD | 0.579126 | 0.267129 |
| 90-360 d AUC/climatology, speed | 0.309277 | 0.274442 |
| 90-360 d AUC/climatology, SST | 0.289608 | 0.269316 |
| 90-360 d AUC/climatology, PHIHYD | 0.182586 | 0.104063 |
| Twin growth per call | 1.013225 | 1.008315 |
| Maximum normalized amplitude through 360 d | 5.257 | 5.398 |

Both selected through the implemented fallback because no candidate met the
declared growth ceiling of 1.0.

**Verified held S0 day-2,000 metrics:**

| Metric | Parent | ft90 child | Reference |
| --- | ---: | ---: | ---: |
| PHIHYD RMSE | 0.039411 | 0.020448 | train climatology 0.082489 |
| SST RMSE | 0.057655 | 0.037402 | train climatology 0.040458 |
| Speed RMSE | 0.006326 | 0.004839 | train climatology 0.003587 |
| Maximum normalized magnitude | 10.328 | 8.434 | -- |
| Day-2,000 anomaly RMS ratio | 1.795 | 1.738 | MITgcm truth 1.0 |
| Day-2,000 WBC/interior anomaly ratio | 5.059 | 8.157 | MITgcm truth 23.098 |

**Inferred.** The child improves several forward RMSE/AUC and long-range
metrics but does not solve perturbation growth, the day-2,000 magnitude gate,
or the weak WBC concentration. It is valuable as a contextual forward
comparator, not as the starting point for the new response-aware model.

### 2.4 Why the primary schedule remains 60 days

**Verified.** The surviving 30-to-60-to-90 curriculum artifact used 11,520
updates and 599,040 state transitions, while also changing `rho` to 0.99. Its
selected checkpoint has growth 1.0216, all three 90-360-day flattening ratios
above 1.9, held maximum normalized magnitude 21.39, and day-2,000 RMSE ratios
to climatology of 3.82, 6.06, and 5.70. Its source config and launcher are not
present in the tracked current tree, and some artifact prose disagrees about
`rho`; it is evidence, not a reproducible baseline.

**Inferred.** The evidence is against automatically bundling a curriculum,
longer exposure, and a tighter spectral cap. It does not isolate which of
those changes caused the failure.

**Proposed.** Both new from-scratch primary arms retain the successful 60-day
schedule. A 90-day continuation is a later matched experiment only if the
predeclared forward gates require it.

### 2.5 Implemented selection nuance

**Verified.** The shared production selector in `src/oceanfno/validation.py`
enforces the within-run 5% short-AUC filter, attempts the growth ceiling, and
then applies its long-range/fallback rule. The ft90 code computes the handbook's
parent-relative short-skill and flattening conditions as acceptance diagnostics,
but those extra conditions are not filters passed into the implemented shared
checkpoint selector. The new contract below follows implemented code where it
differs from older prose and makes every new filter executable rather than
descriptive.

---

## 3. Current adjoint failure evidence

### 3.1 What is established

**Verified.** Phase A is complete despite stale early status paragraphs that
say later steps are queued. The final comparison report is non-provisional,
and the focused tests currently pass (`65 passed` across
`tests/test_adjoint_phase_a.py` and `tests/test_mitgcm_adjoint.py`).

The MITgcm reference uses checkpoint68j commit `f03a2f5e...`, TAF 6.8.11,
source day 7,200, and leads 10, 20, 30, and 90 days. Its 90-day gradient check
passes the scalar `1e-4` error threshold at seven wet points, with worst
relative disagreement `2.10e-6`. The intended interior-epsilon-minimum
evidence holds at six of seven points; the current Boolean gate does not
enforce that plateau flag. Cross-run SSH adjoints agree exactly, the mean-
conservation gate is `3.57e-8`, and all 244 land cells are exactly zero. The
current v2 G0 trajectory check is ETAN-only, not a 46-channel trajectory gate.
These caveats must be resolved or explicitly carried by final Gate A0.

The **ft90 child's** truth-forced FNO adjoint versus MITgcm for the point SSH
anomaly objective is:

| Lead | Pattern correlation | Relative L2 | Amplitude ratio | Sign agreement |
| ---: | ---: | ---: | ---: | ---: |
| 10 d | 0.0594 | 37.895 | 37.941 | 0.431 |
| 20 d | -0.0106 | 16.523 | 16.482 | 0.522 |
| 30 d | 0.0084 | 9.694 | 9.651 | 0.513 |
| 90 d | 0.0213 | 6.620 | 6.565 | 0.483 |

For the five-point meridional kernel objective, smoothing reduces amplitude
error but does not recover the pattern:

| Lead | Pattern correlation | Relative L2 | Amplitude ratio |
| ---: | ---: | ---: | ---: |
| 10 d | 0.0821 | 26.154 | 26.217 |
| 20 d | -0.0109 | 12.367 | 12.316 |
| 30 d | 0.0163 | 6.861 | 6.804 |
| 90 d | 0.0282 | 2.198 | 1.985 |

The trajectory-drift component (`S_free` versus `S_forced`) is 0, 0.021,
0.062, and 0.094 relative L2 at the same leads, two to three orders smaller
than the total mismatch at short lead and still far smaller at 90 days.

The child also destroys the analytically conserved basin-mean SSH mode: the
adjoint amplitude ratio falls from 0.411 at 10 days to 0.0062 at 90 days, and
the 90-day pattern correlation is 0.039. At 10 days, 67% of child sensitivity
variance lies below the FNO's 4.625-cell spectral cutoff versus 4.4% for
MITgcm; low-pass filtering does not expose a hidden correct large-scale map.

### 3.2 What is not established

- **Unresolved:** no current one-input Phase-A result exists for the frozen
  production parent. The old `fno_s0_adjoint_v1` product is a stale two-input
  model and must not be used.
- **Verified limitation:** Phase A is S0-only and its primary control perturbs
  SSH while holding U, V, and Theta fixed. That direction is dynamically
  unbalanced and adjusts through barotropic gravity waves inside the FNO's
  10-day step.
- **Verified limitation:** packaged Phase-A arrays contain SSH adjoints. Raw
  P90 scratch has U/V/Theta adjoint dumps, but direct U/V comparison requires
  the adjoint of the face-to-centre map and is not yet validated.

**Inferred.** The failure is consistent with accurate state forecasts but an
inaccurate local derivative, particularly off the nominal trajectory manifold.
It motivates forward JVP supervision but does not predetermine that the method
will work for the parent or in every tangent direction.

---

## 4. Research hypothesis and identification strategy

MITgcm does not evolve the 46-channel FNO state in isolation. Let $Z_a$ be the
complete 108-record source pickup at anchor $a$, let $E_{a,q}(\eta)$ copy that
pickup and add the declared native-grid edit $\eta v_q$ to only the selected
`Uvel`, `Vvel`, `Theta`, or `EtaN` records, and let $\Phi_{a,k}$ be the full
MITgcm flow for $k$ days with the anchor's regime forcing. The trusted
projection $P$ extracts U/V/Theta/EtaN and centres the native U/V faces. The
conditional forward map actually sampled by this study is therefore

$$
M_{a,q,k}(\eta)=P\Phi_{a,k}[E_{a,q}(\eta)],
\qquad x_{a,q}(\eta)=P E_{a,q}(\eta).
$$

Salt, Adams-Bashforth/history records, and all other restart content in $Z_a$
are held fixed at the initial time, then evolve normally. This definition
does not pretend that an inverse normalization can reconstruct a complete
MITgcm restart from 46 channels.

Write the frozen pointwise FNO-state normalization as
$\hat x=\mathcal N_x(x)=S_x^{-1}(x-\mu_x)$. At an output lead write
$\mathcal N_k(y)=S_k^{-1}(y-\mu_k)$; in this study it is the same
time-independent state-normalizer artifact, so $S_k=S_x$ and $\mu_k=\mu_x$,
with the subscript retained only to distinguish endpoint from input roles. The learned map
$\hat F_\theta$ acts in those normalized coordinates. With a small
dimensionless standardized magnitude $\epsilon_q$, define the exact projected
initial states

$$
x_{a,q}^{s}=x_{a,q}(s\epsilon_q),\qquad s\in\{-1,+1\}.
$$

After the pilot, $\epsilon_q$ is the frozen family amplitude
$\alpha_{h(q)}$ used by the manifests and loaders.

Define oriented signed responses

$$
r_{M,a,q,k}^{s}
=\frac{\mathcal N[M_{a,q,k}(s\epsilon_q)]
       -\mathcal N[M_{a,q,k}(0)]}{s\epsilon_q},
$$

where $\mathcal N=\mathcal N_k$ is the one common strict-training scale used
for external comparison. The FNO response is

$$
r_{F,a,q,k}^{s}=
\frac{\hat F_\theta^{k/10}[\mathcal N_x(x_{a,q}^{s})]
      -\hat F_\theta^{k/10}[\mathcal N_x(x_{a,q}(0))]}{s\epsilon_q}.
$$

In the local regime,

$$
r_{M,a,q,k}^{+}\simeq r_{M,a,q,k}^{-}\simeq
\left.\frac{\partial}{\partial\eta}
\mathcal N[M_{a,q,k}(\eta)]\right|_{\eta=0},
$$

Define
$\hat v_{a,q}=\left.\partial_\eta\mathcal N_x[P E_{a,q}(\eta)]
\right|_{0}$. The FNO expression is the corresponding
$D\hat F_\theta^{k/10}(\hat x_a)\hat v_{a,q}$.
Because $E_{a,q}$ is an affine pickup edit, these are precisely conditional
Jacobian-vector products: $DM\,v_q$ with the unedited restart components held
fixed. Matching both signs therefore constrains the odd/JVP component and
exposes even, finite-amplitude contamination rather than hiding it in an
independently fitted perturbed state.

In implementation, each FNO branch starts from the exact canonical float32
projection of the corresponding edited pickup, not from a reconstructed Zarr
restart. The pilot's 1% magnitude/antisymmetry gates bound the tiny departure
from ideal $x\pm\epsilon v$ introduced by float32 projection; the denominator
remains the declared signed physical coefficient $s\epsilon_q$.

For an autoregressive lead $k=10n$ days,

$$
D\hat F_\theta^n(\hat x_0)
=D\hat F_\theta(\hat x_{n-1})\cdots D\hat F_\theta(\hat x_0).
$$

The sparse long responses constrain this Jacobian product along the forward
trajectory. Transposing reverses the order, exactly as in an adjoint sweep.
For a physical final-state cotangent $w$, the normalized reverse seed is
$S_k^T w$ and the physical initial gradient is
$S_x^{-T}[D\hat F_\theta^n(\hat x)]^T S_k^T w$. All final adjoint maps and metrics are
therefore transformed back to physical units; response training never
silently equates normalized and physical cotangents.

**Primary hypothesis.** Relative to a matched nominal-only from-scratch
control, forward-only response supervision will reduce held-out response error
and later reduce blind MITgcm-adjoint error while keeping nominal forward skill
inside a predeclared tolerance.

**Identification limits.** Finite directional supervision constrains the
Jacobian only on the sampled state/direction distribution. Improvement of a
blind adjoint projection is expected by duality, not guaranteed. The controls
below distinguish the response information from extra updates, extra forward
states, continuation, and a longer rollout horizon.

---

## 5. Experimental arms and controls

| Arm | Initialization | Data/objective | Rollout | Role |
| --- | --- | --- | ---: | --- |
| **A** frozen parent | existing random-init run | legacy production nominal data/loss | 60 d | Historical production reference; no retraining |
| **A90-context** ft90 child | parent checkpoint | legacy nominal data/loss | 90 d | Context only; never a baseline architecture or weight source |
| **B** nominal control | random | strict-v3 nominal data; exact parent loss | 60 d | Matched from-scratch control |
| **C** response-aware | random, paired seed with B | identical to B plus `lambda_resp L_response` | 60 d nominal; response examples to 10 or 90 d | Primary new model |
| **D** perturbed-state-only | random, paired seed | same response states/compute as C, but ordinary perturbed-state loss and no response difference | 60 d nominal; auxiliary 10/90 d | Required “more forward data” ablation |
| **E** short-response-only | random, paired seed | same response-update count as C; every response truncated to 10 d | 60 d | Required sparse-long ablation |

**Proposed names.**

- B: `model_c_adjoint_faithful_nominal_control_v1`
- C: `model_c_adjoint_faithful_response_v1`

**Proposed replication.** B and C use the three paired seeds
`20260911`, `20260912`, and `20260913`. No “best seed” is selected. The first
seed is the declared primary replicate; all three are frozen before blind
evaluation. D and E use `20260911` only in v1; any later replication is a new
version and cannot reuse v1 blind cases for model decisions.

### 5.1 Optional matched continuation, kept separate

If the primary study passes its forward/response gates, a secondary retrofit
experiment may start from the **frozen parent A**, never the ft90 child:

1. parent -> nominal-only 60-day continuation;
2. parent -> response-aware 60-day continuation.

Both use fresh Adam at constant `5e-5`, the parent's existing normalizer,
`1,920` updates, and identical nominal exposure. This asks whether response supervision can repair a
deployed model. It is not the primary result and cannot replace C.
The response continuation uses the already frozen raw P64 response data,
amplitudes, `lambda_resp`, 25% schedule, and loss; it merely expresses raw
responses in the parent's normalizer and recomputes the training-only `d`
scales deterministically. Its nominal-only mate receives the same optimizer
budget. Selection is the section-16 rule relative to that nominal continuation.
Whether to execute this retrofit must be frozen using validation/compute
considerations before any blind test. If it is not completed before the common
blind package, it moves to a new study version with a new blind set.

Only if C preserves 10-90-day skill and improves response validation but still
fails the predeclared 90-360-day forward gate may B and C each receive a
**matched**, separately named 90-day continuation (fresh Adam, constant `5e-5`,
1,920 updates). No 30-to-60-to-90 curriculum and no 500/2,000-day pointwise
target is permitted.

---

## 6. Exact nominal training, validation, and test split

### 6.1 Immutable nominal source

**Verified.** `trajectories_v3.zarr` contains three independently equilibrated
regimes, each with 9,000 daily states and shape
`(9000,46,62,62)`. S0/S1/S2 use wind amplitudes 1.0/0.75/1.25, corresponding
to 0.100/0.075/0.125 N m^-2. The store itself carries a strict split even
though current production code deliberately overrides it in memory.

**Proposed.** The Zarr store, its arrays, split codes, and all existing MITgcm
chains remain byte-unchanged. Response data live in a separate store. B and C
use the store's native chronological split:

| Half-open indices | Inclusive days | Role |
| --- | --- | --- |
| `[0,5040)` | 0-5,039 | nominal training, normalizers, climatology, response pilot/train |
| `[5040,5130)` | 5,040-5,129 | 90-day embargo; never read by model development |
| `[5130,6390)` | 5,130-6,389 | nominal and response validation only |
| `[6390,6480)` | 6,390-6,479 | 90-day embargo; never read by model development |
| `[6480,9000)` | 6,480-8,999 | blind nominal/response test and final truth only |

This is strict, non-overlapping, chronological, and already encoded by
`archive/src/bire_repro/af_data_v3.py` and the Zarr metadata.

### 6.2 Exact nominal records

- **Training:** all 4,980 valid 60-day rollout starts per regime, days
  0-4,979; 14,940 sequences pooled. The last target is day 5,039.
- **Optional 90-day stage:** 4,950 starts per regime, days 0-4,949; the last
  target is day 5,039.
- **Nominal checkpoint validation:** 34 starts per regime,
  $a_m=5130+12m$, $m=0,\ldots,33$, i.e. days 5,130-5,526. Every 360-day
  rollout ends by day 5,886, inside validation. These starts are distinct from
  the three response anchors, although their validation trajectories may
  overlap in time; no sample crosses into an embargo or test block.
- **Blind nominal test:** 15 fixed starts per regime:
  `6480, 6517, 6554, 6591, 6628, 6665, 6702, 6739, 6776, 6813,
  6850, 6887, 6924, 6961, 6999`. Every 2,000-day rollout remains in the test
  block and ends by day 8,999.

### 6.3 Normalization

**Proposed.** Create one immutable strict-training normalizer artifact from
S0/S1/S2 days 0-5,039 using the parent's pointwise recipe:

$$
\hat x_c(j,i)=\frac{x_c(j,i)-\mu_c(j,i)}{\sigma_c(j,i)},
$$

with land set to zero, the per-channel wet-cell fifth-percentile scale floor,
and the per-channel RMS normalized 10-day increment used by
`L_increment`. Static inputs retain the parent philosophy and definitions.
B, C, D, and E must load this byte-identical hashed artifact. An independent
recomputation is only a reproducibility gate and must hash-match before use.
The frozen project-side copy is
`outputs/af_fno/response/strict_v3_normalization_v1/normalization.npz` with its
full calculation report; the scratch copy must share the same SHA-256.

The existing parent/child normalizer is **not** reused because it includes days
5,040-5,999, which are validation/embargo days under this study. The Zarr's
pooled `state_scale` is also not the production pointwise normalizer.

### 6.4 Comparator caveat

**Verified.** A and the ft90 child trained or selected within parts of the
native validation/test chronology because their active split was
train `[0,6000)`, validation `[6000,7200)`, with nested inference. Therefore:

- B versus C is the clean prospective causal comparison;
- A and ft90 remain frozen contextual comparators;
- their common-test metrics are reported, but not called prospectively blind;
- A/ft90 metrics never select B or C.

---

## 7. Exact new MITgcm response-data design

### 7.1 Why annual anchors

**Verified.** Complete, float64 MITgcm pickups already exist annually for every
regime. The trajectory-day/iteration relation is

$$
I(d)=2{,}592{,}000+72d,
$$

with a 1,200 s timestep and 360-day model year. Using annual anchors avoids
constructing a restart from the 46-channel Zarr state and avoids a large new
pickup-generation campaign.

At a segment boundary such as day 3,600, the source resolver enumerates every
candidate pickup, requires all `.meta` and `.data` hashes to agree, and then
chooses the copy in the downstream segment whose half-open interval begins at
the anchor. If hashes disagree or no downstream copy exists where expected,
the anchor fails. The manifest retains every candidate path/hash and the
canonical-choice reason.

### 7.2 Amplitude pilot anchors

**Proposed.** Pilot anchors are days **720 and 3,600** in each regime, six
anchors total. These are inside training, have existing pickups, and are
separated by eight model years. Pilot centre/location IDs are reserved and are
not reused by production response directions, even though the anchor times are
also eligible for production.

At each pilot anchor and regime, run one U, one V, one Theta, and one SSH
direction. SSH is a wet-cell point at day 720 and a smooth 5 by 5 kernel at day
3,600. This gives 24 base directions: six each for U, V, Theta, and SSH.

The six `(regime,day)` cases are ordered
`[(S0,720),(S0,3600),(S1,720),(S1,3600),(S2,720),(S2,3600)]`. Their region
sequence is exactly
`[WBC,interior,eastern,northern,southern,WBC]`. U uses one-based levels
`[1,4,7,10,13,15]`; V uses `[7,10,13,15,1,4]`; and Theta uses
`[13,15,1,4,7,10]`. Thus every group samples upper, middle, and deep water,
all five regions occur, and WBC has two representatives. SSH inherits the
same region sequence and has no vertical level. Within each specified
region/level cell, the section-9 deterministic sampler chooses the centre;
the resulting exact `(j,i)` inventory is frozen before the pilot runs.

Every base direction is run at `alpha={0.025,0.05,0.10}` and both signs: 144
perturbed runs. Twelve base directions, exactly one per input group and regime,
continue to 90 days; the remainder stop at 10 days. The long assignment is:

| Regime | Day 720 long groups | Day 3,600 long groups |
| --- | --- | --- |
| S0 | U, Theta | V, SSH |
| S1 | V, SSH | U, Theta |
| S2 | U, SSH | V, Theta |

Thus the three long SSH pilot controls are the S0 smooth case and the S1/S2
point cases; their exact regions are interior/eastern/southern from the frozen
six-case sequence. The other three SSH cases still constrain day-10 amplitude.

All six paired nominal pilot branches run 90 days, and all six are duplicated
once to measure the deterministic/numerical floor. Pilot responses are used
only to freeze amplitudes and are excluded from train/validation/test losses.

### 7.3 Production response anchors

**Proposed exact days, per regime:**

| Role | Anchor days | Count/regime | Horizon guarantee |
| --- | --- | ---: | --- |
| response train | `0,360,720,...,4680` | 14 | every 90-day endpoint <=4,770, inside train |
| response validation | `5400,5760,6120` | 3 | endpoints <=6,210, inside validation |
| blind response test | `6840,7560,8280` | 3 | endpoints <=8,370, inside test |

The blind response anchors avoid the existing Phase-A source day 7,200. They
are frozen before model training and never read by training, validation,
hyperparameter selection, or checkpoint selection.

### 7.4 Directions at each anchor

At each **training** anchor, create exactly 16 directions:

- four U patches;
- four V patches;
- four Theta patches;
- two SSH point directions;
- two SSH smooth directions.

This is 224 directions per regime: 56 U, 56 V, 56 Theta, and 56 SSH (28 point,
28 smooth).

At each **validation or blind-test** anchor, create exactly 24 directions:

- six U directions;
- six V directions;
- six Theta directions;
- three SSH point directions;
- three SSH smooth directions.

This is 72 directions per regime: 18 per input group, with nine point and nine
smooth SSH directions.

Every production direction has both signs. “Direction count” and “MITgcm run
count” are therefore kept separate in every report.

### 7.5 Sparse-long subsets

All directions provide a 10-day response. Only the following continue to 90
days, writing at days 10,20,...,90:

- **Training:** at days
  `{0,720,1440,2160,2880,3600,4320,4680}`, extend one preassigned direction
  from each input group. This is 8 long directions per group, 32 per regime,
  and 96 total. The other 576 training directions across all regimes are
  short-only.
- **Validation:** at each of the three anchors, extend one direction from each
  input group. This is 12 per regime, 36 total.
- **Blind response test:** the same count and rule, with a disjoint centre and
  level inventory: 12 per regime, 36 total.

Long membership is selected by a constrained deterministic inventory solve
before any response is run. For each training group/regime, its eight choices
must include two WBC cases, at least one case from each other region, and, for
U/V/Theta, at least two cases in each of levels 1-5, 6-10, and 11-15; SSH must
contain four point and four smooth cases. Validation U/V/Theta long choices
are one upper, one middle, and one deep case at three distinct regions; their
SSH sequence is point/smooth/point. Blind-test U/V/Theta use the same depth
coverage with three distinct regions and split-disjoint centres, and SSH uses
smooth/point/smooth. Among
assignments satisfying those constraints, choose the one with maximum minimum
physical centre separation and break ties by the section-9 SHA order.
Validation and test assignments are solved jointly across regimes so each
input group has at least one long direction in all five regions. Failure to
find an assignment stops inventory construction. Membership cannot depend
on response magnitude or model performance.

---

## 8. Exact U/V/Theta/SSH perturbation definitions

### 8.1 Common horizontal kernel

For smooth directions, use a radius-2 isotropic Gaussian on the **native field
grid**:

$$
K(a,b)=\exp[-(a^2+b^2)/2],\qquad a,b\in\{-2,-1,0,1,2\},
$$

then normalize $K$ to unit discrete L2 norm. Thus `sigma_h=1 cell`, the native
support is exactly 5 by 5, and all 25 weights are positive. A centre is eligible
only if all 25 target-grid cells/faces are active at every perturbed level.
There is no clipping, land renormalization, or displaced centroid.

The direction is then scaled so its realized perturbation in the new
pointwise-normalized FNO input has unit RMS over its nonzero support. The pilot
coefficient `alpha_g` is the RMS standardized amplitude, not an arbitrary
physical-unit constant. Every manifest records the resulting physical peak,
RMS, and L2 magnitude.

### 8.2 U

- Native field edited: `Uvel` on the MITgcm W-face grid.
- Horizontal support: one 5 by 5 Gaussian, 25 active W faces per perturbed
  level.
- Primary training vertical support: exactly one of 15 levels.
- FNO input: apply the repository's exact centering
  `U_c(i)=0.5[U_face(i)+U_face(i+1)]` after the edit.
- Realized one-level FNO support: at most, and for the full-support eligible
  inventory exactly, 5 rows by 6 columns = 30 centred U cells.
- No V, Theta, SSH, salt, or history record is edited.

### 8.3 V

- Native field edited: `Vvel` on the MITgcm S-face grid.
- Horizontal support: one 5 by 5 Gaussian, 25 active S faces per perturbed
  level.
- Primary training vertical support: exactly one of 15 levels.
- FNO input: apply `V_c(j)=0.5[V_face(j)+V_face(j+1)]`.
- Realized one-level FNO support: exactly 6 rows by 5 columns = 30 centred V
  cells for eligible full-support patches.
- Every other pickup record remains unchanged.

### 8.4 Theta

- Native field edited: `Theta` at tracer centres.
- Horizontal support: one 5 by 5 Gaussian, exactly 25 wet tracer cells.
- Primary training vertical support: exactly one of 15 levels.
- Realized FNO support: exactly the same 25 cells in one Theta channel.
- Salt and `GtNm1` remain byte-identical; this is a partial derivative with
  unresolved restart histories held fixed.

### 8.5 SSH

Two equally represented training families are required:

1. **Point:** add the perturbation at one wet tracer cell; one affected pickup
   cell and one affected FNO SSH cell.
2. **Smooth:** the same unit-L2 5 by 5, sigma-1 Gaussian; exactly 25 wet tracer
   cells.

The pickup edit changes `EtaN` only. `EtaH` and `dEtaHdt` stay byte-identical.
This is consistent with the corrected adjoint contract: at startup with
`implicDiv2Dflow=1`, MITgcm recomputes `dEtaHdt` and sets `etaH=etaN`. The old
EtaN-plus-EtaH wording is not authoritative.

The pilot must select one common `alpha_SSH` that passes for both point and
smooth families. The maximum physical SSH magnitude is capped at 0.01 m.

### 8.6 Held-out vertical combinations

All training U/V/Theta directions are single-level, and every level appears at
least three times **per variable and regime** among the 56 directions. Order
the four direction slots within each of the 14 anchors and define
`q=4*anchor_slot+direction_slot`, `q=0,...,55`. With regime index
`r={S0:0,S1:1,S2:2}` and variable offsets `{U:0,V:5,Theta:10}`, assign

$$
\operatorname{level}(q,r,h)=1+[(q+5r+o_h)\bmod 15].
$$

This gives every level three occurrences and exactly 11 levels a fourth,
rotated across variable and regime. Direction-slot order is part of the frozen
inventory table, not filesystem enumeration order.

Each validation variable/regime has 18 directions:

- 15 single-level directions, exactly one at every level;
- one upper two-level direction on levels 1-2 with weights
  `[1,1]/sqrt(2)`;
- one mid-depth three-level Gaussian on levels 7-9 with weights
  `[exp(-1/2),1,exp(-1/2)]/sqrt(1+2 exp(-1))`;
- one deep two-level direction on levels 14-15 with weights
  `[1,1]/sqrt(2)`.

The blind test again has 15 single-level directions plus three unseen
combinations: levels 2-3, a Gaussian on levels 6-8, and levels 13-14. Horizontal
support is repeated on each involved level. Its two-level and Gaussian weights
are the same exact vectors above. A two-level U/V patch therefore
edits 50 native faces and maps to 60 centred cells; a three-level patch edits
75 faces and maps to 90 centred cells. Theta uses 50 or 75 tracer cells.

These combinations test linear superposition and vertical generalization.
Vertical weights multiply the native physical horizontal patch before the one
global standardized-RMS scaling in section 10; they are not reweighted level
by level after normalization. In validation, single level $l$ is assigned to
anchor index `[(l-1)+2r+o_h] mod 3`, and the upper/middle/deep combinations to
index `(t+r+o_h) mod 3` for type index `t={upper:0,middle:1,deep:2}`.
Blind-test assignments add one modulo three
to both formulas and use the stated unseen level combinations. Thus each
anchor has five singles and one combination per variable. The combinations
are never used to tune amplitude; their validation response must still pass
the forward numerical/SNR checks before model selection.

---

## 9. Spatial and vertical sampling plan

### 9.1 Exact region masks

The region of a direction is the region of its kernel centre. Masks are built
from the immutable wet/native masks in this precedence order:

1. **WBC:** the first four wet tracer cells east of the western wall in each
   wet row, reusing `src/oceanfno/dataset.py::western_boundary_mask`.
2. **Eastern:** the last four wet tracer cells west of the eastern wall,
   excluding WBC.
3. **Southern:** the ten southernmost wet tracer rows, excluding WBC/eastern.
4. **Northern:** the ten northernmost wet tracer rows, excluding WBC/eastern.
5. **Interior:** remaining eligible centres.

For Theta/SSH the centre is tracer index `(j,i)`. Under MITgcm's C-grid
convention, U face `(j,i)` is assigned to its carrier tracer cell immediately
east, `(j,i)`, and V face `(j,i)` to its carrier tracer cell immediately north,
also `(j,i)`; a face whose carrier is inactive at any requested level is not
eligible. The carrier tracer mask supplies the region label, while full patch
eligibility is checked on `hFacW`/`hFacS`. For smooth Theta/SSH, eligibility is
the erosion of the wet tracer mask by two cells. Inventory tests must verify
this face/carrier convention against the repository's centering equations.

### 9.2 Exact regional counts

Per **training** regime and per U/V/Theta family (56 directions):

| Region | Count | Fraction |
| --- | ---: | ---: |
| WBC | 20 | 35.7% |
| interior | 9 | 16.1% |
| eastern | 9 | 16.1% |
| northern | 9 | 16.1% |
| southern | 9 | 16.1% |

For each training SSH kernel separately (28 directions): WBC 8 and each of the
other four regions 5. The combined SSH family therefore has WBC 16 and 10 in
each other region.

Per **validation/test** regime and U/V/Theta family (18 directions): WBC 6 and
three in every other region. For nine point SSH directions use
`[WBC,interior,east,north,south]=[3,2,2,1,1]`; for nine smooth directions use
`[3,1,1,2,2]`. Combined SSH counts again equal `[6,3,3,3,3]`.

WBC is therefore sampled at roughly one third of directions even though the
four-column band occupies only about 6.7% of wet tracer cells. It is explicitly
oversampled, while every other boundary and the interior retain coverage.

### 9.3 Deterministic centre selection

**Proposed algorithm.** Before any response job is submitted:

1. Enumerate eligible native-grid centres for every family, level support, and
   region.
2. Divide WBC/eastern candidates into four latitude quartiles and
   northern/southern candidates into four longitude quartiles; divide interior
   into four quadrants.
3. Allocate pilot/train/validation/blind-test centre sets **jointly** within
   every `(regime,family,region)` stratum, including all role quotas before
   exposing any response. Centre IDs are constrained to be distinct across
   roles. Lexicographically maximize (i) minimum cross-role great-circle
   separation, (ii) minimum within-role separation, and (iii) summed WBC
   training-only mean surface speed or, outside WBC, proximity to the four
   subregion centroids. Physical distance uses `R=6371 km`, `(XC,YC)` for
   tracer centres, `(XG,YC)` for U faces, and `(XC,YG)` for V faces. Resolve
   every remaining tie by SHA-256 of
   `response-v1|split|regime|family|level-support|region|j|i`.
4. In role order `pilot < train < validation < blind`, require non-WBC
   validation and blind centres to have
   native-index Chebyshev distance at least three from every centre assigned
   to an earlier role; a joint solution that cannot meet that constraint
   fails. WBC is a predeclared capacity-limited exception: its centre IDs must
   be distinct and its joint maximin optimum is reported, but it has no
   hard distance-three threshold.
5. Assign levels by the explicit section-8 formulas, regions by the exact
   section-9.2 quotas, and long membership by the constrained section-7.5
   solve. A centre ID is
   `(regime,family,native_grid,j,i)`; coordinates may repeat across regimes but
   never across pilot/train/validation/test for the same regime/family.
6. Reserve disjoint centre inventories for validation and blind test before
   either is run. The blind list is written to an evaluator-only manifest and
   is not exposed to the training loader.

The five tracer cells used by the existing Phase-A point/kernel target
(`i=1`, zero-based `j=14..18`) are excluded as perturbation centres and as SSH
supports in all response splits; projected U/V/Theta footprints that intersect
those tracer cells are excluded as well. This uses the already frozen objective
location, not any adjoint output, and prevents direct training on the final
target stencil.

**Verified capacity warning.** A direct audit of the current masks finds only
56 full-support U-WBC carrier centres per regime and 47 after the Phase-A
footprint exclusion. The requested production roles consume 20 training + 6
validation + 6 blind centres, plus one pilot centre in the affected S0/S2
strata. This is enough for split-disjoint IDs but not for a global hard
distance-three rule, which is why the WBC exception above is explicit rather
than silently relaxed after seeing data.

**Unresolved until inventory materialization.** No trusted non-WBC sampler or
joint allocator currently exists. The concrete `(j,i)` list and achieved WBC
separations must be generated from this rule, reviewed for counts/full support,
frozen, and hashed before runs. Failure to meet the exact counts, distinct-ID
rule, or non-WBC distance rule is a stop, not permission to clip a kernel.

---

## 10. Perturbation-amplitude calibration protocol

No localized amplitude has been validated in the repository. The global S0
twin amplitudes do not establish a local U/V/Theta/SSH amplitude and will not
be reused as justification.

### 10.1 Standardized direction and candidate magnitudes

For direction $q$, form its native physical kernel and exact centred FNO
projection $v_q$. Scale it so

$$
\left[
\frac{1}{|\operatorname{supp}v_q|}
\sum_{c,j,i\in\operatorname{supp}v_q}
\left(\frac{v_{q,cji}}{\sigma_{c,ji}^{\rm train}}\right)^2
\right]^{1/2}=1.
$$

Test `alpha` in exactly `{0.025, 0.05, 0.10}` separately for U, V, Theta,
and SSH. The physical edit is `delta_x=alpha v_q`. SSH candidates must also
satisfy `max|delta Eta| <= 0.01 m`; a cap-triggered direction is recorded as a
failure of that alpha rather than silently assigned a smaller amplitude.

### 10.2 Precision and noise floor

- Source and endpoint pickups are big-endian float64.
- Nominal and perturbed branches write full pickups every 10 days.
- MITgcm responses are differenced after float64 native-grid extraction and
  face-to-centre projection.
- The FNO initial state uses the repository-canonical P32 projection: cast
  native fields to float32, then centre U/V. Exact nominal, plus, and minus
  P32 initial states are stored. Each realized signed P32 perturbation must be
  within 1% of the intended standardized RMS, and the relative antisymmetry
  error `||delta_P32^+ + delta_P32^-|| / mean(||delta_P32^+||,
  ||delta_P32^-||)` must be <=1%.
- Duplicate nominal pilot branches measure run-to-run noise at all nine leads.
  It is never assumed to be zero.

Define the group-balanced normalized norm

$$
\|z\|_{GB}^2=\frac14\sum_{g\in\{U,V,\Theta,SSH\}}
\operatorname{mean}_{c\in g,\,wet} z_c^2.
$$

Keep raw-response and differentiated-response units explicit. For each pilot
direction and lead, define the **oriented raw normalized responses**

$$
R_{q,k}^{s}=\frac{\mathcal N[M(x+s\alpha v)]-
\mathcal N[M(x)]}{s},\qquad s\in\{-1,+1\}.
$$

Thus $R^+$ is plus minus nominal, $R^-$ is nominal minus minus, both have raw
normalized-state units, and the derivative-like response used in section 4 is
$r^s=R^s/\alpha$. For any output slice $A$ (an individual physical group and
region, or the GB aggregate), the preliminary raw floor is

$$
n^{\rm raw}_{\rm nom,A,k}=\max_a\left\{
\|\mathcal N(M_{a,k}^{(1)})-\mathcal N(M_{a,k}^{(2)})\|_A,
8\epsilon_{64}\max[1,\|\mathcal N(M_{a,k}^{(1)})\|_A,
\|\mathcal N(M_{a,k}^{(2)})\|_A]
\right\},
$$

where `(1),(2)` are duplicate nominal branches and the maximum is over the six
pilot anchors. Candidate screening initially uses this raw floor; section 10.3
replaces it with the combined raw floor.

The sign-symmetry/nonlinearity score is

$$
Q_{\rm lin}=\frac{\|R^+-R^-\|_{GB}}
{0.5(\|R^+\|_{GB}+\|R^-\|_{GB})},
$$

and signal-to-noise is

$$
Q_{\rm SNR}=\frac{0.5(\|R^+\|_{GB}+\|R^-\|_{GB})}
{n^{\rm raw}_{h(q),GB,k}},
$$

where the denominator is initially $n^{\rm raw}_{\rm nom,GB,k}$ and is replaced
by the final input-family-specific combined raw floor after section-10.3
controls.

For adjacent-amplitude convergence use the centred estimate
$\widehat J_\alpha=[\mathcal N M(x+\alpha v)-\mathcal N
M(x-\alpha v)]/(2\alpha)$ and score
$\|\widehat J_\alpha-\widehat J_{\alpha/2}\|_{GB}/
\max(\|\widehat J_{\alpha/2}\|_{GB},
n^{\rm diff}_{h(q),GB,k}(\alpha/2))$, where
$n^{\rm diff}(\alpha)=n^{\rm raw}/\alpha$. Raw and differentiated floors are
never pooled in one maximum. Before the section-10.3 controls,
$n^{\rm diff}(\alpha)$ uses $n^{\rm raw}_{\rm nom}/\alpha$; the final check is
recomputed with the combined floor.

### 10.3 Solver-tolerance and perturbed-repeat controls

The main pilot first identifies a **provisional** group amplitude using the
candidate grid and nominal-repeat floor. Then use the already preassigned 12
long pilot directions—exactly one per input group and regime—at that
provisional amplitude for two additional forward-only controls:

1. duplicate both signed perturbed branches for 90 days at the production
   `cg2dTargetResidual=1e-7` (24 runs, 2,160 model-days);
2. rerun both signs at `cg2dTargetResidual=1e-10` (24 runs, 2,160 days) and
   rerun the six corresponding nominal anchors at `1e-10` (540 days).

For control condition $c\in\{\text{production},\text{duplicate},
\text{tight}\}$, construct $R_c^s$ from that condition's perturbed endpoint
and its own matched nominal endpoint. At every lead and sign define, with the
same normalized group-balanced norm,

$$
Q_{\rm repeat}=\frac{\|R_{\rm prod}^s-R_{\rm duplicate}^s\|_{GB}}
{\max[0.5(\|R_{\rm prod}^s\|_{GB}+\|R_{\rm duplicate}^s\|_{GB}),
n^{\rm raw}_{\rm nom,GB,k}]},
$$

and define $Q_{\rm CG}$ identically with $R_{\rm tight}^s$ in place of
$R_{\rm duplicate}^s$. Both must be <=0.01 for every controlled input group,
regime, sign, and lead. For each input family $h$, output slice $A$, and lead,
freeze

$$
n^{\rm raw}_{h,A,k}=\max\left\{
n^{\rm raw}_{\rm nom,A,k},
\|R_{\rm prod}^s-R_{\rm duplicate}^s\|_A,
\|R_{\rm prod}^s-R_{\rm tight}^s\|_A
\right\},
\qquad
n^{\rm diff}_{h,A,k}(\alpha_h)=
\frac{n^{\rm raw}_{h,A,k}}{\alpha_h},
$$

taking the maximum over the controlled directions, regimes, and signs; the
nominal term already includes the float64 rounding bound. `Q_lin` and
`Q_SNR` use raw $R$ and $n^{\rm raw}$; JVP targets, response loss, and response
validation use $r$ and $n^{\rm diff}$. Recompute final SNR with the combined raw
floor and require it to remain >=20. A failure stops v1; it cannot be fixed by
dropping a case or consulting an adjoint.

### 10.4 Frozen selection rule

For each input group independently, choose the **largest** candidate alpha for
which all of the following hold across the group's pilot cases:

1. `Q_lin <= 0.05` at day 10 for every direction;
2. `Q_lin <= 0.05` at every 10-day lead through day 90 for every long pilot
   direction;
3. `Q_SNR >= 20` for every scored direction/lead;
4. both realized P32 input perturbations pass the 1% magnitude and 1%
   antisymmetry checks and have a nonzero centre weight;
5. the SSH 1 cm cap holds for every point and smooth SSH pilot/planned
   production direction;
6. the centred JVP estimate at the selected alpha differs by <=5% from the
   next smaller passing alpha wherever such an alpha exists.
7. all selected-alpha solver-tolerance and perturbed-repeat controls in
   section 10.3 pass.

If no candidate passes for a group, production response generation stops. A
new, separately versioned pilot may add a smaller amplitude, but the failed
pilot and revised rule must be retained. No adjoint output may inform that
revision.

The frozen amplitude manifest records one selected `alpha_U`, `alpha_V`,
`alpha_Theta`, and `alpha_SSH`, plus every direction's actual physical peak and
RMS. There is no one-size physical amplitude invented in advance.

---

## 11. Short-versus-long response counts and integration cost

### 11.1 Direction and run inventory

| Role | Directions/regime | Directions total | Long directions total | Signed perturbed runs | Long signed runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 224 | 672 | 96 | 1,344 | 192 |
| validation | 72 | 216 | 36 | 432 | 72 |
| blind response test | 72 | 216 | 36 | 432 | 72 |
| pilot, base | 8 | 24 | 12 base | 144 across 3 alphas | 72 |

Every long run also supplies its 10-day response. The train set therefore has
672 distinct 10-day directions and 96 sparse long trajectories; validation and
test each have 216/36.

### 11.2 Exact model-day cost

For a direction pair, a short direction costs `2 signs x 10 = 20` model-days;
a long direction costs `2 x 90 = 180` model-days.

| Component | Short signed runs | Long signed runs | Model-days | Model-years |
| --- | ---: | ---: | ---: | ---: |
| production train perturbations | 1,152 | 192 | 28,800 | 80.0 |
| response validation perturbations | 360 | 72 | 10,080 | 28.0 |
| blind response-test perturbations | 360 | 72 | 10,080 | 28.0 |
| 3-amplitude pilot perturbations | 72 | 72 | 7,200 | 20.0 |
| shared nominal branches | 18 controls | 42 controls | 3,960 | 11.0 |
| duplicate 90-day pilot controls | -- | 6 controls | 540 | 1.5 |
| selected-alpha perturbed duplicates | -- | 24 | 2,160 | 6.0 |
| tight-CG selected-alpha perturbations | -- | 24 | 2,160 | 6.0 |
| tight-CG nominal controls | -- | 6 controls | 540 | 1.5 |
| **Total** | **1,944 perturbed + 18 controls** | **456 perturbed + 54 controls** | **65,520** | **182.0** |

Nominal costs are one shared control per anchor, not one per perturbation:

- training: 24 long controls and 18 short controls = 2,340 days;
- validation: nine 90-day controls = 810 days;
- blind test: nine 90-day controls = 810 days.

The pilot reuses the training-anchor nominal branches. Annual source pickups
already exist, so no pickup-bank integration cost is included. If an annual
pickup or forcing hash is missing, that anchor is a stop; it is not reconstructed
from Zarr.

The complete plan therefore launches 2,400 perturbed branches and 72 nominal
control branches (2,472 total), though long branches are internally chained in
10-day segments. The 5,400 model-days from duplicate nominal,
selected-amplitude perturbed-repeat, and tight-CG branches are calibration
controls, not new training directions.

---

## 12. MITgcm restart/pickup implementation plan

### 12.1 Verified pickup layout

The actual pickup is 62 by 62, big-endian float64, 108 records, about 3.17 MiB:

| Field | Records |
| --- | ---: |
| `Uvel` | 15 |
| `Vvel` | 15 |
| `Theta` | 15 |
| `Salt` | 15 |
| `GuNm1` | 15 |
| `GvNm1` | 15 |
| `GtNm1` | 15 |
| `EtaN` | 1 |
| `dEtaHdt` | 1 |
| `EtaH` | 1 |

The trusted layout parser/editor/verification logic is in
`archive/src/bire_repro/af_s0_twin.py`. It already checks metadata, endian,
field order, record count, finite values, byte-identical untouched records,
and input/output hashes. It will be generalized, not copied into a parallel
implementation.

### 12.2 Exactly what changes

| Perturbation | Modified pickup records | Byte-identical records |
| --- | --- | --- |
| U | selected cells in selected `Uvel` level(s) | all other U; all V, Theta, Salt, Gu/Gv/Gt, EtaN, dEtaHdt, EtaH |
| V | selected cells in selected `Vvel` level(s) | all other V; all U, Theta, Salt, Gu/Gv/Gt, EtaN, dEtaHdt, EtaH |
| Theta | selected cells in selected `Theta` level(s) | all other Theta; all U, V, Salt, Gu/Gv/Gt, EtaN, dEtaHdt, EtaH |
| SSH | selected cells in `EtaN` | U, V, Theta, Salt, Gu/Gv/Gt, dEtaHdt, EtaH |

Holding Adams-Bashforth/history and unrepresented restart variables fixed is a
declared conditional derivative. It is the forward analogue of perturbing one
resolved state component while holding the others fixed. No restart is ever
reconstructed from the FNO state alone.

### 12.3 Regime forcing

Each run stages the exact source pickup plus the source regime's own
`windx_cosy.bin`, bathymetry, SST relaxation target, `data.pkg`, and physics
namelist. S1/S2 never borrow S0 forcing. The MITgcm commit, forward executable,
MPI decomposition (2 by 2, four ranks), convection, and all physical settings
remain identical to the nominal chain. Production, ordinary duplicate, and
perturbed-repeat branches use `cg2dTargetResidual=1e-7`. The sole exception is
the hashed section-10.3 solver-sensitivity condition, whose matched perturbed
and nominal branches use `1e-10`; no production response target uses that
tighter condition.

### 12.4 Output and restart gates

- Run each long branch as nine validated 10-day segments, using
  `pChkptFreq=864000 s`, and archive/hash the endpoint before launching the
  next segment. MITgcm's rotating `pickup.ckptA/B` names are not assumed to
  retain all nine endpoints. A one-job wrapper may chain the segments, but
  each segment must restart from the just-verified preceding pickup.
  Apply the perturbation only to the original anchor pickup; later segments
  restart from the complete evolved pickup, including its evolved history
  records, with no repeated edit.
- Set `dumpFreq=0`, `taveFreq=0`, and disable all gridded diagnostics output in
  the response-run `data.diagnostics`; retain stdout/scalar monitors and the
  declared 10-day pickups only. Diagnostics are passive, but their daily
  float32 files are neither response truth nor part of the storage budget.
- Extract U/V/Theta/EtaN from these pickups; do not subtract daily float32
  diagnostics or Zarr states.
- Canonical P32 gate: casting pickup faces to float32 before the trusted
  centering operator must reproduce the Zarr anchor state bit-for-bit.
- P64 response path: centre float64 faces first, subtract the paired nominal
  endpoint, and store float64 response arrays.
- For every edited pickup, assert metadata byte identity, exact target support,
  exact equality of all untouched records, sign reversal, amplitude, and SHA.
- A plus and minus case must cite the same source/nominal branch and differ
  only in the sign of the declared edit.

### 12.5 Per-run manifest

Every run manifest must contain at least:

- study/contract version and role (`pilot/train/validation/test`);
- source pickup `.meta`/`.data` paths and SHA-256;
- regime, trajectory day, iteration, source-chain segment;
- perturbed variable, native grid, one-based level list;
- centre `(j,i)`, geographic coordinates, region;
- kernel name, sigma/radius, normalization, native and centred support counts;
- sign, candidate/selected alpha, physical peak/RMS/L2;
- horizon and output leads;
- control condition (`production`, `nominal_duplicate`,
  `perturbed_duplicate`, or `tight_cg`) and the exact
  `cg2dTargetResidual`;
- hashes of the edited pickup, nominal branch, endpoint pickups, executable,
  MITgcm commit, forcing files, data namelist, code/config, and Slurm script;
- job ID, ranks, host, start/end times, exit status, and validation gates.

Completed run directories are immutable. A directory without a complete
manifest is quarantined rather than reused.

---

## 13. Proposed response dataset schema

**Proposed development root:**
`/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/forward_response_v1.zarr`,
containing only `pilot`, `train`, and `validation`. The blind test is a separate
evaluator-only store,
`/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/forward_response_blind_v1.zarr`,
created after model freeze. Project-side digest manifests live under
`outputs/af_fno/response/forward_response_v1/` and
`outputs/af_fno/response/forward_response_blind_v1/`.
`trajectories_v3.zarr` is never modified.

The production-role dimensions are frozen as:

| Role | `A` | `A_short` | `A_long` | `Q_short` | `Q_long` |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 42 | 18 | 24 | 576 | 96 |
| validation | 9 | 0 | 9 | 180 | 36 |
| blind test | 9 | 0 | 9 | 180 | 36 |

Long directions are not duplicated in `short`; their day-10 response is the
first element of `long/response_p64`. Each role contains:

| Array/table | Shape/content | Dtype |
| --- | --- | --- |
| `anchors/state_p32` | `(A,46,62,62)` exact FNO input projection | float32 |
| `anchors/state_p64` | `(A,46,62,62)` float64 pickup projection | float64 |
| `anchors/nominal_short` | `(A_short,1,46,62,62)` | float64 |
| `anchors/nominal_short_anchor_row` | `(A_short,)` rows into `anchor_table` | int32 |
| `anchors/nominal_long` | `(A_long,9,46,62,62)` | float64 |
| `anchors/nominal_long_anchor_row` | `(A_long,)` rows into `anchor_table` | int32 |
| `short/anchor_row`, `short/direction_row` | each `(Q_short,)` | int32 |
| `short/input_state_p32` | `(Q_short,2,46,62,62)` exact plus/minus pickup projections | float32 |
| `short/response_p64` | `(Q_short,2,1,46,62,62)` signs x day 10 | float64 |
| `long/anchor_row`, `long/direction_row` | each `(Q_long,)` | int32 |
| `long/input_state_p32` | `(Q_long,2,46,62,62)` exact plus/minus pickup projections | float32 |
| `long/response_p64` | `(Q_long,2,9,46,62,62)` | float64 |
| `lead_days` | `[10,20,...,90]` | int16 |
| `anchor_table.jsonl` | one authoritative row per anchor | canonical JSON Lines |
| `direction_table.jsonl` | one row per direction; array row mappings above point here | canonical JSON Lines |

`anchor_table` columns are `anchor_id`, role, regime, day, iteration, horizon,
canonical source segment/path/hash, all duplicate-source hashes, forcing/static
hashes, and nominal endpoint hashes. `direction_table` columns are
`direction_id`, array group/row, anchor row, input family/native field,
one-based level list, exact vertical weights, native `(j,i)`, lon/lat, region,
kernel/sigma/radius, native/centred support counts, alpha, unit-direction
normalization, physical peak/RMS/L2, both P32 realized magnitudes,
antisymmetry, long flag, sparse edit lists `(record,j,i,value)`, and all input/
response hashes. IDs are UTF-8 strings of the frozen manifest grammar; levels,
weights, hashes, and sparse edits are typed JSON arrays validated against
`config/forward_response_schema_v1.json`. Each line uses sorted keys and compact
separators, and both line-order and file SHA-256 are frozen. This avoids adding
a Parquet engine that the current environment does not provide.

The sign axis is fixed as index 0 = `s=-1`, index 1 = `s=+1`. Stored response
arrays are raw float64 differences

$$
\Delta_{q,s,k}^{64}=P_{64}[M_k(x_q^s)]-P_{64}[M_k(x)],
$$

so the minus record is normally negative. The loader alone forms the oriented
target
$r_{M,q,k}^s=(\Delta_{q,s,k}^{64}/\sigma_{train})/(s\alpha_q)$.
It never guesses sign from a filename. Nominal endpoints plus raw deltas also
reconstruct the perturbed physical targets for D.

The pilot uses the same split short/long layout with 36 short and 36 long
`(base_direction,alpha)` rows, an alpha column, and a separate
`solver_control_table.jsonl`. It additionally stores
`controls/perturbed_response_p64` with shape
`(12,2_conditions,2_signs,9,46,62,62)` for production duplicates and tight-CG
branches, `controls/duplicate_nominal_p64` with shape
`(6,9,46,62,62)`, and `controls/tight_nominal_p64` with the same shape. The
solver-control table maps every array row to its source base direction/anchor,
condition, matched nominal row, exact CG tolerance, and run/endpoint hashes.
All curated numeric arrays use canonical
little-endian `<f4`/`<f8`. Response chunks are one
`(direction,sign,lead,46,62,62)` field, input chunks one
`(direction,sign,46,62,62)` field, and Blosc-Zstd level 3 with bitshuffle is
frozen for v1. Array-level and chunk-manifest SHA-256 digests are mandatory.

Loader reconstruction must hash-match both signed input states and all sparse
edits. Development loaders have no filesystem permission or config route to
the blind store; tests assert both. Every model's external response metrics
use this one strict-training scale even when A/ft90 internally use their legacy
normalizer.

**Estimated combined development-plus-blind curated size:** about 10-12 GB
uncompressed numeric arrays plus metadata; expected compressed size is
unresolved until pilot compression is measured.

---

## 14. Full objective function and response-weight selection

### 14.1 Parent objective stays intact

For every optimizer update, B and C evaluate the unchanged eight-term parent
objective on an effective batch of eight nominal 60-day sequences:

$$
L_{\rm total}=L_{\rm parent,nominal}
+\lambda_{\rm resp}I_{\rm joint}L_{\rm response}.
$$

No parent coefficient changes. No adjoint channel, contraction penalty, tanh
stabilizer, spectral-cap change, or extra state variable is introduced.

### 14.2 Response normalization and group balance

Let input family $h(q)\in\{U,V,\Theta,SSH\}$ and output group
$g\in\{U,V,\Theta,SSH\}$. Compute, from **response-training data only**, an RMS
scale

$$
d_{h,g,k}^2=
\operatorname{mean}_{q:h(q)=h,\,s}
\operatorname{mean}_{c\in g,wet}
\left(r_{M,q,k}^{s}\right)^2.
$$

Floor it at ten times the corresponding differentiated combined
rounding/repeat/CG-disagreement scale from section 10. The pilot SNR gate
should make the floor inactive for the directly excited signal; the floor
prevents a nearly zero cross-group response from producing an infinite loss.

For one response direction,

$$
\ell_{q,k}=\frac{1}{8}
\sum_{s\in\{-1,+1\}}
\sum_{g\in\{U,V,\Theta,SSH\}}
\frac{
\operatorname{mean}_{c\in g,wet}
\left(r_{F,q,k}^{s}-r_{M,q,k}^{s}\right)^2
}{d_{h(q),g,k}^2}.
$$

This gives each output physical group equal weight regardless of units or its
15-versus-1 channel count. The response sampler gives each input family equal
probability, so all 16 input-output blocks receive equal status in expectation.

Short samples use

$$
L_{\rm response}^{\rm short}=\ell_{q,10},
$$

and long samples use an unweighted lead mean

$$
L_{\rm response}^{\rm long}
=\frac{1}{9}\sum_{k\in\{10,20,\ldots,90\}}\ell_{q,k}.
$$

The signed formulation compares perturbation **responses** to responses. It is
not equivalent to fitting the perturbed state alone.

### 14.3 Ordinary state loss on perturbed trajectories

**Primary choice: no.** C does not add ordinary state loss on `x+delta` or
`x-delta`. The nominal parent loss anchors the common forward map, while the
signed response term is the single scientific modification. Adding a
perturbed-state loss would change two things and weaken attribution.

Arm D is the predeclared ablation: it uses the same perturbed endpoints and
auxiliary compute but replaces the response-difference term by the parent's
group-balanced normalized `L_state` form independently at each available
perturbed endpoint. It does **not** apply the eight-term 60-day objective to a
short case or assume ungenerated intermediate truth. C versus D asks whether
explicit response information matters beyond more nearby forward states.

Precisely, D uses

$$
\ell^{\rm pert}_{q,k}=\frac18
\sum_{s\in\{-1,+1\}}\sum_g
\left[
\frac{
\sum_{c\in g,j,i}m_{ji}
\left(\hat F_\theta^{k/10}[\mathcal N_x(x_{a,q}^s)]_{cji}
-\mathcal N[M_{a,q,k}(s\epsilon_q)]_{cji}\right)^2}
{\max\left\{
\sum_{c\in g,j,i}m_{ji}
\mathcal N[M_{a,q,k}(s\epsilon_q)]_{cji}^{2},10^{-12}
\right\}}
\right]^{1/2},
$$

where $m$ is the production wet mask. This is exactly the square-root,
truth-referenced, four-group `group_relative_l2_terms` state form, including
its `1e-12` denominator clamp, applied to each signed endpoint. It uses the
same short/long lead means as C and no subtraction of a nominal branch inside
the loss.

D's auxiliary coefficient is fixed algebraically, not tuned. Define
$\theta_{proj}$ as exactly `fno.projection.fcs.0.weight`,
`fno.projection.fcs.0.bias`, `fno.projection.fcs.1.weight`, and
`fno.projection.fcs.1.bias`; preflight freezes that sorted key list and fails
if it changes. At the common seed-`20260911` initialization, before optimizer update
1, form 12 training-only calibration blocks. Each `(regime,input-family)`
block contains the three lowest-hash short IDs and the lowest-hash long ID, so
the 48 directions reproduce the 75/25 auxiliary mix. Evaluate the response
and D losses on identical cloned weights and spectral buffers and set

$$
G_R=\operatorname{median}_{b=1}^{12}
\|\nabla_{\theta_{proj}}[\lambda_{resp}L_{response,b}]\|_2,
\quad
G_D=\operatorname{median}_{b=1}^{12}
\|\nabla_{\theta_{proj}}L_{pert,b}\|_2,
\quad
\lambda_D=\frac{G_R}{\max(G_D,10^{-12})}.
$$

Nonfinite or zero $G_D$ fails the ablation. The dummy nominal auxiliary branch
is still evaluated and discarded so the branch count matches C. This match is
performed once, uses no validation/test/adjoint data, and all inputs, norms,
key names, and $\lambda_D$ are reported.
E uses the frozen `lambda_resp` and the identical signed response loss, but
uses only `ell(q,10)` for all 1,920 auxiliary updates; former long inventory
members contribute their day-10 endpoint and no 20-90-day target.

### 14.4 Response-loss-weight selection

**Proposed candidates:**
`lambda_resp in {0.03, 0.10, 0.30, 1.00}`.

Use the primary seed and identical initialization/batch order to run a 1,920-
step forward-only screen. For each candidate:

1. evaluate nominal validation and response validation;
2. reject it if any 10-90-day primary-field AUC is >1.05 times the matched
   lambda-zero control at the same step;
3. reject it if growth is >0.005 per call worse than that control or any
   rollout is nonfinite;
4. among remaining candidates, minimize the group/region/lead-balanced
   response relative-L2 composite;
5. candidates within 2% of the minimum tie in favour of smaller lambda.

Freeze the chosen lambda before full three-seed training. If no candidate is
forward-feasible, stop. No MITgcm adjoint, TAF file, FNO adjoint map, blind
response case, or test metric may be read during this screen.
Screen checkpoints and optimizer states are discarded. The full primary-seed
C run restarts from its original random initialization and step zero; it does
not receive 1,920 extra updates from the screen.

---

## 15. Training protocol

### 15.1 Shared B/C protocol

| Setting | Value |
| --- | --- |
| Initialization | neuraloperator default random initialization; local branch zero |
| Parent/ft90 state load | none |
| Normalizer | load the frozen, hash-verified strict-v3 artifact from section 6.3 |
| Nominal rollout | 6 calls / 60 days |
| Teacher forcing | none after initial state |
| Adam | betas `(0.9,0.95)`, weight decay 0 |
| Learning rate | `5e-4` steps 1-5,760; `1e-4` steps 5,761-7,680 |
| Updates | 7,680 |
| Effective nominal batch | 8 = microbatch 4 x accumulation 2 |
| Gradient clipping | none |
| Checkpoints | steps 1,920, 3,840, 5,760, 7,680 |
| Spectral normalization | exact parent machinery, `rho=1`, materialized checkpoints |
| Seeds | 20260911, 20260912, 20260913 paired across B/C |

B and C see exactly the same nominal batch at every update. C does not replace
nominal samples with response samples.

### 15.2 Auxiliary response mixing

On exactly every fourth optimizer update (`I_joint=1`), C adds one response
direction pair. At every autoregressive lead, concatenate the nominal, minus,
and plus states on the batch dimension and make **one** model invocation, then
split and advance the three branches. Sequential sign forwards are forbidden:
the live spectral-normalization layer updates its power vectors on every
training forward, so sequential evaluation would quotient three slightly
different maps. Batched evaluation gives all signs identical normalized
weights at that lead. Thus the **response-update fraction is 25%**, while
nominal exposure remains identical to B.

The auxiliary path must also leave persistent spectral-normalization state
identical to the nominal-only schedule. Immediately before an auxiliary chain,
snapshot every spectral left/right power vector; allow the parent's usual two
power iterations on each batched lead; backpropagate the auxiliary loss; then
restore every vector bit-for-bit before the optimizer step, in a `finally`
guard. Weight gradients are retained, estimator-buffer mutations are not. B
needs no dummy data pass; C, D, and E all use the same context manager. Tests
must establish branch-order invariance, zero response for zero perturbation,
and bit-identical within-arm pre/post auxiliary buffer hashes. A same-weight,
same-buffer counterfactual clone that enters a zero-auxiliary context must
match a clone that skips it after their next identical nominal pass. Across
real B/C runs, require equal buffer-mutation counts and schedule only: their
buffer values are not expected to match after response gradients make their
weights differ. A failure of the schedule-isolation test is a model-contract
failure.

Within the auxiliary stream:

- 75% of samples are short-only;
- 25% are long;
- input families are exactly balanced in blocks;
- regimes are exactly balanced in blocks;
- levels and regions follow the frozen inventory, with deterministic shuffling.

Over 7,680 updates this gives 1,920 response updates: 1,440 short and 480 long.
Because a long loss averages nine leads, 77.78% of total response lead weight
falls at day 10 and 2.78% at each of days 20-90. The study therefore remains
primarily a direct constraint on the fundamental 10-day map.

A short pair adds one batch-three invocation/three state transitions; a long
pair adds nine invocations/27 state transitions. The expected auxiliary load
is nine state transitions on a joint update, or 2.25 averaged over all updates.
The nominal batch performs 48 state transitions/update, so raw transition
count rises by about 4.7%; batched-triplet and snapshot overhead suggests a
practical wall-time increase of roughly 10-25%, to be measured.

### 15.3 Maximum target horizon

The primary nominal target horizon is 60 days. Long response supervision stops
at 90 days. No training loss, amplitude choice, or model selection uses
pointwise truth at day 500, day 2,000, or any horizon beyond 90 days. Rollouts
to 360/2,000 days are diagnostics/evaluation only.

---

## 16. Validation and checkpoint-selection protocol

### 16.1 View 1: nominal forward validation

Use the 102 strict-validation rollouts defined in section 6 (34 per regime),
never a response-validation or test anchor. Reuse the trusted production
numerics for:

- RMSE and ACC by primary field;
- persistence and strict-train climatology curves;
- 10-90-day RMSE-AUC;
- 90-360-day RMSE-AUC/climatology;
- the flattening ratio `(E360-E270)/(E180-E90)`;
- finite-twin perturbation growth and `lambda_hat`;
- maximum normalized magnitude, finiteness, spatial variance, WBC/interior
  structure, and existing stability diagnostics.

The production **RMSE primary fields** are surface speed, SST, and surface
PHIHYD. ACC is not defined for speed: its four established fields are surface
U, surface V, SST, and surface PHIHYD, as uncentered wet-cell pattern
correlations about the strict-training time-mean climatology. Persistence and
climatology guards below refer to RMSE, not to invented ACC baselines.

### 16.2 View 2: response validation

Use only the 216 response-validation directions at days 5,400/5,760/6,120.
Their centres are disjoint from pilot/train/test centres. Every U/V/Theta level
is present, and the three specified multi-level combination types per variable
and regime (27 multi-level directions total) are held out from training.
The two validation views have distinct source starts, positions, and
observables but intentionally share part of the same validation chronology;
they are not claimed to be temporally independent sub-blocks.

Report at each lead, input family, output group, regime, region, and kernel:

- oriented signed response relative L2;
- wet-cell spatial pattern correlation;
- amplitude ratio `||r_F||/||r_M||` and `|log ratio|`;
- sign agreement;
- plus/minus central consistency for MITgcm and FNO;
- WBC versus interior and east/north/south scores;
- point versus smooth SSH;
- single-level versus held-out multi-level combinations;
- response norm, SNR, and lead dependence.

For an evaluation region $\Omega$ define conventional per-case relative L2

$$
E_{q,k,g,s}(\Omega)=
\left[
\frac{\sum_{c\in g,\Omega}(r_F-r_M)^2}
{\max\{\sum_{c\in g,\Omega}r_M^2,
N_{g,\Omega}n_{h(q),g,k}^2\}}
\right]^{1/2},
$$

where $n_{h,g,k}$ is the differentiated training-only combined noise floor
and $N_{g,\Omega}$ is the number of scored wet values. This is distinct from
the training RMS-normalized squared loss. For the primary score set
$\Omega=$ all wet cells, let $R$ denote the **input-centre** region, and first
average $E$ over signs, directions with centre in $R$, and regimes within each
`(input h, output g, R, lead k)` cell. Then define

$$
S_{resp}=\frac1{4\cdot4\cdot5}
\sum_{h,g,R}\left[
\frac79\bar E_{h,g,R,10}
+\frac1{36}\sum_{k=20,\ldots,90}\bar E_{h,g,R,k}
\right].
$$

The exact weights reproduce the training exposure: 77.78% at day 10 and
2.78% at each later lead. The long inventory must populate every displayed
cell; an empty cell fails inventory construction. Re-evaluate $E$ with
$\Omega=$ WBC/interior/east/north/south for output-region diagnostics, but do
not add another hidden weighting dimension to `S_resp`. Pattern correlation
and amplitude are independently reported guards, not folded into this score.
All models are scored in the one strict-training external normalization,
never their model-specific internal scale.
Correlation or amplitude for a truth response at/below its noise floor is
reported as undefined with the norm/SNR, never coerced to zero or one.

### 16.3 Checkpoint selection

First select B with the implemented parent forward rule adapted to the strict
validation starts. Its selected checkpoint becomes the matched forward
reference; B is never selected with response metrics.

For each C seed, a checkpoint is response-eligible only if:

1. every primary 10-90-day AUC is <=1.05 times B for the paired seed and within
   5% of C's own best candidate;
2. worst 90-360-day AUC/climatology is <=`min(0.85,1.05 x B)`;
3. ACC at days 30/60/90 is no more than 0.02 below B in any of surface U,
   surface V, SST, or surface PHIHYD; separately, for every day-90 RMSE
   primary field in which B beats both persistence and climatology, C must
   also beat both;
4. every rollout is finite and maximum normalized magnitude is <=1.05 times B;
5. primary stability uses `lambda_hat<=1.0`; if no candidate passes, a declared
   fallback permits only `lambda_hat <= B+0.002` and labels the selection as a
   fallback;
6. both weighted `S_resp` and its day-10-only counterpart are at least 20%
   lower than B overall and at least 10% lower within each input family;
7. no day-10 input-family/region aggregate is >1.10 times B, including WBC,
   and no primary output-group aggregate is >1.10 times B.

Among eligible candidates, choose the minimum response score. A tie within 2%
is broken by lower worst long/climatology ratio, then earlier step. If no
checkpoint is eligible, the seed/study fails; forward constraints are not
relaxed to obtain an adjoint candidate. The sole predeclared exception is a
separately named 90-day continuation if criterion 2 alone fails—that is, all
of criteria 1 and 3-7 pass. That trigger uses validation only;
the 60-day result remains reported, and the continuation must pass this same
rule before promotion. Its
starting C checkpoint is the minimum-response checkpoint among those passing
criteria 1 and 3-7 except criterion 2; its matched B continuation starts from paired B. Both
starting hashes are frozen before continuation training.

D and E use seed `20260911`, must pass forward criteria 1-5, and are selected
by minimum `S_resp` with the same tie breaks. For criterion 1, replace “C's
own best” with the evaluated arm's own best D or E candidate; the <=1.05
paired-B envelope is unchanged. Criteria 6-7 are
reported as mechanistic outcomes rather than required improvements. They have
no continuation escape. If no checkpoint passes criteria 1-5, freeze the finite
checkpoint with the smallest maximum primary-field 10-90-day AUC ratio to B,
then lower magnitude ratio, then lower growth, then lower `S_resp`; label it
forward-ineligible. This predeclared fallback keeps a mechanistic negative
control available but cannot be promoted as a successful emulator. D/E
identities are frozen before blind tests and cannot select or alter C.

The selected checkpoint is materialized and hashed. No adjoint result is part
of this procedure.

---

## 17. Blind forward-response test

The 216 directions at days 6,840, 7,560, and 8,280 are sealed before training.
They have disjoint centre IDs and held-out vertical combinations. They are not
used for amplitude calibration, lambda selection, early stopping, checkpoint
selection, or architecture choice.

After all B/C/D/E checkpoints, any pre-blind continuation decision/results,
and all associated selection reports are frozen:

1. evaluate A, ft90-context, B, C, D, and E on exactly the same cases;
2. report all validation metrics from section 16.2 without changing them;
3. report each of the three times and regimes, not only a pooled value;
4. retain point/smooth SSH, WBC/interior, and vertical-combination breakdowns;
5. mark A/ft90 results contextual because their old split was not blind to the
   native chronology.

**Predeclared success reading:** the primary `20260911` C checkpoint must lower
`S_resp` by at least 15% versus paired B, the median paired reduction over all
three seeds must be at least 15%, and at least two of three seeds must improve.
The primary seed must improve at least three of four input families and not
worsen any input-family/region aggregate by more than 10%. This test is
reported once. Failure produces a negative result, not a new checkpoint.

---

## 18. Blind MITgcm-adjoint test

### 18.1 Seal and timing

**Verified limitation.** Existing Phase-A numbers are readable in the
repository and summarized earlier in this document because the requested audit
required inspecting them. “Blind” can therefore mean a locked external test
not used for any new decision; it cannot honestly mean the investigators have
never seen the historical result.

Before response work begins, freeze hashes and deny all development code access
to:

- `outputs/af_fno/adjoint/mitgcm_s0_adjoint_v1/**`;
- `outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/**`;
- `outputs/af_fno/adjoint/comparison_phase_a_v1/**`;
- `outputs/af_fno/adjoint/fno_*adjoint*/**`;
- `/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1/**`;
- `/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v2/**`;
- every `ADJ*`, `adxx_*`, TAF log, gradient-check result, and derived copy.

Training configs retain `read_contract.adjoint_state=false`; CI scans for
forbidden paths. Training and selection run under a development identity whose
mount namespace excludes `outputs/af_fno/adjoint/**`, both raw scratch roots,
and the blind-response store; its Unix identity lacks read/search ACLs. A
separate evaluator identity has read-only access and an access log. Merely
marking files read-only is insufficient. The evaluator command becomes
available only after model, normalizer, data, lambda, amplitude, and selection-
report hashes are written to the freeze manifest.

### 18.2 Existing primary suite

Reuse the scalar-gate-validated MITgcm/TAF products; do not rerun them merely
to change provenance, but resolve the G1 plateau caveat at Gate A0. Run the
trusted FNO-side derivative machinery for:

1. frozen parent A;
2. all paired B and C replicates, with `20260911` primary;
3. frozen primary-seed D and E ablations;
4. the existing ft90 result as context.

Use source day 7,200 and leads 10, 20, 30, and 90 days for:

- point SSH anomaly objective;
- five-point meridional Gaussian SSH anomaly objective;
- analytic area-mean conservation probe;
- truth-forced and free FNO chains;
- existing fixed-source and backward-sweep views.

Report, without reselection:

- relative L2, pattern correlation, amplitude ratio, sign agreement;
- WBC/interior structure, radial decay, and lead dependence;
- conservation-mode error;
- absolute 12-bin spectra with the 4.625-cell FNO cutoff;
- the established unweighted metrics for continuity with Phase A, plus clearly
  labeled area-weighted companions if added;
- FNO finite-difference, forward/reverse-mode, masking, dtype, and hash gates.

The FNO spectral path must use the validated complex128 work-buffer correction;
casting weights to float64 without replacing neuralop's hard-coded complex64
buffer is not an acceptable “double precision” adjoint.

Truth-forced relative L2 is the primary local-Jacobian endpoint. The free-chain
metric is secondary because it includes trajectory-linearization drift.

### 18.3 Primary blind scientific endpoint

For model $m$ and seed $s$, define the truth-forced primary score

$$
S_{m,s}=\frac18\sum_{o\in\{point,kernel\}}
\sum_{k\in\{10,20,30,90\}}
\log[\max(E^{\rm relL2}_{m,s,o,k},10^{-12})].
$$

The two primary effects are
$\Delta_{A,s}=S_{C,s}-S_A$ and
$\Delta_{B,s}=S_{C,s}-S_{B,s}$; negative values improve on the historical
parent and prospective paired nominal control. The primary seed succeeds only
if $\Delta_{B,20260911}\le\log(0.8)$, $\Delta_{A,20260911}<0$, at least six of
the eight objective/lead relative-L2 cells improve versus paired B, and none is
>1.10 times B. Across replication, the median $\Delta_B$ must be
<=`log(0.9)` and at least two of three seeds must have $\Delta_B<0$.

Secondary endpoints are the free-chain score, D/E mechanistic contrasts,
higher pattern correlation, amplitude ratio closer to one, and improved mean-
mode preservation. Results are reported per seed; no best seed is chosen after
opening TAF data.

The scientific answer is positive only if the frozen forward gates/tests and
the quantitative adjoint criteria above all pass. A failure is publishable and
closes v1. Any later v2 must use a
new preregistered development cycle and, preferably, new sealed adjoint targets.

### 18.4 Additional final-only exploratory adjoint tests

These may be preregistered before training but generated/opened only after the
primary freeze:

- new runtime-weight SSH objectives at one interior and one eastern target;
- directional projections of raw MITgcm U/V/Theta adjoints onto the exact
  native-grid response kernels, comparing the valid scalars
  $\langle\nabla_{face}J,v_{face}\rangle$ for MITgcm and
  $\langle\nabla_{centre}J,Pv_{face}\rangle$ for the FNO, avoiding an
  unvalidated direct C-grid map;
- balanced/geostrophically consistent U/V/SSH perturbation projections;
- S1/S2 copies of the 10/30/90-day SSH suite with regime-correct forcing.

The existing tape is sized to 200 days. New objective weights do not require a
TAF rebuild, but U/V packaging and face-to-centre adjoint conventions are
**unresolved** and require independent gates before use.

Unless exact target coordinates/direction IDs are materialized and hashed in
the pretraining freeze manifest, these analyses are explicitly exploratory and
cannot rescue or overturn the section-18.3 confirmatory result. A genuinely
new confirmatory S1/S2/interior/eastern suite should be a separately frozen
contract, not chosen after inspecting v1 maps.

---

## 19. Forward figure and anomaly evaluation after training

After checkpoints are frozen and before TAF is opened:

1. Run the established 0-360-day validation package on strict validation.
2. Run a common blind-test package on the 15 exact starts per regime from
   section 6, including persistence and a strict-train climatology.
3. Run the standard S0 streamfunction figures at days 0-40, 60, and 2,000;
   RMSE/ACC through day 200; and RMSE through day 2,000.
4. Run the anomaly package after subtracting the same MITgcm mean over strict
   training days 0-5,039 from truth and every model. Never subtract a model's
   own mean.
5. Report anomaly RMS, spatial variance, streamfunction extrema,
   WBC/interior ratio, and zonal/meridional spectra.
6. Compare A, ft90-context, B, C, D, and E on identical starts and physical
   metrics. Retain the historical A/ft90 figure packages as a separate table.
7. Evaluate the blind forward-response package.
8. Freeze all outputs and only then enable the adjoint evaluator.

The 90-360 and 2,000-day runs are stability/statistical evaluation, not
pointwise training targets. The maximum deterministic supervision horizon
remains 90 days.

---

## 20. Required ablations and controls

### 20.1 Required for the main paper

1. **A versus B:** quantifies legacy frozen parent versus a new strict-split,
   parent-design reproduction. This is contextual because splits differ.
2. **B versus C:** primary causal comparison; only response supervision differs.
3. **B versus D:** determines whether nearby perturbed states alone help.
4. **C versus D:** distinguishes explicit response/JVP matching from ordinary
   perturbed-state fitting.
5. **C versus E:** tests whether sparse 20-90-day response targets add value
   beyond dense 10-day JVP constraints.
6. **Three paired B/C seeds:** separates a response effect from initialization.

### 20.2 Secondary or next-version

- matched parent nominal/response continuations described in section 5.1;
- matched B/C 90-day continuations only under the validation-only trigger in
  section 16.3, before any blind test;
- point-only versus smooth-only SSH and WBC-balanced versus area-proportional
  sampling only in a newly versioned study with a newly sealed blind-response
  set. No v1 blind outcome may trigger either training ablation.

The ft90 child is never a primary control, initialization, normalizer source,
or architecture definition.

---

## 21. Compute and storage estimate

### 21.1 MITgcm

The exact production/pilot/control budget is 182.0 model-years, 4,717,440
forward timesteps. A current S0 production manifest records 680.78 s for 3,600
model-days, or 2.63 ms/step on four MPI ranks. At that rate the exact workload
is about 3.4 four-rank wall-hours of pure integration; other segment manifests
support a conservative 3-6 h integration range. Startup, nine-segment chaining,
filesystem, and scheduler overhead dominate operations: the 2,472 logical
branches expand to exactly 6,552 validated 10-day segment launches. Reserve
**50-70 four-rank node-hours** and use job arrays/bundles. The
measured integration estimate and operational allocation must be reported
separately.

The dataset is intentionally many small forward runs rather than a few long
ones. No TAF license or adjoint tape is used.

### 21.2 FNO

The parent measured 3.215 V100-hours per seed. Budget approximately:

- B, three seeds: 9.7 GPU-hours;
- C, three seeds with 10-25% overhead: 10.6-12.1 GPU-hours;
- four-candidate lambda screen to 1,920 steps: about 3-5 GPU-hours;
- D and E primary-seed ablations: about 7-9 GPU-hours;
- evaluation/adjoints: <3 GPU-hours, mostly CPU for validated double precision.

Total primary allocation: **35-45 V100-equivalent GPU-hours**. Optional
continuations are budgeted separately.

### 21.3 Storage

- curated float64 response arrays plus deltas/nominals: 10-12 GB uncompressed,
  likely 4-8 GB compressed;
- 60 annual source pickups are existing; edited initial pickups need not all be
  retained once their byte edits and hashes are in the curated archive, but a
  recoverability policy must be frozen first;
- transient endpoint MDS/pickup scratch: approximately 50-100 GB;
- manifests/logs: <2 GB;
- models, optimizer checkpoints, reports, and figures: 15-25 GB for all seeds
  and ablations.

Reserve **150 GB scratch** and **40 GB durable project storage**. Compression
ratio and raw-retention policy are unresolved until the pilot is extracted.

---

## 22. Failure criteria and stop/go gates

### Gate D0 — inventory and split

- all source pickup/forcing/executable hashes resolve;
- all anchors lie in their declared split and every endpoint remains inside it;
- exact family/region/level counts hold;
- face/carrier labels, great-circle coordinates, and long-cell coverage pass;
- non-WBC validation/blind centres pass the cross-role three-index separation
  rule; WBC centres pass the distinct-ID/capacity rule and their achieved
  joint-maximin separation is recorded;
- pilot/train/validation/test centre IDs are disjoint within regime/family;
- duplicate boundary pickups hash-match and the downstream-source rule resolves;
- the strict-training normalizer hash is frozen before direction scaling;
- no blind path is visible to training configs.

Failure: stop before MITgcm.

### Gate D1 — pickup surgery

- only declared records/cells change;
- every untouched record is array- and byte-identical;
- plus/minus edits are exact sign reversals;
- native masks/support counts and both centred P32 states/deltas match the
  manifest and pass the 1% antisymmetry gate;
- nominal P32 projection matches Zarr at the anchor.

Failure: stop and fix/retest the generic editor; do not run a batch.

### Gate D2 — amplitude pilot

Every selected group amplitude must satisfy all section-10 linearity, SNR,
precision, and SSH-cap criteria. Failure in any group stops production
generation for all groups so the dataset remains one frozen design.

### Gate D3 — response dataset

- all nominal and signed branches complete;
- control reruns meet the numerical floor contract;
- every train/validation pair satisfies `Q_lin<=0.05` and `Q_SNR>=20` at
  every available lead using the final combined floor; a failure invalidates
  the dataset version rather than dropping/rescaling that case;
- response extraction reproduces direct pickup differences;
- no NaN/Inf; counts, signs, leads, and hashes are exact;
- validation/test groups are inaccessible to the train loader.

Failure: quarantine incomplete cases; do not silently reduce counts.
If a later version changes amplitude after observing a validation linearity/
SNR failure, those cases become development data and that version must create
new response-validation and blind inventories; it may not reuse the failed
held-out cases as if untouched.
Blind-test `Q_lin`/SNR values are computed only after freeze and reported for
every case. A blind failure labels that case outside the calibrated response
regime but cannot trigger amplitude, inventory, model, or checkpoint changes.

### Gate M0 — nominal control recovery

B must have no nonfinite 360-day rollout and must satisfy all of:

- 10-90-day AUC <=0.1011 speed, <=1.027 SST, and <=0.7240 PHIHYD (125% of
  the documented parent values, rounded upward);
- worst 90-360-day AUC/climatology <=0.40;
- day-90 RMSE below at least one of persistence/climatology in every primary
  field and below both in at least two fields;
- `lambda_hat<=1.02` and maximum normalized magnitude <=6.0.

Because A used a different split, these are reproduction-scale gates rather
than a claim of identical validation samples. If B is inadequate,
freeze a negative reproduction report and close v1. Any adjustment to the
shared B/C nominal protocol requires a newly versioned, re-reviewed contract
before response-aware training; it is not an in-run rescue.

### Gate M1 — response-aware eligibility

C must satisfy the section-16 forward envelope and response improvement.
If criterion 2 alone fails while criteria 1 and 3-7 pass, hold the candidate
as provisional and invoke the predeclared validation-only continuation trigger.
Any other failure means no C checkpoint is promoted and TAF remains sealed.

### Gate M2 — frozen forward tests

Blind nominal preservation requires every C 10-90-day primary-field AUC
<=1.05 times paired B, worst 90-360-day AUC/climatology
<=`min(0.85,1.05 x B)`, no nonfinite rollout, maximum normalized magnitude
<=1.05 times B through 360 days, and each day-2,000 primary RMSE plus maximum
magnitude <=1.10 times B. The blind response criteria are frozen in section
17. Both tests are run once. A failure is a negative result; no checkpoint or
lambda is changed. If an M1-eligible model was frozen before
the blind tests, the preregistered adjoint evaluation proceeds regardless of
the blind forward outcome, provided technical Gate A0 passes. This avoids
outcome-dependent absence of the mechanistic Jacobian result; a forward-test
failure still makes the overall “improve while preserve” conclusion negative.

### Gate A0 — adjoint pipeline

Parent/B/C/D/E FNO finite differences, forward/reverse identity, dtype, masks,
checkpoint/normalizer hashes, and weight-field hashes must pass before
comparison. The existing MITgcm G0-G5 scalar gates remain required. Extend the
one offshore G1 curve whose current minimum is at `epsilon=1e-5` with
predeclared `1e-6` and `1e-7` forward differences; an interior minimum must be
obtained or the reference retains a failed plateau flag. The new evaluator
must enforce that flag rather than merely print it.

Because v2 G0 currently checks ETAN only, add a final-evaluation forward-only
F90 extraction of U/V/Theta/ETAN at FNO 10-day nodes and require the canonical
P32 projection to match trajectory-v3 there. The temporal qualifier is
mandatory: final reports may say “46-channel G0 at FNO 10-day nodes; ETAN
daily,” but may not imply that all 46 channels were checked at all 91 daily
outputs. Until that extension passes, reports must say “ETAN-only daily G0.”
These technical extensions occur after model freeze and cannot influence any
model decision.

### Gate A1 — scientific result

No threshold selects a model after TAF. Report the predeclared primary and
secondary endpoints. V1 supports the full hypothesis only if M2 blind nominal
preservation, the section-17 blind response criteria, and every section-18.3
primary adjoint criterion pass. Failure of any component, including a forward/
adjoint tradeoff, rejects “improve the adjoint while preserving forward skill”
for v1 even if a mechanistic sub-result improves.

---

## 23. Exact implementation files to create or modify after approval

No files below are implemented by this planning task.

### 23.1 Create

**Contracts/configs**

- `config/forward_response_amplitude_pilot_v1.json`
- `config/forward_response_dataset_v1.json`
- `config/forward_response_schema_v1.json`
- `config/model_c_adjoint_faithful_nominal_control_v1.json`
- `config/model_c_adjoint_faithful_response_v1.json`
- `config/model_c_adjoint_faithful_perturbed_state_control_v1.json`
- `config/model_c_adjoint_faithful_short_response_v1.json`
- `config/adjoint_faithful_forward_evaluation_v1.json`
- `config/adjoint_faithful_blind_adjoint_evaluation_v1.json`
- `config/adjoint_faithful_firewall_v1.json`

**MITgcm response generation**

- `scripts/build_forward_response_inventory.py`
- `scripts/stage_forward_response_run.py`
- `scripts/extract_forward_response_dataset.py`
- `scripts/verify_forward_response_dataset.py`
- `scripts/freeze_adjoint_faithful_study.py`
- `scripts/verify_adjoint_faithful_firewall.py`
- `slurm/mitgcm/af_forward_response_array.sbatch`

**FNO data/loss/training**

- `src/oceanfno/chronological_dataset.py` (thin strict-split adapter over the
  trusted production loader/normalizer, not a second data implementation)
- `src/oceanfno/response_dataset.py`
- `src/oceanfno/response_objective.py`
- `src/oceanfno/response_spectral_context.py`
- `src/oceanfno/train_response_study.py`
- `src/oceanfno/validation_response.py`
- `src/oceanfno/figures_response.py`
- `src/oceanfno/anomaly_response.py`
- `slurm/models/c/train_adjoint_faithful_nominal_control_v1.sbatch`
- `slurm/models/c/train_adjoint_faithful_response_v1.sbatch`
- `slurm/models/c/figures_adjoint_faithful_response_v1.sbatch`

**Blind adjoint adapters**

- `scripts/fno_adjoint_model.py` (contract-parameterized adapter of the trusted
  ft90 runner, retaining its double-precision spectral fix)
- `scripts/compare_adjoint_models_response_v1.py`

**Tests**

- `tests/test_forward_response_inventory.py`
- `tests/test_forward_response_pickup.py`
- `tests/test_forward_response_dataset.py`
- `tests/test_response_objective.py`
- `tests/test_response_spectral_context.py`
- `tests/test_response_training.py`
- `tests/test_response_validation.py`
- `tests/test_response_blindness.py`
- `tests/test_fno_adjoint_model.py`

### 23.2 Modify minimally

- `archive/src/bire_repro/af_s0_twin.py`: factor its trusted byte-level editor
  into a generic record/cell edit function while retaining the existing global
  U/V scaling wrapper byte-for-byte compatible.
- `pyproject.toml`: add entry points only after the new modules exist; preserve
  the user's current unrelated modification.

### 23.3 Reuse unchanged

- `src/oceanfno/model.py`
- `src/oceanfno/objective.py`
- `src/oceanfno/spectral_norm.py`
- `src/oceanfno/pressure_gradient.py`
- `src/oceanfno/continuity.py`
- `src/oceanfno/barotropic_transport.py`
- numerical routines in `src/oceanfno/figures.py` and `anomaly.py`
- `scripts/adjoint_metrics.py`
- `scripts/stage_adjoint_run.py`
- `af_fno/mitgcm/code_ad/**`, `input_ad/**`, and `tamc.h`
- all existing parent/ft90 configs, outputs, checkpoints, reports, and docs.

The new adapters import trusted numerics; they do not fork or silently edit the
frozen production parent.

---

## 24. Reproducibility and provenance requirements

Before each phase, write an immutable contract plus SHA-256. Every report must
include:

- repository commit, dirty-worktree diff hash, Python lock/environment, NumPy,
  Torch, neuralop, CUDA, compiler, and MPI versions;
- MITgcm commit and executable hash;
- source pickup, metadata, forcing, grid, namelist, and static-input hashes;
- trajectory-v3 metadata/manifest hash (`766cae893593...` currently);
- exact split arrays, anchor/centre inventory, level and region counts;
- pilot raw metrics, nominal/perturbed repeats, tight-CG comparisons, combined
  floor, selected amplitudes, and decision trace;
- response-store metadata/chunk/compressor hashes and per-array digests;
- model architecture/loss contract, seeds, batch order, optimizer, learning
  rates, response schedule, lambda screen, and checkpoint steps;
- pre/post auxiliary spectral power-vector hashes and triplet-order tests;
- normalizer/increment-scale hashes;
- materialized checkpoint and optimizer-step hashes;
- both validation views and the deterministic selection trace, including any
  fallback;
- hardware, Slurm IDs, wall times, failures/retries, and quarantined cases;
- blind freeze timestamp, ACL/mount proof, evaluator access log, and hashes of
  every artifact existing before access;
- final evaluator version and all MITgcm/FNO gate results.

Randomness is limited to the three declared model seeds. Spatial selection is
hash/maximin deterministic. Completed artifacts are write-once. Reports state
inclusive and half-open day conventions side by side to prevent off-by-one
leakage.

The existing adjoint products must be exposed only inside the evaluator mount;
the development identity must have no read permission. Repository readability
in the present audit checkout is not a sufficient blind barrier, so model runs
must use the sanitized development checkout/mount contract above.

---

## 25. Numbered execution order

1. Review and approve this document; make no compute submission before approval.
2. Freeze the strict split and compute/hash the one strict-training normalizer,
   increment scales, and climatology before any standardized perturbation is
   constructed.
3. Freeze model seeds, region masks, kernel definitions, counts, leads,
   candidate alphas/lambdas, and the exact ACL/forbidden-path list in configs.
4. Materialize the deterministic pilot/train/validation/test centre and level
   inventory; verify all counts/full supports and freeze hashes.
5. Generalize and unit-test the trusted pickup editor; prove the old twin path
   remains compatible and only requested bytes change.
6. Stage six nominal pilot branches and their duplicates; run the forward-only
   amplitude pilot at both signs and three alphas.
7. Extract P64 responses and choose four provisional amplitudes; run the 12
   selected-alpha perturbed duplicates plus tight-CG signed/nominal controls,
   apply every section-10 gate, and freeze final amplitudes. Stop if any group
   fails.
8. Generate shared nominal response branches for training and validation only;
   verify their P32 projections against trajectory-v3.
9. Generate signed training and validation response branches, short first and
   then the predeclared long subset. Do not generate/read blind response data.
10. Extract and validate the train/validation response store; freeze response
   scales and all data hashes.
11. Train B for the three seeds with the parent 60-day objective/schedule;
    select B using nominal validation only.
12. Run the four-lambda 1,920-step screen on forward and response validation;
    freeze lambda without adjoint access.
13. Train C for the three paired seeds; train D and E for the primary seed.
14. Apply the two-view checkpoint rule; freeze selected checkpoints,
    normalizers, reports, and the complete model-development manifest. Stop on
    an M1 failure unless its sole cause is the predeclared criterion-2
    continuation trigger.
15. Run the normal forward validation and non-blind figure/anomaly preflights.
16. If and only if the validation-only continuation trigger in section 16.3
    fired, run the separately named matched B/C continuation, apply the same
    validation rule, and freeze it. Make this decision before any blind
    nominal or response result is generated or read. Complete any separately
    preauthorized parent-retrofit pair in this same pre-blind phase or defer it
    to a new version.
17. Run the full common forward figure, anomaly, streamfunction, and blind-
    nominal packages for A, ft90, B, C,
    D, E, and any already-frozen continuation. Freeze the complete forward
    paper package. No blind result can trigger another training run.
18. Generate/extract the sealed blind forward-response store using its already
    frozen inventory; run the response evaluation once and freeze it.
19. Enable the evaluator-only adjoint path. Run FNO gates for parent, B, C, D,
    and E; verify/extend the MITgcm/TAF gates exactly as in A0.
20. Compare parent, B, C, D, E, and ft90 context against the existing Phase-A
    objectives at 10/20/30/90 days. Report all seeds and metrics without
    reselection.
21. Run only the preregistered exploratory adjoint projections/objectives, if
    their independent gates pass.
22. Produce final tables/figures: nominal forward, anomalies, blind responses,
    JVP/adjoint metrics, spectra, conservation, controls, compute, and failures.
23. Archive the full provenance bundle and write the paper conclusion answering
    the single question: did forward-only perturbation-response supervision
    improve the learned Jacobian/adjoint while preserving forward skill?

---

## Frozen proposed contract

This table is the concise proposal to approve. Fields whose numerical outcome
requires the forward-only pilot/screen are frozen as **procedures**, not invented
values.

| Contract item | Frozen proposal |
| --- | --- |
| Baseline model | Frozen `model_c_production_1in_1out_spectralnorm_v1`; architecture/objective source of truth; checkpoint not loaded by primary new arms |
| Context-only child | `model_c_production_1in_1out_spectralnorm_ft90_v1`; no weights, normalizer, or optimizer state used |
| New model | `model_c_adjoint_faithful_response_v1`, random initialization, paired with nominal arm B |
| State channels | U15 + V15 + Theta15 + SSH1 = 46; unchanged; no adjoint outputs |
| Static channels | parent five physical statics; unchanged |
| Nominal train/validation/test | train 0-5,039; buffer 5,040-5,129; validation 5,130-6,389; buffer 6,390-6,479; blind test 6,480-8,999 |
| Nominal validation/test starts | validation `5130+12m`, m=0..33 per regime; blind test 15 listed starts 6480-6999 per regime |
| Response anchors | train 14/regime at 0:360:4680; validation 3/regime at 5400/5760/6120; blind test 3/regime at 6840/7560/8280 |
| Response counts | train 224 directions/regime =672 total, 96 long; validation 72/regime =216, 36 long; blind test same; both signs |
| Perturbation families | native-face 5x5 Gaussian U/V; tracer 5x5 Gaussian Theta; SSH point and 5x5 Gaussian; sigma1, radius2, unit L2, full active support only |
| Spatial sampling | exact region counts in section 9; WBC approximately one third and oversampled; interior/east/north/south retained; joint deterministic maximin/hash allocation; split-disjoint IDs everywhere, hard cross-role distance-three outside capacity-limited WBC |
| Vertical sampling | train single-level, all 15 levels >=3 times per variable/regime; validation/test all levels plus frozen unseen 2/3-level combinations |
| Amplitudes/calibration | forward-only +/- pilot at alpha `{0.025,0.05,0.10}`; choose largest passing <=5% sign asymmetry, SNR>=20, P32 realization/antisymmetry, adjacent-alpha, SSH <=1 cm, perturbed-repeat, and tight-CG 1% gates; separately U/V/Theta/SSH |
| Dense-short/sparse-long | all directions at 10 d; long subsets at 10,20,...,90 d; no target beyond 90 d |
| MITgcm cost | 65,520 model-days =182.0 model-years including pilot, nominal, duplicate-perturbation, and tight-CG controls |
| Restart semantics | edit only selected Uvel/Vvel/Theta or EtaN cells; Salt, AB histories, dEtaHdt, EtaH, and every unselected record byte-identical; never reconstruct pickup from Zarr |
| Parent loss | eight implemented terms and coefficients unchanged |
| Response loss | signed oriented response/JVP error; equal input/output group balance; long lead mean; training-only response scales/noise floors |
| Response mixing | nominal batch8 every update; one batched nominal/-/+ response triplet every fourth update; spectral power-vector snapshot/restore; 75% short/25% long auxiliary sampling |
| Response weight | choose from `{0.03,0.10,0.30,1.00}` by exact forward-only 1,920-step validation screen |
| Primary rollout/training | from scratch, 60 d nominal rollout, 7,680 updates, parent LR/Adam/spectral-norm schedule, seeds 20260911/12/13 |
| Checkpoint selection | forward AUC/ACC/baseline/stability envelope relative to paired B, then weighted dense-short/sparse-long held-out response score and no-catastrophic-cell guard; no adjoint metric |
| Blind forward tests | strict-v3 nominal test and 216 direction pairs / 432 signed perturbed branches; separate evaluator-only store; run once after model freeze |
| Blind adjoint tests | existing scalar-gated S0 point/kernel/mean suite at 10/20/30/90 d plus A0 plateau/full-state extensions; parent/B/C/D/E/ft90 context; exploratory projections only after freeze |
| Required controls | A, B, C; perturbed-state-only D; short-only E; three paired B/C seeds; optional matched parent continuation kept separate |
| Forbidden data | no programmatic access or decision use of any MITgcm/TAF adjoint, `ADJ*`, `adxx_*`, adjoint-derived metric/map, blind response/test product, or new FNO adjoint map during development/selection; historical summaries above are rationale only |

**Approval boundary:** stop here. The next authorized action, if this contract is
approved, is implementation and no-compute inventory/pickup unit testing—not an
MITgcm submission and not FNO training.
