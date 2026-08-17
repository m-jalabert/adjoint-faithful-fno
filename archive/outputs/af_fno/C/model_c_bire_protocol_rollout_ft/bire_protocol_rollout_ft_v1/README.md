# Six-step rollout fine-tune of the Bire-protocol step-15,360 model

Warm start from `model_c_bire_protocol_duration_v1`'s selected checkpoint
(`selected.pt`, optimizer step 15,360); only
`model_state_dict` was loaded and the optimizer was reset.

| quantity | step-15,360 arm | this arm |
| --- | --- | --- |
| rollout calls | 3 | 6 |
| rollout weight | 0.15 | 0.5 |
| initial learning rate | 5e-4 | 2e-05 |
| batch size | 8 | 4 |
| optimizer steps | 15,360 | 3,840 |
| decay step (0.75 x budget) | 11,520 | 2,880 |

Batch 4 over six unrolled calls carries the same activation load as batch 8 over
three (4 x 6 = 8 x 3). The architecture, Fourier modes,
static inputs, positional encoding, 46-channel output, Bire Section 3.2 split,
and train-only pointwise normalizers are unchanged and asserted field by field
against the parent contract; the normalizers are **reused from the parent
package rather than recomputed**, so the fine-tuned weights stay in exactly the
normalized coordinates they started in.

The objective adds three self-generated steps to the rollout term:

    L = L_state + 0.001 L_increment + 0.5 (1/5) sum_(k=2..6) L_state^(k)
      + 1e-5 (1/6) sum_(k=1..6) L_spectral^(k) + 0.065 (1/6) sum_(k=1..6) L_boundary^(k)

with U, V, temperature and SSH weighted 0.25 each, as before. The prediction is
fed back at steps two through six; there is no teacher forcing after the initial
state. Six-step objective SHA-256: `0cc75764a6eec7c7e82bb4cdb97e51929b57b1fbf765a7ee50c290ba3593bdbb`.

Training draws 5,940 starts per regime
(17,820 pooled), the latest being
5,939, so every six-step target sequence stays inside
training 0--5999.

Selection is unchanged: minimise the worst 90--360-day RMSE-AUC relative to
climatology subject to each field's 10--90-day AUC staying within 5% of the best
fine-tuning checkpoint, on 102 pooled validation rollouts.

| step | short AUC 10--90 | long AUC / climatology | short AUC / step-15,360 |
| --- | --- | --- | --- |
| 960 | 0.119/1.348/0.516 | 0.652/0.792/0.344 | 0.961/0.907/0.810 |
| 1,920 | 0.117/1.306/0.483 | 0.612/0.727/0.280 | 0.946/0.879/0.758 |
| 2,880 | 0.114/1.270/0.509 | 0.581/0.690/0.279 | 0.923/0.855/0.799 |
| 3,840 | 0.115/1.271/0.477 | 0.602/0.709/0.293 | 0.926/0.855/0.749 |

The step-15,360 baseline scores
0.124/1.486/0.637 short and
0.716/0.932/0.419 long on the
same rollouts, in the order surface_speed, sst, phihyd_surface.

Selected step 3,840 via `primary_rule`.

Acceptance gate, validation half: no 10--90-day field worsens by more than 5%
against the baseline -- **pass**;
worst 90--360-day climatology ratio 0.709 <= 0.85
-- **pass**. The 2,000-day and
visual conditions are evaluated by the figure package, which is the only stage
that runs a 2,000-day rollout.

Training and validation only; the inference set opens through the figure
contract, S0 only.

Report content SHA-256: `118318b46594e76d05baa6f6c0ca46a7acc199a767451e5d262894aeb549ee67`.
