# Local-24 fine-tune of the six-step Bire-protocol model

Warm start from `model_c_bire_protocol_rollout_ft_v1`'s selected checkpoint
(`selected.pt`, optimizer step 3,840); only
`model_state_dict` was loaded and the optimizer was reset.

| quantity | incumbent | local-24 arm |
| --- | --- | --- |
| Fourier modes (Y, X) | 24 x 16 | 24 x 24 |
| raw-input local branch | none | bias-free 49 -> 46, 3 x 3 |
| rollout calls / weight | 6 / 0.5 | 6 / 0.5 |
| initial learning rate | 2e-05 | 2e-05 |
| batch / optimizer steps | 4 / 3,840 | 4 / 3,840 |

The old 24 x 9 realized spectral tensors are copied into the new 24 x 13
tensors and the four added zonal coefficients are zero. The bias-free local
branch is also zero-initialized. Therefore the warm start is function-preserving
before optimization. Static inputs, positional encoding, output channels,
split, normalizers, loss, optimizer schedule, seed and checkpoint selection are
asserted equal to the archived incumbent.

The unchanged objective is:

    L = L_state + 0.001 L_increment + 0.5 (1/5) sum_(k=2..6) L_state^(k)
      + 1e-5 (1/6) sum_(k=1..6) L_spectral^(k) + 0.065 (1/6) sum_(k=1..6) L_boundary^(k)

with U, V, temperature and SSH weighted 0.25 each. The prediction is
fed back at steps two through six; there is no teacher forcing after the initial
state. Six-step objective SHA-256: `0cc75764a6eec7c7e82bb4cdb97e51929b57b1fbf765a7ee50c290ba3593bdbb`.

Training draws 5,940 starts per regime
(17,820 pooled), the latest being
5,939, so every six-step target sequence stays inside
training 0--5999.

Selection is unchanged: minimise the worst 90--360-day RMSE-AUC relative to
climatology subject to each field's 10--90-day AUC staying within 5% of the best
fine-tuning checkpoint, on 102 pooled validation rollouts.

| step | short AUC 10--90 | long AUC / climatology | short AUC / step-3,840 |
| --- | --- | --- | --- |
| 960 | 0.113/1.260/0.521 | 0.573/0.706/0.273 | 0.983/0.991/1.092 |
| 1,920 | 0.112/1.227/0.467 | 0.540/0.653/0.227 | 0.975/0.965/0.979 |
| 2,880 | 0.109/1.196/0.500 | 0.515/0.659/0.250 | 0.956/0.941/1.048 |
| 3,840 | 0.108/1.194/0.433 | 0.528/0.669/0.233 | 0.947/0.939/0.907 |

The step-3,840 baseline scores
0.115/1.271/0.477 short and
0.602/0.709/0.293 long on the
same rollouts, in the order surface_speed, sst, phihyd_surface.

Selected step 3,840 via `primary_rule`.

Acceptance gate, validation half: no 10--90-day field worsens by more than 5%
against the baseline -- **pass**;
worst 90--360-day climatology ratio 0.669 <= 0.85
-- **pass**; no worse than the
incumbent -- **pass**. The 2,000-day and
visual conditions are evaluated by the figure package, which is the only stage
that runs a 2,000-day rollout.

Training and validation only; the inference set opens through the figure
contract, S0 only.

Report content SHA-256: `4b5392a53aa25785bb6a162de3751b441e592243cd673f33b6e30748f20561c5`.
