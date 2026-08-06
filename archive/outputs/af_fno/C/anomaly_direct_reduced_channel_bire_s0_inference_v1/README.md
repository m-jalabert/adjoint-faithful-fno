# Model C Arm R: S0 Bire-style Figures 3--8

This package evaluates the ten-output reduced-channel causal control under
the S0 control wind (tau0 = 0.1 N m-2) at a ten-day prediction interval.

Arm R retains selected Model C's pointwise anomaly normalization, direct-state
map, width 128, four FNO blocks, 24x16 modes, 10% padding, optimizer, and
three-step objective form. Its autoregressive state contains only surface/mid
U, V, and temperature; surface/mid/bottom PHIHYD; and barotropic
streamfunction.

The 15 starts, continuous day-2000 MITgcm truth, persistence, S0 split-1
climatology, metric reductions, and six filenames match the source
`anomaly_direct_bire_s0_inference_v1` evaluation. Figure 6 compares the
retained 46-channel source map (black) with Arm R (red); it is the direct
channel-count causal comparison.

Training-only checkpoint gate passed: true.
S0 10--90-day deterministic gate passed: true.
Report content SHA-256: `e4b8bb1316d39a7acb346d357cec8890366dab013e9689e8054ffcea6e8d05c1`.
