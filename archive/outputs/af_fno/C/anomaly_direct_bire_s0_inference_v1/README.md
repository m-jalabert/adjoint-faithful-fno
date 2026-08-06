# Selected Model C: S0 Bire-style Figures 3--8

This package evaluates the frozen seed-20260724, step-13440 pointwise-anomaly
direct-state Model C under the control wind (tau0 = 0.1 N m-2). All forecasts
use a ten-day prediction interval.

The 15 initial conditions were randomly drawn without replacement from the
fresh late S0 inference block before model metrics. MITgcm truth through day
2000 combines the immutable trajectories-v2 prefix with evaluation-only job
304735; none of the continuation is training data.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with the
selected anomaly-direct Model C. It is an architecture-direction comparison,
not a literal pretrained/fine-tuned pairing.

Report content SHA-256: `9711016c7fca90478d9c3e14c118b5fb104af19c74876eafba37c528ea12e022`.
