# Model C S0 stability/tangent evaluation recovery

Job 304750 failed during float32 diagnostic reduction after model rollout. This zero-retraining recovery uses float64 physical reductions and explicitly censors metrics after a member's first non-finite model state. It changes no model or checkpoint.

Report content SHA-256: `05e92283b2a067ac703da089dc49b9332f4d8769a14ab0230352e28b674700dd`.
