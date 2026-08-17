# 2-in / 1-out physical-static Model C + pressure gradient + continuity

Loss-only fine-tune of `model_c_2in_1out_new_channels_pressure_gradient_v1`. Architecture, inputs, 32 x 32 modes,
local 3 x 3 branch, six-step autoregression, dataset, normalizers, optimizer
schedule, validation records and checkpoint selection are unchanged.

The retained objective is

    L = L_parent + 0.05 * L_pressure_gradient + 0.05 * L_continuity

`L_pressure_gradient` is inherited unchanged from the parent: it reconstructs
total MITgcm PHIHYD from predicted THETA and ETAN, forms neighboring-tracer
horizontal gradients on C-grid velocity faces, and compares them to truth with a
dimensionless relative-L2 metric.

The sole scientific addition is `L_continuity`. For each ten-day step it
depth-integrates the predicted velocity channels with the tutorial's exact 15
`delR` thicknesses into a barotropic transport `Q`, forms the free-surface
residual

    R = (eta_next - eta_now) / 10 days + div((Q_now + Q_next) / 2)

at interior tracer points, and scores the prediction against the *truth*
residual as `||R_pred - R_truth||^2 / (||R_truth||^2 + eps)`. It is
truth-referenced rather than driven to zero because ten-day sampling and the
centered U/V representation only approximate MITgcm's native discrete
continuity operator. The residual is chained through the rollout exactly as the
states are, and all six calls carry equal status.

Selected optimizer step: 3,840.
Parent initialization is strict same-shape and function-preserving; Adam state is
reset.
