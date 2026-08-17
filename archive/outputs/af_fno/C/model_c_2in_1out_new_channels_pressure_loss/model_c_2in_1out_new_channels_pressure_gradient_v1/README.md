# 2-in / 1-out physical-static Model C + pressure-gradient consistency

Loss-only fine-tune of `model_c_2in_1out_new_channels_v1`. Architecture, inputs, 32 x 32 modes,
local 3 x 3 branch, six-step autoregression, dataset, normalizers, optimizer
schedule, validation records and checkpoint selection are unchanged.

The sole scientific addition is

    L = L_parent + 0.05 * L_pressure_gradient

where `L_pressure_gradient` reconstructs total MITgcm PHIHYD from predicted
THETA and ETAN, forms neighboring-tracer horizontal gradients on C-grid velocity
faces, and compares them to truth with a dimensionless relative-L2 metric. The
hydrostatic integration uses the tutorial's exact 15 `delR` values and reference
temperature profile already validated against MITgcm PHIHYD in the archived
pressure diagnostic.

Selected optimizer step: 3,840.
Parent initialization is strict same-shape and function-preserving; Adam state is
reset.
