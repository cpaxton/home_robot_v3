# Direct hand-to-base readout removal is not a standalone fix

Reject the intervention. On the rejected262k context final, removing direct
hand-pose-error contributions from base outputs improves full arm success
29/64→32/64 but regresses coupled success4/8→3/8. Base success stays42/50.
Arm tracking improves63/64→64/64, while32 arm jerk failures remain. No recorded
force/wheel/invalid-physics/retention flags trip in this paired free-space test.

This is a causal simulator diagnostic, not a modified production actor. The
baseline reproduces all122 prior context physical cases exactly. Only when
base x/y/yaw and hand masks are all active, zero the direct current/future hand
error columns in the two base readout rows. Retain every arm/grip output,
context/history, base-error and joint/base-velocity feedback and yaw-mode blend.
Base-only, free-base and partially constrained goals remain unchanged. All
physical limits, inputs, weights and teacher budgets remain fixed.

[All paired cases and artifact identities](../../paper/data/whole_body_control/mars_constrained_base_readout_v1/results.json)
keep the unchanged weights hash plus the explicit intervention/script digest.
The in-memory hook is ineligible for canonical qualification. No policy is
exported, adopted or sent to hardware. Direct hand-error leakage contributes
to some failures, but removing it does not explain or repair the larger
stability regression; do not hard-code this restriction into the controller.
