# Exact objective replay points toward optimization, not a reward-only fix

All292 deterministic source/final rollouts reproduce the saved native actions,
rewards, poses, episode lengths and physical contract outcomes exactly under
the actual full-fraction `TeacherContractReward` wrapper. All67 arm cases losing
full success earn lower final discounted shaped return; none earns more. The
two base cases losing full success also earn less.

| Deterministic shaped reward per control step | Source | Rejected final |
|---|---:|---:|
| Base |0.114287|0.120813|
| Arm |0.364793|0.328380|
| Coupled |0.174864|0.174039|
| Declared stage/primitive-budget weighted |0.251655|0.235242|

The weighted diagnostic worsens6.52%. It uses only training cases and their
declared control-step stage/primitive budgets and within-bucket case cycles.
It is an undiscounted deterministic episode-derived average, not the stochastic,
discounted, bootstrapped PPO objective. Per-case discounted and undiscounted
returns and all terms remain explicit, with validation cases separate. This
does not establish which optimizer, critic or exploration effect caused the
regression, but it does not support assuming that these failures earn better
reward or that a larger jerk penalty alone repairs learning.

Actual noisy training continues to receive few full-contract successes after
the reward ramp: arm48/495, base41/293 and coupled8/225 completed episodes
in full-fraction rollout records after36864transitions. These are observations
of changing actors and sampled cases, not matched deterministic success rates.

A separate local input-sensitivity audit uses the same56 saved pregrasp inputs
for the source and both rejected finals. The median yaw-command sensitivity to
measured yaw rate changes from−0.00214 in the source to+0.10439 in the context
final and+0.09699 in the translation final. These are dimensionless diagonal
entries in a local input/output Jacobian, not measured acceleration or a
closed-loop stability certificate; positive values below one do not by
themselves establish instability of the underlying velocity servo.

The direct yaw-rate gain changes−0.01269→+0.57379 in normalized readout units.
Its state-independent bias changes−0.00229→+0.52799; context contributes the
remainder. About90% of the direct-gain change on these inputs comes through
the bias. Forward-velocity gain shows a similar large bias contribution.
Context itself depends on velocity, so direct-gain decomposition and total
Jacobian are distinct measurements.

The prior recipe applies actor LR3e-6 but a300× gain-bias multiplier, giving
bias LR9e-4. That is a concrete next optimization variable, not proof of the
cause. The matched control changes only that multiplier300→1. Keep the same
source, data, reward, seed, noise, critic multiplier100, context trainability,
normalization and grip head. Require exact initialized source behavior and
both self/previous-run physical equality at the4096 critic-only boundary.
The fixed final remains262144, with all prior subset/fresh/TAMP gates and no
intermediate selection. No runtime filter, sign projection, release-authority
change or hardware motion is introduced.

[Full replay, terms and sampling ledger](../../paper/data/whole_body_control/mars_objective_audit_v1/results.json),
[local velocity sensitivity](../../paper/data/whole_body_control/mars_objective_audit_v1/velocity-feedback-sensitivity.json),
and [gain decomposition](../../paper/data/whole_body_control/mars_objective_audit_v1/gain-velocity-decomposition.json)
retain source/script/input identities and every case. Raw traces and models
remain ignored. Nothing in this audit qualifies a policy for hardware.
