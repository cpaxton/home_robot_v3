# Stationary-reference exploration training

This fixed-budget pair follows the [completed critic pilot](finite_critic_pilot_20261117.md)
and [architecture review](../rl/WHOLE_BODY_ARCHITECTURE_REVIEW.md). The retained
policy is insufficient; the experiment is designed to improve physical tracking
and settling, rather than merely preserve its current score.

Both arms start from the same retained actor and fresh critics, seed 107,
16,384 decisions with 1,024 critic-only warmup, unchanged legacy rewards,
corrected finite-contract deadlines, three independent critic epochs, std 0.02,
and the previous actor LR 5e-5 → 1e-5 / clip 0.05 / target KL 0.01 settings.
Outer general/TAMP exposure is 50/50 by control ticks; general is 80/20 hand/base.
The identical twelve train and twelve audit cases, ten existing coupled cases,
and six unloaded TAMP prefixes retain their goals and thresholds.

`ordinary_noise` uses the original distribution. `quiet_reference_noise` scales
only motion exploration by 0.125 when present/future pose targets and masks are
constant and announced twist is exactly zero. Upcoming motion/mask changes keep
full noise. Measured error or motion never suppresses correction commands.
Gripper noise stays unchanged; sampling, likelihood and entropy use the same
conditioned Gaussian. No output smoothing, teacher loss or actor inference change.

The initializer, critic parameters and first observation must match exactly.
The first sampled rollouts are expected to diverge because exploration differs;
this pair must not claim identical first actions. Audit cases never enter PPO,
no intermediate checkpoints are selected, and no automatic extensions occur.
Evaluate fixed final actor means on identical physics/reference fingerprints,
reporting hand accuracy, quiet settling, base drift, jerk, contact, wheel support
and confirmed stops. All case failures remain in the denominator.

The smaller frozen-actor probe is diagnostic: 0/8 → 3/8 stochastic successes,
all 16 stops confirmed, same one unsafe-contact identity. It is not a trained
policy improvement. This new source requires its own complete regression receipt
and fresh native teacher admission before PPO; the prior 1,574-test receipt
cannot qualify it. The source-bound runner and exact protocol remain under
ignored `logs/recovery/quiet-reference-pilot-v1/`; raw native evidence,
checkpoints and offline W&B records go to ignored
`logs/manipulation/quiet-reference-pilot-v1/`. No hardware motion or promotion.
