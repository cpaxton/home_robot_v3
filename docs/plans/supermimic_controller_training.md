# SuperMimic ideas for Mars/Stretch training

Reviewed 2026-11-16. This is a research handoff, not an installed backend or a
measured training speedup. Mars remains first; native simulator results remain
the authority for tracking, contact, retention, support and confirmed stops.

## What the method contributes

[SuperMimic](https://arxiv.org/abs/2610.28480) learns shared differentiable
robot dynamics from simulator transitions, then improves a policy through short
imagined rollouts. The paper reports eight-step model fitting and 32-step policy
rollouts with critic bootstrapping. This is a different optimization method from
teacher-action imitation; it can optimize beyond a teacher. Its compute results
on large GPUs do not predict performance on Caliban's Jetson Orin.

The [released code](https://github.com/HybridRobotics/supermimic) targets G1,
builds on mjlab/MuJoCo Warp, and declares Python 3.13 plus a sibling
`mjlab_research` dependency in its
[package configuration](https://github.com/HybridRobotics/supermimic/blob/main/pyproject.toml).
Our Python constraint is below 3.13. Any port needs an isolated environment;
do not upgrade the working training environment to try it.

## First experiment: reset/phase exposure

The official [reset sampler](https://github.com/HybridRobotics/supermimic/blob/main/src/supermimic/runners/reset_sampling.py)
combines 50% uniform, 40% current low-return, and 10% historical difficult-region
sampling. Borrow this allocation **inside each family** of our existing 50/50
physical-control-tick general/TAMP mixture. Preserve train/audit separation.
Our short PPO pilot reached only five of six general cases and underexposed
late TAMP phases, making this a concrete coverage problem to address.

Before phase resets, serialize and restore physically validated checkpoints:
physics state, actuator targets, slew state, integrators, measured history,
reference clock and executor authority. Do not teleport a robot into an object,
clear a stop latch, synthesize a grasp, or carry labels across holdouts. Keep
full-reset episodes as a separate evaluation cohort. Log physical ticks, unique
case IDs, phase starts and completed stages, not just episode counts.

Compare fixed-budget uniform and adaptive-reset runs from the same initializer,
with the same actor, rewards, random seed stream and native evaluation cases.
Report collection and evaluation time as well as policy-update time. This is a
proposed experiment; the sampler is not yet integrated.

## Second experiment: differentiable dynamics

Collect teacher and actual actor transitions from unloaded Mars tracking.
Include physical state, executed actions, command targets, native servo/slew
state, timing, reference progress and applied disturbance information in the
model's training state. Keep privileged state outside the deployed actor.
Measure multi-step errors on complete held-out trajectories before optimizing
through the model. Compare a bounded imagined-rollout update against equal-cost
PPO, including model fitting and data collection in compute accounting.

Retain our history and trajectory preview; a transformer is not a prerequisite.
The paper's history setup and released runner defaults differ, so verify the
actual configuration rather than copying defaults.

Do not adopt the released [environment's](https://github.com/HybridRobotics/supermimic/blob/main/src/supermimic/tasks/env_cfgs.py)
removed self-collision penalty/deviation termination or its much broader tracking
reward scales as our acceptance criteria. Its training-motion benchmark does
not establish millimeter hand accuracy or loaded manipulation reliability.
Evaluate every resulting actor in native physics with the original safety and
held-out task contracts. No automatic promotion or hardware motion follows.
