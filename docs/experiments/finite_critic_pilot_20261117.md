# Corrected finite-contract critic pilot

This comparison follows the [actor/reward/critic review](../rl/WHOLE_BODY_ARCHITECTURE_REVIEW.md).
It holds the original reward, actor, exploration and task distribution fixed,
and tests joint versus independently scheduled critic fitting after correcting
finite tracking deadline semantics. The completed previous reward pilot is
historical; its negative physical results are not relabeled as a corrected run.

Completed runtime: `2acc3b52ed68009fc81ff7cc4923e999e7e1fec2`, 3,572 packed
inputs, with **1,574 fresh passing source-bound regressions** (1,044.46 s).
Source/archive record and compact exact producers are in
[the evidence package](../../paper/data/whole_body_control/finite_critic_pilot_v2/README.md).
The wrappers physically stop and score a finite deadline as terminal; generic
collection truncations still bootstrap. Actor inference and guards are unchanged.

Both arms use retained scoped-feedback v4 actor weights
`46291c71ea4dccd93feaf94d4b0e88f3deb1ee5945667964300f3c9f1c165813`, fresh critics,
seed 107, 16,384 decisions, 1,024 critic-only warmup decisions, three PPO actor
epochs, 64-sample minibatches, 256-decision rollouts, initial std 0.02, actor LR
5e-5 → 1e-5, clip 0.05 and target KL 0.01. Both use the unchanged legacy wrapper
reward and corrected finite-contract targets. `finite_joint` keeps historical
joint value updates; `finite_independent` schedules three full value epochs
after actor optimization, including when actor KL stops early. This changes
critic update scheduling/minibatch ordering, not just a scalar budget. Critic
permutations preserve the actor RNG stream.

The exact twelve training and twelve disjoint audit cases from the preceding
pilot are unchanged. Outer general/TAMP exposure remains 50/50 by physical ticks;
within general it is 80/20 hand/base. TAMP is still unloaded, open-jaw grasp-pose
prefix execution. Fixed final checkpoints are evaluated on the same 24 cases,
existing ten-case coupled suite and six TAMP prefixes. No audit labels enter
training, no intermediate checkpoint selection, and no automatic budget extension.
This does not establish loaded carry, full agent deployment or hardware readiness.

The frozen native admission replays the existing results: teacher 12/12 training,
11/12 audit and 6/6 TAMP. Retained actor is 20/24 new cases, 6/10 existing coupled
and 6/6 TAMP, with all 40 stops confirmed. Known contact failures remain counted.
The corrected full gate is `20261117_005528_167274`; admission is
`20261117_005605_ab7700`. Both pass before paired PPO job
`20261117_005750_a59bcd` starts. All initial actor/critic states, observations and
first actions match exactly across arms. The completed run produces:

| Controller | New train | New audit | Existing coupled | Unloaded TAMP prefix |
| --- | ---: | ---: | ---: | ---: |
| Retained comparison | 11/12 | 9/12 | 6/10 | 6/6 |
| Corrected finite, joint critic | 10/12 | 9/12 | 6/10 | 6/6 |
| Corrected finite, independent critic | 11/12 | 9/12 | 7/10 | 6/6 |

The independent arm solves existing training case `innate_mars:v2:train:coupled:4`
without losing an existing-coupled success. On the new cases it gains audit hand
3 and loses audit hand 2, leaving 20/24 unchanged. Across the new and existing
coupled cohorts, worst active-hand endpoint error changes **18.19 → 13.10 mm**
and orientation **0.0782 → 0.0619 rad**. These are scoped gains, not general
mastery. Worst yaw jerk increases **27.80 → 53.36 rad/s³**, peak contact force
**34.65 → 34.96 N**, and hold-base drift **0.402 → 0.437 mm**. There is no clear
joint accuracy/stability improvement and neither candidate is promoted.

All **120/120 actor evaluation stops confirm**. Each controller has the same
three unsafe-contact case identities, no wheel lift, invalid physics or retention
loss. TAMP remains unloaded open-jaw prefix execution, not grasp/carry success.
The original reward, acceptance thresholds, geometry and hard guards are fixed.
The independent arm receives all **720/720** planned post-warmup critic updates;
the joint arm receives 612, with 16 actor-KL stops. Independent actor updates
number 624 with 15 KL stops. The value-budget implementation works as intended;
that alone does not establish better deterministic physical control.

Each arm completes 16,384 decisions, including 15,360 actor-update decisions.
Physical ticks are 19,571 joint and 19,662 independent. Offline W&B IDs are
`va12m86z` and `84m8vafn`; the comparison ID is in the evidence package.
Raw traces, checkpoints and per-update diagnostics remain ignored under
`logs/manipulation/finite-critic-pilot-v2/`. Compact results preserve every case,
initial/reference/model fingerprints, safety outcomes and active-hand metrics.
The retained-policy viewer is restored on localhost:8766 after completion.
Retained remains an experimental comparison, not a finished controller.

### Historical first gate

The preceding source `a179dceb` packs 3,564 inputs and produces 1,571 passing
tests with two compact test-policy interface failures in the new value-fit
diagnostics. No PPO starts under that failed receipt. The learner is repaired to
fall back to its existing `evaluate()` interface when a policy has no separate
`values()` method. The fresh corrected-source gate above includes that repair.
The original failed-source records remain in `finite_critic_pilot_v1/`; they are
not relabeled as successful runtime evidence.

### Next controlled improvement

The [architecture review](../rl/WHOLE_BODY_ARCHITECTURE_REVIEW.md) finds that
constant std 0.02 repeatedly interrupts the two-second quiet hold in stochastic
training. A frozen-actor diagnostic reduces motion exploration only for an
announced stationary reference: 0/8 → 3/8 stochastic successes, with all stops
confirmed and the same contact-failure identity. This changes neither actor mean
nor deterministic control. The next paired training experiment holds reward,
finite semantics, three independent critic epochs, task mix and final budget
fixed, and isolates that exploration change. It must pass a fresh source gate;
this completed receipt does not qualify later source changes.
