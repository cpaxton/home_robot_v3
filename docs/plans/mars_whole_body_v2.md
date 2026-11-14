# General whole-body PPO with actual TAMP primitives

This is a new simulator training prototype, not a replacement for the retained
source, a qualified controller, or authorization to move Herman. Mars is the
first training target. The actor and sensor encoding also support Stretch;
the actual public-tool primitive environment is currently Mars-only.

The [completed first validation](../experiments/mars_whole_body_validation_20261113.md)
passes all 1,170 regressions and teacher 164/164, but the cold 8k final achieves
only 2/164 full general cases and 0/6 TAMP pregrasp. Reject it and retain the
original source. The independent jaw-unit trial also completes and is rejected.
Further scale is conditional on diagnosing the measured learning failures.

The [controlled normalization diagnostic](mars_whole_body_advantage_pair.md)
subsequently passed 1,187 regressions and the unchanged teacher admission.
Its historical-normalization control completed 4,096 decisions; the candidate
stopped before its first update on a settings serialization mismatch. No paired
normalization conclusion is available. The optional legacy-MLP work in
[PR #232](https://github.com/cpaxton/home_robot_v3/pull/232) is a separate
experiment and is not required to run this prototype.

Correct experimental code can merge with measured behavioral limitations.
Passing all performance gates is a checkpoint-promotion decision, separate
from implementation correctness and from permission to test real hardware.

## Controller and reference contract

Keep one masked SE(2) base target, one SE(3) target per supported hand, and the
robot's gripper coordinates. Current Mars/Stretch implementations each have
one hand. Hand-only goals may use base motion; active base masks constrain it.
The robot profile still bounds actual wheel, arm and gripper commands.

The new bundle (`emet-whole-body-v2`) consumes existing history and pose preview
plus a nine-column announced reference twist: world base vx/vy/yaw rate, hand
linear velocity and hand angular velocity. Static goals give zero twist.
Differentiate the actual retimed reference, including wrapped yaw; do not feed
the actor private planner joint witnesses, object truth, contact solver outputs
or the teacher action. A finite-window GRU processes all 20 measured samples
(one second at the current 20 Hz cadence); no hidden state survives a call.

One actor produces ordinary bounded velocity actions. Its learned feedback
readout keeps zero pose/velocity errors at a stationary motion equilibrium.
Only task-owned retain/grip commands enable grip correction; release remains
executor-owned. The existing hold/stop handshake is still active. A successful
executor quiet hold must not be described as an actor-only stability test.

TAMP's opt-in `s_curve` reference uses a seven-phase jerk-limited S curve with
trapezoidal velocity where move length permits. Short moves omit cruise or
constant-acceleration phases. Preserve polyline corners with stops and certify
speed, acceleration and jerk, including phase-governor derivatives. Apply the
same profile to the terminal Cartesian correction. Existing quintic/legacy
defaults and frozen acceptance cases are unchanged. Experimental jerk limits
are not manufacturer ratings or a dynamic-feasibility certificate.

## Training distribution

Use [the development manifest](../../configs/benchmarks/mars_whole_body_v2_development.json):

- 50% general / 50% actual TAMP primitive **physical control ticks**. Select the
  next family by accumulated tick deficit at episode boundaries; report both
  decision and physics counts because TAMP's executor also owns hold ticks.
- Within the general half, base/arm/coupled shares are 30/50/20. Preserve the
  existing nearby reachable references, admission, time horizons and gates.
- TAMP runs the actual `scene_tasks`, `plan_pick_place`, and
  `execute_pick_place_plan` tools in a spawned, single-owner simulator. The
  first primitive ends after pregrasp, before closure/lift/carry/release. It
  includes approach, execution, terminal hold and confirmed stop.
- Four new development fixtures use seeds 34600000–34600003; two audit fixtures
  use 34610000–34610001. Frozen public benchmark seeds stay outside training.
  Do not discard rejected initializations or failed executions.

One development fixture is a declared challenge: the analytic reference
triggers `object_changed_replan`. Both reference profiles moved its supported
object by 0.0596 rad during approach; idle hold moved it 0.0112 rad. Additional
0.5/1.0 s settling caused planning rejection and is not used in the curriculum.
Retain this failure in the training pool and denominator. This is evidence of
contact sensitivity, not proof of controller infeasibility.

Experimental launch readiness requires all original general teacher cases to
pass, all six TAMP cases to admit execution and confirm a safe stop, and every
TAMP outcome to be either success or that one explicitly declared challenge.
Unknown failures, rejected plans, wheel lift, excessive force, invalid physics
or unconfirmed stops block training. The report separately records TAMP
successes/attempts and `tamp_all_success`; **readiness does not mean qualification**.
The preliminary reference result is 5/6, not 6/6. No prior readiness or promotion
gate is loosened by this new experimental entrypoint.

## Objectives and privileged critics

Use direct PPO, without a residual teacher or teacher actions in its loss.
Three independent critics estimate tracking, stability and contact/safety
returns. Normalize their advantages separately before combining them; record
per-head losses, advantage scales/correlations and the selected weights.
Clip critic gradients separately so one reward's units cannot clip actor or
other critic gradients through a shared norm.

The tracking group includes active hand position/orientation, active base
region/heading, measured versus announced hand/base velocity, and contract
completion/failure. Pose rewards remain different at 5, 10 and 20 mm. The
stability group charges measured jerk, command changes and effort. The contact
group allows benign contact, charges current force excess above 3 N, and
penalizes independent hard-guard failures. Do not repeatedly charge a stale
running contact-force maximum as if it were current force.

Critics additionally receive a separately named 34-column simulator vector:
payload mass/relative pose, finger normal/tangential-load bounds and contact
counts, applied wrench, support/wheel loads, friction, retention intent, base
tilt, pose errors and episode time. Critic gradients never update the actor
encoder. Export only the actor; strict runtime loading rejects privileged
inputs. `--no-privileged-critic` provides an ablation with zeroed privileged
features. Start at the existing zero-randomization model; widen randomization
only after measurable tracking/stability improvement.

## Intermediate tests and launch

1. Test derivative bounds, C2 continuity, yaw wrapping, velocity masks,
   oldest-history influence, stationary equilibrium and robot-specific widths.
2. Test actor/critic gradient isolation, actor-only bundle roundtrip, grouped
   GAE timeout/reset behavior, actual PPO updates and unequal-length tick mix.
3. Run native general observations, actual successful TAMP pregrasp, and the
   retained failure with its original replan threshold and confirmed stop.
4. Run the complete fresh source-bound regression suite. Then run a fresh
   teacher preflight bound to the development manifest, all original general
   cases, actual TAMP tools, agent/executor sources and model fingerprints.
5. Train an 8,192-decision pilot before scaling. Log offline W&B scalars, every
   episode outcome and step shares. Atomically checkpoint training state after
   each rollout. Do not overwrite or promote the retained source.
6. Compare exported final and retained source on identical original general
   cases and all development/audit primitives. Keep position/orientation,
   settling, drift, jerk, force, wheel support and all failure denominators.
   Run full pick/place and actor-active quiet-hold tests before any broader
   claim. Extend primitives to grasp→lift, loaded transport and placement only
   after this first pilot is understood; none is covered by pregrasp success.

Commands (run from the repository, with BLAS/Torch threads bounded):

```bash
python -m emet.rl.regression_gate --output logs/eval/whole-body-v2/regressions.xml
python -m emet.rl.manipulation.whole_body_training preflight \
  --output logs/eval/whole-body-v2/preflight
python -m emet.rl.manipulation.whole_body_training train \
  --preflight logs/eval/whole-body-v2/preflight/preflight.json \
  --regression-receipt logs/eval/whole-body-v2/regressions.receipt.json \
  --steps 8192 --output logs/manipulation/mars-whole-body-v2-pilot
```

The UI, mastery evaluator and admitted simulator TAMP backend can load the new
actor bundle. TAMP still requires archived training evidence and unchanged
robot/inference sources; adding a format does not bypass admission. Existing
SB3 component training remains a separate v1 path. Hardware, full TAMP carry,
observed-agent integration and automatic promotion remain separate gates.
