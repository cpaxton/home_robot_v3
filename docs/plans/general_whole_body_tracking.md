# General whole-body tracking and continuation audit

The controller accepts one masked SE(2) base pose, one masked SE(3) pose per
supported hand, and the robot's gripper coordinates. It can receive a timed
preview or a changing pose goal. Mars is the first physical target; Stretch
uses the same task contract with its own model and native limits. This work is
simulator-only and does not authorize Herman motion.

TAMP is a downstream command source and acceptance suite. Its approach, lift,
carry and placement paths must be executable, but the controller must also
handle ordinary base motion, hand-only targets that may require base movement,
combined constrained targets, rotations, translations, goal changes and
recovery. A grasp-planning IK witness establishes geometric admission; the actor
still receives pose targets and emits bounded velocities.

## Coupled control and embodiment comparison

Current direction, 2026-11-16: learn coupled base/arm control for both Mars and
Stretch. General-purpose control remains a goal for both; it is not restricted
to robots with a six-axis stationary arm. Use the
[active work index](../rl/ACTIVE_WORK.md) and [ordered current plan](mars_wbc_next_steps.md)
for launch decisions. The continuation experiments below are historical
protocols/results, not instructions to restart those runs or their old mixture.
The current intended outer allocation is 50/50 general/task by physical ticks,
after stage-specific admission.

The common contract is one component-masked SE(2) base target plus one SE(3)
target and native gripper/hand coordinates per supported hand. In hand-only
mode the authorized base may move. In base-region/partial-mask mode coordinate
within the declared constraints; fully specified base/hand references must be
jointly feasible. Do not independently sample arbitrary endpoints and interpret
failures as a deficiency of the controller. Region and multi-hand adapter
support are requirements to verify, not claims about existing runtime coverage.

1. Construct reachable start/goal pairs and smooth paths using full-robot FK/IK
   witnesses, including nonholonomic base motion where applicable. Preserve
   rejected references and identify geometric, timing, collision, support and
   solver failures separately. FK endpoint reachability alone does not certify
   the intervening trajectory or its dynamics.
2. Test hand-only/free-base, fixed-world-hand with moving base, constrained
   coupled motion, changing goals and quiet holds. Preserve precise hand
   tracking and native force/jerk/support limits. Pair learned/teacher cases
   within each embodiment and keep the frozen task/audit populations distinct.
3. Diagnose model, controller and learning failures before declaring an
   embodiment limitation. On demonstrated executable references, failure of
   the learned actor remains a tracking/training problem to investigate.

### Optional MABEL diagnostic

Keep Mars first and Stretch in scope. If those diagnostics suggest an
embodiment limitation or leave it unresolved, add a more redundant robot as a
separate simulator comparison. [MABEL's official project](https://robotmabel.github.io/website/)
and [source repository](https://github.com/robotmabel/open_mabel) describe two
7-DoF arms, a lift/torso, dexterous hands and a holonomic base, with a published
MuJoCo model. These make it a candidate for the comparison; its assets and
runtime have not been imported or validated here.

Before a MABEL rollout, pin its source revision and model/assets/licenses;
validate joint coupling, actuator units/limits, contacts, gravity, wheel/base
mapping and support/stop behavior. Begin with one active hand and a declared
stationary posture for other coordinates, then explicitly test both hands and
their self-collision/shared-body constraints. Represent its native hand
coordinates explicitly; do not silently reinterpret them as a scalar gripper.
Use the same task semantics and outcome definitions with native physical
limits and comparable workspace-relative difficulty. Report differences in
redundancy, actuation, contacts and base mobility: better performance on MABEL
alone would not prove that Mars or Stretch failures are hardware limitations.
No MABEL integration, policy, hardware readiness or cross-robot transfer is
claimed by this plan.

## First gate: preserve moving-goal performance during updates

The completed goal-gated settling experiment retains accurate stationary holds,
but both 16,384-step continuations regress from the initial checkpoint on moving
goals. Both start from exactly matching validation behavior. Recorded-action
counterfactuals identify larger effects from changed policy weights than from
changed normalization on the same recorded observations; these are offline
comparisons, not counterfactual physics rollouts. The continuation also restarted
a fresh optimizer at 5e-5 after the source run finished at 1e-5.

`configs/benchmarks/mars_general_continuation_v1.json` specifies four sequential
8,192-step pilots from the same original weights and seed:

| Variant | Learning-rate decay | Observation statistics |
| --- | --- | --- |
| adaptive_restart | 5e-5 to 1e-5 | adaptive |
| frozen_restart | 5e-5 to 1e-5 | frozen |
| adaptive_small | 1e-5 to 2e-6 | adaptive |
| frozen_small | 1e-5 to 2e-6 | frozen |

All variants use a fresh optimizer, goal-gated settling, the corrected current
retention-error cost, the existing 70% reach / 30% manipulation reset mixture,
identical physical limits, 0.05 randomization, exploration and 6 s training
horizon. Do not change task distribution or architecture while isolating these
factors. Frozen statistics require an initializer and do not freeze PPO weights.
The native action mapping and runtime normalization implementation stay unchanged.
No checkpoint is promoted automatically.

Launch requires a fresh, source-bound full regression receipt after merging the
latest harness. Record exact commands, source states, configuration/weight
hashes, sampled episode/step shares, scalar training metrics and all failures.
Keep old training provenance intact; a later reward/harness fix does not
requalify old reports.

## Validation beyond TAMP

Evaluate the original checkpoint and every completed pilot against:

- Six strict stationary/nearby/wrist probes at two existing development postures.
- Seeded ordinary base-only, hand-only and combined pose targets, with two cases
  per mode for the initial diagnostic. Generate target poses by FK from sampled
  reachable configurations; preserve failed admission and rollout outcomes.
- Two mid-motion retarget cases and two bounded hand-push cases, using the same
  announced-goal and wrench timing semantics as the general environment.
- The unchanged three public TAMP development programs with no interstage reset,
  plus the existing per-category teacher comparison during training.

Use independent, fixed development seeds and pair controller/teacher resets.
Report admission, conditional execution and end-to-end denominators separately.
Measure arm position/orientation, active base-region/heading error, goal changes
actually delivered, force, jerk, retention, wheel unloading, and measured quiet.
Teacher failures remain visible; they do not establish geometric infeasibility.
Two cases per mode diagnose behavior and do not qualify a robot or population.
The general hand contract stays 3 mm / 0.02 rad, while ordinary base goals retain
their declared region. Fixed-base manipulation still obeys the 1 cm / 0.02 rad
safety boundary. Benign contact is allowed within existing force/support limits.

## Base endpoint references

`BaseGoalReference` compiles base-only endpoint tests into continuous, axle-aware
forward/reverse turn–drive–turn S-curves for Mars and Stretch. Native profile
speed/acceleration and an explicit experimental reference jerk bound apply;
scratch geometry, initial measured motion and dynamic execution remain separate
checks. Its custom record must not be silently decoded as a sampled linear
`task_pose_trajectory_v1` wire schedule.

The opt-in `BaseNavigationFeedback(reference_tracking=True, heading_gain_scale=2)`
keeps the inner XY region active through motion and uses the announced twist's
known physical-unit mapping as feedforward. The heading multiplier is explicit
development calibration. Learned pose gains remain trainable; active-hand and
partial-base mappings remain unchanged. The nested v2 navigation contract makes
older v3 actor runtimes reject unsupported semantics. Legacy defaults are kept.

These are rest-to-rest reference tests. Before integrating interactive goal
dragging, implement and validate moving-start braking/replans that match the
previous reference velocity/acceleration, preserve physics/history/faults, and
remain inside the native derivative and geometry bounds. Do not restart a
zero-velocity reference while the robot is moving. Keep the
[endpoint diagnostic](../experiments/base_goal_reference_20261116.md) distinct
from raw static-goal convergence, loaded carry and hardware qualification.

## Next gate: expand the distribution after updates are reliable

Within the tracking share, explicitly cover near-constant-orientation Cartesian
translations in all reachable directions, near-constant-position rotations,
combined motions, and terminal holds. Retain base-only, hand-only and combined
masks, rather than conditioning every episode on a fixed base. Sample new
training configurations and paths; do not optimize against frozen public
regression trajectories. Add TAMP-shaped motions as a bounded subset and keep
ordinary goals in every validation comparison.

Increase difficulty and randomization only when accuracy and settling pass at
the preceding level. Keep smoothness/energy terms subordinate to completing the
active pose contract, with native speed/acceleration and force/support guards
unchanged. Loaded retention, public task success, observed-agent testing and
Herman remain independent subsequent gates.

## Recovery and translation-coverage follow-up

All four continuation pilots completed before the reboot. The smaller/frozen
setting preserves position better but still fails angular and public task gates.
The next matched pair adds reachable fixed-orientation translation coverage in
a bounded part of the existing nearby-reach share, retaining the 70/30 macro mix.
See the [recovery report](../experiments/mars_recovery_translation_20261026.md) for
full negative results, the corrected action audit, target export and physical
translation probes. Persistent scripts preserve completed variants and numbered
checkpoints; recovered optimizer continuation resets environment/RNG state and
is explicitly not claimed bitwise identical. No candidate can promote itself.

The matched pair has now completed with no task gain: 1/8 translation success
for source/control/treatment and 0/3 public programs each. The immediate acceptance
gate is an actual agent delivery through public tools, alongside the general
coverage above. Reference execution passes one development delivery; the learned
backend now reaches grasp and begins lift in explicit gain diagnostics but still
fails loaded orientation/base tracking. See the
[integrated report](../experiments/mars_agent_controller_20261026.md). Prioritize
loaded transition coverage and first-failure evidence over longer unmeasured
continuations; offline gain scaling adds no PPO training or qualification.

## Coupled simulator implementation (2026-11-16)

The new `emet.rl.manipulation.coupled_tracking` module supplies a reusable
`CoupledTrackingEnv` and continuous `CoupledReference` for both initial robots.
The versioned [case set](../../configs/benchmarks/coupled_tracking_v1.json)
contains six train and four audit cases per robot with distinct IDs/reset seeds.
It covers free-base hand targets, full/partial base constraints, base-only
motion, compatible fixed-world-hand self-motion and a mid-motion goal reversal.
This is a small diagnostic distribution, not workspace or reliability coverage.

Ordinary references use a differential-drive axle arc and full-robot FK.
Fixed-hand references sample a compatible whole-body Jacobian nullspace on
scratch data; they do not prescribe an arbitrary straight base shift that a
five-coordinate arm cannot cancel. Their base endpoint is generated with the
arm path, and `self_motion_scale` is a sampling amplitude, not a promised travel
distance. Neither private joint witnesses nor case metadata enter actor inputs.
The existing history/preview/motion actor and grouped PPO collector consume the
new environment; simulator-only critic observations remain separate. Native
limits, contact/wheel monitors and precise hand tolerances remain in use.

A rest-to-rest quintic phase drives the continuous arm spline. Coordinatewise
arm derivative bounds and base reference limits are checked before emission.
Full-path geometry is sampled every 25 ms with joint margins, collision checks
and a 5 mm static support margin. Stretch uses its two drive wheels and the
single modeled caster sphere; Mars retains its existing support computation.
Geometric admission does not prove dynamics/contact feasibility. A goal change
constructs a new admitted reference from measured state without clearing history
or faults; its rest-to-rest derivative certificate does not claim C2 continuity
across the retarget. Physical transition jerk is measured and penalized.

Export targets, including orientation and heading data, or run the physical
teacher using a fresh output directory on the reserved lane:

```bash
EMET_UV_RUN=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=disable \
taskset -c 14 .venv/bin/python -m emet.rl.manipulation.coupled_tracking \
  --robot innate_mars --teacher --output logs/eval/coupled-mars-FRESH
```

Use `--robot stretch` for Stretch; omit `--teacher` for geometry export only.
Outputs are ignored `references.json`, `sources.json` and `references.svg`.
The plot shows base paths/headings, hand references versus measured paths, and
tracking error. Every rejection remains in the denominator. A teacher readout
requires all declared cases, delivered retargets, tracking/settling, measured
whole-body stopping and no safety flags before marking its distribution ready.
It is not policy promotion or hardware qualification. Full source-bound
regressions and a separately declared mixed-training protocol remain required.

The TAMP primitive environment separately accepts
`planning_scope="unloaded_prefix"` with `primitive="pregrasp"` or `"grasp_pose"`.
It requests that exact scope through the public tools; its default remains
`"full_task"` for reproducibility of old experiments. The public plan handle
cannot upgrade unloaded admission into closure/lift/placement. Prefix training
and the new coupled general share must be admitted separately before the next
50/50 physical-tick training run. No loaded grasp or additional robot is
qualified by these interfaces.

The partial-mask teacher also retains coupled base/arm authority. Active base
components enter the resolved-rate objective alongside the precise hand pose;
inactive coordinates do not become hidden targets. This matters for Stretch: an
XY-constrained, yaw-free target can require yaw motion to close the hand error.
The fully constrained-base teacher path remains unchanged.

## Coarse general base region, 2026-11-16

The versioned `configs/benchmarks/coupled_tracking_v2.json` distribution starts
with 5 cm XY and 5° heading acceptance, keeping the hand at 3 mm / 0.02 rad.
This is a first general-controller operating region, not a claim that wheeled
robots cannot be more precise. Required base paths travel 15 cm; staying still
does not pass. Small fixed-hand nullspace paths are explicitly hand-hold
diagnostics, not required-base progress. V1 and frozen TAMP results are retained.

Use `CoupledTrackingEnv(robot, suite_path=COARSE_SUITE)` for the new distribution;
its CLI accepts `--suite configs/benchmarks/coupled_tracking_v2.json`. The v2
base pose reward is flat inside both base acceptance bands. Fine hand tracking,
measured motion/jerk and force terms remain active. Existing task-specific
base contracts are not changed by this general benchmark.

A separate opt-in `WholeBodyActor(base_rate_feedback=False)` removes the direct
measured joint/base-rate path only from base output rows. Arm rows retain it;
history, hand/base pose errors and announced motion remain available. The flag
is bound in exported weights and manifest; old bundles preserve legacy behavior.
This is an experimental ablation, not a promoted default. See the
[complete diagnosis](../experiments/coupled_tracking_20261116.md) and
[SuperMimic coverage proposal](supermimic_controller_training.md).

## Optional signed base feedback, 2026-11-16

`BaseNavigationFeedback` supplies an explicit coarse base-only feedback
parameterization. Learned gain magnitudes remain trainable. Forward-position
and heading-error responses have restoring signs; lateral-to-yaw correction
changes with forward/reverse travel. Unsupported cross-axis gains are omitted.
A stationary constant preview uses an inner hold region at half the declared
XY/yaw acceptance radius, leaving margin to enter the accepted region.

```python
from emet.rl.manipulation.base_navigation_feedback import BaseNavigationFeedback
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy

policy = MultiCriticPolicy(
    robot="innate_mars",
    base_rate_feedback=False,
    base_navigation_feedback=BaseNavigationFeedback(
        xy_region_m=0.05,
        yaw_region_rad=0.08726646259971647,
    ),
)
```

It applies when the hand is inactive and all three base coordinates are
constrained. Active-hand and partial-base commands retain the existing coupled
mapping; the base can still supply missing hand task directions. Native wheel
speed/slew, arm, gripper and stop guards remain authoritative. No output filter,
teacher action or private joint witness is added. This prior is not a global
stability certificate.

Configured exports use `emet-whole-body-v3`, which v2-only runtimes reject.
The manifest and weights must agree on both feedback options. The loader checks
the declared environment tolerance against the inner hold region. Legacy v2
bundles remain readable; they retain their original feedback semantics. The
policy loader, TAMP adapter, current-feedback validator and test-drive picker
recognize v3. Existing qualification checks still apply.

The fixed-weight prototype completes 24/24 base trajectory cases, 10/10 small
general cases, 6/10 larger general cases, 6/6 open-jaw TAMP cases and 4/4 live
base-to-hand handoffs. Repeated source-bound readouts are separate producers,
not independent reliability trials. V3-source validation passes 1,451 regressions and the complete native readout.
Static endpoint goals pass 2/8: lateral/diagonal cases enter the XY region but
fail final heading convergence. They need feasible local reference generation
or broader control coverage. Mars owns these physical results; Stretch has
unit coverage of the parameterization, not a trained v3 policy qualification.
