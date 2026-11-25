# Whole-body RL control: multi-arm torque contract

Status: **schema v2 implemented; actuator adapter prototyped; no hardware policy
promoted.** This document is the design authority for the contract in
[`wbc_contract.py`](../../src/emet/controller/manipulation/wbc_contract.py),
the example profiles in
[`wbc_profiles.py`](../../src/emet/controller/manipulation/wbc_profiles.py), and
the adapter in
[`torque_adapter.py`](../../src/emet/controller/manipulation/torque_adapter.py).

## Program goal

The goal is one **torque-based whole-body control stack** that spans Mars,
Sourccey, RB-Y1, Stretch, and future mobile manipulators. A robot supplies an
`EmbodimentProfile` (base, named arms, torso, grippers, DOF, limits) and an
actuator adapter. The policy, observation, reward, case library, and training
loop are shared. Mars is the first target and the first place where the
position-servo adapter is being validated; it is not a Mars-only design.

## Decision summary

- The **policy action is per-arm joint torque**, not a single flat joint vector
  and not a single hand. Torque is the most general policy contract for contact,
  compliance, and cross-embodiment training.
- The **robot-facing command is an actuator adapter**, not the policy. Mars
  currently uses position-controlled servos, so the default adapter converts
  torque to a bounded position-target offset plus a current/PWM limit. True
  torque or impedance control can replace the adapter without changing the
  policy schema.
- The schema supports **zero, one, or many arms**, **torso/head groups**, and
  **per-arm grippers**. Arm names are explicit; there is no implicit `left` or
  `primary` assumption.
- The existing velocity/position tracker remains the free-space baseline. The
  torque contract is additive and targets contact-rich / carry work first.

## Why torque at the policy layer

Torque is the natural action for:

- contact-rich manipulation and force regulation;
- compliant recovery when a reference is blocked;
- a shared action space across position-servo, current-controlled, and
  torque-controlled robots;
- multi-arm coordination, where each arm has its own torque vector and target.

Torque is **not** the natural low-level command for every actuator. In
particular, Mars has geared Dynamixel servos with position goals. The contract
therefore separates:

1. **policy action** — torque, embodiment-independent;
2. **actuator adapter** — maps torque to the robot's native command;
3. **robot capability** — position servo, current-controlled position, or true
   torque.

Calling a position-servo adapter "torque control" would be inaccurate. The
adapter is a model-based feedforward; its accuracy depends on identified
stiffness, gravity, friction, and current limits.

## Schema

### Profile

`EmbodimentProfile` is the canonical layout:

```text
base: BaseSpec                  # planar (vx, omega), holonomic (vx, vy, omega), or fixed
joint_groups: JointGroupSpec[]  # named ordered joints; role = arm | torso | head
arms: ArmSpec[]                 # named arm -> joint group + EE link + optional gripper
torso_group: str | None
head_group: str | None
```

`JointSpec` carries position/velocity/torque limits and the optional actuator
model fields used by the adapter (`kp_eff`, `kd_eff`, `torque_constant`,
`gear_ratio`, `current_limit`, `max_lead`). `ArmSpec` owns the EE link and
optional `GripperSpec`. The arm order in `arms` is canonical for actions,
observations, rewards, and vector flattening.

### Action

```text
WbcAction:
  base_velocity: (d_base,)
  torso_torques: (n_torso,)
  arms: [
    { name, joint_torques: (n_arm,), gripper: {values, mode} | null }
  ]
```

- `d_base` is 2 for planar, 3 for holonomic, 0 for fixed.
- Gripper commands are separate from arm torques. A parallel jaw has one value;
  a multi-finger hand can have several. `GripperSpec.command_limits` supplies
  the executor-level command bounds; if absent, the gripper channel is passed
  through and the servo/executor owns its limit.
- `JointSpec.control_mode` records the actuator capability:
  `position`, `current_position`, or `torque`. `PositionServoAdapter` accepts
  the first two and rejects `torque`; a true torque backend must provide its own
  adapter.
- `clamp_action(action, profile)` is the shared base/torque/gripper safety clamp.
  Every adapter calls it before mapping to hardware; it is not optional.
- The head group is **observed-only** in this schema. It has no action channel
  and is excluded from `controlled_joint_names`, so a head feedforward torque
  is rejected rather than silently ignored.
- `action_vector(action, profile)` flattens in **base, torso, arms (profile
  order), per-arm gripper** order. `action_from_vector` reverses it.
- The policy never sees joint-name strings; the training loop uses the profile
  to build tensors. The profile id/hash must be stored in the policy bundle.

### Observation

```text
WbcObservation:
  base_pose: (x, y, theta)
  base_twist: (d_base,)
  torso: JointGroupObservation | null
  head: JointGroupObservation | null
  arms: [
    { name,
      joint_position, joint_velocity, joint_torque?,
      ee_pose, ee_target?, target_mask=(position, orientation),
      gripper? }
  ]
  occupancy: (H, W, D) uint8 | null
  failure_mode: str
```

- `ee_target` is per arm; an arm with `ee_target=None` is observed but inactive
  for the task. This is how a one-arm task runs on a dual-arm robot.
- `target_mask` allows position-only or orientation-only targets.
- `occupancy` is a local crop in the base frame. No GT object pose.
- `joint_torque` is measured/estimated when available; otherwise it is empty.
  The policy must not assume torque feedback exists.

### Reward

`reach_reward(action, next_obs, profile=..., ...)` sums/weights active arms:

- position and orientation errors honor `target_mask`;
- `arm_weights` controls task priority;
- `success_mode="all"` (bimanual) or `"any"` (single-arm task on a multi-arm
  robot);
- action and jerk penalties normalize by each joint's torque limit and base
  velocity limit, so coefficients transfer across robots;
- collision is a global penalty; per-arm components are logged as
  `arm.<name>.pos_err`, etc.

## Example layouts

These are schema examples in
[`wbc_profiles.py`](../../src/emet/controller/manipulation/wbc_profiles.py).
They use published ranges where available and clearly-marked placeholder torque
values; they are not hardware calibration.

| Robot | Base | Torso | Arms | Grippers | Action dim |
|---|---|---|---:|---:|---:|
| Innate Mars | planar 2 | — | 1 × 5 | 1 × 1 | 8 |
| Sourccey | planar 2 | lift 1 | 2 × 5 | 2 × 1 | 15 |
| RB-Y1 (proxy) | holonomic 3 | 4 | 2 × 6 | 2 × 2 | 23 |
| Stretch 3 | planar 2 | — | 1 × 8 (lift/telescope/wrist) | 1 × 1 | 11 |

The same code handles all four; no `left`/`right` or single-gripper branch is
hard-coded. A robot with a different number of joints only changes its profile.

| Robot | Profile | Actuator path | Current evidence |
|---|---|---|---|
| Innate Mars | yes | position-servo adapter; current-controlled wrist/gripper | external innate-os MuJoCo smoke: adapter vs direct torque, limits, gravity, contact |
| Sourccey | yes | position-servo adapter at each 5-DOF arm; lift holds | schema/vector tests only; no bench or sim validation yet |
| RB-Y1 (proxy) | yes | torso/arm adapter; current-control capability must be supplied by the backend | schema/vector tests only; base/torso actuator mode open |
| Stretch 3 | yes | position/velocity telescope + wrist adapter | schema/vector tests only; no bench validation yet |

The point of the table is that the policy/training contract is shared; only the
profile and actuator adapter are robot-specific. A backend that cannot provide
torque feedback still participates through the position-servo adapter and must
declare that limitation in its capability manifest.

## Actuator adapter

### Position servo (default for Mars)

For each joint:

```text
tau_total = clip(tau_cmd + gravity + friction + kd_eff * qd,
                 +/-tau_limit)
tau_total = clip(tau_total, +/-current_limit * kt * gear_ratio)   # if known
q_cmd     = clip(q_meas + clip(tau_total / kp_eff, +/-max_lead),
                 position_limit)
i_limit   = |tau_total| / (kt * gear_ratio)                      # if known
```

`kp_eff` and `kd_eff` are **identified effective position-loop gains**, not
datasheet values. The `kd_eff * qd` term compensates the servo's own velocity
damping so the policy's torque is the net requested torque. `max_lead` is the
safety-critical bound on how far the policy can pull the position target from
the measured joint. The adapter fails closed if
either is missing. It sends position targets and, when supported, a current/PWM
limit through the robot-side transport.

### Other adapter modes

| Mode | Policy contract | Hardware command | Use |
|---|---|---|---|
| Position servo | torque | position + current limit | Mars now; stable, approximate |
| Current-controlled position | torque | current limit + position | available on Dynamixel X-series; needs driver/bench work |
| True torque | torque | current/torque | requires torque loop, thermal model, torque sensing/observer |
| Impedance | pose + Kp/Kd + tau_ff | position + current feedforward | hardware-native alternative; can be a future adapter |

Impedance is attractive because it is closer to how these servos actually
behave. It should be an **adapter-level** option first. If the policy needs to
choose stiffness, add it to the action schema in a later version rather than
overloading torque.

## Bench identification (before hardware policy motion)

Start with the elbow (joint 3), the current failure point.

1. **Static torque mapping**: hold several poses with the forearm supported;
   command current/PWM or torque offsets; measure joint torque with a load cell
   (or a known lever arm). Fit `tau(I, q)` and `kp_eff`.
2. **Gravity model**: slow sweep through the workspace; record holding current
   at a grid of poses; fit `g(q)` for the gravity feedforward.
3. **Friction**: slow up/down sweeps at constant velocity; fit stiction and
   viscous friction.
4. **Step/frequency response**: command small torque steps; measure `q`, `qd`,
   current, delay, rise time, overshoot; identify bandwidth.
5. **Thermal/current limits**: sustained 50% and 100% of the planned limit;
   record temperature, voltage, current; derate before the policy is deployed.
6. **Supported → unsupported hold**: repeat the same commands with and without
   forearm support; verify no stall or latched overload.

Deliverable: `actuator_model.json` with `kp_eff`, `max_lead`, gravity/friction
coefficients, current/torque constants, thermal limits, and uncertainty. Store
its hash in the policy bundle. A profile without a completed identification is
research-only.

## Training plan

1. **Simulation uses the same adapter.** Do not train on ideal torque and deploy
   on position servos. Start with the position-servo adapter in MuJoCo, then
   compare against true torque actuators.
2. **Randomize the actuator model**: `kp_eff`, friction, armature, delay,
   payload, current limit. Report the identified nominal and the range.
3. **Teacher torques**: inverse dynamics or a PD target from the analytic
   teacher. Record `tau_cmd`, `tau_meas`/current, `q`, `qd`, and the adapter
   mode in the case library.
4. **Start with one arm and one target**, then dual-arm tasks. Use
   `success_mode="all"` only when both arms genuinely have targets.
5. **Keep the existing velocity tracker** for free-space tracking until the
   torque actor matches it. Compare on the same held-out cases.
6. **Acceptance for the first torque actor**:
   - bench model predicts measured torque/motion within a documented tolerance;
   - free-space reach tracks a 5 cm target without contact or overshoot;
   - gentle contact push is stable and stops on the guard;
   - no wheel lift, joint-limit overrun, thermal trip, or unconfirmed stop;
   - no regression on the retained velocity actor's free-space cases.

## Migration and versioning

- `SCHEMA_VERSION = 2`. The v1 flat single-arm contract is **not** silently
  reinterpreted.
- Policy bundles must include `embodiment` name, profile hash, schema version,
  action mode, and actuator-model hash. A mismatch is rejected.
- Old velocity checkpoints remain v1 artifacts. They are not loaded as torque
  actors.
- The case library stores nested multi-arm goals/state in JSON. Legacy
  single-arm records remain readable; new records add `meta.embodiment` and
  `meta.schema_version=2`.
- `synthetic_obstacle_target(..., profile=...)` emits the multi-arm layout;
  without a profile it preserves the legacy single-target format.

## Safety

Torque actions require tighter constraints than velocity actions:

- `clamp_action` is the mandatory action-boundary clip for base velocity,
  per-joint torque, and gripper command limits; adapters may only tighten it;
- per-joint torque and current limits, plus a total power/thermal budget;
- position-target lead (`max_lead`) and joint-limit clipping;
- torque-rate limiting;
- a robot-side watchdog that commands a measured hold on stale feedback,
  stream loss, overcurrent, or thermal fault;
- independent contact/wheel-lift guards (existing executor behavior);
- no mode switch while powered without the native hold/recovery procedure.

The adapter is feedforward; it is not a force limit. A contact force estimate
or a compliant guard remains a separate requirement.

## Robot backend contract

The multi-robot abstraction lives here, not in any one robot driver. Each
backend (innate-os, a ROS 2 Control system, MuJoCo, Isaac, Gazebo, …) must
publish enough capability information to build an `EmbodimentProfile`:

- base kind and action dimension;
- arm/joint-group names, joint order, DOF, EE links, grippers;
- per-joint position/velocity/torque limits and control modes;
- current/torque feedback availability and units;
- contact/force sensing and gripper force/contact state;
- command lease/TTL, measured stop, and state timestamps.

For Mars specifically, see
[innate-os-wbc-interface.md](innate-os-wbc-interface.md). The working prototype
lives in the fork PR
[cpaxton/innate-os#1](https://github.com/cpaxton/innate-os/pull/1); it is not
vendored and not a hardware qualification. The short version is
that the current XL430/XC430 shoulder and elbow joints have no current control;
they can realize a position-servo impedance adapter but not true joint torque.
The wrist and gripper are current-capable. Do not present the Mars adapter as
full-arm torque control.

## Relationship to the existing whole-body stack

The existing `src/emet/rl/manipulation/whole_body_*` code is the older
gain/velocity tracker: it emits seven motion outputs, consumes pose targets,
and has its own `test_whole_body_v2.py`. That stack remains the free-space
baseline and is not redefined here.

This document/schema is the new **torque contract**: a multi-arm, per-arm
action/observation API with an actuator adapter boundary. It currently has
contract/adapter tests and the external innate-os smoke test, not a training
consumer. When a training loop adopts it, the two stacks should be compared on
the same held-out cases rather than merged by name.

## Open questions

- **Mars servo mode**: current-controlled position vs true current control on
  the XL430s. The merged recovery patch accepts `CURRENT_CONTROLLED_POSITION`,
  but the normal app path still writes position goals.
- **Gripper force**: keep position commands or expose force/current as a
  separate channel with its own limits.
- **RB-Y1 base**: holonomic velocity is the current schema; wheel torque and
  swerve steering are deferred.
- **Torso**: torque vs position hold for RB-Y1; the adapter can support both,
  but the profile must declare which.
- **Impedance**: fixed adapter gains vs policy-selected stiffness/damping.
  Start fixed; add a versioned action extension if needed.
- **Torque feedback**: none of the current arms has calibrated joint torque
  sensing. `Present Load`/current is a proxy until bench identification.

## File map

| Path | Purpose |
|---|---|
| [`wbc_contract.py`](../../src/emet/controller/manipulation/wbc_contract.py) | Schema v2, profiles, actions/observations, reward, vector helpers |
| [`torque_adapter.py`](../../src/emet/controller/manipulation/torque_adapter.py) | Position-servo adapter and adapter protocol |
| [`wbc_profiles.py`](../../src/emet/controller/manipulation/wbc_profiles.py) | Example Mars/Sourccey/RB-Y1/Stretch layouts |
| [`case_library.py`](../../src/emet/controller/manipulation/case_library.py) | Deduplicated scenario records |
| [`data_generators.py`](../../src/emet/controller/manipulation/data_generators.py) | Synthetic obstacle/target scenarios; multi-arm mode |
| [`failure_record.py`](../../src/emet/controller/manipulation/failure_record.py) | Failure/trace -> scenario record |
| [`src/test/motion/test_wbc_foundations.py`](../../src/test/motion/test_wbc_foundations.py) | Contract, multi-arm, case library, generator tests |
| [`src/test/motion/test_torque_adapter.py`](../../src/test/motion/test_torque_adapter.py) | Adapter mapping, limits, fail-closed tests |
| [`scripts/validate_innate_torque_adapter.py`](../../scripts/validate_innate_torque_adapter.py) | External innate-os MuJoCo adapter/contact smoke test |
| [`innate-sim-validation.md`](innate-sim-validation.md) | Setup, metrics, and limits of the external sim validation |
