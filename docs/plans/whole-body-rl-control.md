# Whole-body RL control: multi-arm torque contract

Status: **schema v2 + adapter-matched trainer implemented; no hardware policy
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
- `action_bounds(profile)` returns the per-channel physical bounds in vector
  order. `normalized_action_vector` / `action_from_normalized_vector` map
  bounded base/torque/gripper channels affinely onto `[-1, 1]` for PPO-style
  training; a gripper with no declared `command_limits` passes through
  unbounded so the gap is visible rather than silently rescaled.
- `active_action_mask(observation, profile)` marks the channels the active task
  may use. Arm channels follow that arm's target; base and torso channels stay
  active for arm tasks so the controller can extend reach. The training loop
  masks inactive channels; the executor still holds those joints rather than
  interpreting zero torque as a safe command.
- The head group is **observed-only** in this schema. It has no action channel
  and is excluded from `controlled_joint_names`, so a head feedforward torque
  is rejected rather than silently ignored.
- `action_vector(action, profile)` flattens in **base, torso, arms (profile
  order), per-arm gripper** order. `action_from_vector` reverses it and
  validates the result against the profile.
- The policy never sees joint-name strings; the training loop uses the profile
  to build tensors. `EmbodimentProfile.to_dict()` / `from_dict()` and the
  `fingerprint` property are the checkpoint/dataset identity. The profile hash
  must be stored in the policy bundle, not only the robot name.

### Observation

```text
WbcObservation:
  base_pose: (x, y, theta)
  base_twist: (d_base,)
  base_target?: (x, y, theta)     # masked SE(2) goal
  base_target_mask: (x, y, heading)
  torso: JointGroupObservation | null   # position/velocity/torque + target?
  head: JointGroupObservation | null    # observed-only
  arms: [
    { name,
      joint_position, joint_velocity, joint_torque?,
      ee_pose, ee_target?, target_mask=(position, orientation),
      gripper?, contact? }
  ]
  occupancy: (H, W, D) uint8 | null
  occupancy_origin?, occupancy_resolution?   # required together
  failure_mode: str
```

- `ee_pose` and `ee_target` are both expressed in the arm's
  `ArmSpec.target_frame`; an arm with `ee_target=None` is observed but inactive
  for the task. This is how a one-arm task runs on a dual-arm robot. The Mars
  simulator profile uses `world` goals so base motion can assist reach; a
  base-frame target instead moves with the robot.
- `base_target` is the masked SE(2) goal in the odometry frame. It is optional,
  so base-only tasks can run on a multi-arm robot, and a fixed base rejects it.
- `torso.target` is a joint-space posture target. It is optional; an active
  torso target is included in success and reward. Head targets are rejected
  because the head has no action channel in schema v2.
- `target_mask` allows position-only or orientation-only arm targets; the base
  has independent x/y/heading masks.
- `occupancy` is a local crop in the base frame. When supplied, `origin` and
  `resolution` locate the voxels so different embodiments cannot silently
  disagree about geometry. No GT object pose.
- `joint_torque` is measured/estimated when available; otherwise it is empty.
  The policy must not assume torque feedback exists.
- `contact` is an optional measured `ContactObservation` in the EE link frame:
  `in_contact`, a 6D wrench, and a measurement source. It is a wrist F/T sensor
  or contact-normal estimate, never an object identity or privileged contact
  label. The raw wrench may include the distal gravity/inertial load. A backend
  without force sensing leaves it `None`; the layout carries a presence flag so
  a checkpoint cannot confuse missing contact with zero load.
- `gripper.force` is the measured per-finger load when available. It is separate
  from the wrist wrench so a two-finger grasp can report bilateral contact.
- The training encoder adds derived, sensor-only error features that are not
  part of the semantic contract: base x/y and wrapped heading error, plus EE
  position and orientation (rotation-vector) error per arm. These helped the
  combined task after DAgger; the encoder layout is versioned separately
  (`OBSERVATION_LAYOUT_VERSION`).

### Reward

`reach_reward(action, next_obs, profile=..., ...)` is the whole-body tracking
reward, not just a single-arm reach:

- position and orientation errors honor each arm's `target_mask`; orientation
  error and `quat_tolerance` are geodesic radians, not a quaternion
  dot-product residual;
- an active `base_target` contributes masked x/y and wrapped-heading error;
  an active `torso.target` contributes joint-space posture error (wrapped for
  revolute joints);
- `arm_weights` controls task priority;
- `success_mode="all"` (bimanual) or `"any"` (single-arm task on a multi-arm
  robot) controls only the arm terms; active base and torso targets are always
  required. Base-only and torso-only goals are valid;
- action and jerk penalties normalize by each joint's torque limit and base
  velocity limit, so coefficients transfer across robots;
- collision is a global penalty; per-arm components are logged as
  `arm.<name>.pos_err`, etc., plus `base_xy_err`, `base_heading_err`, and
  `torso_err`.

## Example layouts

These are schema examples in
[`wbc_profiles.py`](../../src/emet/controller/manipulation/wbc_profiles.py).
They use published ranges where available and clearly-marked placeholder torque
values; they are not hardware calibration. `mars_sim_profile()` is the separate
simulator-matched Mars profile (`kp=80`, `kv=4`, effort `2.0`) used by the
MuJoCo trainer; the example Mars profile remains the bench-replacement target.

Additional sim-matched profiles are derived from the packaged MJCFs:

- `stretch_sim_profile()` — five motion axes (lift, virtual telescope, three
  wrists) + gripper;
- `sourccey_sim_profile()` — planar base, lift, two 5-DOF arms, two grippers;
- `rby1_sim_profile()` — holonomic base, 4-DOF torso, two 6-DOF arms, two
  2-finger grippers, matched to the packaged Galaxea R1 proxy.

`sim_profile(robot)` is the registry. RB-Y1 is intentionally a proxy profile;
the direct-MuJoCo base must map `(vx, vy, yaw_rate)` through its three
steer/wheel modules, and the RoboCasa RB-Y1 adapter is the benchmark source of
truth for that robot.

### Robot and RoboCasa matrix

`wbc_robots.py` is the robot registry; `wbc_robocasa.py` is the explicit
RoboCasa/Robosuite hook boundary. A missing adapter fails closed rather than
silently training against the wrong controller.

| Robot | Sim profile | Direct MuJoCo | RoboCasa hook |
|---|---|---|---|
| Innate Mars | `mars_sim_profile()` | trained DAgger 0.973 held-out | not a RoboCasa target |
| Stretch 3 | `stretch_sim_profile()` | trained DAgger 0.953 held-out | `Stretch` |
| Sourccey | `sourccey_sim_profile()` | trained DAgger 0.952 held-out | `Sourccey` |
| RB-Y1 | `rby1_sim_profile()` | trained DAgger 0.996 held-out | Galaxea R1 proxy path |
| PandaOmron | `pandaomron_sim_profile()` | composed Omron+Panda plant; clean 1.000 / robust 0.991 held-out | `PandaOmron` hook |

Held-out numbers are 50 episodes per subtask on a fresh seed. All are direct
MuJoCo results; none is a RoboCasa benchmark result.
Backend/teacher/DAgger details:
[wbc_multirobot_20261125.md](../experiments/wbc_multirobot_20261125.md).

PandaOmron is an Omron holonomic base + 1-DOF torso height + one 7-DOF Panda
arm + two-finger gripper. `PandaOmronWbcBackend` composes those Robosuite
assets into a direct MJCF: the Panda arm and torso are torque motors, so it uses
`TorqueAdapter`; the base is velocity-controlled and the gripper is positional.
The plant resets to Robosuite's collision-free Panda initial posture; the
composed model's zero pose folds the wrist into link 5. Stable 20 Hz tracking
uses identified 300/30 gains for the arm. The teacher and promoted DAgger policy
both clear the direct-MuJoCo suite. The hard-posture family is not yet
promoted: the first export contained stale successful-grasp records, and only
one genuine reach failure remains, so the family is planned until #267 adds
more failure data. Its RoboCasa composite controller is
OSC/JOINT_POSITION, not the torque contract, so the benchmark adapter must map
explicitly. A reach-tracking RoboCasa adapter now drives the native `PandaOmron` kitchen
sim; benchmark task targets and the merged-scene Stretch/Sourccey/RB-Y1
adapters are still open.

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
| True torque | torque | current/torque | `TorqueAdapter`; requires torque-capable joints, torque loop, thermal model, torque sensing/observer |
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

Deliverable: an actuator-identification JSON consumed by
`load_actuator_identification` with per-joint `kp_eff`, `max_lead`, current/torque
constants, gear ratio, control mode, and limits, plus a separate
`FeedforwardModel` JSON for static gravity/load bias. Store both hashes in the
policy bundle. A profile without a completed identification is research-only.
The calibration CLI only fits what the failure traces contain; torque constants,
friction, bandwidth, and thermal derating still require the hardware sweeps
above.

## Training plan

1. **Simulation uses the same adapter.** Do not train on ideal torque and deploy
   on position servos. Start with the position-servo adapter in MuJoCo, then
   compare against true torque actuators.
2. **Randomize the actuator model**: `kp_eff`, friction, armature, delay,
   payload, current limit. Report the identified nominal and the range.
3. **Teacher torques**: inverse dynamics or a PD target from the analytic
   teacher. Record `tau_cmd`, `tau_meas`/current, `q`, `qd`, and the adapter
   mode in the case library.
4. **Start with one arm and one stationary target**, then arm-only tracking,
   base-only driving, combined base+arm, and dual-arm tasks. The Mars base is
   differential-drive `(vx, yaw_rate)`, so a lateral base goal requires a
   turn-and-drive maneuver rather than direct translation. Use
   `success_mode="all"` only when both arms genuinely have targets.
5. **Keep the existing velocity tracker** for free-space tracking until the
   torque actor matches it. Compare on the same held-out cases.
6. **Acceptance for the first torque actor**:
   - bench model predicts measured torque/motion within a documented tolerance;
   - free-space reach tracks a 5 cm target without contact or overshoot;
   - gentle contact push is stable and stops on the guard;
   - no wheel lift, joint-limit overrun, thermal trip, or unconfirmed stop;
   - no regression on the retained velocity actor's free-space cases.

## Training wiring

The contract now has a runnable vertical slice:

- [`wbc_env.py`](../../src/emet/rl/manipulation/wbc_env.py) builds a fixed-width,
  profile-derived observation vector and a normalized action Gymnasium space.
  It decodes through the contract, calls an `ActuatorAdapter`, and scores the
  next observation with `reach_reward`.
- [`wbc_mujoco.py`](../../src/emet/rl/manipulation/wbc_mujoco.py) is the first
  backend. It is **adapter-matched Mars**, not a true-torque plant: the policy
  emits torques, the position-servo adapter turns them into bounded position
  targets, and the existing MuJoCo servo model executes them. The base uses the
  planar `MarsControl` `base_x/y/yaw` velocity actuators, not the weak
  physical-wheel contact model (whose 0.05 Nm actuators make base goals
  unreachable). Goals are sampled by FK from a randomized reachable arm
  configuration; base goals are optional. The backend applies the retained
  base-acceleration, joint-target-rate, and gripper-rate slew limits, and holds
  the gripper open for reach training because the current contract has no
  gripper target or reward. `--teacher-steps` optionally behavior-clones a
  simulator-privileged PD teacher before PPO; the teacher's sampled joint
  target never enters the actor observation or exported checkpoint. The BC
  policy is evaluated and saved as `bc_policy.pt` before PPO, and
  `--bc-kl-coefficient` anchors PPO to it so a short fine-tune cannot silently
  erase the warm start.
- [`wbc_training.py`](../../src/emet/rl/manipulation/wbc_training.py) is a small
  PPO trainer and checkpoint format. The checkpoint carries the full profile
  payload/fingerprint, observation/action width, observation-layout version,
  PPO config, and metadata. Loading with a different profile is rejected
  before inference. The first trainer is single-arm Mars; the environment
  exposes `info["action_mask"]` for future multi-arm masked PPO, but this
  trainer does not yet consume it.
- [`wbc_calibration.py`](../../src/emet/rl/manipulation/wbc_calibration.py) fits
  the static per-joint bias from the manipulation-failure sink. Pass the saved
  feedforward to training with `--feedforward`; the trainer converts bias to
  torque with the profile's identified `kp_eff` and gives it to the adapter.

Smoke commands:

```bash
python scripts/fit_wbc_feedforward.py \
  --records ~/runs/emet/manip_failures/manip_failures.jsonl \
  --embodiment innate_mars --out runs/wbc/feedforward.json

EMET_WBC_FEEDFORWARD=runs/wbc/feedforward.json \
  python -m emet.rl.manipulation.wbc_training \
  --steps 2048 --output runs/wbc --device auto \
  --target-range 0.15 --base-goal-mode forward \
  --teacher-steps 4096 --teacher-epochs 30 --bc-kl-coefficient 0.1 \
  --checkpoint-every 512 --eval-episodes 10

# Continue an interrupted run (steps are additional, not total)
python -m emet.rl.manipulation.wbc_training \
  --resume runs/wbc/checkpoint.pt --steps 2048 --output runs/wbc --device auto
```

`--device auto` selects CUDA when available. `metrics.jsonl` receives one row
per update, and `checkpoint.pt` carries policy, optimizer, step/update count,
profile fingerprint, and observation-layout version for resume. `wbc_calibration.py`
provides the same fit as a tested module API; the script adds the operator-facing
dry-run/feedforward-contamination flags.

### DAgger subtask suite

[`wbc_dagger.py`](../../src/emet/rl/manipulation/wbc_dagger.py) adds the
actor-visited-state loop that plain BC needs:

- suite: `arm` (held base, EE target), `base` (held arm, planar base target),
  and `combined` (forward-cone base + arm target);
- robots with two arms default to `arm_left` / `arm_right` / `base` /
  `combined_left` / `combined_right`: reaching with both arms at once is a much
  harder imitation target than either arm alone;
- Stretch uses a 5 s arm and 8 s combined window because its underdamped
  position-servo wrists need longer to settle;
- initial teacher rollouts and BC warm start (200 epochs by default; the first
  multi-arm runs were undertrained at 30);
- each DAgger iteration rolls out the current actor with small action noise,
  queries the privileged teacher at the visited states, aggregates the labels,
  retrains the actor, and evaluates each subtask separately;
- teacher-only evaluation is recorded as the ceiling; the best checkpoint is
  chosen by aggregate held-out success.

```bash
python -m emet.rl.manipulation.wbc_dagger \
  --robot innate_mars --iterations 4 \
  --teacher-steps 2048 --dagger-steps 1024 \
  --bc-epochs 200 --eval-episodes 10 --device auto --output runs/wbc-dagger
```

The teacher uses simulator-only target/phase information; it is never part of
the actor observation or exported checkpoint.

The MuJoCo trainer uses `mars_sim_profile()`, so a checkpoint must be loaded
with that same sim-matched profile; the example Mars profile is the
bench-replacement target and has a different fingerprint. These are
wiring/prototype commands. The example profile's actuator fields are still
placeholders; replace them with bench identification before interpreting the
reward or claiming contact performance.

### First smoke evidence

On 2026-11-25, the contract-native Mars path was exercised on the local RTX
4070. These are wiring checks, not a benchmark or a promoted controller:

- Cold PPO from scratch (18k steps, arm-only, 0.15 rad target range, held base)
  stayed at 0/10 eval success; the reward curve was flat.
- A simulator-privileged PD teacher behavior-cloned for 2,048 samples / 30
  epochs (MSE 0.054) reached **16/20 (80%)** eval success on the arm-only
  distribution without any PPO. This is the warm-start reference.
- Two thousand PPO steps with a 0.1 BC-KL anchor and 3e-4 actor LR retained
  **15/20 (75%)**. The same fine-tune without the KL anchor and at 1e-3 LR
  reached 1/10 after 2k, then degraded to 0/20 after 10k. The BC-KL anchor is
  therefore required for short fine-tunes.
- The contact-wheel base model was unusable for base goals (0.05 Nm wheel
  actuators); the planar `MarsControl` base is used instead.
- The first base+arm curriculum uses a forward cone, staged base-then-arm
  teacher, and arm IK against the current base.
- Plain teacher BC on the three-task suite starts around **0.45** aggregate
  (arm ~0.35, base 1.0, combined 0.0). Combined is the distribution-shift
  failure case.
- Adding derived sensor-only error features (base x/y/heading error, EE
  position/rotation-vector error) and running DAgger changed the result
  sharply. DAgger iteration 1 reached **0.92** aggregate, and iterations 5–6
  reached **0.95** (arm 1.0, base 1.0, combined 0.85).
- Fresh held-out validation of the best checkpoint, 50 episodes per subtask:
  **0.973 aggregate** (arm 1.0, base 1.0, combined 0.92). The teacher ceiling
  on the same seeds is **0.98 aggregate** (combined 0.94). Artifacts:
  `logs/eval/wbc_dagger_features_20261125/`. Full protocol and limitations:
  [wbc_dagger_20261125.md](../experiments/wbc_dagger_20261125.md).

Treat these as evidence that the contract, DAgger loop, derived features, and
teacher-parity path work on the current simulator distribution. The policy is
still simulator-only; broader base goals, contact tasks, and hardware
qualification remain.

## Migration and versioning

- `SCHEMA_VERSION = 2`. The v1 flat single-arm contract is **not** silently
  reinterpreted.
- Policy bundles must include `embodiment` name, `EmbodimentProfile.fingerprint`,
  schema version, action mode, and actuator-model hash. A mismatch is rejected.
  The fingerprint covers the full profile, including limits and actuator-model
  fields, so a renamed profile cannot hide a changed control contract.
- Old velocity checkpoints remain v1 artifacts. They are not loaded as torque
  actors.
- The case library stores nested multi-arm goals/state in JSON. Legacy
  single-arm records remain readable; new records add `meta.embodiment`,
  `meta.embodiment_fingerprint`, and `meta.schema_version=2`. The scenario
  fingerprint includes all three, so a profile revision is a distinct case.
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
action/observation API with an actuator adapter boundary. It now has contract
and adapter tests, the external innate-os smoke test, and a contract-native
Gym/MuJoCo/PPO wiring. The two stacks should still be compared on the same
held-out cases rather than merged by name; the new trainer is not automatically
better than the retained velocity baseline.

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
| [`torque_adapter.py`](../../src/emet/controller/manipulation/torque_adapter.py) | Position-servo and true-torque adapters |
| [`wbc_profiles.py`](../../src/emet/controller/manipulation/wbc_profiles.py) | Example layouts and bench-identification patch loader |
| [`case_library.py`](../../src/emet/controller/manipulation/case_library.py) | Deduplicated scenario records |
| [`data_generators.py`](../../src/emet/controller/manipulation/data_generators.py) | Synthetic obstacle/target scenarios; multi-arm mode |
| [`failure_record.py`](../../src/emet/controller/manipulation/failure_record.py) | Failure/trace -> scenario record |
| [`failure_dataset.py`](../../src/emet/controller/manipulation/failure_dataset.py) | Reader/validation for recorded manipulation attempts |
| [`wbc_feedforward.py`](../../src/emet/controller/manipulation/wbc_feedforward.py) | Static per-joint feedforward fit from tracking failures |
| [`wbc_env.py`](../../src/emet/rl/manipulation/wbc_env.py) | Contract observation layout, Gym env, reward adapter |
| [`wbc_mujoco.py`](../../src/emet/rl/manipulation/wbc_mujoco.py) | Adapter-matched Mars MuJoCo backend |
| [`wbc_training.py`](../../src/emet/rl/manipulation/wbc_training.py) | PPO trainer and profile-checked checkpoint |
| [`wbc_calibration.py`](../../src/emet/rl/manipulation/wbc_calibration.py) | Failure-sink -> feedforward library/CLI |
| [`wbc_dagger.py`](../../src/emet/rl/manipulation/wbc_dagger.py) | Subtask suite, actor-visited teacher labeling, DAgger loop |
| [`wbc_robots.py`](../../src/emet/controller/manipulation/wbc_robots.py) | Robot/profile registry and RoboCasa names |
| [`wbc_backends.py`](../../src/emet/rl/manipulation/wbc_backends.py) | Direct-MuJoCo / RoboCasa backend factory |
| [`wbc_robocasa.py`](../../src/emet/rl/manipulation/wbc_robocasa.py) | Explicit RoboCasa/Robosuite registration hooks |
| [`scripts/fit_wbc_feedforward.py`](../../scripts/fit_wbc_feedforward.py) | Operator feedforward-fit CLI with dry-run/contamination flags |
| [`src/test/motion/test_wbc_foundations.py`](../../src/test/motion/test_wbc_foundations.py) | Contract, multi-arm, case library, generator tests |
| [`src/test/motion/test_torque_adapter.py`](../../src/test/motion/test_torque_adapter.py) | Adapter mapping, limits, fail-closed tests |
| [`src/test/motion/test_wbc_env.py`](../../src/test/motion/test_wbc_env.py) | Env layout, reward, PPO/checkpoint tests |
| [`src/test/motion/test_wbc_calibration.py`](../../src/test/motion/test_wbc_calibration.py) | Failure-record feedforward tests |
| [`scripts/validate_innate_torque_adapter.py`](../../scripts/validate_innate_torque_adapter.py) | External innate-os MuJoCo adapter/contact smoke test |
| [`innate-sim-validation.md`](innate-sim-validation.md) | Setup, metrics, and limits of the external sim validation |
