# Mars reactive control in simulation

The optional `emet.rl` package provides a synchronous Gymnasium environment,
PPO training, a deterministic baseline, and reactive control in the existing
Mars MuJoCo server. `BaseZmqServer` continues to own transport and command
receipts. The default controller remains `legacy`.

The reactive ZMQ runtime currently supports Innate Mars in the MuJoCo simulation.
It does not enable Mars hardware motion or replace Nav2. Stretch support in this
package is limited to simulation, training, and offline TAMP regression; no live
Stretch command adapter is provided. Hardware publishers and perception adapters
are follow-on work, and no learned policy is claimed to beat the baseline.

## Start with base-only control

The CLI now defaults to `--task base --architecture mlp`: two 64-unit hidden
layers with ten samples of history at 20 Hz (300 scalar inputs). Actions are
forward velocity and yaw rate. The arm targets stay fixed; its collision geometry
remains active. Each sample contains relative goal position/heading, measured
body velocity, applied commands, nine obstacle distances plus validity bits, and goal distance/direction.
Obstacle inputs currently come from simulator geometry, not physical sensors.

Normal commands are limited to 0.20 m/s, 0.80 rad/s, 0.40 m/s² and 1.20 rad/s².
Retargeting preserves velocity continuity. Obstruction and emergency stops can
bypass acceleration limits. These are simulator tuning values, not verified Mars
hardware limits. Innate's official example commands the base at 0.2 m/s, but does
not document a hardware maximum ([Innate OS example](https://github.com/innate-inc/innate-os)).
Success requires position error <=2 cm, heading error <=2°,
and low measured velocity continuously for 0.30 seconds. The arm is held through
position servos, not pinned in the physics state.

Start the conventional base controller through the existing server:

```bash
emet serve mujoco --robot innate_mars --headless --no-cameras --controller base
```

Send `send_reactive_goal(client, base=[x, y, yaw])` using world-frame coordinates.
Arm goals are rejected in this mode. A learned base bundle selects the same
base-only adapter automatically when loaded with `--controller rl --policy-path`.
The default server controller is still `legacy`.

Train and compare with the conventional controller on identical held-out seeds:

```bash
python -m emet.rl train --task base --architecture mlp --envs 1 --seeds 0 \
  --steps 16384 --output logs/mars-base-smoke
python -m emet.rl evaluate --task base --episodes 20 --output /tmp/base-baseline.json
python -m emet.rl evaluate --task base --episodes 20 \
  --policy logs/mars-base-smoke/seed-0 --output /tmp/base-policy.json
```

Stage 0 isolates short pose goals on an empty floor. Stage 1 expands the area,
varies base drive gains/damping and adds 0–100 ms of action delay; stage 2 adds
mid-motion goals and disturbances. Stages 3–4 add static/moving obstacles.
Randomization and command queues reset deterministically with the episode seed.

Rewards combine pose-error progress and settled success with measured acceleration,
command-jerk and positional-overshoot penalties. Evaluation separately reports
final errors, settling time, peak overshoot along the goal approach direction,
RMS measured acceleration/jerk, peak commanded acceleration, collision force and
success. Differenced velocity metrics are sampled at 20 Hz; they do not measure
all high-frequency vibration. Ordinary evaluation stops on contact; use the
explicit `--contact-training` experiment to test the obstruction guard.

Task identity, observation/action layout, acceleration limits and control profile
are validated in exported bundles. Coordinated reaching remains available with
`--task reach`; its transformer is optional, not part of the base MLP.

The initial 16,384-step base MLP trial had 0/20 held-out successes, versus 19/20
for the conventional controller; both had zero collisions. The baseline's one
miss was within pose tolerance at the 10 s horizon but had not completed the
settling interval. The learned checkpoint is experimental, not a replacement.
Reports and smoothness comparisons are in `logs/mars-base-smoke/`.

## Install and run

Install the optional dependencies into your existing Emet environment:

```bash
uv pip install -e '.[rl]'
python -m emet.rl evaluate --task reach --episodes 50 --output /tmp/mars-baseline.json
python -m emet.rl train --task reach --output logs/mars_rl --steps 200000 --seeds 0 1 2 --envs 4
python -m emet.rl evaluate --task reach --policy logs/mars_rl/seed-0 --episodes 50 --output /tmp/mars-policy.json
python -m emet.rl replay --task reach --policy logs/mars_rl/seed-0 --output /tmp/mars-replay.json
```

`--envs 1` avoids subprocess workers in restricted environments. Training uses
CPU PPO and one torch thread per trainer. On workstations with known unreliable
CPU cores, use the existing CPU-affinity tools before starting training.

Replay optionally accepts `--video /tmp/mars.mp4`; this requires working MuJoCo
offscreen rendering. Training and ordinary evaluation need no rendering context.
Output files and policy bundles are local artifacts, not tracked source assets.

The stages are explicit so experiments are reproducible:

| Stage | Task |
| --- | --- |
| 0 | Small free-space base and joint target changes |
| 1 | Larger base displacement |
| 2 | Stage 1 plus a target change at 3 s and velocity disturbance at 5 s |
| 3 | Stage 2 plus three static box obstacles |
| 4 | Stage 3 with moving obstacles |

Progress a trained bundle with `train --resume logs/mars_rl/seed-0 --stage 2
--output logs/mars_recovery --seeds 0`. Resume restores the policy and observation
normalization. Use separate output directories to preserve earlier experiments.
Evaluate each stage on the same held-out seeds (`--seed 10000` by default), and
also run `evaluate --scene table` to exercise the existing table scene.

## Control contract

### Coordinated pose-policy contract

The next policy generation uses one masked target set rather than selecting a
separate base or arm controller. `PoseTargets` in `emet.rl.pose_targets`
represents one world-frame SE(2) base target and a variable-length set of
world-frame SE(3) hand targets. Each target carries a mask; a cleared base mask
holds the measured base pose, and cleared/padded hand entries are ignored.
Hand orientation uses position plus the first two rotation-matrix columns.
Mars's coordinated actor receives padded target tokens and masks, then combines
masked target pooling with a two-layer causal transformer over measured history
at 20 Hz. Stretch's simulator has one hand; its 20 Hz transformer receives the
same base/hand goals as part of the measured history token. The shared contract
supports additional hands with padding and masks. A goal may require coordinated
base and arm movement, while a fixed-base goal disables the base target.

For the initial Mars coordinated curriculum, hand targets are generated by
forward-kinematics from small perturbations around the reset configuration,
then checked with the runtime's base-locked inverse-kinematics solver. Targets
whose IK solution takes a discontinuous joint branch are rejected. The policy
reward and settling test include end-effector position and orientation errors;
the current simulator tolerances are 1 cm and 0.12 rad. These local targets are
for learning the tracking task first; wider workspaces and TAMP trajectories
need separate staged evaluation.

Coordinated training can exercise measured multi-waypoint retargeting with
`--trajectory-waypoints 5`. The environment advances only after active targets meet the controller's settling
rule; masked hand targets cannot block a base-only goal. History is retained
across each target change. Coordinated base waypoints use turn/drive/final-heading
segments shared by training and evaluation, rather than asking a differential-drive
robot to perform a fresh sideways parking maneuver at each interpolated XY point. Evaluate the same contract with
`python -m emet.rl evaluate --task coordinated --trajectory-waypoints 5 ...`.
The separate simulator gate `python -m emet.rl.trajectories --coordinated`
checks five reachable pose waypoints without resetting controller history.
The current best sequence-trained policy reached 65% on 20 held-out waypoint
episodes, compared with 95% for the hand-designed controller; both had the
same one low-force contact stop. The policy is not ready for TAMP handoff or
hardware use. Keep PPO behind its bootstrap gate until sequence success and
contact severity improve.

Stretch samples bounded joint references and derives hand targets through FK.
When a base target is active, FK now evaluates the arm reference at that target
base pose, so combined base/hand goals describe a consistent reachable state.

The base action is a bounded differential-drive velocity for Stretch and a
bounded planar velocity for Mars. Arm actions adjust smooth joint references,
with gripper adjustment only in authorized grasp phases. The learned policy
handles feedback, coordination, disturbances and when to settle. A settling
deadband and explicit zero-velocity action are trained and evaluated so the
robot stops once the target is reached instead of wandering. Safety monitors
remain outside the policy and can stop it on sustained tracking failure,
excessive contact, invalid state or wheel unloading. A disabled base target is
also enforced at the action adapter, not merely indicated to the actor.

This is a new, versioned observation/action contract. Existing base and
manipulation bundles remain untouched on disk and are rejected by the new
contract. The first coordinated Mars and Stretch paths are now implemented in
simulation; policy quality is not established by the smoke runs.
Promotion requires held-out evaluation of position
and orientation error, settling time, post-settle drift, measured acceleration
and jerk, contact force and impulse, object retention, and wheel support. The
simulator's contact limits are screening thresholds, not calibrated damage
limits. Current Mars base and Stretch manipulation results do not establish
performance for this coordinated controller.

The policy runs at 20 Hz with 25 MuJoCo substeps of 2 ms each. Each environment
owns its model, data, random generator and reset state; stepping never sleeps or
uses ZMQ. Physics advances through actuators, without per-step teleportation or
state pinning.

Mars-first TAMP trajectory cases can be generated from the GT joint-path planner
without touching the live simulator state:

```bash
python -m emet.rl.manipulation.tamp_cases --robot innate_mars --episodes 10 \
  --output /tmp/mars-tamp-cases.json
```

The Mars manipulation simulator uses a free chassis with two driven wheels and a
passive swivel caster extracted from the pinned upstream `base.STL`. Add
`--execute` to replay planned arm paths through actuators and the support monitor.
A free-space reaching reset keeps the tabletop outside the short arm's workspace.
The older Mars base/coordinated navigation simulator retains planar joints and
cannot evaluate wheel lift; use the physical manipulation model for that gate.
Full public-tool Mars TAMP execution and hardware arm trajectories are still
separate integration work. See [TAMP support](api/tamp.md#mars-first-rl-trajectory-and-tamp-support).

The physical manipulation feedback controller now tracks base targets as well as
arm references. Base policy actions are bounded velocity **residuals** around an
SE(2) turn/drive/final-heading servo; a zero residual is the working teacher, not
a zero wheel command. Goal changes retain command slew limits and history.
Wheel actuator targets use transmission velocity (`gear * joint_velocity`),
including Stretch's non-unit wheel gears. Free-space reaching keeps the table
outside the workspace so base turns do not strike an incidental fixture.

Interactive hand targets prefer a local solution using forward/yaw base controls,
respect joint bounds, and check sampled robot/scene clearance before acceptance.
Mars joint1 compensates measured base yaw during the turn to limit hand sweep.
The physical base marker now accepts SE(2) nudges directly. Updated profile
contracts reject older policy bundles; earlier training results on this page need
re-evaluation against this corrected teacher and tighter planar hand tolerance.

Inspect the physical chassis in the local UI with:

```bash
MUJOCO_GL=egl python -m emet.rl.test_drive --mode mars-arm --port 8765
python -m emet.robots.innate_mars.chassis > /tmp/mars-chassis-audit.json
```

The source-derived drive wheel radius is 37.22 mm, track is 171.9 mm, and the rear
caster wheel radius is approximately 13 mm. The source's 0.89 kg assembly mass is
redistributed over CAD solids with uniform density to estimate COM and inertia.
Friction, caster mass partition and wheel torque remain simulator assumptions,
recorded in `reference/chassis_support.json`; these are not measured hardware
dynamics. New model fingerprints reject checkpoints trained with the old contact
model. Contact forces now decode both pyramidal and elliptic constraints in
newtons, with a linear-force bound that excludes torsional moments.

- Seven normalized actions: forward speed, yaw rate, and five arm target
  increments. The adapter limits these to 0.20 m/s, 0.80 rad/s, and joint-target
  changes of 0.03 rad per control tick. These are simulation defaults, not
  calibrated hardware limits.
- `joint1` through `joint5` are reaching joints. `joint6` is the gripper,
  `joint6M` is its mimic, and `joint_head` is held during a reactive command.
- Goals contain a world-frame base `[x, y, yaw]`, named arm targets, or both.
  Missing dimensions hold their configuration at goal acceptance. Partial arm
  dictionaries are supported. Cartesian targets remain an upstream IK task.
- Observations contain 25 state values (including held actuator-target errors),
  15 goal/error/mask values, 26 measured/target hand and body pose values, and 45 local
  obstacle values: conservative bounding-sphere clearance, link-frame direction and
  validity for each of nine collision proxies. The supporting plane is omitted
  from obstacle features; floor and self-collisions are still checked.
  A 20-token history stacks these 111 values at 50 ms intervals. Reset repeats
  the first measurement; reads at the same timestamp do not advance history.
  Hand poses use position plus two rotation columns; body poses use planar
  position and sine/cosine heading. Target poses come from FK of the sampled
  reachable joint/body configuration; the command API still accepts joint goals.
- Obstacle features currently use privileged MuJoCo geometry. Deployment needs
  a depth/map adapter with equivalent units, frames and missing-data handling.

The robot's visual meshes remain unchanged. `model.py` adds prototype primitive
collision proxies and contact exclusions when building the RL model. They are
not a calibrated CAD collision model. Both training and policy-enabled table
simulation use this preparation, including the same inertials and modified reaching servos.

The planar Mars coordinated controller requires active base errors below 2 cm
and 2 degrees, active hand errors below 1 cm and 0.12 rad, and active joint errors
below 0.05 rad, plus low measured velocities for 0.25 seconds. By default, collisions and invalid physics
terminate an episode; the 10-second horizon truncates it. Rewards use goal-error
progress, settled success, and penalties for error, collision, action changes,
action magnitude and joint-limit proximity. Goal changes rebase the progress
term. Recovery stages do not finish before their scheduled disturbance.

## Use through the existing simulator

```bash
emet serve mujoco --robot innate_mars --headless --no-cameras --controller baseline
# After evaluation, opt into a trained bundle:
emet serve mujoco --robot innate_mars --headless --no-cameras \
  --controller rl --policy-path logs/mars_rl/seed-0
```

Omit `--no-cameras` for normal image streams. Reactive selection currently accepts
only Mars and the default table scene. Legacy `xyt` and joint commands retain
their existing semantics; coordinated policy control uses the explicit new goal
commands. Selection never silently replaces a failing policy with the baseline.

Using an existing connected Emet client:

```python
from emet.core.command_client import (
    send_reactive_goal, update_reactive_goal, wait_reactive_goal,
    cancel_reactive_goal,
)

command = send_reactive_goal(
    client, base=[0.1, 0.0, 0.0], arm={"joint1": 0.1}, timeout=10.0,
)
update_reactive_goal(client, command, {
    "base": [0.2, 0.0, 0.0], "arm": {"joint1": 0.2},
})
reached = wait_reactive_goal(client, command, timeout=10.0)
# For an early stop: cancel_reactive_goal(client, command)
```

Updates replace the complete target, rather than patching the old target. They
are applied at the next control tick; if several arrive first, the latest wins.
The original command identity and deadline persist. Update receipts acknowledge
dispatch; only the original motion receipt reports settled completion.

`command_protocol.reactive_control` advertises support. The v2 identity envelope,
ownership, deduplication and bounded receipts remain in force. Reactive motion
and legacy navigation share one motion slot; unrelated joint/velocity/posture
commands cannot interrupt either. Cancellation targets the original session and
sequence. A confirmed stop zeros base commands and holds measured arm positions;
an unconfirmed stop retains ownership until inspection/restart. Client shutdown
attempts cancellation before releasing control.

The controller fails on invalid actions, invalid physics, forbidden contact,
observation/inference over the 50 ms budget, or physics state older than 250 ms.
The original command deadline also applies when the client disconnects. These
are software checks for the simulator, not a hardware safety controller.

## Validation and extension

```bash
python -m pytest src/test/rl src/test/core
python scripts/sync_core_runtime.py --check
```

Tests cover seeded stepping, Gymnasium behavior, actuator mapping and limits,
gripper separation, rotated base frames, collisions, training/runtime parity,
bundle normalization and compatibility, target updates, ownership and timeouts,
and coordinated completion through the real simulator and ZMQ server threads.

Bundles contain `policy.zip` and `manifest.json`: profile, model fingerprint,
ordered joint names, limits, timestep, observation normalization, weight checksum,
and training settings. Load only trusted policy bundles. Incompatible models or
profiles are rejected before control starts. Training dependencies never enter
the lightweight `emet-core` package.

Initial validation on 2026-10-13 ran three stage-0 PPO seeds for 100,096 steps
each. Held-out free-space success over 50 episodes was **14%, 2%, and 8%**,
with no collisions. All three had **0/20 table-scene successes**. The deterministic
baseline succeeded on every free-space and table trial. These checkpoints are
development artifacts, **not qualified replacements** for existing control.
Local bundles and episode reports are under `logs/mars_rl/`. Those results used
profile version 2. The history/compliance experiment is version 3 and rejects
those old bundles; retrain before continuing the curriculum. In particular, table
geometry is out of distribution for stage-0 training, whose obstacle features
are constant. Obstacle training and normalization/generalization experiments
are necessary before promotion.

Evaluation exports individual episode results, success/collision rates, tracking
errors, clearance, action changes, joint-limit violations, throughput, and the
95th-percentile observation/inference time. Compare all three training seeds to
the same deterministic baseline. Keep learned control opt-in until it matches
free-space success and improves recovery or obstacle response without increasing
collisions. A short smoke-training run is not sufficient evidence for promotion.

Next: implement Mars hardware publishers matching Stretch's semantics and verify
feedback freshness, stop behavior, limits and geometry; add a perception adapter;
then implement robot profiles/adapters for Stretch and Sourccey and evaluate
RoboCasa/MolmoSpaces scenes. No universal cross-robot policy is assumed.

## History and contact experiment (profile 3)

The coordinated experiment uses `--task reach --architecture transformer` for
a two-layer causal transformer (64-wide, four attention
heads, no dropout), predicting one action per tick. `--architecture mlp` supplies
an ablation using the same observations. PPO stores the full history in each
transition; serving uses the same history builder and exported normalization.

The simulation reaching servos now use kp=80, kv=4 and a +/-5 Nm actuator limit.
Joint targets stay within 0.08 rad of measured positions. A detector runs every
physics step: residual above 0.06 rad with speed below 0.03 rad/s for 0.20 s
latches the affected joint, removes its accumulated target error, and suppresses
further commands in the blocked direction. A reverse command permits retreat.
The base detects measured progress below half the commanded velocity, with
leaky evidence accumulation to tolerate brief feedback jitter. These
are tracking-failure heuristics; gravity, friction or delay can produce false
positives. The numeric settings are uncalibrated simulation experiments, not
Mars hardware ratings. The gripper and head retain their original held servos.

`--contact-training` permits light contact during training/evaluation and adds
force/impulse penalties. Aggregate non-support robot contact force above 30 N
terminates at the physics substep; force is privileged reward/metric information,
not policy input. Evaluation records episode peak force, integrated impulse,
obstruction steps and whether any collision occurred. Runtime serving retains
its stricter collision-stop behavior. This experiment does not implement a
hardware impedance loop or certify physical compliance.

```bash
python -m emet.rl train --task reach --architecture transformer --contact-training --stage 3 \
  --steps 2048 --envs 1 --seeds 0 --output logs/mars-history-smoke
python -m emet.rl evaluate --task reach --policy logs/mars-history-smoke/seed-0 \
  --contact-training --stage 3 --episodes 10 --output /tmp/mars-history-eval.json
```

The 2,048-step smoke run completed training/export/reload and ten held-out
obstacle episodes: 0/10 successes and 0/10 collisions, with approximately 1 ms
p95 observation/inference time. This verifies the pipeline only. Longer
curriculum training, matched MLP comparison, contact-recovery trials, dynamics
randomization and hardware telemetry freshness/validity handling remain needed.
The Mars bridge currently exposes joint positions and optional velocities;
effort/current availability and hardware gain controls still need verification.

A deterministic box-contact probe starts collision-free and drives into a box.
With contact training enabled, the tracking guard suppresses forward motion and
reduces contact force from a 29.7 N peak to approximately 0.02 N by 3 seconds.
The same probe verifies the default collision stop and a lowered force-threshold
termination. These are regression tests of the controller safeguards, not
learned recovery performance. Results are in
`logs/mars-history-smoke/box-contact-probe.json`.

## Training runs and Weights & Biases

Install `.[rl,rl-tracking]` to enable W&B. Logging is optional and disabled by
default. Use `--wandb-mode online` with an existing W&B login or `WANDB_API_KEY`
in the process environment; use `--wandb-mode offline` without network access.
The trainer does not accept or write an API key in CLI arguments or manifests.
Project/entity/name are controlled by `--wandb-project`, `--wandb-entity` and
`--wandb-name`. Each seed gets a separate run. `wandb-run.json` records its link.

```bash
python -m emet.rl train --task base --envs 8 --seeds 0 --steps 262144 \
  --output logs/mars-base-new/ppo-stage0 \
  --eval-every 32768 --eval-episodes 20 --wandb-mode online
```

Training logs PPO losses, episode reward/success, final errors and collision/
obstruction rates. Every evaluation interval it exports weights and normalization,
runs deterministic validation on seeds 50000+, and logs success, accuracy,
settling time, measured acceleration/jerk, commanded acceleration and contact
force. A separate `best/` bundle is selected by success, collision rate, then
pose error; the root bundle is the last checkpoint. Evaluation metrics also
remain in local JSON/JSONL files when W&B is disabled. The best checkpoint is
logged as a W&B model artifact. Resuming restores normalization and cumulative
PPO timesteps and supports changing the number of workers.

A teacher-initialized alternative collects conventional-controller trajectories,
then queries that controller on states visited by a teacher/student mixture.
It trains the same small MLP, with desired-velocity labels and the common acceleration
adapter, and fine-tunes it using PPO. The final test seeds are never collected.

```bash
python -m emet.rl train --task base --architecture mlp --envs 8 --seeds 1 \
  --bootstrap-steps 16384 --bootstrap-rounds 3 --bootstrap-epochs 60 \
  --steps 65536 --output logs/mars-base-trained/bootstrap-ppo \
  --eval-every 16384 --eval-episodes 20 --wandb-mode online
```

`bootstrap-best/` retains the best supervised initialization separately from
PPO checkpoints, so regressions during RL fine tuning remain visible. The local
`demonstrations.npz` stores raw histories and teacher labels for reproduction;
bootstrap metrics log dataset size, loss and validation. Imitation and PPO
sample counts are reported separately. These are simulation experiments; success
on short free-space goals does not qualify obstacle handling or hardware use.

## Audited model (profile 4)

The current generated RL model is aligned to the pinned vendor URDF; older
profile-2/3 checkpoints are incompatible and remain historical experiment data.
See the [asset provenance and changes](../src/emet/assets/robot/innate_mars/NOTICE.md).
The independent audit compares every represented URDF link over 50 joint
configurations sampled across the full joint ranges, joint limits, each link's
mass/center of mass/inertia tensor, actuator effort caps, and collision coverage.
It also checks the pinned URDF/SRDF and all nine STL hashes against provenance. The
arm and head FK agree to numerical precision; source mass is 1.365 kg plus 4 mg
of planar-helper/visual bookkeeping mass in MuJoCo (0.000004 kg).

```bash
python -m emet.rl.audit --output /tmp/mars-model-audit.json
```

This does **not** verify hardware dynamics: the vendor URDF supplies no wheel
joints or tire/contact model, and drive gains, delay, payload response and effort
ratings require measured command/odometry data. The RL model remains a planar
velocity-actuated approximation. Geometry correctness and calibrated dynamics
are separate requirements; neither a successful policy nor an FK check establishes
physical compliance. The default legacy model is not modified by this audit.

At the pinned innate-os revision, `maurice_bringup/config/robot_config.yaml`
sets I²C and odometry updates to 30 Hz and driver limits to 0.4 m/s and 2.5 rad/s.
The ROS driver negates yaw before its firmware command; the simulation uses the
ROS yaw convention. The optional navigation `velocity_smoother.yaml` runs at
40 Hz, with forward/reverse limits 0.45/-0.2 m/s, yaw limits 0.6/-0.8 rad/s,
and deadbands 0.01 m/s and 0.02 rad/s. These are configuration evidence, not
measured motor dynamics, and the current 20 Hz RL simulator does not reproduce
that complete downstream pipeline. Hardware transfer needs timestamped velocity
commands and odometry to identify delay, deadband, saturation, stopping response,
and tracking residuals under obstruction, using the actual deployed command path.

Teacher initialization is a starting point only. PPO optimizes task reward without
an imitation penalty thereafter; it can improve time-to-goal, smoothness and
recovery beyond the teacher. Promotion requires independent measurements showing
that improvement. Local occupancy/voxel-map input, dynamic-map updates and
obstacle/replanning evaluation remain later tasks, not capabilities established
by free-space pose-goal success.

### Precision initialization and PPO fine tuning

The strongest current initialization uses the latest goal and ten samples of
motion/command history. A binary position-arrival feature helps the MLP represent
the teacher's switch from translation to final heading. It is an observation,
not an action restriction. Its threshold uses normalized distance, so these
bundles explicitly freeze observation normalization during PPO and resume.

The corrected-model dataset contains 131,072 teacher-labeled examples. To
reproduce the precision refit and conservative PPO experiment after collecting
that dataset:

```bash
python -m emet.rl.refit \
  --dataset logs/mars-base-trained/corrected-model/seed-2/demonstrations.npz \
  --reference-bundle logs/mars-base-trained/corrected-model/seed-2/bootstrap-best \
  --precision-feature --epochs 100 --eval-every 50 \
  --output logs/mars-base-trained/precision-refit --wandb-mode online
python -m emet.rl train --task base --architecture mlp --envs 8 --seeds 5 \
  --resume logs/mars-base-trained/precision-refit/best \
  --critic-steps 16384 --learning-rate 0.00003 --clip-range 0.1 \
  --steps 131072 --eval-every 16384 --eval-episodes 30 \
  --output logs/mars-base-trained/precision-warm-ppo --wandb-mode online
```

Critic warm-up fits discounted returns from deterministic policy rollouts while
leaving the actor unchanged. PPO subsequently updates both networks using task
reward. The initial checkpoint participates in validation selection so an RL
regression cannot silently replace a better initialization.

### Corrected-model results

The completed critic-warm PPO run performed 131,072 PPO transitions after
16,384 warm-up transitions, following the precision refit of 131,072 labeled
examples. Validation selected step 49,152; later PPO updates regressed. The
selected bundle is `logs/mars-base-trained/selected`, with selection provenance
in `selection.json`. Training metrics and its best artifact are in
[W&B run t30v2s0c](https://wandb.ai/cpaxton/emet-mars-control/runs/t30v2s0c).

The separately held-out 100 free-space goals (seeds 60000–60099) give:

| Metric | Teacher | Selected learned policy |
| --- | ---: | ---: |
| Success | 95/100 | 94/100 |
| Mean final position error, all episodes | 11.43 mm | 7.82 mm |
| Mean final yaw error, all episodes | 0.01535 rad | 0.01522 rad |
| Mean successful episode duration | 7.00 s | 7.00 s |
| Mean RMS linear jerk | 1.245 m/s³ | 1.211 m/s³ |
| Mean RMS angular jerk | 4.406 rad/s³ | 4.549 rad/s³ |
| Collisions | 0 | 0 |

Accuracy improved, but success and angular smoothness did not; this does not
establish superiority over the teacher. These are small local pose goals in
stage 0, not obstacle navigation, learned contact recovery, or hardware tests.
Full per-episode data and aggregates are in `selected-test.json`,
`corrected-baseline-test.json`, and `comparison.json` under the same log root.
The test set was evaluated after selection and was not used to choose weights.

```bash
python -m emet.rl evaluate --task base \
  --policy logs/mars-base-trained/selected --episodes 100 --seed 60000 \
  --output /tmp/mars-selected-test.json
emet serve mujoco --robot innate_mars --headless --no-cameras \
  --controller rl --policy-path logs/mars-base-trained/selected
```

## Recovery curriculum and matched stress tests

`--base-scenario mixed --horizon 15 --contact-training` mixes short free-space
moves, goal changes at 1.5–3 seconds, 300 ms external pushes, and approach/retreat
exercises with a fixed box. These scenarios randomize drive gains, damping and
0–100 ms of action delay. Scheduled events must occur before an episode can
succeed. Reset keeps scenario obstacles outside the start-sampling region until
the start has been chosen. The 30 N simulation force cutoff remains active.

The contact exercise changes the goal to the starting pose after four seconds;
the box remains in place. This tests unloading and retreat in response to a new
goal. It does **not** establish autonomous route planning around an obstacle or
an autonomous decision to retreat. The shared tracking-residual guard suppresses
continued commands into a sustained stall and permits reverse motion. Safety
stops can exceed the normal acceleration limit; command-jerk metrics include
those overrides.

The standalone comparison uses separate seeds and reports each scenario rather
than hiding contact failures in aggregate free-space success:

```bash
python -m emet.rl.robustness --episodes 20 --seed 90000 \
  --policy previous=logs/mars-base-trained/selected \
  --policy learned=logs/mars-base-robust/selected \
  --output logs/mars-base-robust/reproduced-test
```

The `contact_retreat` case uses the same teacher-controlled approach before
handing the retreat to each tested policy. Thus all policies encounter the same
initial contact. `approach_retreat` tests the complete learned approach as well;
its approach-distance metric exposes a policy that merely waits at the start.
Reports include force cutoff/invalid-state rates, final and peak contact force,
contact impulse, recovery time, position/yaw errors, acceleration and jerk.
Collision rate is expected to be nonzero in the deliberate-contact case; inspect
recovery and force metrics, not collision count alone.

A free-space-only normalizer has essentially zero variance in obstacle features.
Reusing it froze distinct distances at the observation clip limit. The optional
`emet.rl.refit --geometry-normalization` uses physical scales for distance and
validity features, while fitting the remaining feature statistics to the dataset.
The normalizer remains frozen with the exported model, including on PPO resume.
The balanced recovery dataset combines 32,768 recovery examples with 32,768
seeded samples of the previous free-space dataset; its provenance is saved in
`logs/mars-base-robust/balanced-input/provenance.json`.

The serving path still stops on collision and confirms measured stop. Permissive
light-contact training does not silently enable permissive runtime behavior.
These tests qualify simulator behavior only, not physical compliance or safety
of uncalibrated hardware. Arm learning and the real-robot velocity adapter remain
subsequent work. Arm coverage should use joint-space stratification, FK-derived
reachable targets, collision checks and explicit coverage near workspace/joint
limits after this base-control gate is met.

### Residual feedback experiment

The balanced MLP recovered contact behavior but regressed on held-out nominal
navigation (16/20 successes versus the teacher's 19/20). It is retained as an
experiment, not promoted. PPO fine tuning of that MLP also failed to improve its
validation success, so checkpoint selection retained the initialization.

`emet.rl.residual` instead initializes the actor to the exact conventional
feedback calculation. It supplies goal/history features, the feedback command,
and clearance-dependent command features to an unrestricted linear residual
actor. The initial command difference against the teacher was below 1.4e-6 over
12 rollouts. PPO can change every actor coefficient; there is no imitation loss
or residual-amplitude cap beyond the shared velocity/acceleration constraints.
This keeps a working feedback law at initialization and lets RL learn corrections.

```bash
python -m emet.rl.residual \
  --reference-bundle logs/mars-base-robust/balanced-refit/best \
  --output logs/mars-base-residual/initial
python -m emet.rl train --task base --architecture mlp \
  --resume logs/mars-base-residual/initial --base-scenario mixed \
  --contact-training --horizon 15 --terminal-accuracy-weight 100 \
  --critic-steps 8192 --learning-rate 0.0001 --clip-range 0.1 --target-kl 0.005 \
  --steps 65536 --envs 8 --seeds 11 --eval-every 16384 --eval-episodes 30 \
  --output logs/mars-base-residual/ppo --wandb-mode online
```

The additional training-only terminal penalty is
`weight * (position_error + 0.2 * yaw_error)` on successful episodes. It encourages
finishing accurately without changing evaluation tolerances or the runtime
control contract. `--target-kl` stops individual PPO update epochs when policy
changes become too large. Force/invalid-state aborts explicitly zero motion
commands; a terminated simulation episode alone is not evidence of a measured
hardware stop.

### Residual held-out result: not promoted

The residual run completed 65,536 PPO transitions plus 8,192 critic warm-up
transitions. Validation selected the final checkpoint, without using the test
set. The independent test used 20 episodes per scenario, with seed ranges
100000+, 101000+, 102000+, 103000+, and 104000+ respectively:

| Scenario/metric | Conventional teacher | Learned residual |
| --- | ---: | ---: |
| Nominal success | 20/20 | 18/20 |
| Nominal mean final position error | 11.45 mm | 8.74 mm |
| Nominal mean final yaw error, including failures | 0.01496 rad | 0.05143 rad |
| Retarget success | 20/20 | 20/20 |
| Retarget mean final position error | 10.76 mm | 8.00 mm |
| Push recovery success | 20/20 | 20/20 |
| Push mean final position error | 11.34 mm | 5.70 mm |
| Contact recovery after matched handoff | 16/17 | 17/17 |
| Full approach/retreat success | 15/20 | 15/20 |
| Full approach force-cutoff events | 4/20 | 5/20 |

Three of the twenty forced-approach episodes hit the force cutoff before handoff
for both controllers; they are excluded only from the explicitly conditioned
17-episode recovery metric. Full reports retain all twenty. Terminating at the
force cutoff is not evidence that physical forces never exceeded it. Final-force
metrics on an aborted episode describe the last solved step, not a measured stop.

The learned policy improves position accuracy and contact recovery, but nominal
heading/settling failures and the extra approach force cutoff fail the promotion
gate. It does **not** beat the teacher overall. Keep the conventional baseline
as the reference controller; the learned bundle in
`logs/mars-base-residual/selected` is experimental. No hardware or arm-controller
promotion was made. Two nominal failures (seeds 100000 and 100005) remained near
the position target with substantial yaw error/angular motion at the ten-second
deadline; their phase-transition behavior needs further work.

Artifacts: `logs/mars-base-residual/comparison.json`, `promotion.json`, per-episode
`test-*` reports, the checkpoint, and a source snapshot. The full held-out reports
and checkpoint are logged in
[W&B comparison m583k7jp](https://wandb.ai/cpaxton/emet-mars-control/runs/m583k7jp);
training is in [tdbzonf5](https://wandb.ai/cpaxton/emet-mars-control/runs/tdbzonf5).

## Arm curriculum and longer base training (2026-10-14)

The earlier ordinary-goal regression was primarily a settling failure, not a
large final translation error: two held-out episodes were within about 12 mm
but still had 0.26/0.51 rad heading error at the 10-second cutoff. Mixed training
and validation had allowed 15 seconds. Ordinary episodes inside the mixed
curriculum now truncate at 10 seconds; explicitly configured standalone horizons
remain available. Evaluation also reports success by 3, 5 and 10 seconds. Extra
training is an experiment, not evidence that the regression is fixed.

`train/evaluate --task reach --arm-only` holds the base command at zero and uses
five named arm joints through the existing seven-action contract. The sampler
proposes joint configurations across audited limits with an 0.08 rad margin,
rejects geometric contact, derives reachable EE targets by FK, and checks straight
joint interpolation at at most 0.04 rad increments. Half of target proposals are
local (within 0.25 rad per joint). Starts without an accepted connecting segment
are rejected and counted. This is rejection-biased joint-space coverage, not
uniform Cartesian sampling or continuous swept-volume certification. Loaded-arm
and contact-recovery curricula remain separate work.

The first arm run failed during sampling, before completing training; it is
retained at `logs/mars-arm/ppo/seed-12`. The corrected run is
`logs/mars-arm/connected-ppo/seed-12`, with 65,536 PPO steps. Its initializer is a
joint-feedback prior with a trainable linear actor over current state, obstacle
features, and five recent state tokens (250 ms); it is not a transformer.
`python -m emet.rl.arm --output PATH` reproduces that initializer. The exported
arm-only policy refuses base goals and zeros base actions. Checkpoint ranking now
includes joint error; end-effector position and orientation are reported too.
No base weights are transferred into the incompatible arm action layout.

Coverage is reproducible with:

```bash
PYTHONPATH=src .venv/bin/python -m emet.rl.coverage \
  --samples 500 --seed 80000 --output logs/mars-arm/coverage.json
```

The audit includes start and goal joint histograms, EE bounds/voxel occupancy,
rejected starts, and motion-size quantiles. All ten bins of each joint are
represented. Accepted motions are predominantly local: median maximum joint
travel is about 0.20 rad. This does not establish broad long-distance arm planning.

The additional base experiment resumes `logs/mars-base-residual/selected` for
131,072 PPO steps (196,608 cumulative), seed 13, mixed scenarios, 15-second
recovery horizon, matched 10-second ordinary horizon, learning rate 3e-5,
clip 0.08, target KL 0.005, and terminal accuracy weight 100. Artifacts live under
`logs/mars-base-extended/ppo/seed-13`. Both runs select checkpoints on validation
seeds 50000 onward; fresh arm tests use 130000 onward and base tests use 140000
plus scenario offsets. Test results do not select the checkpoint.

### Navigation and TAMP trajectory contract

The exact linked [TAMP API](https://github.com/cpaxton/home_robot_v3/blob/fix/native-tamp-live-validation/docs/api/tamp.md)
was retrieved on 2026-10-14. It is newer than this checkout's `docs/apis/tamp.md`:
the native adapter is explicitly stationary Stretch/privileged geometry, with
fresh measured snapshots, validated arm/payload paths, and sustained hold/release
verification. Mars is not made a supported native TAMP adapter by this work.

`python -m emet.rl.trajectories --arm --policy BUNDLE --output REPORT` tests five
arm interpolation waypoints using the repository's `interpolate_arm_waypoints`.
Omit `--arm` for `differential_drive_waypoints`: turn, bounded translation segments,
final yaw. Intermediate points advance from measured error and preserve control
history. Final success requires measured settling; collisions and tracking
timeouts are separate. The benchmark uses a 15-second whole-path deadline, not a
15-second allowance per waypoint, and reports each episode's completed waypoint,
tracking error and contact force. It does not execute public pick/place tools,
validate carried payloads, verify release, or certify hardware.

A future Mars native adapter must preserve atomic measured joint/timestamp
feedback, cancellation and unknown-command outcomes, scene/model binding, fresh
collision validation, and the API's partial-result semantics. Controller endpoint
accuracy cannot replace these requirements. Grasp/release verification stays with
the task executor; the policy only tracks the motion references it receives.

Arm held-out results (100 goals, seed 130000; separate 20 waypoint paths,
seed 120000): teacher and validation-selected PPO both achieved 100/100 goals
and 20/20 paths, with no detected collisions. Mean goal completion was 0.8145 s
for both; EE position error was 1.1601 mm teacher vs 1.1486 mm learned, and maximum
joint error averaged 0.002768 vs 0.002734 rad. These tiny changes do not establish
meaningful superiority. Mean waypoint-path time was 0.9300 vs 0.9325 s. No new
contact/payload safety claim is supported. The selected arm checkpoint is step
49152; the full 65536-step run is retained. Reports are in
`logs/mars-arm/{comparison,coverage,test-teacher,test-learned,trajectory-teacher,trajectory-learned}.json`.

Training dashboards:
- [Arm PPO](https://wandb.ai/cpaxton/emet-mars-control/runs/aw8i05ug)
- [Extended base PPO](https://wandb.ai/cpaxton/emet-mars-control/runs/a4zhj4o9)

Extended base held-out results (20 episodes per scenario, new seeds 140000 +
scenario offsets):

| Scenario | Teacher success | Extended PPO success | Teacher / PPO mean XY error |
| --- | ---: | ---: | ---: |
| Ordinary goals | 18/20 | 18/20 | 11.43 / 5.81 mm |
| Retarget | 20/20 | 19/20 | 11.39 / 6.88 mm |
| Push | 20/20 | 20/20 | 10.86 / 5.16 mm |
| Forced-contact retreat | 17/20 | 17/20 | 42.20 / 37.91 mm |
| Full approach/retreat | 12/20 | 12/20 | 94.76 / 91.63 mm |

The previous learned checkpoint achieved 17/20 ordinary goals on these same
fresh seeds, averaging 8.28 mm error. Additional training improved that result,
but successful ordinary completion barely changed versus teacher (6.54 vs
6.59 s). Force cutoffs remained 3/20 for forced approach and 8/20 for the full
approach/retreat condition. Retarget success regressed. Neither learned policy is
promoted as safer or more reliable than the teacher.

The selected extended checkpoint is cumulative step 196608. Training also had
large late critic-loss spikes (recorded in `training-diagnostics.json`); additional
steps did not monotonically improve validation. These results argue for a better
trajectory/settling objective and training-stability checks before simply scaling
training duration again. The reports retain unsuccessful episodes in error means.

Final navigation waypoint gate: teacher and selected base PPO both passed 20/20
paths without collisions. Mean whole-path duration improved from 9.385 to 9.085 s
under the same 15-second deadline. This is modest progress, not the desired fast
reactive navigation result. Full local reports and explicit non-promotion decisions
are in `logs/mars-base-extended/{comparison,promotion}.json` and
`logs/mars-arm/{comparison,promotion}.json`.

Validation: 205 tests passed across RL/core/navigation/TAMP API/Mars bridge suites;
the subsequently added mixed-deadline regression and focused arm/metric tests also
passed. Ruff, formatting, core-runtime synchronization, and diff whitespace checks
passed. Training runs are logged online. A separate upload of evaluation reports
and selected checkpoints was blocked by automatic approval review pending explicit
artifact-export approval; local evaluation is complete.

## Revised safety objective: gentle contact and wheel support

The user's safety target permits contact. Avoid harsh impacts, high contact
forces, abrupt acceleration/jerk, damaging sustained loads, and especially any
loss of wheel support. Wheel unloading can result from arm-floor contact as well
as a chassis/obstacle collision. Arm and navigation safety therefore cannot be
certified independently of whole-robot support dynamics.

In contact-enabled training, collision occurrence no longer carries a separate
reward penalty or prevents settled arrival. Force, impulse, smoothness, obstruction
and unsafe-force penalties remain. Unsafe force still stops the episode and zeros
commands. Checkpoint selection prioritizes unsafe-force/invalid-physics rates,
then task success, force/impulse severity and measured jerk before accuracy; raw
collision count is not a ranking criterion for contact-enabled runs. Strict
non-contact mode and runtime stop behavior remain available. Existing trained
checkpoints have not been retrained under this revised objective.

**Unsupported safety gate:** the present three-DOF planar base cannot rise,
roll, or pitch, and has no wheel/support dynamics. It cannot demonstrate resistance
to wheel lift caused by the arm pressing into the floor or by external collisions.
Evaluation reports now explicitly mark wheel-lift evaluation as unsupported.
All robot contacts, including arm-floor contacts, contribute to the experimental
force measurement, except the base's nominal support-plane contact. The existing
30 N threshold is not a hardware damage or stability threshold.

Before training/promotion against the wheel-lift requirement, the simulator needs
whole-body vertical/roll/pitch dynamics with physically modeled wheel/support
contacts and drive forces. Validate support geometry, mass/COM and payload effects,
then add separate tests for arm-floor levering, arm-obstacle loading, chassis
impacts, acceleration/braking, and combined loaded-arm/base motion. Record wheel
clearance and support-force unloading, body tilt/rates, per-contact peak forces
and impulses, joint loads, and measured motion jerk. Ground truth support loss
must be a failure independently of whether a force threshold is exceeded. Real
hardware inference from available IMU/odometry/joint tracking signals needs its
own validation; missing wheel-load sensors must not be replaced with fabricated
measurements or a claimed no-lift guarantee.

## Physical manipulation and execution-risk work in progress

`emet.rl.manipulation` introduces a separate versioned manipulation policy and
physical grasp curriculum. The Stretch adapter uses the archived contact model;
training resets randomize object placement and do not select the frozen TAMP
cases. Carry initialization closes the fingers and lifts through actuator-driven
simulation. There is no runtime object attachment. Actor observations contain
proprioceptive/reference history, while retention and contact outcome labels use
simulator ground truth. Training randomizes payload mass and payload-surface
friction without changing wheel or table traction. The disturbance scenario
applies a randomized, ramped 1–4 N and 0.03–0.20 N·m wrench to the end effector
for 0.30–0.60 s, with one fixed wrench per event; the object moves only through
simulated grasp contacts. Training ramps disturbance magnitude over its first
20 disturbance episodes, starting at 0.5–1 N and 0.01–0.04 N·m. Evaluation uses
the full force range. Full release remains an explicit executor operation.

The simulated position servo also has a slow integral tracking trim, capped at
0.04 rad for revolute joints and 0.04 m for linear joints, to remove steady load
sag while preserving the trajectory slew and acceleration limits. This trim is
a simulation calibration and is not evidence that Stretch hardware has matching
gains or force limits. Under it, the existing seed-0 carry checkpoint passed
all five fixed carry validation seeds; the disturbance cases still need targeted
training and larger held-out evaluation.

This implementation is experimental and not ready for hardware deployment. With
the bounded servo trim, the seed-0 checkpoint passed all five fixed carry seeds,
but passed only two of five randomized disturbance episodes; three of those
failures crossed the unchanged retention limit before the disturbance pulse began.
Frozen cases are archived with checksums, but their historical event traces lack
complete physics snapshots. The oracle harness below plans and physically
executes from those fixtures; it does not claim exact replay of the original live
CHAT timing. The Mars physical chassis now derives support geometry from the
pinned CAD mesh; mass distribution and contact/motor parameters remain estimates
that require calibration before hardware transfer.

The separate `risk.py` predictor estimates explicit outcomes for a particular
robot, model and controller: completion, retention loss, excessive contact and
wheel lift. It does not expose the PPO critic as an infeasibility oracle. An
expected return combines time, accuracy, smoothness and failures, so low return
does not establish that a trajectory is unexecutable.

Risk fitting requires independent episode IDs and rejects frozen regressions,
failed initializations and mixed controller/model identities. Episodes are split
60/20/20 for fitting, temperature calibration and independent audit. Each output
requires positive and negative examples in every partition and must pass the
audit calibration gates. Unsupported outputs, unfamiliar feature ranges and
controller changes yield unknown estimates. Even supported estimates are advisory,
not physical infeasibility or safety certificates.

The current feature prototype summarizes arm history, joint waypoints and total
duration. It has no map, object identity or per-waypoint timing input, and has not
been validated on real outcome data. It must not be treated as a general TAMP
trajectory checker. Integrating local geometry, payload/grasp context and timed
trajectory representations requires new data and a new feature contract. Known
joint, clearance and other planning constraints should continue to be checked
directly. A predicted execution failure can inform retiming, increased clearance
or a different grasp without declaring the overall task impossible.

### TAMP oracle execution and generated component cases

`scripts/eval_manipulation_tamp_oracle.py` loads the native branch's existing
privileged public-tool harness. It calls `scene_tasks`, `plan_pick_place`, and
`execute_pick_place_plan`; the planner and executor keep their original collision,
retention, freshness and placement checks. The command adapter runs either the
original direct position commands, bounded feedback, or a manipulation policy.
Only actuator commands move the live scene. The policy receives proprioceptive
history, not oracle object poses or contact forces. Safety monitoring records
physics-step force, wheel support and retention outcomes separately.

The harness requires `--native-root` because the native table adapter is not yet
present in this branch. Source hashes record the actual external implementation;
the checkout is not modified. Every report contains the actual oracle waypoints,
phase-start physics snapshots, commands and control traces. `left` and `right`
retain their frozen object positions, placement regions and 0.02 m / 0.1 rad
retention limits. `generated` perturbs the ordinary fixture's object placements
with explicit seeds before planning. Generated test cases and frozen regressions
are not used as training resets. A failure stops the task; no partial result is
counted as full success or automatically replayed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  scripts/eval_manipulation_tamp_oracle.py \
  --native-root /path/to/native-tamp-checkout \
  --controller feedback --cases familiar left right generated --repeats 3 \
  --output logs/manipulation/oracle-feedback

OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  -m emet.rl.manipulation.tamp_cases --episodes 10 --execute \
  --output logs/manipulation/tamp-arm-execution.json
```

The second command uses this branch's real `PhysicalPickPlaceExecutor` planner
and executor on generated reaching/carrying configurations. Planning uses a
private physics copy. Execution uses a regular-cadence actuator controller and
requires an additional one-second settled hold. These are arm component tests,
not full pick-and-place success. The initial 20-case feedback report produced
9/10 valid reaching paths and 10/10 valid carry paths. All nine valid reaching
paths executed; only 3/10 carry paths passed execution and retention/settling.
The rejected reaching goal remains visible. Reports are local at
`logs/manipulation/tamp-{generated-arm-cases,arm-execution-teacher}.json`.

Oracle testing caught two command-adapter defects: blocking gripper opening must
allow travel time at the bounded servo speed, and an explicit release must retain
the open reference through the next object's approach. Both are corrected and
covered by regression tests. A separate physical mechanism test pushes an arm
against the floor and checks that the support monitor detects the resulting wheel
lift; this tests the detector, not the accuracy of either robot model.

With these adapter fixes, the bounded feedback controller completes both frozen
two-object tasks in 3/3 repeats each, retaining the original regions and thresholds
(`logs/manipulation/oracle-feedback-v5`). The familiar control and three generated
cases still fail release-clearance validation, so the overall oracle gate does
not pass. The direct position baseline fails all 12 cases under the added loading
and wheel-support monitoring (`oracle-direct-v4`). These results do not establish
that a learned policy is better: the first diagnostic learned checkpoint passed
the right case once but lost retention on the left (`oracle-first-policy`).

### Active manipulation training and follow-up audit

The initial `physical-v1` experiment was stopped after finding that running
contact/retention costs were charged per tick rather than integrated over time.
Two ordinary eight-second teacher timeouts returned about -50 and -46, worse
than the -20 terminal failure penalty. The corrected objective integrates running
costs with `dt`; regression tests require ordinary safe holding to be preferable
to early failure and invariant to sampling cadence. The old checkpoint remains
diagnostic and is not promoted.

The corrected three-seed Stretch experiment is running under
`logs/manipulation/physical-v2`: seeds 0/1/2, each with 32,768 feedback-initialization
steps and 262,144 PPO steps. Per-seed `run.json` records settings and source hashes;
`metrics.jsonl` records training and stratified validation. W&B receives scalar
metrics only. No frozen reports or checkpoint artifacts are uploaded. Training
completion alone never sets `safety_promoted`.

`emet.rl.manipulation.post_training` is queued locally at
`logs/manipulation/physical-v2-final-audit`. Its `status.json` distinguishes waiting,
evaluation, completion and errors. After all seeds complete, it selects a checkpoint
using validation only, then runs paired fresh 100-episode-per-category teacher and
policy evaluation, fresh TAMP arm component tests and three repeats of each oracle
case. It records the unchanged safety and performance gates in `acceptance.json`.
It separately collects fixed-controller risk outcomes, fits/calibrates/audits the
outcome predictor, and leaves unsupported outputs unknown. Even passing simulator
gates does not promote hardware use. The full training and audit are not yet
complete; consult those status files before claiming learned improvements.

### Local 3D policy test drive

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m emet.rl.test_drive --port 8765
```

Open `http://localhost:8765`. The UI streams actual MuJoCo renders and supports
camera orbit/zoom, pause/run/single-step, seeded resets, goal changes and bounded
0.25-second disturbances. Choose Stretch physical manipulation, Mars physical
wheels + arm, or the existing Mars planar base/coordinated simulation. Controller selection loads a validated local checkpoint
or the feedback baseline; refresh discovers newly exported checkpoints without
changing a running controller automatically. A live panel separates initialization
from PPO training steps and links the W&B run.

The default checkpoint roots are `logs/manipulation/physical-v2` and
`logs/mars-base-extended`; repeat `--policy-root` to choose other local roots.
The server binds only to loopback. If the browser is on another machine, forward
port 8765 over SSH. It has no robot client, connection-profile lookup, hardware
endpoint, or actuator transport. The same-seed reset permits manual comparisons;
interactive interventions do not count as acceptance evidence. Mars wheel support
is measured in `mars-arm` and remains unsupported in the planar modes. Episode limits in this interactive tool are
30 seconds, separate from the training/evaluation budgets.

The UI was checked with a real headless Chromium screenshot and live HTTP controls,
including loading and stepping the selected Mars base policy. Tests cover rejection
of hardware modes, reference changes without teleportation, checkpoint discovery,
and partially written training progress records.

### Herman read-only inspection

Herman (`192.168.1.43`) was reachable over SSH during this session. The bridge was
not running, its observation/state ports were closed, and the ROS graph contained
only `/rosout` and `/parameter_events`. No drive/arm/gripper/mode commands or service
starts were issued. The onboard URDF was copied to `logs/herman-readonly` for local
comparison. It adds fixed `base_footprint` and `base_laser` frames to the pinned
description, but still provides no separate wheel/caster links. The physical
simulator now derives those supports from the matching source CAD mesh. This
read-only inspection does not calibrate that model or certify hardware readiness.

### Mars tracking and carry audit (2026-10-17)

The test-drive starts running on startup/reset and resumes when a target is edited.
Pause remains available. The overlay shows simulated seconds per wall second,
and the configured physical base caps (0.12 m/s, 0.5 rad/s). These caps are
conservative simulation settings, not verified Mars hardware speed ratings.
Lateral goals require differential-drive turning, translation and alignment;
playback speed and controller settling time are distinct measurements.

Run the paired, seeded tracking audit with:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m emet.rl.manipulation.tracking_audit \
  --episodes 10 --output logs/manipulation/mars-corrected-audit/tracking.json
```

Add `--policy CHECKPOINT` to evaluate a policy on the identical teacher seeds.
Cases cover FK-reachable nearby poses, randomized initial chassis headings,
base/hand/combined masks, a mid-motion retarget and a finite 0.5 N hand push.
Reports include position/orientation error, settling time, measured jerk, contact
loads, support loss and two-second post-success drift. FK reachability does not
certify a collision-free dynamic trajectory. Initialization failures and rejected
cases must remain in the results.

Mars grasp resets now use the rectangular jaws once (the duplicated union-box
proxy is removed), positive-angle opening, bounded experimental gripper torque,
a narrow support pedestal and 16 mm wide box/cylinder payloads. The old 30–40 mm
payloads exceeded usable clearance in this reset. Ground-truth Jacobian lift
construction is confined to simulator reset assistance; scored motion uses real
contacts without an object weld. Lifts preserve hand orientation and use small
vertical waypoints. MuJoCo uses pyramidal contacts with five no-slip iterations;
wheel contacts use finite-width cylinders without additional point torsion.
These settings, gripper gains and the 0.08 Nm torque cap remain uncalibrated
simulator estimates. Frozen Stretch fixtures and 20 mm/0.1 rad retention gates
are unchanged. Passing initialization is not scored carry success.

Training accepts `--scenario`, `--horizon`, `--target-scale`,
`--randomization-scale` and `--initial-heading-range`. Initial defaults use a
20-second horizon, quarter-range targets and 5% randomization. Mass, object
friction, observation delay and hand disturbances respect the randomization
scale. After at least 20 validation episodes per category, progression requires
90% success in every category, zero initialization failures and zero observed
unsafe contact, wheel lift or retention loss. Progression increases target range
and randomization by 0.1, capped at one. It is a simulator curriculum gate, not a
statistical hardware safety certificate.

Checkpoint selection and paired teacher comparison always use full target range
and full randomization on fixed seeds. Separate curriculum validation controls
progression, preventing easy-stage success from being reported as full-contract
performance. W&B receives scalar metrics and paired success deltas; detailed
case traces remain local. Mars carry ranges now match generated TAMP component
motions rather than applying Stretch-sized jumps to all revolute joints.

The final development audit passed 50/50 cases. A separate fresh audit at seed
1010000, with ten cases per category, also passed 50/50 without observed wheel
lift or unsafe contact. Detailed local reports are
`logs/manipulation/mars-corrected-audit/tracking-final.json` and
`tracking-heldout.json`. These are teacher results, not learned-policy results.
Three generated TAMP reaches and three generated TAMP carries executed through
the controller successfully (`tamp-oracle.json`); maximum carry retention error
was 1.12 mm and 0.063 rad. This is arm-planner component coverage, not public
end-to-end native Mars TAMP support.

At the initial low-randomization carry curriculum, 20/20 resets succeeded but
only 13/20 scored transports succeeded. Seven crossed the unchanged rotation
retention gate. No unsafe contact or wheel unloading was observed in those
trials. Carry remains below its progression gate. The reaching-only run is
`logs/manipulation/mars-corrected-v1/reach/seed-17`, with four environments,
8192 critic-initialization steps and 262144 PPO steps requested. The UI discovers
this run under its new default policy root. Training includes a 25% chance of a
seeded absolute retarget and a 25% chance of a finite ramped hand push. These
inputs are frozen at reset for paired replay. Reach pushes have a 0.5 N full-scale
cap, multiplied by randomization difficulty. They are separate from the UI's
1 N stress push and the larger loaded disturbance curriculum.

The first reaching checkpoint (8192 critic warm-up steps, zero learned residual)
matched the teacher on fixed full-distribution validation: 18/20 successes,
including two unsafe-contact terminations. Its separate initial curriculum
validation passed 20/20 with zero unsafe contact or wheel unloading, allowing
one curriculum increment. Full-range safety remains an unmet gate; the small
nearby-pose audit must not be substituted for it. See live scalar metrics at
https://wandb.ai/cpaxton/emet-mars-control/runs/3zlg1l58.

Completed physical free-space reaches can accept another target without a
physics reset. Startup/reset and target edits run automatically. Safety faults
still require an explicit reset; changing a target does not clear them.

### Dynamic reaching references and admission (2026-10-18)

The new reaching training defaults use `certify_reach_paths=True` and
`dynamic_references=True`. Reset-time geometry checks reuse the actual TAMP
scene checker, with its unchanged penetration tolerance. They check sampled
quintic arm references, endpoints, and, for coordinated Mars targets, a
conservative sweep over joint 1's compensation range. A compensation rejection
means this nominal execution strategy is unsafe; it does not prove the hand
pose is universally unreachable. The check uses scratch state and never moves
the scored robot. Rejected candidates, contact evidence, masks and reference
coefficients remain in local evaluation reports. Sampling exhaustion is an
initialization failure, never a successful rejection episode.

Timing varies across cases, including short initial pauses. Mid-motion goal
changes use quintic boundary conditions that preserve reference position,
velocity and acceleration. Reference speed, acceleration and sampled jerk
bounds are checked before admission. These are reference bounds; measured
robot jerk and force remain rollout metrics. The policy still executes closed
loop and can lag, yield or recover within its residual/actuator limits; the
reference generator does not itself implement a learned phase/timing action.
It currently produces arm references alongside base pose goals, not a dynamically
certified mobile-base trajectory. Physics execution remains a separate gate.

A new, untouched 30-case batch with 50% retarget probability and 50% ramped-push
probability passed 30/30 after sweep screening, with one candidate rejected.
No unsafe-contact or wheel-lift termination occurred. Mean completion time was
6.44 seconds; this is a corrected-teacher baseline, not learned improvement.
An earlier fresh batch passed 29/30 and exposed the compensation collision;
its failure remains recorded in `teacher-heldout.json`. The later batch is
`logs/manipulation/mars-dynamic-v2-audit/teacher-heldout-sweep.json`.
Benign contact still occurs (maximum observed contact-force bound 18.9 N);
these results do not establish hardware safety.

Reproduce and inspect reference curves:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m emet.rl.manipulation evaluate --robot innate_mars \
  --scenario reach --horizon 20 --episodes 30 --seed 500000 \
  --retarget-probability .5 --reach-push-probability .5 \
  --output /tmp/mars-dynamic-report.json
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m emet.rl.manipulation.dynamic_trajectory \
  --report /tmp/mars-dynamic-report.json --output /tmp/mars-references.png
```

`--no-certify-reach-paths --no-dynamic-references` retains the unchecked static
reference mode for explicit legacy comparisons. Evaluate reaching explicitly;
use `--scenario mixed` to evaluate all manipulation categories. Dynamic mode
always includes geometric screening. The visualization shows nominal joint
references and their derivatives, not measured robot motion.

Checkpoint selection now requires matched episode identities and identical
initial conditions/reference schedules. A faster candidate is ineligible if
it loses any teacher success, has a safety/retention failure, or worsens measured
pose accuracy, force/impulse, or jerk metrics, including maximum force and jerk.
Time comparisons use common successful cases so early failures cannot appear
as a speed improvement. This selection gate is separate from hardware promotion.
W&B logs the paired eligibility and timing results. The new reaching run is
`logs/manipulation/mars-dynamic-v2/reach/seed-18`; carry remains separately gated.

The run's first 8192-step critic-only checkpoint passed 20/20 full-range
admitted references, matched the teacher exactly, and passed the paired
no-regression gate. Three candidates were rejected and logged during that
validation. This zero-residual checkpoint establishes the new baseline; no
learned speed gain has been established. Live run:
https://wandb.ai/cpaxton/emet-mars-control/runs/jizxb9wp.

### Direct velocity policy and continuous task benchmark (2026-10-18)

New training defaults to `--action-mode direct_velocity`. The actor produces the
complete bounded base and arm velocity command, without adding a teacher pose
servo at inference. Joint target integration, speed/acceleration limits, measured
tracking trim, contact stops, wheel support checks, and executor-only release
remain in the actuator layer. Existing bundles without an action-mode declaration
retain residual semantics. Bundle contracts explicitly distinguish the two modes;
the evaluator, test drive, TAMP component adapter, and oracle select those semantics.

The teacher is an initialization source and comparison baseline. Direct training
now uses DAgger: five collection/fitting rounds, teacher intervention probabilities
1, .75, .5, .25, 0, full-sized reachable targets, and labels at learner-visited
states. PPO then optimizes independently. Raw histories are stored during
initialization and renormalized consistently for each fit. Motion labels receive
extra weight so masked/holding zeros cannot dominate the imitation loss. A first
teacher-only imitation ablation obtained low fitting error but 0/20 initial
full-distribution success; that checkpoint was rejected. More identical PPO was
not treated as the solution to that distribution failure.

The objective includes an integrated active world-pose tracking cost, negative
progress on overshoot, and a motion penalty within the goal region, in addition
to elapsed time, measured jerk, force, and retention costs. Base XY error inside
an explicitly configured task tolerance is not penalized. New training uses a
50 mm base region; explicit contracts may tighten it. Hand tolerances cannot be
loosened beyond the existing 15 mm / 0.1 rad training contract. Redundant joint
solution error remains diagnostic rather than defining world-pose success.
Masked targets do not contribute hand/base error statistics. Paired selection
compares base constraint violation rather than rewarding additional precision
inside the permitted region, while preserving hand/load/jerk no-regression gates.
Base settling requires <5 mm/s linear and <0.02 rad/s yaw velocity, not merely
remaining somewhere inside the base region.

`emet.rl.manipulation.integrated` is a continuous physical task benchmark, not a
concatenation of reset-assisted carry cases. It begins with an open hand above a
supported object, physically approaches/closes/lifts, executes an out-and-back
transport with the ground-truth TAMP arm planner, descends to loaded support,
releases, retreats, and requires two seconds of supported object settling.
There are no object attachments, pose edits, or physics resets between stages.
Only a successfully completed reach/grasp contract is re-armed; faults are not
cleared. The oracle executor uses solved support load for expected-contact
handoff, not a ground-truth contact signal passed into the learned actor.

The fixed task contract uses a 45 mm lift, 40 mm support half width, 5 mm hand
position and 0.05 rad orientation tolerance, 20 mm final object XY tolerance,
and unchanged 20 mm / 0.1 rad retention limits. The base hold is 5 mm for this
stationary manipulation fixture. Expected placement contact ends descent;
opening is allowed only after at least half the object's weight is supported.
The support remains present from initialization. A slow, bounded probe is limited
to 5 mm, and the fingers retreat 60 mm after opening. The grasp pose, fixture
size, and lift height are recorded settings, not frozen Stretch fixture edits.
The lift fixes its world orientation and XY reference rather than accumulating
measured sag at each incremental target. Gripper release uses the gentler slew
rate. Footprint checks use the actual support dimensions.

Reproduce the successful box fixture, with the velocity teacher driving the
same direct actuator interface that a learned actor will use:

```bash
MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0-7,12-31 .venv/bin/python -m emet.rl.manipulation.integrated \
  --seed 1800006 --episodes 1 --action-mode direct_velocity \
  --output /tmp/mars-pick-place.json
```

Pass `--policy CHECKPOINT` to run the actual learned controller on this task.
Reaching success alone does not establish manipulation success. The integrated
fixture passes with the teacher under both action semantics; additional box
seeds have exposed retention failures, so it is not yet a robust manipulation
policy. These are simulation task tests with privileged planning/support handoff,
not full public TAMP-tool execution or hardware acceptance.

Cylinder placement remains blocked by a reproducible support-contact problem:
an isolated initially tilted cylinder, with no robot contact or commanded forces,
does not settle under the current model. Over the last two seconds of a ten-second
audit it reached roughly 0.052 m/s and 1.70 rad/s, whereas the box settled to
numerical zero. Do not waive the object velocity criterion or train a policy to
compensate for this simulator artifact. Primitive/prismatic collision shapes,
cone/no-slip settings, contact dimensions, and time-step ablations did not establish
a reliable replacement model; the canonical contact model was left unchanged.
Reproduce the isolated failure with:

```bash
MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0-7,12-31 .venv/bin/python -m emet.rl.manipulation.support_audit \
  --output /tmp/mars-support-audit.json
```

The direct DAgger/PPO experiment is logged at
https://wandb.ai/cpaxton/emet-mars-control/runs/i5cc7n9g under
`logs/manipulation/direct-v4-dagger/reach/seed-20`. It uses 32,768 requested
initialization samples and 262,144 PPO steps. It is reaching-only training, not
an integrated manipulation promotion. Run manifests retain source hashes;
checkpoints must be re-evaluated against the current runtime and integrated gate
before adoption. Existing selected policies and the running simulator UI are
not automatically replaced by this experiment. No hardware commands were sent.

### Simpler tracking baseline and optional DAgger (2026-10-19)

Direct-control training no longer requires imitation. The default is a plain MLP
with four recent observation tokens flattened into its input, trained by PPO
without initialization demonstrations. `--dagger` explicitly enables the existing
aggregation initializer, and `--architecture history` selects the transformer.
Scratch direct PPO defaults to normalized action standard deviation 0.3;
initialized/residual training defaults to 0.03. `--exploration-std` overrides it.
`run.json` and W&B configuration record initialization, architecture, and noise.
Existing policy bundles keep their saved architecture and action semantics.
`--initialization-steps` is unused for direct scratch training; it controls DAgger
samples when enabled, or critic warmup for the legacy residual mode.

This is an experimental baseline option, not a demonstrated robot-policy
improvement. UMI on Legs §3.2 uses an MLP with task-frame trajectory preview and a
coupled exponential position/orientation tracking reward with a precision
curriculum. Its §7.1 reports no benefit from privileged distillation and notes
history timing problems. Reference: https://arxiv.org/html/2407.10353v1

Our current history tokens still lack an explicit future trajectory preview, and
the reward is still the v3 progress/error/physical-cost objective. These must not
be described as a reproduction of UMI on Legs. The intended next controlled
experiment is a preview-conditioned tracking PPO baseline, compared on the same
references with optional DAgger and transformer variants. TAMP owns task sequencing
and grasp/placement selection; the controller owns accurate, stable execution.
Integrated physical pick/place and retention remain acceptance gates. The paper's
legged jumping/tossing behaviors and reward coefficients are not safety guidance
for Mars or Stretch; wheel unloading remains a failure in our experiments.

### Preview-conditioned tracking baseline implemented (2026-10-19)

The primary baseline now uses direct bounded velocity actions, a feed-forward MLP,
and PPO from scratch. DAgger and the transformer remain explicit options. Unlike
the preceding baseline-options change, explicit future commands and the coupled
tracking objective are now implemented. This is inspired by UMI on Legs §3.2,
not a reproduction of its legged robot, actuator setup, control frequency, or
reported performance.

**Inputs and reference frames.** `history` retains its existing tensor shape for
bundle compatibility. In the new pose-only contract, IK joint-reference slots and
inactive pose targets are zeroed. A separate `(7, 18)` `preview` tensor contains
reference offsets `[-.05, 0, .05, .1, .2, .5, 1]` seconds, masked base error in the
current base frame, hand position/rotation error in the current hand frame, masks,
and the gripper reference error. Desired poses themselves remain fixed in the task
frame; they do not move with chassis translation, roll, or pitch. Reference FK uses
private anchored scratch state and kinematics only. No contact/payload ground truth
or future scheduled pushes enter the actor. Reset-frozen unexpected retargets
remain hidden until the retarget is announced. Bundle manifests record and validate the
preview contract and store normalization for every observation key. Legacy bundles
retain their old input/action/mask semantics.

A hand-only goal no longer disables base actuators. Reach generation samples a
latent base configuration as well as arm joints for such goals, yielding reachable
whole-body hand targets instead of only the fixed-base arm's pose manifold. The
latent base solution and joint solution are not actor inputs. Explicit base masks
remain task constraints. The runtime `TaskPoseTrajectory` adapter accepts actual
timed SE(2)/SE(3) commands from a planner without an execution-time IK loop; see
`docs/api/tamp.md` for its prototype scope and gripper authorization contract.

**Objective and curriculum.** The primary hand reward is
`exp(-position_error / sigma_position - orientation_error / sigma_orientation)`,
weighted by four. Active base constraints add a separate score, with no XY cost
inside their declared region. Physical running costs retain measured jerk, contact
force, grip retention, and action regularization. There is additional action-change
regularization and normalized motion cost near a stationary final target. The
position/orientation precision stages are `(0.15 m, .8 rad)`, `(.06, .35)`,
`(.025, .15)`, `(.01, .08)`, and `(.005, .05)`. Reward precision tightens only after
at least twenty validation cases meet the current error envelope with no
initialization, force, wheel, or retention failures. Target coverage and disturbance
randomization retain their separate success-gated curriculum. Final hand acceptance
for new training is 5 mm / .05 rad; reward curricula do not relax that contract.

Reach episodes continue through the horizon even after attaining the goal, with
one success bonus per goal. This both tests post-arrival drift and avoids an
incentive to delay completion to keep collecting positive tracking rewards.
Reports distinguish elapsed rollout time from first-settling time, retain peak/RMS
hand error, and record post-settling base drift. Active unconstrained base motion
must settle too. Policy selection still requires paired task/safety checks; no
checkpoint is automatically qualified for hardware. An MLP PPO smoke run exercised
collection, optimization, checkpoint reload, and evaluation without demonstrations.
The reference and actuator cadence remains the profile's 20 Hz; a 50 Hz actuator
comparison is not claimed by this baseline.

**Validation artifacts.** `logs/manipulation/preview-v5-audit/tamp-cases.json`
contains four fresh GT TAMP component paths (two reach, two carry), all physically
executed successfully by the teacher adapter, plus their retimed pose schedules.
`task-references.png` visualizes their positions and endpoint orientation frames;
these are command references, not learned rollouts. The separate integrated box
pick/transport/place regression remains at 5 mm hand accuracy with physical grip,
loaded support, release, retreat, and settling. Existing cylinder settling failures
remain unresolved and must not be counted as successful manipulation.

The fixed-frame twenty-second hold audit exposed deficiencies in the manual
teacher that short terminal tests hid: 10/20 passed in the first audit, including
one wheel-unloading failure. This audit preceded the expanded hand-only base
sampling and used 15 mm hand acceptance, so it is diagnostic, not the acceptance
rate for the final distribution. Base region tolerance must not be interpreted as
permission for the hand target to drift with the chassis. Teacher failures do not
prove the trajectory infeasible, and teacher actions are not a constraint on PPO.

**Training.** The Mars baseline is running under
`logs/manipulation/preview-v5-scratch/reach/seed-23` with 2,097,152 requested PPO
steps, eight simulator workers, no imitation initialization, and twenty paired
validation cases every 131,072 steps:
https://wandb.ai/cpaxton/emet-mars-control/runs/qa0dv3ha

```bash
MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0-7,12-31 .venv/bin/python -m emet.rl.manipulation train \
  --robot innate_mars --scenario reach --horizon 20 --architecture mlp \
  --steps 2097152 --initialization-steps 0 --envs 8 --seeds 23 \
  --retarget-probability .5 --reach-push-probability .5 \
  --eval-every 131072 --eval-episodes 20 --wandb-mode online \
  --output logs/manipulation/preview-v5-scratch/reach
```

A separate local acceptance worker waits for completion, then evaluates twenty
fresh full-distribution cases, paired teacher results, four TAMP component paths,
and four continuous pick/place cases including the known cylinder failure. Its
status/reports are written to the run's `acceptance/` directory. It never promotes
hardware; failed gates remain failed. The worker has a 24-hour wait timeout and
records evaluation errors explicitly. The previous selected UI policy is unchanged.
Reaching-only training does not establish grasp/carry competence. Optional DAgger
must be evaluated as an ablation on this same distribution; the historical v4 run
is not an isolated DAgger comparison because its inputs/objective differed.

### Completed v5 run and next experiment (2026-10-21)

The 2,097,152-step v5 run completed and failed simulation acceptance. Fresh paired
reaches: policy 0/20, teacher 5/20; policy wheel unloading in 1/20. All four
continuous pick/place cases timed out before the grasp completed. This is not a
usable manipulation policy. The teacher's failures do not establish infeasibility.

The final curriculum validation reached position-error p95 7.6 mm and orientation
p95 0.093 rad, with zero strict successes. Target scale remained 0.25 for the entire
run because increasing coverage required 90% final-contract success. Recomputed
endpoint diagnostics on fresh cases found 10/20 within hand pose tolerance, but
6/20 within hand tolerance while the base was still moving. Those endpoints are
not successes; the one-second continuous settling requirement remains unchanged.
Local diagnosis: `logs/manipulation/preview-v6-audit/baseline-diagnosis.json`.

The next experiment changes two training mechanisms:

- Reach target coverage may increase by 0.1 after twenty current-distribution cases
  have p95 hand error at most 25 mm / 0.15 rad and base-region violation at most
  25 mm, with no initialization, contact-safety, retention, or wheel failures.
  Dynamics randomization still requires 90% strict success. This coarse gate is
  curriculum scheduling only, never policy acceptance or a hardware safety claim.
- At the end of a timed reference, a continuous accurate-and-still reward augments
  pose tracking. Its pose scale is fixed at 5 mm / 0.05 rad; measured joint and
  base velocities reduce it continuously using the existing settling velocity
  scales (0.03 rad/s joints, 0.005 m/s base, 0.02 rad/s yaw). It has no discontinuous
  threshold or premature-stop bonus for inaccurate poses. Existing jerk/contact
  costs, physical limits, and all acceptance contracts remain intact.

The new reward identifier is `coupled_task_pose_tracking_and_hold_v5`. Observation
and action schemas are unchanged. The comparison run uses the same seed, number
of steps, exploration settings, retargets, disturbances, and paired validation
cases as v5, under `logs/manipulation/preview-v6-scratch/reach/seed-23`.
This tests the combined curriculum/hold change; it does not isolate each change.

The v6 comparison run is active on W&B:
https://wandb.ai/cpaxton/emet-mars-control/runs/7mbs98vy
A matching acceptance worker writes to the run's `acceptance/` directory after
completion, using the same fresh seeds and unchanged contracts as v5. A 256-step
single-worker Mars smoke completed collection, PPO, export/reload and evaluation;
that smoke establishes pipeline operation only, not control quality.

Verification for these changes: the full RL suite passed (179 tests), followed by
six curriculum tests including the additional endpoint-diagnostic regression.
Ruff and `git diff --check` passed. None of these software checks substitutes for
the pending learned-policy acceptance evaluations.

### Fixed-case learning audit (2026-10-22)

The v6 run completed without an accepted checkpoint: fresh reach success 0/20,
wheel unloading 1/20, position p95 55.8 mm and orientation p95 0.310 rad. Expanding
coverage did not resolve the learning failure. Further broad runs are deferred
until a fixed-case diagnostic establishes that the production training objective
and action interface can learn and hold a reachable reference.

`train --debug-reset-seed SEED` now repeats one seeded reset in the training
simulator only (requires reach and one worker). Checkpoints write
`debug-training-case.json` separately; ordinary held-out evaluation, paired
comparison and acceptance never use the wrapper. The reset seed is recorded in
`run.json` training options. This is an overfitting diagnostic, not a coverage or
policy-quality result.

Two 65,536-step diagnostic runs use the same robot, seed 26, fixed reset 250000,
10-second horizon, target scale .25, no randomization/pushes/retargeting and no
DAgger. Only reward precision differs: stage 4 under
`logs/manipulation/fixed-case-audit/seed-26` versus stage 2 under
`logs/manipulation/fixed-case-coarse-audit/seed-26`. Each evaluates the training
case every 8192 steps. Validation still uses the strict 5 mm / .05 rad endpoint
contract. With one evaluation episode the curriculum does not advance. The case
has both hand and base targets active and initially has 39.8 mm / .091 rad hand
error; a stationary actor fails as expected. The existing joint-reference teacher
also fails it (30.4 mm hand error), so teacher imitation is not an acceptance
shortcut for this case. These local diagnostics do not upload new W&B runs.

### Reporting contracts and horizon audit (2026-10-22)

Evaluation summaries now persist active hand/base case counts, overlapping
endpoint position/orientation/base-region/heading/velocity failure counts and
rates, plus terminal timeout/contact/wheel/retention counts. Training's existing
summary logger sends these scalar metrics to W&B on subsequent runs. Hand-only
statistics exclude cases with inactive hand targets; an inactive hand is not an
accurate hand. Missing historical measurements remain unavailable rather than
being counted as passing. New episodes record wrapped base heading error and
maximum measured joint speed to diagnose additional settling failures.

Diagnostic hand endpoint bands are 5 mm/.05 rad, 10 mm/.1 rad, and 20 mm/.1 rad.
These are pose-only diagnostics, not task success: they do not demonstrate
continuous settling, collision safety, or grasp retention. Strict contract success
is reported separately. Frozen TAMP regression thresholds remain unchanged.

The normal base XY region is a 50 mm radius disk around the target base frame's
XY position, not independent +/-50 mm bounds in X and Y. Heading tolerance is
separate (0.04 rad when active). Some manipulation fixtures explicitly request a
5 mm base hold region. A masked base target permits repositioning to reach an
active hand target, but settling still requires the base to stop. Current limits
are <5 mm/s translation and <.02 rad/s yaw, arm speed <.03 native joint units/s (rad/s for revolute joints, m/s for
prismatic joints), and one second
continuously within final reach tolerances. Intermediate trajectory samples do
not each require a stop; settling begins only after the timed reference ends.

An executor can submit an entire `TaskPoseTrajectory`. The actor observes a
rolling preview out to one second, plus state history, rather than the full
variable-length trajectory. This preserves reactive replanning without requiring
IK targets in actor inputs. Unannounced future goal changes remain hidden.

`logs/manipulation/horizon-audit-v6/` compares the v6 final checkpoint on the same
20 fresh seeds at 20 and 40 seconds with all tolerances unchanged. The comparison
explicitly verifies identical initial actor states and announced references.
This is a diagnostic for slow convergence versus drift; extra time is not a
substitute for a stable hold and does not extend hardware commands.

Completed horizon audit: initial states/references matched exactly. Success stayed
0/20 at both 20 and 40 seconds; endpoint base-moving count increased from 13/20
to 15/20. Both horizons had one wheel-unloading stop and one unsafe-contact stop.
Hand-only denominators were 14 active cases: 2/14 within 5 mm/.05 rad, 6/14 within
10 mm/.1 rad, 8/14 within 20 mm/.1 rad, unchanged by extra time. These results
support investigating stabilization rather than increasing the default horizon.
Hold drift is now reported as unavailable when an episode never achieved a hold,
rather than interpreting its uninitialized zero as successful low drift.

### Endpoint motion ramp and conservative execution (2026-10-22)

`--endpoint-motion-weight` enables an optional post-reference velocity penalty;
it defaults to zero to preserve the baseline. It ramps linearly from zero to
full weight over two seconds after the scheduled reference ends. Installing a
new timed trajectory resets its reference clock. It uses measured base/joint
velocities at settling scales and saturates each base/arm group as x/(1+x), so
the maximum cost density is twice the weight. Inactive arm targets omit the arm
term. The cost is integrated with dt and does not change actuator commands,
pose tolerances, success or safety termination. The new reward identifier is
`coupled_task_pose_tracking_endpoint_ramp_v6` when enabled.

Position and rotation use a coupled exponential. At precision stage 3, position
errors 5/10/20 mm with zero angle error score .607/.368/.135 before the hand
weight of four. At stage 4 they score .368/.135/.0183. Rotation error multiplies
these scores down further. The reward-shape plot and numeric table are saved in
`logs/manipulation/endpoint-ramp-audit/`. A weight of .25 bounds the new cost at
.5 units/second rather than the potentially dominant 2 units/second at weight 1.
A fixed-case paired experiment uses stage 2, identical seeds and 131072 PPO
samples per arm: baseline weight 0 versus ramp .25. Runs execute sequentially,
with two simulator workers confined to cores 12–15 and one BLAS/OpenMP thread.
The comparison lives in `logs/manipulation/endpoint-ramp-paired/`; status and
per-checkpoint fixed-case reports distinguish training progress from acceptance.
No failures are automatically retried. Held-out validation stays separate.

All twenty v6 fresh reset cases were audited using nominal FK configuration
witnesses at the target base pose. No joint-limit or sampled geometric collision
violations were found. Nominal arm speed and acceleration maxima were .6703 and
.6825 of profile limits. Report/witnesses are in
`logs/manipulation/target-witness-audit-v6/report.json`. This is not a certificate
of a dynamically executable base path, safe force profile, contact recovery, or
whole-body settling. Existing admission checks use the reset base configuration;
this supplementary audit also checks at the latent target base configurations.

After the reported machine reboot, memory was not under pressure and the current
boot had no logged hardware-error/panic/OOM evidence. Older kernel logs contain
many MuJoCo/Python segmentation faults, often on logical CPU 8. This is evidence
for continuing CPU-affinity mitigation, not proof of the reboot's cause. No BIOS,
voltage, kernel, or hardware settings were changed. The two-worker restriction
supersedes plans for larger parallel training on this unstable host.

### Continuing training without restarting the policy (2026-10-22)

`train --resume-from BUNDLE --initialization-steps 0` now restores PPO policy,
value function, optimizer state, total step count, and per-observation-key running
mean/variance/count. It validates robot/model fingerprint, bundle checksum,
architecture, action semantics, preview layout and pose-only input semantics.
Rollout buffers/observations are recreated for the new simulator. Older bundles
without per-key counts use the saved shared history count. Fresh reset physics,
not a saved training episode state, starts the continuation. Initialization/DAgger
cannot be combined with resume. The selected output directory is a new experiment;
completed experiment directories are never overwritten.

The fixed-case baseline finished its first 131072 steps at 3.22 mm hand error and
.0482 rad angle error, but still failed settling: measured base speeds .00824 m/s
and -.03684 rad/s. This is evidence of improving pose tracking, not task success.
The ramp comparison remains in progress. A sequential continuation worker waits
for both arms to complete, then gives each another 131072 steps with precision
stage 4 and exploration std .1 (original stage 2/std .3). Both receive the same
refinement schedule, preserving the ramp/no-ramp comparison. This schedule tests
precision refinement with lower exploration; it does not isolate which of those
two refinement choices helps. Results/status live under
`logs/manipulation/endpoint-refinement-paired/`. A single two-worker run at a time
uses cores 12–15; errors stop the sequence without automatic retries. No hardware
promotion occurs. Fixed-seed debugging now supports multiple workers, each
repeating the same reset while policy actions provide independent exploration.

Resume regression tests verify restored deterministic predictions, running
statistics/counts and optimizer tensors, reject incompatible action semantics,
and actually continue PPO updates after loading. Both preview and history-only
bundles are covered (four training-options tests passed including reset checks).

### Checkpoint inspection before publishing the implementation (2026-10-22)

The completed coarse ramp arm reached .20 mm / .00525 rad hand error on its fixed
training case, compared with baseline 3.22 mm / .0482 rad at the same 131072 steps.
Neither established a one-second hold. During ramp refinement, another active
constraint is visibly failing: the latest inspected checkpoint (139264 steps)
had 2.05 mm / .00317 rad hand error but .203 rad base heading error, versus the
.04 rad heading contract. The arm can compensate for a base at the wrong heading;
excellent hand accuracy alone does not establish whole-body success. Base XY
violation was zero and no unsafe contact/wheel unloading occurred in this
particular checkpoint's fixed-case rollout.

The current additive hand/base rewards may allow this tradeoff; that remains a
hypothesis to test with paired active-target metrics after the current comparison
finishes. Heading accuracy and continuous low-speed hold both remain required.
This inspection does not justify loosening acceptance or claiming generalized
manipulation success. Training continues independently of source publication.

Publication checks: the complete RL suite passed (182 tests); the Mars backend
suite passed (22 tests) with local simulator ZMQ sockets allowed. The sandboxed
backend attempt could not create those sockets and its stalled process was
stopped after the socket-enabled rerun passed. Ruff over all changed Python files
and `git diff --check` passed. Unrelated lockfile regeneration is excluded from
the control/training commit. These are software/simulation interface checks,
not learned-policy acceptance or hardware qualification.

### Stability-focused refinement and goal age (2026-10-22)

Base heading is task-dependent: .2 rad is about 11.5 degrees and may be acceptable
when the hand task and clearance constraints are satisfied. The default heading
contract remains .04 rad. New `base_heading_tolerance` / CLI
`--base-heading-tolerance` makes alternatives explicit in configuration, episode
metadata, and initial task contracts. Frozen fixtures keep the default; no old
acceptance result is reclassified. The same final ramp policy evaluated under
.04 versus .25 rad tolerances had exactly identical endpoint motion/pose errors.
The relaxed case accumulated only .2 seconds of settling, still short of one
second. Thus heading flexibility alone does not resolve the continuous hold.

Training now accepts `--learning-rate` and `--final-learning-rate`. The linear
schedule covers the additional training phase, not the model's entire lifetime:
a resume at 262144 steps starts at the requested initial LR and reaches the final
LR after the additional budget. Regression tests verify phase endpoints and actual
PPO optimizer LR after resumed updates. `logs/manipulation/stability-lr-paired/`
compares constant 2e-5 with 2e-5 decaying to 2e-6, both initialized from the final
ramp policy, same .25 rad declared heading tolerance and .1 exploration std, same
65536 additional samples, and unchanged strict hand/velocity/hold requirements.
Runs remain sequential, two workers on cores 12–15. Final evaluation reports both
the old strict heading contract and the explicitly relaxed diagnostic contract.
This remains fixed-case training, not generalized task acceptance.

For untimed static goals the endpoint motion ramp now uses elapsed time since
an active base/hand target or mask actually changed. Numerically duplicate target
packets and changes confined to inactive targets do not reset this clock. A new
active target restarts it. For timed references the scheduled trajectory end
remains authoritative, so valid motion along a long path is not penalized merely
because the original command arrived long ago. A replacement trajectory uses its
own clock. Episode metadata records goal age; this is not packet/network age.
The penalty remains bounded to permit correction and disturbance recovery; pose
tracking is still rewarded. No extra actor inputs or action semantics were added.

The static-goal age regression also exposed caller-array aliasing: mutating a
previously submitted target array could change the controller's stored goal
without an update call. Base and hand setters now take owned copies. Duplicate
packets therefore remain distinguishable from actual target updates. Thirty-two
focused control/tracking/training tests passed, including this ownership case,
phase-LR optimizer behavior, and unchanged default heading constraints.

The complete RL regression suite passed after the goal-age and LR changes:
184 tests, no failures. Ruff and `git diff --check` also passed. The stability LR
comparison is still running; no successful stable controller is assumed from
these software checks.

### Measured/commanded hold diagnosis and UI (2026-10-22)

`python -m emet.rl.manipulation.hold_audit --policy BUNDLE --output DIRECTORY`
records raw actor requests, slewed base/arm velocity commands, position actuator
targets, measured joint/wheel/body velocities, pose-difference velocity and hand
errors. Paired interventions use the same reset and policy prefix, switching at
six seconds; summary windows exclude the first two seconds of intervention and
the timeout's final stop-command rewrite. Plot/JSON artifacts are under
`logs/manipulation/hold-command-audit-expanded/` and
`logs/manipulation/explicit-hold-audit/`. Explicit holds are interventions, not
learned-policy successes or hardware qualification.

The zero-action intervention still moved the base: approximately .0022 m/s
forward RMS and .0044 rad/s yaw RMS. Disabling the supplementary base feedback
while sending zero reduced these to about 1.8e-6 m/s and 1.9e-5 rad/s, while hand
error stayed below 1.6 mm in that audit. The fault-stop latch was similarly quiet.
However, disabling that feedback globally worsened active yaw tracking and failed
the existing pick/place regression. That candidate was rejected. Original
tracking feedback and limits remain the default, including during zero actor
requests unless an explicit hold is engaged. This separates plant/servo behavior
from policy-induced small nonzero commands; additional training alone cannot
remove every low-level zero-command oscillation.

`ManipulationControl.hold_position()` is a normal hold distinct from a latched
fault stop. It captures measured arm position, zeros velocity/base requests,
freezes tracking trim, preserves the gripper target, and uses native zero wheel
velocity targets without the supplementary feedback. Physics, gravity and safety
monitoring stay active. Duplicate pose packets leave hold engaged; a changed
active pose or a newly installed trajectory resumes tracking and resets the trim
clock. Fault stops cannot be cleared by this API. No extra learned action output
has been added; deciding to enter hold is currently an explicit caller/UI action.

The simulation UI exposes **Hold pose** for Mars/Stretch arm modes, reports hold
state and keeps physics running; moving a goal resumes tracking. Its checkpoint
selector also now validates each bundle's action mode rather than assuming
residual actions, fixing the hidden direct-policy entries. Loopback URL remains
`http://localhost:8765`; from another machine forward it with
`ssh -N -L 8765:127.0.0.1:8765 cpaxton@olympia`. Server execution remains pinned to
cores 12–13 and offline audits to 14–15, with one BLAS/OpenMP thread.

The explicit-hold audit passed its declared relaxed-heading contract: about
.59 mm maximum tail hand error, 5.9 seconds continuously settled, first one-second
hold at 7.1 seconds. The policy-only counterpart never achieved a one-second hold.
Both retain about .21 rad base heading error; this success is under the explicit
.25 rad diagnostic heading tolerance, not the old .04 rad contract. No unsafe
contact or wheel unloading occurred in either audit rollout. Forty-one targeted
UI/control tests passed, including both existing physical box pick/place teacher
regressions. The new hold mode is caller-selected, not a trained binary policy.

UI verification selected `stability-lr-paired/constant/seed-26/last`, issued an
explicit hold through `/api/command`, confirmed `holding: true` with no UI error,
and observed simulation time advance from .1 to .6 seconds while holding. This
checks real server command handling rather than merely displaying a button.

Final verification: all 186 RL tests passed after explicit hold and checkpoint
selector changes. Ruff and whitespace checks passed. The rejected global
feedback-removal candidate is not included in the final controller behavior.

### Visible base editing and Cartesian target modes (2026-10-22)

The UI now has visible target buttons: 1 Base, 2 EE/base held, 3 EE/base allowed.
The previous arm UI incorrectly masked the base target off in its labelled
base-held mode. Mode 2 now constrains the base pose; mode 3 leaves the base target
inactive for learned Cartesian policies so they may move it to satisfy the hand.
For the feedback comparator, mode 3 still executes the private IK-generated base
reference. Base-only mode requests an SE(2) target and disables the hand target.
These choices expose controller masks, not a guarantee of learned base behavior.

Learned Cartesian policies receive the requested SE(3) pose directly, without
running the UI IK loop or using its joint solution. Rotation columns are checked
for orthonormality. Feedback control retains IK for its joint-reference interface.
The UI tracks Cartesian error rather than an obsolete joint-reference error.
`Targets to robot` restores measured targets and engages normal hold without
resetting/teleporting physics; changing a goal resumes tracking. Faults still
require reset. The mode choice remains explicit instead of silently replacing a
learned controller with feedback control.

Fixed-case bundles are labelled using their saved training options: their local
precision is not evidence of distant-goal recovery, base-only control, or general
whole-body coverage. The current selected checkpoint has precisely that limited
scope. Thirty UI tests passed, including mask semantics, direct pose input without
IK, physical feedback base motion, reset/policy contracts and fault handling.
JavaScript syntax, Ruff and whitespace checks passed.

The served UI was verified after restart: visible target buttons and recovery
control, active mode 1 with a blue base marker and three SE(2) goal components,
normal hold engaged, saved policy retained, fixed-case scope warning visible,
and no UI error. The policy's base-only behavior is still unvalidated. Use the
explicit Feedback controller selection for the tested manual base comparator;
the UI never substitutes it silently for a learned policy.

A nearby masked-coverage continuation was started at
`logs/manipulation/nearby-masked-coverage/seed-27`, initialized from the final
constant-LR fixed-case policy. It samples fresh reset cases rather than repeating
one seed: base-only/hand-only/combined targets, target scale .25, heading range
.3 rad, randomization .05, retarget probability .25, no pushes initially, precision
stage 2, LR 3e-5 to 5e-6 over 262144 additional steps, twenty paired validation
cases every 65536 steps. Workers are limited to two on cores 14–15. Full-distribution
validation and current-distribution curriculum reports remain separate; default
safety gates and the explicitly declared .25-rad heading tolerance are logged.
This addresses missing coverage, not a claim of improved distant-target behavior.
No UI checkpoint is automatically promoted and no hardware is commanded.

### Differential-drive base diagnosis

The coverage continuation was stopped after an audit found that 29/29 active
base targets started inside the accepted 5 cm XY region (maximum displacement
3.82 cm). It did not teach meaningful translation. In paired 20 cm cardinal-goal
tests, feedback succeeded 4/4 with 5–11 mm final error; the fixed-case learned
policy succeeded 0/4 with 0.69–1.29 m final error. Reports are under
`logs/manipulation/base-direction-audit/`. The UI was explicitly switched to the
Feedback comparator while retaining the goal.

Training now supports a forced base-only target mode and a minimum displacement
outside the accepted region. A 12 cm minimum samples 12–18 cm translations,
independent of the arm-target curriculum scale. An optional differential-drive
reward emphasizes forward/reverse travel alignment away from the goal and final
heading near the accepted region, allowing the initial turns needed for lateral
targets. Nominal base-only endpoints receive a collision check; this does not
certify the intervening dynamic path. Existing defaults remain unchanged.
Thirty-one initial targeted tests passed. After adding the distance curriculum,
27 tracking, curriculum, and training-option tests passed again. Corrected
teacher validation admitted all 20 cases: all settled at least once, 14/20 were
still successful at the endpoint, and six lost sustained settling. Final XY
error mean/p95 was 18.2/23.7 mm; heading mean/p95 was .0098/.0148 rad. There were
no unsafe contacts or wheel lifts. Mean first-settle time was 8.54 seconds.
These results expose remaining feedback chatter rather than certify stability.

The replacement scratch PPO run is
`logs/manipulation/differential-base-curriculum-v1/seed-28`, W&B
<https://wandb.ai/cpaxton/emet-mars-control/runs/9du79bdg>. It requests 1048576
steps with two workers on cores 14–15, LR 1e-4 decaying to 1e-5, endpoint motion
weight .25, low randomization .05, and no retargets or pushes initially. Base
distance samples start at 8–12.05 cm (minimum .08 m, final maximum .35 m,
target scale .15). The maximum grows with the existing validation-gated target
scale; the minimum always exceeds the 5 cm accepted radius. Base-only cases
require the strict success gate (at least 90% over 20 cases and no safety or
initialization failures) to advance. Fixed full-distribution tests use 8–35 cm
and full randomization, separately from current-stage tests. Checkpoints are
compared to the teacher on identical cases every 65536 steps. Logs include
current-stage outcome metrics, distance bounds, ever-settled rate, subsequent
settling loss, and measured base-axis jerk. No corrected learned policy has yet
been validated or promoted; this run isolates base control before reintroducing
whole-body goals and dynamic goal changes.

### Frozen base stress benchmark

`python -m emet.rl.manipulation.base_benchmark` freezes seven strata before
scoring: near targets, longer translations, full initial-heading range,
mid-motion goal changes, finite end-effector pushes, full dynamics
randomization, and a 40-second hold horizon. The default is 20 fresh cases per
stratum (140 per controller), starting at seed 9300000, disjoint from training
validation seeds. Acceptance tolerances come from the template bundle and remain
unchanged. Raw initial base pose, goal and masks are recorded alongside seeded
reference and disturbance information for replay. Initialization failures remain
in the denominator. Success intervals are Wilson 95% intervals; per-stratum
metrics remain visible rather than relying on pooled success alone.

The first run is `logs/manipulation/base-stress-v1`, with a frozen `suite.json`
and serial teacher results saved per stratum. It will evaluate the immutable
`differential-base-curriculum-v1/seed-28/step-65536` bundle after that bundle is
available, then save candidate reports and a paired comparison. It shares the
existing two-core CPU cap and adds no simulator worker pool. Training continues;
this audit does not select curriculum stages or promote a UI/hardware policy.

```bash
MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 14-15 .venv/bin/python -u -m emet.rl.manipulation.base_benchmark \
  --template logs/manipulation/differential-base-curriculum-v1/seed-28/step-0 \
  --policy logs/manipulation/differential-base-curriculum-v1/seed-28/step-65536 \
  --wait-seconds 43200 --cases 20 --seed 9300000 \
  --output logs/manipulation/base-stress-v1
```

This is base-control evidence, not a manipulation/TAMP or dedicated obstacle
impact test. The paired comparison now skips hand accuracy metrics only when
both reports explicitly have zero active hand targets; safety, force, jerk and
base-contract comparisons remain active. Thirteen targeted tests passed,
including seed separation, preserved tolerances, confidence intervals and
base-only paired comparisons that still reject increased forces.

### Payload-aware TAMP trajectory progression

The current Mars primitive pick/return fixture has a nominal 20 g payload. It is
not evidence of heavier-object manipulation. `ManipulationConfig.payload_mass_kg`
and training's `--payload-mass-kg` now set a nominal mass before initialization;
existing mass randomization is applied around that value. Inertia scales by the
same density factor at fixed geometry and center of mass, and MuJoCo constants
are recomputed before any scored motion. Each reset records actual mass,
principal inertia and local COM in the randomization metadata, which generated
TAMP component cases and integrated stage reports carry forward. Uniform density
scaling does not represent arbitrary real-object mass distributions.

The continuous integrated benchmark accepts `--payload-masses-kg`, retaining
identical seeds, trajectories, controller speed limits and acceptance thresholds
across the sweep. It saves each completed case immediately. A teacher sweep of
.02/.05/.10 kg at box seed 1800006 is queued after the base stress suite, at
`logs/manipulation/tamp-payload-sweep-v1/teacher.json`. These are simulator stress
values, not hardware payload ratings. The existing nominal teacher regressions
under residual and direct velocity interfaces remain separate from these new
load cases.

```bash
MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 14-15 .venv/bin/python -m emet.rl.manipulation.integrated \
  --robot innate_mars --seed 1800006 --episodes 1 --action-mode direct_velocity \
  --payload-masses-kg .02 .05 .10 --output /tmp/mars-payload-sweep.json
```

The next manipulation gate should use the actual TAMP benchmark object assets
and mass/inertia/COM data, with generated GT/oracle plans converted to timed hand
SE(3), masked base SE(2), and gripper trajectories. Execute approach, grasp, lift,
carry, lower, release and settle continuously, with no stage resets or object
attachments. Compare teacher and a whole-body checkpoint on identical scenes
and loads; the current base-only policy is not that checkpoint. Start near the
nominal physical distribution and broaden load, grip friction, speeds, COM and
persistent disturbance windows only after validation. Keep frozen retention,
fingertip-clearance and placement thresholds unchanged. Report failures by stage,
load, geometry and plan feasibility as well as tracking, slip, sag, jerk, support
loads and wheel unloading. The current integrated fixture uses actual GT TAMP
arm transport paths but is not yet full public-tool oracle pick-and-place.

### Native TAMP payload gate and reward audit (2026-10-23)

`--payload-asset-set native_table` transfers the canonical TAMP table's actual
box and cylinder bodies into the manipulation fixture. Both are 0.5 kg at source
scale. The importer preserves compiled mass, inertia, COM, geometry and contact
properties, explicitly overriding inherited robot defaults. The compiled model
has a distinct fingerprint; incompatible policy bundles remain rejected. Source
path, SHA256 and actual mass properties are recorded. `--payload-object` fixes
object identity so a suite cannot accidentally omit one shape.

Run the object-transfer gate with:

```bash
MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 14-15 .venv/bin/python -m emet.rl.manipulation.tamp_benchmark \
  --robot innate_mars --episodes 3 --seed 1800006 --output /tmp/native-payload-tamp
```

This is continuous physical pick/return-place with GT TAMP transport, transferring
objects into the existing support fixture. It does not reproduce the full native
table scene or add Mars to the Stretch-only public native adapter. `--policy`
adds the matching whole-body policy on identical cases; task completion never
sets promotion. Suites are frozen and partial reports are saved atomically.

Initial native-object execution (`native-payload-tamp-v1`) failed all six cases:
three box timeouts and three cylinder wheel-unloading stops during approach.
Reset diagnostics exposed a setup error: the oversized box had already fallen
from its support, while the cylinder wedged between open fingers. The declared
nominal centered pinch at 30 mm depth has about 21.8 mm open clearance; projected
native box/cylinder widths are 80/40 mm. A scratch geometry admission now rejects
that specific pinch before any scored motion. It does not prove all grasps are
infeasible. `native-payload-tamp-v2` preserves all six rejections in its denominator
rather than counting them as controller successes. The existing 16 mm training
box is admitted. Neither objects nor hardware gripper limits are resized.

The small-object teacher sweep (`tamp-payload-sweep-v1`) succeeded at 20 g but
failed retention during lift at 50 and 100 g: peak relative rotations were
.1154/.1084 rad against the unchanged .1 rad limit. A timed minimum-jerk lift
trial did not fix the heavier loads and caused wheel unloading at 100 g; that
trial was discarded. A grasp-depth diagnostic at 25/35/40 mm likewise failed to
qualify either heavier load. Its configurable `--grasp-depth` is recorded as a
fixture setting; frozen fixture goals and acceptance thresholds remain unchanged.
No grip-force increase was made. These failures block manipulation qualification;
more PPO is not evidence that these grasps or assets are suitable for Mars.

Separately, the frozen 140-case base suite completed: teacher 102/140 sustained
success, first learned checkpoint 0/140, with no wheel lifts in either report.
The original run was stopped after its 131072-step validation also remained at
0% success. The v7 objective rewarded stationary behavior outside the goal and
charged an endpoint motion penalty before a static goal was reached. New opt-in
`--goal-gated-settling` restricts both terms to the accepted active pose region;
force, measured jerk, action-rate and other safety terms remain active everywhere.
The per-control-step PPO discount is now explicit through `--discount` (legacy
.99 default). At 20 Hz, .99 discounts over roughly five seconds, whereas .9975
covers roughly twenty seconds.

`python -m emet.rl.manipulation.reward_audit --output PATH` compares teacher and
idle returns on identical fresh seeds before training. In four local cases,
v7/.99 preferred idle in one case. Goal-gated settling preferred the teacher in
all four at both discounts. With .9975, teacher-minus-idle discounted return was
10.13–17.27, compared with -.14–1.49 under v7/.99. This is an incentive diagnostic,
not learned-controller success or broad qualification. The full RL suite passed
200 tests before the final admission/metadata checks; targeted final checks also
cover source-asset fidelity, pre-motion rejection, the reward counterexample and
unchanged nominal physical pick/place.

Next manipulation work remains gated on a suitable grasp distribution: select
Mars-admissible real object grasps, fix heavier-load retention without raising
unsafe forces, and then train/retest full timed hand/base/gripper trajectories.
Use the native assets that fit each robot, retain inadmissible cases as planning
rejections, and keep controller failures distinct from reachability/geometry
failures. The public native adapter still needs an explicit Mars arm/gripper
facade before claiming public-tool oracle task coverage for Mars.
