# Mars reactive control in simulation

The optional `emet.rl` package provides a synchronous Gymnasium environment,
PPO training, a deterministic baseline, and reactive control in the existing
Mars MuJoCo server. `BaseZmqServer` continues to own transport and command
receipts. The default controller remains `legacy`.

This is a simulation development path. It does not enable Mars hardware motion,
replace Nav2, or claim that a learned policy is better than the baseline.
Stretch's base-velocity and nonblocking joint-position commands are the reference
actuation interface. Hardware publishers and perception adapters are follow-on work.

## Start with base-only control

The CLI now defaults to `--task base --architecture mlp`: two 64-unit hidden
layers with ten samples of history at 20 Hz (300 scalar inputs). Actions are
forward velocity and yaw rate. The arm targets stay fixed; its collision geometry
remains active. Each sample contains relative goal position/heading, measured
body velocity, applied commands, nine obstacle distances plus validity bits, and goal distance/direction.
Obstacle inputs currently come from simulator geometry, not physical sensors.

Normal commands are limited to 0.30 m/s, 0.80 rad/s, 0.40 m/s² and 1.20 rad/s².
Retargeting preserves velocity continuity. Obstruction and emergency stops can
bypass acceleration limits. These are simulator tuning values, not verified Mars
hardware limits. Success requires position error <=2 cm, heading error <=2°,
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

The policy runs at 20 Hz with 25 MuJoCo substeps of 2 ms each. Each environment
owns its model, data, random generator and reset state; stepping never sleeps or
uses ZMQ. Physics advances through actuators, without per-step teleportation or
state pinning.

- Seven normalized actions: forward speed, yaw rate, and five arm target
  increments. The adapter limits these to 0.30 m/s, 0.80 rad/s, and joint-target
  changes of 0.03 rad per control tick. These are simulation defaults, not
  calibrated hardware limits.
- `joint1` through `joint5` are reaching joints. `joint6` is the gripper,
  `joint6M` is its mimic, and `joint_head` is held during a reactive command.
- Goals contain a world-frame base `[x, y, yaw]`, named arm targets, or both.
  Missing dimensions hold their configuration at goal acceptance. Partial arm
  dictionaries are supported. Cartesian targets remain an upstream IK task.
- Observations contain 25 state values (including held actuator-target errors),
  15 goal/error/mask values, 26 measured/target hand and body pose values, and 45 local
  obstacle values: nearest scene-surface distance, link-frame direction and
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

Success requires errors below 5 cm, 5 degrees, and 0.05 rad for active goals,
plus low measured velocities for 0.25 seconds. By default, collisions and invalid physics
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
simulator ground truth. Full release remains an explicit executor operation.

This implementation is experimental. The 1,024-step smoke run completed, but
loaded tasks did not pass their success gates; it is not evidence of improved
grasping or safe deployment. Frozen cases are archived with checksums, but their
historical event traces lack complete physics snapshots. The oracle harness below
now plans and physically executes from those fixtures; it does not claim exact
replay of the original live CHAT timing. Mars manipulation training is blocked
by missing authoritative wheel/support geometry; diagnostic free-base simulation
is not a substitute for that geometry.

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
0.25-second disturbances. Choose Stretch physical manipulation or the existing
Mars base simulation. Controller selection loads a validated local checkpoint
or the feedback baseline; refresh discovers newly exported checkpoints without
changing a running controller automatically. A live panel separates initialization
from PPO training steps and links the W&B run.

The default checkpoint roots are `logs/manipulation/physical-v2` and
`logs/mars-base-extended`; repeat `--policy-root` to choose other local roots.
The server binds only to loopback. If the browser is on another machine, forward
port 8765 over SSH. It has no robot client, connection-profile lookup, hardware
endpoint, or actuator transport. The same-seed reset permits manual comparisons;
interactive interventions do not count as acceptance evidence. Mars wheel support
is explicitly marked unsupported. Episode limits in this interactive tool are
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
description, but still provides no wheel/caster support geometry. This inspection
does not resolve the missing Mars stability model or certify hardware readiness.
