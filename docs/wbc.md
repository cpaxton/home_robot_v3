# Whole-body control (WBC) guide

Status: simulator-first. Direct MuJoCo policies are trained and gated for Mars,
Stretch 3, Sourccey, RB-Y1, and PandaOmron. RoboCasa has a native PandaOmron
reach adapter and privileged teacher; benchmark task targets and the
merged-scene adapters for the other robots are still being wired.

This is the how-to guide. Result tables and caveats live in:

- [WBC training report](experiments/wbc_training_report_20261126.md)
- [multi-robot result matrix](experiments/wbc_multirobot_20261125.md)
- [RoboCasa adapter](experiments/wbc_robocasa_20261126.md)
- [whole-body control plan](plans/whole-body-rl-control.md)

## 1. Mental model

There are two layers:

1. **WBC policy (low level).** A normalized-torque actor that tracks a base
   pose, torso posture, and per-arm EE target. This is what the checkpoints in
   this guide contain.
2. **Skill/TAMP layer (high level).** Chooses object, grasp, lift, place, and
   base targets. The WBC policy does not know about objects or task success; it
   tracks the targets it is given.

The actor observation is sensor-only: base pose/twist, joint encoders, EE pose,
EE target, gripper state, measured contact (layout v3), and derived target
errors. A privileged teacher may use simulator targets and phase state during
training, but **teacher targets never enter the exported actor or checkpoint**.

Actions are base velocity plus per-arm joint torques in `[-1, 1]`. A profile and
an actuator adapter turn them into the robot-specific command (position servo,
current, or true torque).

## 2. Quick start

Train a robot's default DAgger suite:

```bash
uv run python -m emet.rl.manipulation.wbc_dagger \
  --robot innate_mars --output runs/wbc-mars \
  --iterations 4 --teacher-steps 2048 --dagger-steps 1024 \
  --bc-epochs 200 --eval-episodes 10 --selection balanced
```

For a dual-arm robot, the default suite is `arm_left`, `arm_right`, `base`,
`combined_left`, `combined_right`; for Stretch/PandaOmron it is `arm`, `base`,
`combined` with a longer combined window.

Evaluate a checkpoint over held-out seeds:

```bash
uv run python scripts/eval_wbc_policy.py \
  --robot pandaomron --checkpoint runs/wbc-panda/best.pt \
  --output /tmp/wbc-gate --episodes 50 --seeds 30000,40000,50000
```

Run the strict coverage gate:

```bash
uv run python scripts/emit_wbc_coverage.py \
  --cases configs/wbc/pandaomron_cases.json \
  --policy-report /tmp/wbc-gate \
  --out /tmp/wbc-coverage.json
uv run python -m emet.rl.manipulation.wbc_coverage \
  --cases configs/wbc/pandaomron_cases.json \
  --results /tmp/wbc-coverage.json \
  --out /tmp/wbc-gate-report.json
```

For the other four robots, one registry consumes per-robot reports:

```bash
uv run python scripts/emit_wbc_coverage.py \
  --cases configs/wbc/multirobot_cases.json \
  --policy-report innate_mars=/tmp/wbc-gate/mars \
  --policy-report stretch=/tmp/wbc-gate/stretch \
  --policy-report sourccey=/tmp/wbc-gate/sourccey \
  --policy-report rby1=/tmp/wbc-gate/rby1 \
  --out /tmp/wbc-multirobot-coverage.json
uv run python -m emet.rl.manipulation.wbc_coverage \
  --cases configs/wbc/multirobot_cases.json \
  --results /tmp/wbc-multirobot-coverage.json \
  --out /tmp/wbc-multirobot-gate-report.json
```

A checkpoint must load with the exact profile fingerprint. Layout-v2
checkpoints are migrated to v3 automatically: old observation columns are
remapped, new contact inputs are zeroed, and the normalized gripper-force
columns are rescaled. New DAgger checkpoints also store the task contract
(subtask goals, target range, base-goal probability, hold-base, control period,
success hold, collision threshold). `eval_wbc_policy.py` and
`eval_wbc_hard_cases.py` refuse to evaluate a checkpoint without one, so a
held-out row cannot silently use a different task. Legacy DAgger checkpoints can
be migrated explicitly:

```bash
uv run python scripts/stamp_wbc_contract.py \
  --checkpoint old_best.pt --robot pandaomron \
  --output stamped_best.pt
```

`eval_wbc_policy.py` / `eval_wbc_hard_cases.py` also accept
`--allow-missing-contract` as an explicit escape hatch for an old artifact;
the manifest records `contract_source=legacy_default_suite` (or
`legacy_default_timing`). Prefer stamping so the stored contract is checked.

## 3. Adding a robot

A robot needs a profile, a plant, and a registry entry.

### 3.1 Profile

`src/emet/controller/manipulation/wbc_profiles.py`:

- base kind and velocity limits,
- torso/head groups (optional),
- one `JointGroupSpec` per arm with joint names, position/velocity/torque
  limits, `kind`, `control_mode`, `kp_eff`, `kd_eff`, `max_lead`,
- `ArmSpec` with EE link, target frame, and gripper limits,
- metadata (`robot`, `limits_source`, `sim_model`).

The profile fingerprint is part of every checkpoint. Changing joint order,
limits, control mode, or gripper limits invalidates old checkpoints.

### 3.2 Plant

`src/emet/rl/manipulation/wbc_<robot>.py` implements the `WbcBackend` protocol:

```python
reset(seed=None, options=None) -> WbcObservation
step(ServoCommand, dt) -> (WbcObservation, terminated, info)
load_hard_case(case) -> WbcObservation | None   # optional but needed for replay
teacher_action() -> np.ndarray                  # training/teacher only
close()
```

Bind profile joint names to MJCF qpos/dof/actuator addresses, validate the
intersection of profile and model limits, and report non-finite physics. A
`preferred_adapter` (e.g. `TorqueAdapter`) is used by `WbcGymEnv` when present.

### 3.3 Registry/factory

- `wbc_robots.py`: robot name, profile factory, direct-MuJoCo availability,
  RoboCasa name.
- `wbc_backends.py`: direct-MuJoCo factory and `sim='robocasa'` dispatch.
- `wbc_robocasa.py`: explicit RoboCasa registration; missing adapters fail
  closed.

Add a sim test that resets, steps zero/teacher actions, validates observation
shapes/finiteness, and checks the model fingerprint.

## 4. Failure-driven data from a benchmark

The goal is to turn a failed benchmark attempt into a **replayable posture and
target** plus enough scene information to reconstruct the state and train on it.

### 4.1 What a failure record contains

The manipulation failure sink is JSONL. Each line has `context`, `attempt`, and
`ts`:

```json
{
  "context": {
    "robot_id": "panda_omron",
    "op": "grasp",
    "object_query": "pear",
    "receptacle_query": "sink",
    "object_body": "obj_main",
    "receptacle_body": "sink_main_group_main",
    "layout": 1,
    "style": 1,
    "job_id": "20261125_233017_3e5d1e",
    "message": "grasp_failed:lift_tracking_failed"
  },
  "attempt": {
    "joint_names": ["robot0_joint1", "..."],
    "reached": false,
    "failure_mode": "tracking_failed",
    "error_m": 0.42,
    "trace": [{"qpos": [0.0, "..."], "ee_xyz": [0.0, 0.0, 1.0]}],
    "planned_waypoints": [[0.0, "..."]]
  },
  "ts": 1795670000.0
}
```

**A hard case is only a failed reach.** Require `attempt.reached is False`.
Records from a place failure often carry the preceding successful grasp as
`last_attempt` with `reached=True` and `failure_mode=None`; those are not
failure postures and must be rejected. The 2026-11-25 sink had five such stale
records (014–018); only `failure-019` was genuine.

The exporter uses `attempt.trace[0].qpos` (or `attempt.initial_q`) as the
starting posture and `attempt.planned_waypoints[-1]` as the target posture. It
infers the arm and contract joint order from the recorded joint names, so a
Sourccey/RB-Y1 left-arm record is not mislabeled as the right arm; ambiguous
multi-arm records are skipped unless `--arm` is supplied. `attempt.op`,
`attempt.failure_mode`, `error_m`, object/receptacle bodies, layout/style, and
job id are retained as provenance.

### 4.2 Export a hard-case split

```bash
uv run python -m emet.rl.manipulation.wbc_hard_cases \
  --records ~/runs/emet/manip_failures/manip_failures.jsonl \
  --robot pandaomron \
  --split train \
  --out configs/eval/my_robot_hard_cases_train.json

uv run python -m emet.rl.manipulation.wbc_hard_cases \
  --records ~/runs/emet/manip_failures/manip_failures.jsonl \
  --robot pandaomron \
  --split test \
  --out configs/eval/my_robot_hard_cases_test.json
```

The JSON wrapper declares `schema_version`, `split`, and `cases`. A case stores
`initial_q`, `target_q`, `failure_mode`, `source`, and a `scene` mapping. The
scene mapping is where you add benchmark context (`layout`, `style`,
`object_body`, `receptacle_body`, poses, model path). The training suite ignores
scene metadata, but it keeps the replay reproducible.

**Splits are not optional.** Train on `split='train'` only; keep test cases out
of teacher collection, BC, DAgger, and checkpoint selection.
`dagger_train --hard-cases` rejects non-train and empty train files.

### 4.3 Add scene information explicitly

For a direct MuJoCo robot, put the reconstructable scene in the case's `scene`:

```json
"scene": {
  "simulator": "mujoco",
  "profile": "pandaomron_sim",
  "mjcf": "third_party/robosuite/.../PandaOmron.xml",
  "layout": 1,
  "style": 1,
  "object_pose": [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0],
  "receptacle_pose": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
}
```

For RoboCasa, record the env name and layout/style, plus the object/fixture
poses. `RoboCasaPandaBackend` reconstructs the Omron + Panda model from the env;
the reach target is an EE pose, and the high-level layer supplies
object/cabinet targets through `set_ee_target(world_pose)` or
`set_local_ee_target(local_pose)`.

### 4.4 Generate loaded data from a failing scene

For RoboCasa, the sweep script loads the scene once, samples targets, runs the
policy or teacher, and writes actor-visited observations with privileged
teacher labels:

```bash
uv run python scripts/sweep_wbc_robocasa_panda.py \
  --checkpoint runs/wbc-panda/best.pt \
  --episodes 20 --target-mode offset \
  --offset-range 0.05 --z-range 0.08 \
  --backoff 0.45 --layout 1 --style 1 --save-data \
  --output /tmp/wbc-robocasa-sweep
```

- `--target-mode joint` samples postures around the ready pose.
- `--target-mode offset` samples Cartesian EE offsets for task-like targets.
- `--teacher` runs the privileged teacher instead of the policy.

Outputs:

- `sweep.json`: per-episode success, position/orientation error, target, final
  EE, scene.
- `loaded_data.npz`: `observations`, `teacher_actions`, `episode_ids`,
  `step_ids`.
- `loaded_data_meta.json`: checkpoint, observation count, action size, scene
  (env/robot/layout/style/backoff), target mode/ranges.

The observation vector is the **current layout** (v3); the teacher actions are
normalized contract actions. This is the same schema `WbcPPO.pretrain` consumes.

### 4.5 Train from the loaded data

Fine-tune a policy from a loaded NPZ:

```python
import numpy as np
from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_training import WbcPPO

data = np.load("/tmp/wbc-robocasa-sweep/loaded_data.npz")
trainer, metadata = WbcPPO.load("runs/wbc-panda/best.pt", pandaomron_sim_profile())
metrics = trainer.pretrain(
    data["observations"],
    data["teacher_actions"],
    epochs=200,
    batch_size=64,
    learning_rate=1e-3,
)
trainer.save("runs/wbc-panda/robocasa_finetune.pt", env, metadata=metadata)
```

Then re-run the sweep and the held-out direct gate. Do not use the loaded data
as an evaluation set; keep a fresh scene/seed for the gate.

Hard-case replay training uses the suite:

```bash
uv run python -m emet.rl.manipulation.wbc_dagger \
  --robot pandaomron --output runs/wbc-panda-hard \
  --init runs/wbc-panda/best.pt \
  --hard-cases configs/eval/my_robot_hard_cases_train.json \
  --hard-case-probability 0.25 \
  --iterations 4 --bc-epochs 300 --selection balanced
```

### 4.6 Validate the data

- `attempt.reached is False` for every accepted case.
- Robot aliases normalized (`panda_omron` → `pandaomron`, `rb_y1` → `rby1`,
  `mars` → `innate_mars`).
- Train/test names do not overlap.
- Initial and target postures are finite, have the profile's joint count, and
  are not identical.
- The scene is reconstructable: MJCF/profile for direct MuJoCo, env/layout/style
  for RoboCasa.
- No object ids, GT poses, or teacher targets in the actor observation.
- Coverage metrics are finite scalars: episodes/collisions integer, success and
  errors finite. Missing tracking/duration/collisions fails closed, and
  collision counts require `collision_measured=True`.
- The checkpoint stores the exact task contract; evaluate scripts reconstruct
  that suite instead of the current defaults.

## 5. Evaluation and promotion

- **Held-out clean gate:** 50 episodes per subtask, three fresh seeds.
  `evaluate_suite` reports success, measured collisions, settled tracking error,
  final error, settled-window max, episode max, duration, and contact-state rate.
  The gate's `tracking_error_m` is the mean over the last 0.5 s (10 steps) after
  the first successful step (last 10 steps for a failed episode), not the single
  final step.
- **Collision provenance:** every gate row must carry
  `collision_measured=True`. Stretch, Sourccey, RB-Y1, PandaOmron, and RoboCasa
  now measure external/self contacts from solved MuJoCo contacts. Wheel/caster
  floor contacts are support; intentional hand-object contacts are not
  collisions; chassis/arm contact and non-gripper self-collision are. A report
  that says `collisions=0` without this flag fails.
- **Coverage gate:** per-family cases with `min_success`,
  `max_collision_rate`, `max_tracking_error_m`, optional `max_duration_s`, and
  contact-state requirements. Missing metrics raise; there are no zero
  defaults. A policy is promoted only when active test cases pass and older
  families do not regress.
- **Hard posture:** currently `planned`; the first failure sink contained stale
  successful grasps, so the train split is empty. Regenerate after #267.
- **Checkpoint selection:** `--selection balanced` ranks by aggregate plus the
  weakest subtask; later checkpoints win ties. Do not select a hard-case
  checkpoint from the clean aggregate; use the coverage report once genuine
  cases exist.

## 6. RoboCasa adapter

`RoboCasaPandaBackend` wraps a Robosuite/RoboCasa env and writes contract
torques/velocities directly to the Omron + Panda actuators. It is registered
through `make_wbc_backend("pandaomron", sim="robocasa", env=...)`. It uses a
reset-anchored robot frame for both EE and base observations (including a yaw
rotation), a 0.45 m chassis backoff, and a torso PD/bias hold. The profile keeps
canonical names (`right_hand`, `panda_finger_joint*`); the direct and RoboCasa
plants resolve them through explicit alias maps and verify the result.

- **Teacher:** joint-space PD for sampled postures; Cartesian PD through a
  finite-difference EE Jacobian for task targets. Privileged only.
- **Reach results:** 8 cm smoke = 1.31 cm / 0.029 rad; 20-episode joint-target
  sweep = 20/20, mean 1.13 cm, p95 1.67 cm, max 1.70 cm; Cartesian-offset
  sweep = 20/20, mean 0.87 cm, p95 1.54 cm.
- **Task-target bridge:** `wbc_robocasa_task.py` derives object/cabinet poses and
  pregrasp/grasp/lift/place/retreat targets via IK; `sweep_wbc_robocasa_tasks.py`
  runs the sequence with teacher or policy and records task success. A 10-episode
  teacher sweep is **0/10** (5 closed cabinets skipped); pregrasp and centered
  grasp track, but one object was lost during lift and another was lifted ~12 cm
  but not placed inside the cabinet. The free-space policy degrades after
  pregrasp; a 728-sample BC fine-tune still gave 0/3 task success and worse
  post-pregrasp tracking. This is a scaffold, not task success.
- **Not task success:** object/cabinet place planning, grasp retention, base
  goals, and the merged Stretch/Sourccey/RB-Y1 scenes remain open.

## 7. Troubleshooting

- **Hard cases look wrong:** check `reached`; place failures can carry a
  successful grasp as `last_attempt`. Reject, do not relabel.
- **Checkpoint layout mismatch:** layout v2 checkpoints migrate automatically.
  A custom occupancy layout needs an explicit migration.
- **Resume refused:** the checkpoint has no task contract. Periodic checkpoints
  now carry it; old checkpoints must be re-saved or retrained.
- **`--hold-base`:** base goals are forced off; base target observations are
  absent.
- **Multi-calibration feedforward:** rejected until nearest-posture torque
  selection exists.
- **Coverage gate fails on tracking:** the gate uses the mean EE error over the
  last 10 steps, with final and max fields reported for diagnosis. It is not a
  transient average over the base stage, and it is not just the final step.
- **RoboCasa reach drifts:** verify the chassis backoff, torso hold, and target
  frame; the teacher should track to sub-centimeter error in the same scene.

## 8. Plan

1. Fix the task-target bridge's collision-aware base/cabinet approach and
   place path; the current teacher loses the place target (31–34 cm).
2. Fine-tune from the 728-sample loaded task data and re-run the task sweep;
   the free-space policy degrades after pregrasp (15–20 cm at grasp).
3. Improve grasp retention; the teacher reaches the object but does not
   consistently lift it.
4. Reuse the direct-write adapter for the merged Stretch/Sourccey/RB-Y1
   RoboCasa scenes.
5. Add genuine hard-posture data after #267; train and gate the family.
6. Add shelf-entry, carry, and contact-recovery cases with occupancy/contact
   observations.
7. Longer term: one embodiment-conditioned policy instead of one per robot.
