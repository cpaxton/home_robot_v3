# PandaOmron WBC in a RoboCasa kitchen

Date: 2026-11-26

Scope: first RoboCasa adapter for a trained WBC policy. This is a **reach
tracking smoke**, not a RoboCasa task-success result. The policy is the
direct-MuJoCo PandaOmron checkpoint; the adapter drives the same robot model
inside RoboCasa's `PickPlaceCounterToCabinet` scene.

## Adapter

[`wbc_robocasa_panda.py`](../../src/emet/rl/manipulation/wbc_robocasa_panda.py)
wraps a Robosuite/RoboCasa env:

- Reuses the env's `sim.model`/`sim.data`; RoboCasa builds the same Omron +
  Panda actuators as the direct plant (`robot0_torq_j1..7`, base velocity
  actuators, torso torque, `gripper0_right_*` fingers).
- Writes contract torques/velocities directly to those actuators, bypassing
  Robosuite's composite controller.
- Expresses the base and end-effector poses in a reset-anchored robot frame so
  the policy sees the same layout as its direct training suite.
- Backs the chassis 0.45 m along the forward planar slide before raising the
  arm; the task spawn otherwise puts the ready posture inside a counter fixture.
- Holds the torso at 0.2 m with a PD/bias servo; the arm-only policy's torso
  channel is not a reliable height servo in the kitchen.
- The default target is a free-space offset from the current end effector. A
  skill/TAMP layer can call `set_ee_target(world_pose)` for real task targets.

The fail-closed `make_robocasa_backend("pandaomron")` hook is intentionally
unchanged: the adapter requires an env and a task target provider, which the
generic factory cannot construct by itself.

## Privileged teacher

`RoboCasaPandaBackend.teacher_action()` is the privileged labeler:

- **Joint mode.** Samples a target joint posture within `target_range` of the
  ready pose, computes its EE pose by forward kinematics, and applies joint PD
  with the profile kp/kd gains plus gravity bias. The teacher tracks a sampled
  target to **0.48 cm / 0.0115 rad** in the kitchen scene.
- **Cartesian fallback.** For an EE target supplied with `set_ee_target()`, it
  applies a Cartesian PD wrench through a finite-difference EE Jacobian. This is
  the path for object/receptacle targets, where a joint-space target is not
  known a priori.
- The teacher also holds the torso at 0.2 m with the same PD/bias servo the
  backend applies. Teacher actions are never part of the actor observation or
  exported checkpoint.

## Reach sweep and loaded data

```bash
uv run python scripts/sweep_wbc_robocasa_panda.py \
  --checkpoint /tmp/opencode/wbc_panda_mixed_heldout/best.pt \
  --episodes 20 --target-range 0.15 --backoff 0.45 \
  --layout 1 --style 1 --save-data \
  --output /tmp/opencode/wbc_robocasa_sweep_policy
```

- 20/20 contract successes on sampled joint targets (±0.15 rad) in the loaded
  `PickPlaceCounterToCabinet` scene.
- Position error: mean **1.13 cm**, p95 **1.67 cm**, max below the 2 cm
  tolerance; orientation error stays well inside 0.05 rad.
- The earlier 1.31 cm smoke is therefore typical, not a marginal one-off.
- `loaded_data.npz` stores actor-visited observations and teacher labels
  (`observations`, `teacher_actions`, episode/step IDs) for BC/DAgger
  fine-tuning; `sweep.json` stores per-episode metrics and targets.

A second loaded sweep samples **Cartesian EE offsets** (±5 cm in x/y, 0–8 cm
up) with the same scene and policy:

- 20/20 contract successes.
- Position error mean **0.87 cm**, p95 **1.54 cm**, max below 2 cm.
- Data: `/tmp/opencode/wbc_robocasa_sweep_cartesian/loaded_data.npz`.

Both the joint-target and Cartesian-offset distributions stay inside the 2 cm
tolerance, so no DAgger fine-tune is required for this reach envelope. Farther
object/cabinet targets, grasp/lift/place sequences, and obstacle/contact cases
still need a task-target sweep and, if their tail crosses 2 cm, a DAgger
fine-tune from `loaded_data.npz`.

## Task-target bridge (development)

[`wbc_robocasa_task.py`](../../src/emet/rl/manipulation/wbc_robocasa_task.py)
derives task poses from the loaded scene:

- object pose from `obj_main` and its bounding-box long axis;
- a top-down grasp orientation with the finger-closing axis across the short
  side;
- pregrasp, grasp, lift, place, and retreat joint targets from damped
  least-squares position IK on the composed Panda arm;
- the lowest open cabinet shelf as the place target.

[`scripts/sweep_wbc_robocasa_tasks.py`](../../scripts/sweep_wbc_robocasa_tasks.py)
runs the full phase sequence with the teacher or a policy, records per-phase
tracking, object displacement, collisions, and the env's own
`_check_success()`, and exports policy observations plus teacher labels.

```bash
uv run python scripts/sweep_wbc_robocasa_tasks.py \
  --teacher --episodes 3 --layout 1 --style 1 --backoff 0.0 \
  --output /tmp/opencode/wbc_task_sweep_teacher
uv run python scripts/sweep_wbc_robocasa_tasks.py \
  --checkpoint logs/eval/wbc_multirobot_20261126/strict/pandaomron_robust/best.pt \
  --episodes 3 --layout 1 --style 1 --backoff 0.0 --save-data \
  --output /tmp/opencode/wbc_task_sweep_policy
```

Development result (10 teacher seeds, one scene):

- **0/10 task success**; 5/10 seeds had a closed cabinet and were skipped.
- The 3 open-cabinet seeds below show where tracking breaks.
- Teacher: pregrasp **0.7 cm**, grasp **3.2–3.8 cm**, centered grasp **0.6 cm**,
  lift **8.7–10.2 cm**, preplace **12–19 cm**, place **3.4–11.6 cm**, release
  **1.7–2.6 cm**; task success **0/3**. Object displacement was 5.3 and 16.2 cm.
- Policy (robust checkpoint): pregrasp **3.7–5.9 cm**, grasp **15–20 cm**,
  centered grasp **4–7 cm**, lift **29–98 cm**, place **79–117 cm**; task
  success **0/3**.
- The 10-episode teacher sweep moved the object 0–28 cm but never placed it in
  the cabinet. The latest policy-mode loaded data has 728 actor-visited
  observations and teacher labels
  (`/tmp/opencode/wbc_task_sweep_policy/loaded_task_data.npz`).

**Interpretation:** pose extraction, IK target generation, pregrasp tracking,
and finger centering work. The teacher can now reach the place/release area,
but grasp retention is not reliable and the free-space policy degrades as soon
as the object-approach/grasp phase begins. This is a task-target scaffold, not
task success; the next gate is grasp retention plus collision-aware
base/cabinet planning, followed by DAgger from the loaded task data.

## Smoke result


```bash
uv run python scripts/smoke_wbc_robocasa_panda.py \
  --checkpoint logs/eval/wbc_multirobot_20261126/pandaomron/best.pt \
  --backoff 0.45 --layout 1 --style 1
```

- Scene: `PickPlaceCounterToCabinet`, layout 1, style 1.
- Target: 8 cm above the ready end effector.
- 100 control steps at 20 Hz.
- Tracking: **1.31 cm position / 0.029 rad orientation** → contract success.
- No robot contact with the kitchen at the final step.

The direct plant reaches the same target to ~0.5 cm; the remaining gap is the
scene contact/placement and the approximate torso hold.

## Limitations

- This is not task success: the policy does not yet receive object/receptacle
  targets, grasp/lift sequences, or base goals from the benchmark task.
- The base is held; combined base+arm benchmark skills are not wired.
- Stretch, Sourccey, and RB-Y1 RoboCasa configs use the emet merged-scene path,
  not a native Robosuite robot. Their WBC adapters need the merged model (or the
  same direct-write pattern against the merged MJCF) before benchmark runs.
- The kitchen layout/backoff is scene-specific; a task-target adapter should
  choose a collision-free stance per task rather than a fixed 0.45 m.

## Next

1. Add collision-aware base/cabinet planning and a place approach that does not
   require the arm to reach through the cabinet front.
2. Run a small RoboCasa task DAgger/fine-tune from the loaded task data; the
   free-space policy currently fails after pregrasp.
3. Improve grasp retention: the teacher reaches the object but the object is
   not consistently lifted, so the task bridge is not yet a task-success gate.
4. Reuse the direct-write adapter for the merged Stretch/Sourccey/RB-Y1 scenes.
