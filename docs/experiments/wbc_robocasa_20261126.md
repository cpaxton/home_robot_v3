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

> **Follow-up (2026-11-27):** Items 1 and 3 below are now implemented and verified in [`robocasa_pandaomron_pick_place_20261127.md`](robocasa_pandaomron_pick_place_20261127.md) (`11 / 11` dynamic-base privileged teacher `env._check_success()` across `PickPlaceCounterToSink` and `PickPlaceCounterToCabinet`, plus `--save-rollout` export for task-posture DAgger).

1. Add a task-target bridge from the benchmark/TAMP skill layer into
   `set_ee_target` (object grasp, lift, place), then sweep those Cartesian
   targets with the Cartesian teacher.
2. Run a small RoboCasa DAgger/fine-tune from `loaded_data.npz` if the
   task-target tail crosses the 2 cm tolerance.
3. Wire base goals so the WBC can approach the counter instead of using a fixed
   backoff.
4. Reuse the direct-write adapter for the merged Stretch/Sourccey/RB-Y1 scenes.
