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
  skill/TAMP layer can call `set_ee_target(world_or_local_pose)` for real task
  targets.

The fail-closed `make_robocasa_backend("pandaomron")` hook is intentionally
unchanged: the adapter requires an env and a task target provider, which the
generic factory cannot construct by itself.

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

1. Add a task-target bridge from the benchmark/TAMP skill layer into
   `set_ee_target` (object grasp, lift, place).
2. Wire base goals so the WBC can approach the counter instead of using a fixed
   backoff.
3. Reuse the direct-write adapter for the merged Stretch/Sourccey/RB-Y1 scenes.
