# Multi-robot WBC MuJoCo backend matrix

Date: 2026-11-25

Scope: direct-MuJoCo WBC plants for Mars, Stretch, Sourccey, RB-Y1, and
PandaOmron. This is simulator-only; none of these is a RoboCasa benchmark
result.

## Robot matrix

| Robot | Profile | Direct MuJoCo backend | Teacher | DAgger status |
|---|---|---|---|---|
| Innate Mars | `mars_sim_profile()` | `MuJoCoWbcBackend` | 3/3 | 0.973 held-out |
| Stretch 3 | `stretch_sim_profile()` | `StretchWbcBackend` | 1.00 | 0.90 in-suite, 0.84 held-out |
| Sourccey | `sourccey_sim_profile()` | `SourcceyWbcBackend` | 1.00 on the single-arm suite | right-arm specialist 0.90; full suite pending |
| RB-Y1 | `rby1_sim_profile()` | `Rby1WbcBackend` | 1.00 on the single-arm suite | right-arm specialist pending |
| PandaOmron | `pandaomron_sim_profile()` | composed Omron+Panda torque plant | 0.93 (arm/base 1.0, combined 0.9) | not trained |

Teacher success is measured per subtask with 5–10 episodes at the evaluation
seed. It is a ceiling, not a policy result. Stretch's 0.84 held-out number is
50 episodes per subtask; the others are smaller gates.

## What landed

- `wbc_robots.py`: robot/profile registry and RoboCasa names.
- `wbc_backends.py`: direct-MuJoCo vs RoboCasa factory.
- `wbc_robocasa.py`: explicit registration hooks; missing adapters fail closed.
- `wbc_stretch.py`: wheel/freejoint plant through `ManipulationControl`; five
  contract motion axes plus gripper. Resets to a raised, collision-free ready
  pose (the packaged zero pose folds the wrist into the chassis), resamples
  targets that start against the chassis/table, and re-anchors the combined
  reach goal to the measured base pose after the drive.
- `wbc_sourccey.py`: planar base, lift, two 5-DOF arms, two grippers, optional
  `active_arms` for single-arm subtasks.
- `wbc_rby1.py`: Galaxea R1 proxy; freejoint base + three steer/wheel modules,
  4-DOF torso, two 6-DOF arms, two grippers, optional `active_arms`.
- `wbc_pandaomron.py`: composes the Robosuite Omron base + Panda arm + gripper
  into a direct MuJoCo model. The arm/torso are torque motors, so the plant uses
  `TorqueAdapter`; the base is velocity-controlled. It resets to Robosuite's
  Panda initial posture instead of the composed model's self-colliding zero
  pose, samples combined targets at the base goal, and re-anchors them to the
  measured base after the drive. The visual floor is collision-disabled because
  the Omron base has no wheels and otherwise pins against the floor.
- `wbc_dagger.py`: per-robot default suites. Dual-arm robots train single-arm
  variants (`arm_left`, `arm_right`, `combined_left`, `combined_right`, `base`)
  because a simultaneous 10–12-DOF reach is a much harder imitation target.
  BC defaults to 200 epochs; the first multi-arm runs used 30 and were
  undertrained. `--init` warm-starts from a profile-checked checkpoint and
  `--max-bc-samples` bounds the replay buffer.

## Observed limitations

- **Stretch:** the position servos are underdamped and slow; the arm and
  combined subtasks need a 5 s / 7 s window. The combined policy is the weakest
  part (0.62 held-out) because the base stage consumes most of the episode.
- **Sourccey / RB-Y1:** single-arm imitation works (Sourccey right-arm
  specialist 0.90 in-suite), but one multi-task actor covering both arms, base,
  and both combined variants is data-hungry. Per-task teacher data is the
  bottleneck; the earlier dual-arm 0.33 plateau was under-training and a
  simultaneous dual-arm target, not just action plumbing.
- **RB-Y1:** the packaged wheels are visual-only; the backend adds wheel
  collision proxies and a steer/wheel swerve mapping. The direct plant is a
  training proxy, not RoboCasa benchmark parity.
- **PandaOmron:** the direct plant tracks arm/base/combined teacher targets, but
  there is no trained policy and no RoboCasa adapter. RoboCasa's benchmark
  composite controller is OSC/JOINT_POSITION; the adapter must map explicitly.
  The composed model's zero arm posture self-collides with the wrist; the
  ready-pose reset and 300/30 torque gains are required at 20 Hz.

## Next experiments

1. Finish the Sourccey/RB-Y1 full-suite fine-tune from the arm specialists;
   report honest held-out numbers per subtask.
2. Add a combined specialist or staged resume if the full-suite actor still
   drops the arm skills.
3. Train a PandaOmron policy from the now-working teacher.
4. RoboCasa adapters for RB-Y1 and PandaOmron using the same profile/checkpoint
   contract.

Artifacts:

- Stretch: `/tmp/opencode/wbc_stretch_full/` (`best.pt`, `heldout_50.json`).
- Sourccey specialists: `/tmp/opencode/wbc_sourccey_arm_left/`,
  `/tmp/opencode/wbc_sourccey_arm_right/`.
- Sourccey full suite: `/tmp/opencode/wbc_sourccey_multi/`.
- RB-Y1 specialist: `/tmp/opencode/wbc_rby1_arm_right/`.
- The earlier dual-arm plateau runs remain under `/tmp/opencode/wbc_sourccey_full/`
  and `/tmp/opencode/wbc_rby1_full/`; they are superseded.
