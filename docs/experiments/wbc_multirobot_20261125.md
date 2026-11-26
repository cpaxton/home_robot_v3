# Multi-robot WBC MuJoCo backend matrix

Date: 2026-11-26 (started 2026-11-25)

Scope: direct-MuJoCo WBC plants for Mars, Stretch, Sourccey, RB-Y1, and
PandaOmron. This is simulator-only; none of these is a RoboCasa benchmark
result.

## Robot matrix

| Robot | Profile | Direct MuJoCo backend | Teacher | DAgger status |
|---|---|---|---|---|
| Innate Mars | `mars_sim_profile()` | `MuJoCoWbcBackend` | 1.00 | 0.973 held-out |
| Stretch 3 | `stretch_sim_profile()` | `StretchWbcBackend` | 1.00 | 0.84 held-out |
| Sourccey | `sourccey_sim_profile()` | `SourcceyWbcBackend` | 1.00 | full-suite BC 0.86; DAgger 0.86 in-suite; held-out pending |
| RB-Y1 | `rby1_sim_profile()` | `Rby1WbcBackend` | 1.00 | 0.996 held-out |
| PandaOmron | `pandaomron_sim_profile()` | composed Omron+Panda torque plant | 1.00 | 0.927 held-out |

Teacher success is a ceiling, not a policy result. Held-out numbers use 50
episodes per subtask on a fresh evaluation seed (30000).

| Robot | Arm subtask(s) | Base | Combined | Aggregate |
|---|---|---|---|---|
| Innate Mars | arm 1.00 | 1.00 | 0.92 | **0.973** |
| Stretch 3 | arm 0.90 | 1.00 | 0.62 | **0.84** |
| PandaOmron | right 1.00 | 1.00 | 0.78 | **0.927** |
| RB-Y1 | left 1.00, right 1.00 | 1.00 | left 1.00, right 0.98 | **0.996** |
| Sourccey | left 1.00, right 1.00 (BC) | 1.00 (BC) | left 0.70, right 1.00 (BC) | held-out pending |

The Sourccey row is the current full-suite BC/DAgger checkpoint, not a fresh
held-out gate yet.

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
  `active_arms` for single-arm subtasks. Resets at the raised, collision-free
  home posture with small joint noise; sampling the full arm range made the
  imitation target much harder.
- `wbc_rby1.py`: Galaxea R1 proxy; freejoint base + three steer/wheel modules,
  4-DOF torso, two 6-DOF arms, two grippers, optional `active_arms`. Also
  resets near the packaged home posture instead of sampling every joint range.
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

## What fixed the plateau

The earlier Sourccey/RB-Y1 0.33 aggregate was not an adapter bug. Three
curriculum choices were compounding:

1. **Full-range joint resets.** Sourccey and RB-Y1 sampled every arm and torso
   joint uniformly across its limits. The actor had to learn IK from an
   enormous 10–12-DOF pose distribution with a few thousand labels. Resetting
   near the collision-free packaged home pose made single-arm BC go from
   ~0.2 to 1.0.
2. **Too few BC epochs.** 30 epochs left the actor underfit; 200–600 epochs
   lowered the arm action error enough for closed-loop success.
3. **Simultaneous dual-arm targets.** Asking for both arms at once doubled the
   effective action dimension. The default suite now trains one active arm at a
   time; the inactive arm is held and publishes an empty target mask.

The combined subtasks remain the hardest because the actor must react to the
reach goal after the base stage.

## Observed limitations

- **Stretch:** the position servos are underdamped and slow; the arm and
  combined subtasks need a 5 s / 7 s window. The combined policy is the weakest
  part (0.62 held-out) because the base stage consumes most of the episode.
- **Sourccey:** full-suite BC reaches 0.86 with combined_right 1.0 and
  combined_left 0.7. A fresh held-out gate is still running; combined_left is
  the likely weak spot.
- **RB-Y1:** the packaged wheels are visual-only; the backend adds wheel
  collision proxies and a steer/wheel swerve mapping. The direct plant is a
  training proxy, not RoboCasa benchmark parity.
- **PandaOmron:** the direct plant now has a trained policy (0.927 held-out),
  but no RoboCasa adapter. RoboCasa's benchmark composite controller is
  OSC/JOINT_POSITION; the adapter must map explicitly. The composed model's
  zero arm posture self-collides with the wrist; the ready-pose reset and
  300/30 torque gains are required at 20 Hz.

## Hard cases

`wbc_hard_cases.py` exports recorded arm postures from the manipulation failure
sink into DAgger cases: the actor starts at the posture it actually reached when
the attempt failed, and the teacher is asked to reach the recorded target
posture. `--hard-cases` and `--hard-case-probability` inject them into arm and
combined subtasks; base-only subtasks are excluded. The first export is
`configs/eval/wbc_pandaomron_hard_cases.json` (six Panda sink/cabinet records).

These are free-space arm postures, not contact/carry geometry. Shelf-entry and
carry failures still need a dedicated carry subtask or an obstacle-conditioned
backend before they are fully represented.

## Next experiments

1. Finish the Sourccey held-out gate; if combined_left stays low, train a
   combined-left specialist or fine-tune with a higher combined weight.
2. Improve Stretch combined with a longer window or a two-stage curriculum;
   it is the weakest promoted policy.
3. RoboCasa adapters for RB-Y1 and PandaOmron using the same profile/checkpoint
   contract.
4. Consider a single embodiment-conditioned policy once per-robot policies are
   stable; the current artifacts are one policy per robot.

Artifacts:

- Mars: `logs/eval/wbc_dagger_features_20261125/`.
- Stretch: `/tmp/opencode/wbc_stretch_full/` (`best.pt`, `heldout_50.json`).
- Sourccey: `/tmp/opencode/wbc_sourccey_ready/` (full suite; fresh run),
  `/tmp/opencode/wbc_sourccey_arm_left/`, `/tmp/opencode/wbc_sourccey_arm_right/`
  (specialists, superseded).
- RB-Y1: `/tmp/opencode/wbc_rby1_multi/` (`best.pt`, `heldout_50.json`).
- PandaOmron: `/tmp/opencode/wbc_pandaomron_full/` (`best.pt`, `heldout_50.json`).
- The earlier dual-arm plateau runs remain under `/tmp/opencode/wbc_sourccey_full/`
  and `/tmp/opencode/wbc_rby1_full/`; they are superseded.
