# Multi-robot WBC MuJoCo backend matrix

Date: 2026-11-25

Scope: direct-MuJoCo WBC plants for Mars, Stretch, Sourccey, and RB-Y1, plus
PandaOmron profile/RoboCasa hook. Mars remains the only trained WBC result.
This is simulator-only.

## Robot matrix

| Robot | Profile | Direct MuJoCo backend | Teacher smoke | DAgger status |
|---|---|---|---|---|
| Innate Mars | `mars_sim_profile()` | `MuJoCoWbcBackend` | 3/3 | trained: 0.973 held-out |
| Stretch 3 | `stretch_sim_profile()` | `StretchWbcBackend` | base only (1/3) | not run |
| Sourccey | `sourccey_sim_profile()` | `SourcceyWbcBackend` | 3/3 | 8 iterations, 0.33 |
| RB-Y1 | `rby1_sim_profile()` | `Rby1WbcBackend` | 3/3 | 8 iterations, 0.33 |
| PandaOmron | `pandaomron_sim_profile()` | composed Omron+Panda torque plant | base teacher 5/5; arm not settled | none |

Teacher smoke is one episode per subtask (`arm`, `base`, `combined`), not a
qualified result. Sourccey/RB-Y1 DAgger used the Mars-scale 4k teacher / 2k
actor / 40 epoch / 8 iteration configuration.

## What landed

- `wbc_robots.py`: robot/profile registry and RoboCasa names.
- `wbc_backends.py`: direct-MuJoCo vs RoboCasa factory.
- `wbc_robocasa.py`: explicit registration hooks; missing adapters fail closed.
- `wbc_stretch.py`: wheel/freejoint plant through `ManipulationControl`; five
  contract motion axes plus gripper.
- `wbc_sourccey.py`: planar base, lift, two 5-DOF arms, two grippers.
- `wbc_rby1.py`: Galaxea R1 proxy; freejoint base + three steer/wheel modules,
  4-DOF torso, two 6-DOF arms, two grippers.
- `pandaomron_sim_profile()`: Omron holonomic base, 1-DOF torso, 7-DOF Panda
  arm, two-finger gripper.
- `wbc_pandaomron.py`: composes the Robosuite Omron base + Panda arm + gripper
  into a direct MuJoCo model. The arm/torso are torque motors, so the plant uses
  `TorqueAdapter`; the base is velocity-controlled. The visual floor is
  collision-disabled because the Omron base has no wheels and otherwise pins
  against the floor.

## Observed limitations

- **Stretch:** the PD teacher tracks the arm but does not settle within the
  0.02 m / 0.05 rad contract tolerance in 3 s. Base-only succeeds. Needs an IK
  teacher or longer/looser curriculum before DAgger.
- **Sourccey / RB-Y1:** teachers reach 3/3, but plain BC and 8-iteration DAgger
  stay at the base-only 0.33 aggregate success. The likely cause is the
  high-dimensional dual-arm/torso IK mapping and insufficient demonstration
  coverage; the actor fits teacher states (BC MSE 0.006–0.008) but does not
  generalize to reset states.
- **RB-Y1:** the packaged wheels are visual-only; the backend adds wheel
  collision proxies and a steer/wheel swerve mapping. The direct plant is a
  training proxy, not RoboCasa benchmark parity.
- **PandaOmron:** direct model is composed from Robosuite assets; the Omron
  base teacher passes 5/5, but the torque PD arm teacher still oscillates
  around the orientation/position tolerance. RoboCasa's benchmark composite
  controller is OSC/JOINT_POSITION; the adapter and trained policy do not exist
  yet.

## Next experiments

1. Single-arm variants for Sourccey/RB-Y1 (`ee_target` on one arm, other held)
   to reduce action dimensionality before dual-arm/combined.
2. More teacher demonstrations and/or IK teachers for Stretch and the dual-arm
   tasks; actor BC MSE is not the bottleneck.
3. RoboCasa adapters for RB-Y1 and PandaOmron using the same profile/checkpoint
   contract.
4. Then rerun the subtask DAgger gate on each robot.

Artifacts for the Sourccey/RB-Y1 smoke runs are under
`/tmp/opencode/wbc_sourccey_full/` and `/tmp/opencode/wbc_rby1_full/`; they are
not promoted.
