# S0 torso-collapse reproduction — 2026-09-06

The failed query-conditioned run `20260906_074618_5d4314` recorded camera
height declining from 1.43 m to -0.06 m during its initial eight-frame scan.
Those observations alone did not distinguish chassis tipping from torso bending.

A render-free MuJoCo control experiment uses the same merged default-table
model, fixes the base upright, commands torso1 to -30 degrees, and advances
eight intervals of 1,000 physics steps. It compares preserving commanded
joint targets against resetting targets to measured positions between intervals.
This isolates control behavior; it is not a replay of the full ZMQ trajectory.

| Target policy | First camera height | Last camera height | Base upright |
| --- | ---: | ---: | --- |
| Retarget measured posture | 1.379 m | 0.048 m | Yes |
| Preserve commanded posture | 1.379 m | 1.370 m | Yes |

Navigation called `_sync_actuator_ctrl_from_joint_positions`, clearing client
pins and adopting gravitational tracking error as the next posture target.
Repeated base commands therefore allowed torso drift to accumulate. Base motion
now reapplies the existing hold targets instead. Explicit pose initialization
and manipulation synchronization retain their existing behavior.

Tests cover velocity and teleport action paths and the above physics control:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  src/test/config/test_navigation_posture_stability.py \
  src/test/config/test_robosuite_load_utils.py -q
python -m pytest --noconftest src/test/simulation/test_robosuite_nav_world_clamp.py -q
```

`--noconftest` bypasses the simulation directory's automatic Robocasa asset
download fixture; these navigation tests do not need kitchen assets.
Result: 34 passed, one missing cached-merge asset skip.

Remaining: validate the complete ZMQ scan after the control fix and correct
camera aiming independently. In the merged model, torso1 motion leaves the
camera forward vector approximately +Y while changing image roll, so the
current head-tilt adapter does not actually aim down at the table. No learned
perception success is claimed by this control experiment.
