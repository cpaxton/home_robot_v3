# Herman captured-scene rehearsal — 2026-11-17

This is an offline simulator rehearsal based on Herman's actual reported arm,
head and end-effector frames, head stereo images and depth calibration. No motor
commands, driver starts, active deployments or physical repositioning were sent.
It exposes a deployment distribution gap; it does not qualify a controller for
hardware.

## Capture and geometric checks

An eight-second subscriber-only ROS capture used the robot's existing
Zenoh/domain-0 configuration. It received 1,598 arm states, 240 odometry states,
64 synchronized left RGB/depth frames, 115 right images and 31 lidar scans. The
selected arm and depth captures differ by **3.947 ms**. No wrist image arrived and
no wrist optical TF was available. The live URDF was captured and checksummed.

At the selected capture, the reported arm joints are approximately
`[1.593806, -1.745670, 1.563126, 1.090660, -0.019942]` rad, with gripper
`0.088971` rad and head tilt zero. Published joint1 and joint2 limits disagree
with that parked state by **1.318° and 10.019°**, respectively. The pinned model
has the same limits. This is a state/declared-envelope discrepancy, not proof that
physical hard stops were exceeded. No limit is widened and no captured state is
silently clamped. The captured posture is displayed using static FK, with **zero
physics steps** and a blocked admission result.

Live-URDF FK and the simulator agree with the paired head, hand and stereo TF to
floating-point precision. These are consistency checks against the same reported
joint/model chain, **not independent physical calibration** of the robot.

A second model issue is visible at this posture. Direct MuJoCo geom-distance
queries report **12.920 mm penetration** for the conservative link3 collision
proxy against the chassis, versus **0.112 mm separation** for the source CAD
convex hull. That tiny CAD gap is not a physical-clearance guarantee. The
canonical proxy, collision masks and safety checks remain unchanged. Review the
stow envelope and physical geometry before selecting a real initial posture.

The observed depth encoding is signed `16SC1`. Read-only inspection of
`maurice_cam/depth_estimator/publishing.cpp:134–175`, SHA-256
`9b862dccbffa958b8e9f4e5faa044719f57ace694cdefe35a5d3e09059fb978f`, establishes
millimetres per positive count; nonpositive values are invalid. That pins source
semantics, not the installed binary or physical depth accuracy. Only **18.715%**
of pixels have valid positive depth. Scaled rectified **P**, inverse rectification
**R**, and image-capture-time optical TF generate measured surface points.
Camera calibration is checked by rendering a known off-axis point, including
principal-point offsets, and by comparing the full optical mount transform.

Observed points become 6 cm collision boxes over a bounded near-robot region.
Holes and space outside the view remain unknown. The canonical floor remains an
explicit uncalibrated assumption; there is no inferred floor-to-ceiling wall.
The selected lidar scan is **137.9 ms** from the arm capture, so the final attempt
omits it rather than treating it as a paired room observation. Earlier local
attempts retain their results separately.

Scene points and test goals use the captured local `base_link` frame, with yaw
normalized to zero and the canonical 3 mm chassis/floor offset added. They do not
reuse the physical robot's global odometry coordinates as simulator world poses.

## Explicit synthetic start and matched tests

Because the captured state is blocked, a separate static staging search retains
all rejected candidates and selects a pose within the existing joint limits,
without scene collisions, with at least 10 mm proxy/chassis clearance and 10 mm
support margin. It is a **synthetic proposal**, not a robot reposition trajectory.
The admitted arm posture is
`[1.5308, -1.5308, 1.063126, 1.090660, -0.019942]` rad. Relative to the capture,
that changes the first three joints by roughly `[-3.61°, +12.31°, -28.65°]`.
This is meaningfully different from the parked pose and requires its own physical
reposition validation. Its proxy/chassis clearance is 12.399 mm and nominal
support margin is 28.422 mm.

Six unfiltered cases test static hold, two small arm motions, opposite base yaw
requests (±0.05 rad), and a two-stage coupled handoff. Joint-space witnesses
produce reachable Cartesian paths with the existing smooth reference compiler;
geometry/support admission still runs. Each teacher/actor receives the same
initial-state and reference hashes, configured limits and scene. These are
unloaded tracking cases, not grasp/carry/contact qualification.

The source actor's original canonical plant identity is verified before loading.
The partial observed scene and calibrated camera produce a distinct fingerprint;
running the frozen actor there is an explicitly labeled transfer experiment.
The normal actor observation supplies history, masked preview and reference
velocity. No privileged critic features or task-specific correction is provided.
Zero initial simulator velocity is a hypothesis; published real velocity units
remain unverified.

All paths enforce a measured 5 cm XY / ±47° yaw boundary. A native fault is never
re-armed. Executed episodes include a bounded physical stop check; all failures
and rejected proposals remain in the denominator. Pose/velocity traces are
recorded before terminal stopping so a final stopped state cannot hide active
wandering.

## Results and interpretation

| Controller | Admitted | Success | Confirmed simulator stops |
| --- | ---: | ---: | ---: |
| Analytic teacher | 6/6 | 6/6 | 6/6 |
| Previous long continuation | 6/6 | 0/6 | 6/6 |
| Goal-change continuation | 6/6 | 0/6 | 6/6 |

There are no wheel-lift or unsafe-contact flags in these 18 unloaded episodes.
The older actor hits the translation boundary on five cases; the newer actor on
both yaw cases. Its four arm-active cases end with **0.46–1.59 mm** hand errors,
but fail quiet settling. For example, the last active hold sample has 0.515 mm
position error while an arm joint still moves at 0.0743 rad/s and base yaw at
−0.0215 rad/s. Final accuracy therefore does not establish a quiet stable hold.
The final room model contains 588 observed boxes. The staging search preserves
22 rejected candidates before one accepted candidate; no physical staging path
is claimed.

See the compact [result record](../../paper/data/whole_body_control/herman_rehearsal_v1/results.json)
for the final source/artifact identities, all case outcomes and active-loop
measurements. The two checkpoints are the previous long continuation and the
completed goal-change continuation. Neither is promoted. The prior 1,604-test
training receipt belongs to its frozen source and does not qualify this new
scene-adapter code.

The nearby staged posture is already enough to expose learned-policy failures
on cases the analytic controller executes. Strong final hand accuracy alone is
insufficient: continued arm/base motion and boundary violations remain failures.
This rehearsal should become a dedicated deployment-envelope regression family
before any additional actor is presented as ready for Herman. Keep held-out
postures and target sequences separate from any training examples.

Next: reconcile published versus actual stow limits and the conservative link3
proxy; select and validate a physical staging path; expand reachable posture/hold
and handoff coverage around that admitted envelope while preserving the general
training mixture; then compare checkpoints on the same recorded-scene cases.
Driver feedback units/read validity, robot-side watchdog and measured real stop
are independent prerequisites. More PPO does not verify those interfaces.

## Run and inspect locally

Raw images, depth, reconstructed room geometry, MuJoCo binaries and replay NPZ
files stay under ignored `logs/`; do not upload home imagery/geometry in a PR.
The repository fixture contains only sanitized robot/calibration metadata.

From the canonical controller worktree, capture without actuator publishers:

```bash
PYTHONPATH=src .venv/bin/python scripts/capture_herman_rehearsal.py \
  --ssh jetson1@192.168.1.43 --seconds 8 --output logs/captures/herman-new
```

Run the documented captured fixture with one simulation worker and numerical
threads limited on this unstable workstation:

```bash
env -u WANDB_API_KEY PYTHONPATH=src CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 15 .venv/bin/python scripts/rehearse_herman.py \
  --capture logs/recovery/herman-rehearsal-v1/capture-v2 \
  --output logs/eval/herman-new \
  --policy logs/manipulation/long-finetune-v1/continued/checkpoint-65536/actor \
  --include-proposals
```

Fresh output directories are required. `artifacts.json` binds the completed
report, model, photos and individual replays. The runner checks that source hashes
have not changed during evaluation; the viewer verifies all replay checksums
before loading. A checksum inventory detects accidental changes, not malicious
rewriting of the entire inventory.

```bash
PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 15 .venv/bin/python -m emet.visualization.herman_rehearsal \
  --rehearsal logs/eval/herman-rehearsal-v4 --port 8771
```

Open **http://localhost:8771**. Over SSH, forward from your laptop:

```bash
ssh -N -L 8771:127.0.0.1:8771 cpaxton@olympia
```

Choose a recorded controller/case, scrub or play, and use the orbit view to
inspect motion. The real image stays fixed at capture time; the head simulation
moves with each recorded posture. The wrist view is visibly model-only. This
viewer loads recorded states with `mj_forward`; it never steps physics, opens a
robot connection or accepts motion commands. The separate live passive console
remains on port 8770.

Calibration references: [ROS CameraInfo contract](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg),
[MuJoCo camera frustum implementation](https://github.com/google-deepmind/mujoco/blob/3.5.0/src/engine/engine_vis_visualize.c).

Validation: **169 distinct targeted/regression tests pass**, plus a repeated
seven-test viewer check after display illumination changed. This includes reset,
canonical model identity, compressed depth/calibration, optical projection/mount,
source binding, command guards, fault holds, goal transitions, retention, console
and replay endpoints. Chromium desktop/mobile interaction checks pass. Runtime
package parity and Ruff pass. The initial targeted test attempt had one test
fixture missing depth dimensions; the corrected tests and that failed receipt
are retained locally. No new full training qualification receipt is claimed.
[Compact validation record](../../paper/data/whole_body_control/herman_rehearsal_v1/validation.json).
