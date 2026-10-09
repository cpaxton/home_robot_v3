# Model integrity review after placement-video inspection

The September 4 Sourccey video is **rejected as positive placement evidence**.
Its historical `success=True` was recorded before measured-arrival fixes and with
an invalid robot model. Object displacement alone does not establish attachment.

## Confirmed defects and fixes

- The URDF converter emitted revolute joints on parent bodies, rotating the wrong
  subtree about the wrong pivot. Emit each joint on its child link. Independent
  URDF FK now matches every arm link at zero and nonzero configurations.
- Right-arm frames were reflected but CAD mesh vertices were not. Reflect the
  mesh assets and inertia cross terms too; test actual world mesh symmetry, not
  merely body-frame symmetry. This fixes the visibly detached right arm.
- Coincident collision/visual meshes were both visible. Collision copies now
  use hidden group 3; collision properties are unchanged.
- Robot preview floors were included alongside the default environment floor.
  Remove direct robot-world preview planes only in the default scene assembler;
  preserve the source robot asset and the environment floor.
- External recording reused a renderer with head-camera self masks. Explicitly
  reset visible geometry groups for chase/overhead views. Hide rangefinder debug
  rays in generic RGB rendering, matching the native Stretch camera policy.

The corrected arm needs a different compact home posture; it is not certified
collision-free. These kinematic changes invalidate historical Sourccey baselines.
Both gripper-frame origins are now approximately 19 mm from their nearest visual
finger mesh vertices. This is a model sanity check, not proof of object contact or
retention. Actual grasp/attachment success still requires a new measured run.

## Stretch

A native MuJoCo lift/telescope/wrist diagnostic is available, with no object
attachment or placement claim. Reproduce using
`scripts/render_robot_model_diagnostic.py` (CPU-render command in its help).

Stretch's existing server/controller route does not advertise the kinematic TAMP
executor or base endpoint query. Its telescope is tendon-coupled, so assigning
four independent extension joints to the generic IK/streaming path would be
incorrect. A genuine Stretch placement port must map that coupling consistently
in IK, observed FK, command transmission and collision checks, then implement and
test measured attachment/release plus endpoint validation. A teleport-only video
would not address the user's complaint and is not presented as a substitute.

## Artifacts

Local review: `/tmp/emet-model-review/index.html`, before/after scene stills and
`stretch-model-motion.mp4`. These use Mesa llvmpipe, not GPU rendering. The
Stretch clip is labeled `model motion diagnostic` and `not a placement test`.

## First live corrected-model run

Managed CPU job `20261009_180642_71f2af` at `d029f355` saved video and failed
`pregrasp_ik_failed` (0.672 m error), before attachment or placement. Inspection
found the direct video harness hardcoded front-facing approach yaw even for a
side-reaching robot. It now uses the shared robot-specific approach helper. This
is a harness correction, not permission to count the failed run as success.
