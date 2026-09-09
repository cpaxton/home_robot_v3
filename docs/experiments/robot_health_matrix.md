# Cross-robot scene health checks

## Current run and preliminary findings

The active replacement is `20260906_163000_725ac2`, source `e9a83300`, output
`/tmp/emet-robot-health-matrix-v2-20260906`. Results are pending.

The first matrix below was cancelled after detecting a probe measurement bug:
raw camera Y is not image-up when signed intrinsics encode a vertical flip.
The corrected probe uses `R @ inverse(K) @ [0, -1, 0]`, with regression tests.
Innate Mars's first-run inversion flags are invalid, not robot failures.
The worktree also now links the installed MolmoSpaces venv; the early rby1
and Sourccey Molmo rows failed environment discovery and require reruns.

Valid first-run evidence:

- rby1/Robocasa: base remained upright (>0.9999 up dot), commanded torso
  targets stayed fixed, but torso joints diverged and camera height fell from
  1.617 m to 0.333 m by pose 2. This is a remaining dynamics/contact/tracking
  failure, distinct from the corrected target-retargeting bug.
- Sourccey/Robocasa: completed all nine captures (baseline plus eight turns),
  camera height remained 0.853 m. This passes the gross posture screen only.
- Stretch/Robocasa: partial captures before cancellation had stable 1.299 m
  camera height; generic base/target telemetry was missing. Not a completed row.

## Scope

Job `20260906_162502_6917ea` runs source `25492216` sequentially on the
CPU-safe allocation with numerical library threads set to one and an exclusive
GPU lock. No learned perception models, hardware commands, or full benchmark
sweeps. Results live in `/tmp/emet-robot-health-matrix-20260906`.

Each row uses eight relative base turns after navigation posture/look-front,
or eight idle intervals for fixed-base Franka. Scenes are frozen to Robocasa
`PickPlaceCounterToCabinet`, layout/style 1, seed 0, and MolmoSpaces iTHOR
train index 0, seed 0. Each case is bounded to 420 seconds.

| Robot key | Motion probe | Notes |
| --- | --- | --- |
| rby1 | Base turns | Primary torso-collapse regression |
| sourccey | Base turns | Planar base |
| innate_mars | Base turns | Simulation only; Herman untouched |
| stretch | Base turns | Separate server; generic posture telemetry may be unavailable |
| xlerobot | Base turns | `xlerobot_dual` currently resolves to the same backend/spec |
| nori | Base turns | Scene-loader support must be measured, not inferred from registry |
| galaxea_r1 | Base turns | Shared rby1 model; checks launch-key compatibility, not independent embodiment evidence |
| franka_fr3 | Idle only | Fixed base, wrist camera; no mobile-navigation claim |

`rb_y1`, `nori_a3`, and `franka` are aliases, not extra embodiments.
`mobile_aloha` and `yor` are stubs without vendored MJCFs. Molmo-native floating
grippers and other IDs without an EMET backend/MJCF are outside this harness's
currently runnable set. They must not appear as passing rows.

## Evidence and interpretation

`scripts/probe_rby1_camera.py` now accepts any scene config and robot override.
The historical filename is retained. Each case produces:

- `sim.log`: server diagnostics, including existing actuator-control logs.
- `observations.jsonl`: base uprightness/position, measured joints, actuator
  targets, camera height/up/forward, RGB statistics, and depth statistics.
- `view_NN.png`: source RGB for visual inspection.
- `summary.json`: completed probe, detected failure, or incomplete telemetry.

The probe stops on motion exceptions, an inverted mobile head camera, a base
tilt beyond approximately 55 degrees, or camera height loss above 0.3 m.
These are gross-failure screens, not calibrated robot safety specifications.
Missing telemetry is explicitly incomplete. `completed_probe` is not an
overall benchmark pass: inspect yaw achievement, drift, target preservation,
camera axes, and depth visibility from artifacts. A load failure or timeout
may have no summary; `CASE_RESULT` in the parent job log preserves its exit code.

Still required beyond this first matrix: translation/collision checks, head
aiming response, wrist/point-cloud registration, and task-level manipulation.
Do not infer success in those areas from stable camera height.

Example single-row reproduction (use the installed simulation Python):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/probe_rby1_camera.py \
  --sim configs/sim/robocasa_pick_place_rby1.yaml --robot sourccey \
  --poses 8 --port-offset 610 --output-dir /tmp/sourccey-health
```

The worktree uses a local, untracked symlink to the existing Robocasa asset
checkout. The initial job `20260906_162108_93817e` failed asset discovery before
this link was established; that was an environment failure, not a robot result.
