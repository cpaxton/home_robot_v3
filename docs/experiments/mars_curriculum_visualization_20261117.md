# Mars curriculum: seven 4×4 videos and full target coverage

All seven videos are saved and served at
[localhost:8771/curriculum/](http://localhost:8771/curriculum/), using the existing
SSH forward. The [interactive 3D view](http://localhost:8771/curriculum/assets/coverage.html)
adds family/split filters, orientation axes, base paths, drag-to-orbit and zoom.
The same trajectories are exported as PNG/SVG. Reproduction, storage and
serving commands are in [CURRICULUM_VISUALIZATION](../rl/CURRICULUM_VISUALIZATION.md).

## What is recorded

Each video is a 1536×1232, 8 fps, real-time 4×4 mosaic of **actual analytic-teacher
simulation**, with 16 distinct case/seed pairs. The renderer loads checksummed
recorded qpos/models and calls `mj_forward`; it neither steps physics nor
animates the private FK witness in place of an executing robot. Per-cell raw
joint/base motion spans and decoded image differences verify physical motion
in all 112 displayed cells. Failures remain visible; no trials are replaced.

Cyan axes are the hand target, orange axes the measured hand. Blue base-goal
heading arrows are elevated on a pole so the chassis does not hide them. Base
videos report measured **base** XY/yaw error, with angle wrapping; other videos
report hand position/orientation error. Moving-base views are wider to preserve
framing. A shorter recording freezes at its last recorded state and is marked
END. All 112 stops confirm separately; general-case stop tails are not in the
videos, while TAMP traces include executor holds/stopping.

| Motion family | Teacher success | Failures retained |
| --- | ---: | --- |
| General hand/base | 15/16 | 1 unsafe-contact stop |
| Base-only | 13/16 | 3 tracking deadlines |
| Goal changes | 13/16 | 3 tracking deadlines |
| Small arm splines | 16/16 | None |
| Medium arm splines | 14/16 | 1 contact stop, 1 path-tracking failure |
| Full arm splines | 12/16 | 1 contact stop, 3 path-tracking failures |
| TAMP unloaded approach/grasp pose | 16/16 | None |

These are visual samples, not a benchmark success-rate estimate or a learned
policy improvement. Three contact faults, zero wheel-lift/invalid-physics flags.
Level zero is the initial PPO spline level. Medium/full remain future stages;
their failures prevent claiming those stages are reliable or advancing solely
from the count of geometry-admitted paths. TAMP stops before task-owned grip
closure: no loaded grasp/lift/carry/place is represented.

General/goal cases use their existing train/audit pools. Base has eight original
identities and TAMP six; their extra eight/ten examples use an explicitly
visualization-only 480M seed namespace. They never enter PPO. Spline videos
sample eight train/eight audit cases per level for inspection, rather than
claiming that PPO trains on audit data.

## Coverage findings

The plot covers **208 targets** (192 active hand paths and 16 base-only paths): all 144 declared spline identities plus 16
paths each for hand, base, goal change and TAMP. All 144 spline cases admit after
bounded geometric sampling, with rejected proposals retained; none exhausts
sampling. Base-only hand masks are false: their passive hand sweeps are excluded
from active hand position/orientation coverage metrics and hand path plots.
All 48 spline-video reference hashes match the corresponding full
coverage references exactly. General coverage includes both announced stages
for goal changes; TAMP coverage includes executed unloaded-prefix goals only.

| Measure | Observed |
| --- | ---: |
| Active hand X span | 0.380 m |
| Active hand Y span | 0.381 m |
| Active hand Z span | 0.368 m |
| Occupied 2 cm active-hand-position voxels | 1,053 |
| Maximum orientation distance from the first sampled pose | 106.9° |
| Spline starts per posture, across three levels | 36 |

This is much wider than the existing tiny joint-delta hand distribution, but
it still has **four posture-centered clusters**, with gaps between them. It is
not uniform reachable-workspace or SO(3) coverage. The full spline stage adds
vertical/curved motion; the existing hand/base/goal families are comparatively
flat. TAMP's unloaded-prefix cloud is particularly narrow. The visual evidence
supports retaining broader arm trajectories and later filling measured gaps,
while holding the initial teacher-reliable curriculum level fixed.

## Validation and provenance

Implementation begins at `28844d03`; display fixes preserve the same physical
recordings. Compact source/model/media hashes, sample outcomes and coverage
statistics are in
[results.json](../../paper/data/whole_body_control/curriculum_visualization_v1/results.json).
The media, models, raw traces and full target exports stay under ignored
`logs/eval/curriculum-videos-v1`, with original-camera media retained separately.
Compressed, content-deduplicated model snapshots avoid storing a separate raw
32 MB model for every tile. Raw sensor/captured-room data are not used here.

49 focused integration tests pass, including existing replay streaming, media
whitelisting/checksums, byte-range seeking, traversal/symlink rejection, distinct
case identities, coverage summaries, base-heading wrapping, source gates and
snapshot packaging. Ruff, runtime parity and diff checks pass. Decoded frames
were inspected, with per-cell image/physical motion checks. Chromium plays and
seeks the videos, verifies changing decoded pixels, displays all seven cards,
and passes mobile width, coverage filtering and orbit checks. Codec-specific
H.264 detection selects the saved VP9 WebM when that browser cannot decode H.264.

The separate frozen training source `614dfcd6` passes **1,679 full regressions**
and **48/48 small-spline teacher preflight cases**, all stops confirmed. Those
full regressions apply to that frozen training source, not later viewer edits.
The initial visualization job froze the simulator service and exposed a TAMP
IPC wall-clock timeout after 2,048 critic-only decisions. No actor update was
lost; preserve that failed run and logs. The same frozen source, teacher report,
regression receipt, initializer, reward and fixed budget were restarted in
`logs/manipulation/arm-splines-v1-restart`. Its matched policy evaluation remains
pending. Do not freeze live TAMP IPC; schedule rendering before PPO or stop at
an explicit checkpoint boundary. No hardware motion or policy promotion.
