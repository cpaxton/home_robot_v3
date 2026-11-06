# Sensor navigation repair and live acceptance

Follow-up to [the bounded collection failures](exploration_history_20261106.md).
Retain every original/seeded failure. This is a physical/input-contract repair,
without Qwen inference, answer changes, or scene-specific motion policy.

## Diagnosis and repair

The reported camera principal point was (320, 240) for a 640x480 raster. OpenGL
samples pixel centres at framebuffer coordinates (u+.5, v+.5); OpenCV arrays
index their centres at integer (u,v). Read the renderer's projection matrix and
convert that convention, including off-axis projections, instead of assuming
the focal length and principal point. The same helper is used by the sensor-only
adapter, ordinary Habitat simulator client and learned Hydra capture probe.
Existing saved packets keep their original calibration and provenance.

On the exact saved RGB-D trajectory, changing **only** the principal point by
(-.5,-.5) with the original pairwise estimator leaves q2's diagnostic floor cell
certified. Its initial full-turn estimated XY displacement falls from 22.85 to
8.02 mm. q6's second turn displacement falls from 25.17 to 9.34 mm. These are
replays on fixed images, not live acceptance or ground-truth error measurements.
An exploratory local-keyframe implementation also helps the repeated q2 view,
but is not retained: the calibration bug has a smaller, independently testable
repair. Save the experimental driver/outputs without promoting that design.

The retained odometry initializer check handles blocked motors: choose identity
as the optimizer initializer only if measured overlap, depth and appearance all
dominate the command prior. Always solve and validate the resulting registration;
neither identity nor a commanded pose becomes an unchecked observation.

Expose floor height conflicts separately from missing returns in failed
footprint receipts. Both remain uncertified floor. Drop/raised/occupied checks,
the 30 mm floor band, 25 mm motion tolerance, and collection budgets stay fixed.

## Declared acceptance

1. CPU tests: projection/raster ray equivalence (also off-axis), blocked motion
   over repeated captures, missing depth/tracking loss, floor conflicts, true
   drops/raised obstacles, unknown-footprint admission and bounded collection.
2. Scorer-only q6 replay: execute its recorded primitives without a controller,
   require exact raw RGB agreement, then read simulator base motion to separate
   physical displacement from odometry drift. Privileged states are saved only
   under a separate scorer directory and never reach agent tools/model inputs.
   Save the native projection and unprojected-ray comparison for calibration.
3. Fresh q2, q6, q12, in that order, seed 0, same fixed collector. Each must
   complete two 0.75 m prefixes and both arrival scans within 200 attempted
   primitives / 240 seconds, with accepted tracking and no reported collision.
   Retain failed rows and diagnose before subsequent repair; do not choose a
   favorable seed, goal, answer or relaxed threshold to obtain acceptance.

Use one managed GPU job at a time, CPU 12, single-thread libraries, offline
assets and 20 GiB disk reserve. RL retains admission priority. No real robot
motion, model downloads, environment rebuilds or full benchmark sweep.

Passing this gate qualifies the bounded sensor/navigation collector, not full
EQA/OVMM quality. The subsequent paired active comparison still uses the existing
agent and must match pose, footprint, perception, map sampling and motion budgets.

## First live repair check

Job `20261106_075142_875ea8`, source `d48e516c`, reproduces every one of the
176 original q6 RGB frames in the scorer replay. The simulator base never
translates during its turns. The old turn residual is therefore an odometry
error, not actual creep. Native `unproject` addresses integer viewport boundaries;
its half-pixel difference from array-centred rays is recorded explicitly.

Fresh q2 passes the old floor blockage and reaches the same first target
(-0.5,-0.05) m with 0.050 mm position residual relative to odometry. It then
exhausts 200 primitives during its second leg (202 frames, 69.20 s). This remains
a failed full collection. The floor repair is useful but does not close the gate.

The grid route introduces many turn-drive-turn operations around small lattice
steps. Add a shared geometric shortcut operation: a direct drive segment replaces
a grid subpath only after checking the full turn/translation footprint sweep.
Keep the exact terminal XYT, the 0.2 m route-segment bound and fresh per-primitive
checks. Obstacles and unknown cells can force retention of the original corners.
The collector ranks paths by translation length instead of waypoint count so
compression cannot change its distance metric. Partial legs now clip the last
admitted segment to the full 0.75 m allowance rather than discarding that segment.
The 200/240 budgets and observation scans stay fixed; no shorter-distance gate
or relaxed clearance is introduced. Retain interrupted-motion events as well.

Repeat all three scenes at the resulting frozen source, with the same seed/order.
The scorer also checks native raster-centre rays by averaging opposite viewport
boundary rays on the unit-depth plane; do not confuse those two conventions.
