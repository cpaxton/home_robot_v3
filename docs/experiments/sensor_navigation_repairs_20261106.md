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

The first centre-ray oracle at `b35ba00f` uses the wrong Y boundary and fails
before q2 launches (job `20261106_080022_ed2f5e`). The native routine first
converts Y with `height-v-1`, so the opposite queries are `(u,v)` and `(u+1,v-1)`.
This is explicit in the [native implementation](https://github.com/facebookresearch/habitat-sim/blob/main/src/esp/gfx/RenderCamera.cpp).
An independent CPU test pins that convention. The oracle is corrected without
changing sensor calibration or navigation. A 1e-4 unit-depth ray tolerance
accommodates native float32 far-plane unprojection (<0.02 pixel here); the
half-pixel convention error is much larger. Preserve the failed oracle result.

## Exploration effort and fixed-turn control

The corrected ray oracle passes at `355bd05f` (maximum centre-ray error 6.74e-6,
all 176 original RGB frames matched). Distance-only q2 reaches **both** goals
but exhausts 200 actions during its last fixed camera scan. Removing obsolete
intermediate grid yaw reduces the second arrival from frame 199 to 190; the
full prescribed collection still needs six more scan actions. These remain
failed 200-action collections, not successful episodes.

The task-independent collector now ranks the *executed prefix* by drive length
and turning angle in calibrated primitive equivalents (0.1 m and 5 degrees).
Previously its closest frontier could require a half-turn even when a reachable
frontier lay ahead. This general control-cost objective uses no question, target
label, answer, scene identifier or case-specific parameter. Distance-only remains
an explicit `--frontier-policy distance` control. Record the chosen policy and
all candidate scores. This is an exploration-policy change, separate from the
camera-calibration repair; do not attribute its outcome to memory or VLM quality.

Predeclare q2/q6/q12 again, seed 0, `--frontier-policy motion_effort`, with the
same 200 actions, 240 seconds, two prefixes and all fixed scans. Also run the
independent q6 `check_grapheqa_navigation.py --turn-deg 180` gate: fresh translation,
fixed half-turn, and unknown-goal rejection. New frontier choices must not be
used as a substitute for proving that the original turning capability works.

## Acceptance result: all bounded navigation gates pass

All three effort-policy runs use frozen `818066c9`, seed 0, the same calibrated
camera/body/map, 200 attempted primitives, 240 seconds and the complete scans.
The raw results remain under `/home/cpaxton/runs/emet/sensor-nav-repairs-20261106/`.

| Scene | Complete legs / 2 | All scans | Actions / 200 | Frames | Wall s |
| --- | --- | --- | --- | --- | --- |
| q2 | 2 | yes | 137 | 140 | 48.70 |
| q6 | 2 | yes | 138 | 141 | 48.98 |
| q12 | 2 | yes | 139 | 142 | 50.65 |

Jobs: `20261106_081902_30dd25`, `20261106_082100_42c048`, and
`20261106_082246_8ece00`. All **423** frames have accepted registration and zero
reported collisions. Six planned prefixes each stay within 0.75 m. Maximum
measured arrival residuals are **24.19 mm** and **0.383 degrees**, within the
unchanged 25 mm / 3 degree route tolerances. These are relative to odometry,
not ground-truth localization accuracy.

The independent q6 control passes fresh translation, the fixed 180-degree turn,
and unknown-goal rejection before motion. The turn finishes with 2.40 mm position
residual and 5.84e-7 rad yaw residual, using 58 motion primitives across the
execution controls. Its scorer-only reconstruction matches all 176 original RGB
frames and records zero true base displacement during turns. No simulator state
enters an agent or model input.

The final CPU regression run has **167 passing pytest tests**, with one Habitat
configuration test skipped only because the root interpreter lacks Habitat-Sim.
That exact test function also passes standalone in the existing Habitat
interpreter (using the root's installed pure-Python pytest module; no install or
environment rebuild). An additional 32 extracted-acceptance tests pass. Coverage
includes actual MuJoCo ray controls for floor/drop/raised surfaces, continuous
sweeps, obstacle/unknown rejection, calibrated rays, blocked motion, head arrival,
route efficiency, evidence packets and collector budgets.

The ordinary Habitat client also passes the existing full EQA-import-stack +
rendered-frame preflight at `bec398ac`, job `20261106_082811_b729cf`, with semantic
sensors explicitly disabled. The first direct invocation omitted the existing
Habitat launcher's library path and hit a CXXABI import-order error. Repeating
with `LD_LIBRARY_PATH=/path/to/habitat-env/lib` (as already used by
`scripts/install_habitat.sh` and acceptance runners) passes. Both logs remain;
no environment was rebuilt and no import-order workaround was added to inference.

### Recomputable evidence and figures

`paper/data/sensor_navigation_repairs_20261106/summary.json` records the validated
matrix, controls, source pins and artifact/figure hashes. Its adjacent
`validate.py` checks raw packet hashes, source/policy/budgets, every action and
tracking ledger, six measured arrivals, initial and arrival scans, renderer
oracle and ordinary-client preflight. It rejects both the original floor failure
and the later distance-policy budget failure as negative controls.

```bash
.venv/bin/python paper/data/sensor_navigation_repairs_20261106/validate.py \
  --root /path/to/sensor-nav-repairs-20261106 --output /tmp/navigation-summary.json
```

The `figures/` directory contains question-labelled uniform raw-RGB / estimated
map-path panels as PNG/PDF. Question captions are added only by offline analysis;
the collector never receives them. Manual inspection of all three confirms
visible movement and varied headings. It also shows the limits: q2 remains in
the living room, q6 does not provide an unambiguous ladder view, and q12 does not
enter the bedroom. The final q12 scan includes a nearby wall. These are human
observations, not ground-truth visibility annotations.

This closes the bounded navigation/collection gate. It does **not** establish
better EQA answers, target-view coverage, OVMM success, or full matched-agent
acceptance. Calibration-only replay, route execution and exploration-policy
changes are separate evidence. Keep the earlier failures and controls in the
record. Next bind the existing agent to the common physical/input contract for
task-conditioned acquisition and paired active evaluation.
