# Native public-tool integration — 2026-10-13

Status: experimental integration, **not merge-ready acceptance**. Native execution
is wired through the existing public JSON tools on a stationary Stretch table
scene, with explicit privileged inputs. Observed exploration remains unavailable.

The earlier contact baseline (#207) is 6/6 scripted actuator transfers. Its
controller path and fixture are different from this live ZMQ tool experiment;
those results must not be pooled.

## Live attempts and diagnosis

Artifacts are under `/home/cpaxton/runs/emet/` on the experiment host.

| Run | Source | Result | Finding |
| --- | --- | --- | --- |
| `tamp-native-agent-20261013-r1` | `cac71267` | Rejected before discovery | Generic merged model differed from the packaged Stretch scene; model binding caught it. |
| `tamp-native-agent-20261013-r2` | `568e1c5e` | Discovery passed; planning rejected | Initial planner diagnostics hid the underlying collision. |
| `tamp-native-agent-20261013-r3` | `736159d1` | Discovery passed; grasp planning rejected | Saved replay showed the left finger intersecting the neighboring cylinder by about 3 cm. |
| `tamp-native-agent-20261013-r4` | `1e495a5c` | Discovery and planning passed; execution partial at lift | Separated-object fixture; client “stopped moving” heuristic rejected a 0.028 joint residual within the physical executor's 0.03 tolerance. 75 audited commands, zero forbidden commands. |

Runs r1–r3 use the original cluttered table. Run r4 changes only the distractor's
**initial** X position to 0.3 m in the scene definition, matching the separated
contact-baseline arrangement. It performs no runtime object relocation. The
original clutter remains an unresolved negative case: bounded grasp orientations
find a grasp, but a full stationary placement plan has not passed.

Runs r5 (`9e3d2e21`) and r6 (`aea983b1`) also failed at lift on the separated
fixture. R5 passed measured arrival but crossed the unchanged retention limit:
6.0 mm translation and 0.10033 rad rotation relative to the grasp reference.
R6 added endpoint settling and cumulative gripper-drift rejection; it stopped on
wrist-pitch tracking error of 0.03169 rad (limit 0.03). R5/R6 had 74/72 audited
commands respectively, with zero forbidden commands. These are **zero completed
native tool transfers**, not new successes or regression percentages.

Offline inspection of the r6 snapshot found bilateral fingertip/object contacts
and no fingertip/table contacts. The object was about 7 mm above its initial
center height and tilted about 0.13 rad. The remaining issue is native loaded
tracking/grasp stability. Do not widen retention/tracking thresholds simply to
turn these trials green. Diagnose contact centering and controller behavior,
with separate tests for free motion, allowed object contact, and obstacles.

The shared benchmark launcher previously forced navigation teleport even when
physical execution requested it off. It now forces both teleport settings off
for physical runs and retains assisted defaults for other runs. The simulator
audit independently rejects forbidden actions. No forbidden actuation appeared
in these trials.

## Implementation and validation

- Physics-owned model signatures and timestamped qpos/qvel snapshots bind the
  privileged adapter to the actual executing model and server boot identity.
  Free-body initial poses are excluded from the geometry signature because
  measured qpos determines those poses; geometry and inertial properties remain bound.
- Planning uses private MuJoCo state, bounded grasp frames, the shared
  obstacle-subtracted support search, and dense arm/payload collision checks.
- Plans are one-shot; changed scenes and stale capture timestamps reject motion.
- The physical executor now owns native joint arrival checks. The client sends
  ordinary nonblocking commands; acknowledgement alone does not establish success.
  Measured convergence, base drift, and payload retention checks remain mandatory.
- Native articulation is unavailable rather than falling back to joint teleport.
  Observed mode rejects before reading privileged metadata.

At `f6e638f4`, **218 targeted tests passed, zero failures/skips**. Release regressions
cover wrong support height and a still-spinning object. JUnit and runner artifacts
are archived under `tamp-native-agent-20261013-r6/validation/`. The independently
reviewable executor fix is [#208](https://github.com/cpaxton/home_robot_v3/pull/208),
with 73 motion/contact tests passing on its own branch. Ruff passed on changed Python files.
A prior repeated model-compilation test run stalled inside MuJoCo; the test fixture
now compiles once and gives each test fresh state. A diagnostic CPU replay also
segfaulted; later diagnostic jobs used the managed CPU-safe launcher.

## Remaining acceptance gates

Complete live native transfers, including the cylinder and fresh repeats; retain
and resolve the cluttered negative. Then validate LLM task selection and consecutive
tasks before adding exploration with observed object and voxel geometry. See the
[native-to-exploration plan](../plans/2026-10-13_native_tamp_agent.md).

## Artifacts and reproduction

Each live run contains `protocol.json` (source SHA and exact command),
`process.log` (public JSON outcomes), `actuation.jsonl`, and `summary.json`.
Runs r3 onward also contain `native-debug.json` with privileged snapshots and
executor events. Separated runs archive their initial scene under `fixture/`.
These paths are local to the experiment host, not public downloads. Absolute
mesh paths in archived fixtures refer to the source checkout.

The tool sequence is the same as CHAT: `scene_tasks` filtered to the blue cube,
`plan_pick_place` using the returned task handle, then `execute_pick_place_plan`
using the returned plan handle. The runner uses `scripts/scripted_sim_pick_place.py`
with `--manip-mode physical --tamp-inputs privileged --physical-audit PATH
--native-debug-output PATH`. Launch live runs through `emet jobs run --need-mib 3000`
and retain the exact source checkout until the job finishes. No LLM is used in
these deterministic tool-routing tests.

## Execution follow-up: contact calibration and simulator timing

Controlled CPU replay isolated two effects. R6's loaded wrist tracked about
0.023 rad short after settling, versus about 0.005 rad without the payload.
The live simulator advanced only 7.32 physics seconds in a 46.4-second run;
wall-only motion deadlines therefore did not allow the intended settling time.
The native adapter now uses physics time for settling, with an independent
30-second wall watchdog. Hardware/default executors retain wall-clock timing.

The grasp also targeted the named EE frame 3 cm above the object center. That
frame is not the contact point. Even centering the tilted pad boxes leaves their
inward contact edges about 17 mm off-center, generating payload rotation under
load. Native grasp calibration now derives the opposing support edges from the
robot's collision geometry and centers those on the payload. Grasp IK uses a
3 mm positional target tolerance. Model gains, gripper preset, joint arrival
limits, and payload-retention limits are unchanged.

Managed CPU diagnostics `20261013_131331_d22d6e` and
`20261013_131416_eaddd3` completed a box transfer with the calibrated grasp.
The second is an ablation with feedback correction disabled; consequently the
experimental feedback correction was removed. These are in-process diagnostics,
not live ZMQ acceptance. Live retesting and both-shape repeats remain required.

Reproducible CPU regression (public JSON tool registry, in-process actuators):
`scripts/eval_native_table_contact.py --output PATH --object both --trials 3`.
Use `emet jobs run --no-gpu-exclusive --cpu-safe`. `--cluttered` preserves the
original object arrangement. The separated fixture only changes initial object
positions before physics; execution uses actuator commands exclusively. JSON
artifacts include tool responses, execution events, commands, and final measured
state. This does not test IPC, cameras, an LLM, navigation, or observed perception.
