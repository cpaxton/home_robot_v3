# Bounded local collision maps and mobile manipulation — November 17

Branch: `fix/tamp-observed-admission`, development base `41c0007a`.
The [compact JSON index](local_map_mobile_20261117.json) retains every scored
development attempt, cancellations and final source hashes; these retries do
not define a benchmark denominator.
This extends the [stationary mapped contact control](mapped_contact_20261117.md)
with bounded map construction/ownership and a mobile MuJoCo development episode.
It does not change default OVMM/Habitat behavior or qualify observed-only TAMP.

## Memory control

Separate CPU-only processes construct the same 2,304,000-cell, 5 mm workspace
from a 121×161 depth image with missing returns. The reference loads the
`ObservedWorkspace` source at `41c0007a`; the candidate batches geometry at
4,096 cells. Both produce cell SHA-256
`d92a78c67d499d93ec71e3808c4b56ee75e401d45fe132c0c90af70c367d4f4f`.

| Constructor | Peak process RSS | Construction wall time |
| --- | ---: | ---: |
| Reference, full-grid geometry | 1,877,245,952 bytes | 51.491 s |
| Batched geometry | 136,491,008 bytes | 51.076 s |

This is about 13.8× less peak RSS for this isolated constructor. It is one paired
measurement, with no speedup or full-agent memory claim. Runs used separate safe
CPU allocations concurrently. A whole robot episode includes its native model,
collision data and software renderer, so its RSS is reported separately.

Raw results: `.tamp-runs/local-map-ovmm-20261117/memory-{reference,chunked}/result.json`.
Managed jobs: `20261117_190820_9c4195` / `20261117_190820_04a233`.
The reference source is archived at the raw run root, outside tracked documents.

## Mobile development gate

The explicit assisted setup uses the existing simulated Stretch contact fixture,
normal arm/gripper actuators and wheel velocity actuators. Native state and
full-scene collision geometry remain privileged. A standalone simulated overhead
RGBD scanner provides background geometry with the task target absent before
release. It is not an onboard acquisition or observed robot-ownership result.
After release the target is included in the new scene scan.

The goal is unloaded approach → source scan → public-tool plan and grasp/lift →
fresh stopped-handoff scan → checked loaded transport → destination scan →
place/release/retreat → released-object scan → checked departure. A single-entry
64 MiB local scene store issues weak generation leases; pending plan handles are
cleared on scene replacement. Commands check their expected lease, and ongoing
base control checks it again. A mapless initial approach is explicitly checked
by native geometry. Live pose teleports are not used.

The loaded sweep checks full robot/payload geometry and the same declared
tracking envelope as unloaded approaches. It checks actual retention before
and after expensive preflight and throughout motion. These are sampled checks,
not continuous collision/dynamics guarantees. Model-specific wheel-controller
feed-forward compensates declared joint friction; it is not a force sensor.

### Retained failure

`20261117_190648_decddd` completed initial wheel navigation, transfer planning,
grasp and lift, then rejected the carry preflight with
`support_geometry_unknown`. Planning took 140.421 s and the carry check took
92.662 s; the original source scan's fixed 299 s static-evidence window expired.
No loaded carry command was dispatched. Its first-run tool scope fields still
use inherited stationary metadata; the outer record explicitly labels mobile
assistance. The scope fields are corrected before the subsequent repeats. The tool returned `partial`, completed
`["grasp"]`, failed stage `place`, recovery `inspect`. Its store released all
counted storage at teardown. This failure is not a successful mobile task.

The repair reacquires at the stopped loaded handoff before the carry sweep.
It does not renew the old scan or enlarge its lifetime. Review also added base
dispatch lease checks, checks before depth conversion/sanitation, and complete
post-release collision rejection telemetry.

Mobile outcomes and regression results are recorded below.
Raw artifacts stay under `.tamp-runs/local-map-ovmm-20261117/`; only compact
summaries belong in versioned documentation.

## Regression and reproduction

The focused map/background/support/release/payload/native-approach/adapter batch
passes **225 tests**, zero failures/errors/skips, in 219.714 s
(`20261117_191726_1d5155`). This includes loaded-payload-only obstacle rejection,
retention loss before/during wheel motion, expired/retired leases at base command
dispatch, failed reacquisition, same-key generation invalidation, weak-reference
release and pre-allocation budget rejection. These are regression counts, not
successful task episodes. Independent read-only review found no remaining blocker
in the changed lifecycle/loaded-sweep code. Ruff is clean on changed Python.

Prepare the declared development fixture with
`python scripts/prepare_observed_stretch_fixture.py --output OUT/fixture`.
Run the software-rendered experiment through the managed supervisor, with the
local OSMesa library configured, for example:

```bash
emet jobs run --name mobile-mapped --no-gpu-exclusive --cpu-safe \
  --out-dir "$PWD/OUT/mobile" -- \
  env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/eval_mobile_mapped_contact.py \
  --fixture OUT/fixture --output OUT/mobile --object object2 --resolution 0.005
```

Use the active checkout's environment (`uv run emet`/`uv run python` where
appropriate). `--object object1` runs the cube case; the default is `object2`
(cylinder). The two supported map resolutions are 0.02 and 0.005 m. Original
acquisition IDs and scan arrays, tool receipts, command/event/physics traces,
per-phase wall time, counted resident scene storage and cumulative kernel peak
RSS are archived. Physics advances on controller sleeps in this in-process
facade; these wall timings do not qualify asynchronous real-time execution.
`eval_local_map_memory.py --output RESULT.json` reproduces the isolated candidate
constructor; `--reference-source PATH.py` runs an archived reference in its own
process. Keep the two measurements separate from the whole native episode.

### Carry repeat and release rejection

`20261117_191624_d26a9e` passes the real loaded carry after handoff reacquisition:
measured base XY is `[0.136148, 0.000181]` m and yaw `0.0000813` rad, with
confirmed stop and retained payload. Carry validation/execution takes 135.782 s.
The source, carry and destination maps each take about 24 s to acquire/build.
Peak counted local storage is 24,111,690 bytes, versus 2,013,245,440 bytes whole
process RSS including the native model/renderer. All three maps unload at
teardown; old generation checks reject as expected.

This is still **not a successful full task**. The placement approach executes,
but release corrections consume their unchanged two-adjustment budget. Measured
command/FK height shortfall grows from 0.018899 to 0.027318 m; the release planner
cannot certify an opening pose within the unchanged raise/drop limits. The tool
returns `partial`, `no_plan_within_budget`, completed `["grasp"]`, failed stage
`release`, recovery `inspect`. No opening or departure success is claimed.

A proposed endpoint-settling change was discarded: inspection showed that
`NativeTableAdapter._executor` already passes `endpoint_settle_s=.3`. Both the
successful cube and failed cylinder already use that gate. The duplicate runs
`20261117_192455_e76ad4` / `20261117_192757_0cc794` were cancelled and are not
new policy evidence or failed-task denominator entries. No settling setting or
release bound is changed in the final implementation. Persistent measured
tracking shortfall under the existing policy remains an investigation; it is
not established to be the sole cause of this shape-specific release rejection.

### Full cube mobile sequence

`20261117_191753_aee05b` completes the entire existing-policy mobile cube sequence:
public-tool pick/carry/place/release/retreat, fresh released-object map, then
checked departure with confirmed stop. The released object drifts 0 m during
departure at the archived trace precision. The map contains its occupied returns;
the object is not omitted from the post-release scene. Four scenes load and
unload, three deliberate old-generation checks reject, final counted storage is
zero and peak counted storage is 24,113,850 bytes. Whole-process peak RSS is
1,999,306,752 bytes. Loaded carry takes 135.202 s and departure 47.988 s. This is
one successful assisted development episode, not an OVMM benchmark success rate.

The existing physical tracking/settling and boundary regressions also pass
**99 tests**, zero failures/errors/skips (`20261117_192528_c8af0d`, 13.390 s).
Those checks overlap the earlier batch; the final standalone boundary run
passes two checks (`20261117_192210_c48d27`) under the normal source path.


[Figure: exact-grid memory comparison and measured cube trajectory](figures/local_map_mobile_20261117.pdf).
The trajectory uses independent simulator measurements and includes normal base
actuation, grasp/lift, carry, placement, release, retreat and departure. The
constructor RSS panel is separate from full-model episode RSS. This is a
single development trajectory, not a task-rate plot or hardware observation.

Remaining gates are cylinder release feasibility under sustained tracking bias,
larger layouts/objects, actual onboard map acquisition and whole-robot occupancy
qualification, then the normal OVMM/agent navigation/exploration integration.
The bounded local-map owner is opt-in; it does not provide automatic whole-house
load/unload integration for an existing OVMM runner. Sweep/search CPU cost remains
substantial (about 140 s initial planning and 135 s carry validation/execution
here); this experiment establishes correctness evidence, not interactive speed.
