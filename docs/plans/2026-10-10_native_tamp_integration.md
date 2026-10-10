# Native TAMP integration and merge ledger

Accepted direction: incremental merges; simulation acceptance first; explicit
native-observed and native-debug presets; existing defaults unchanged. Privileged
inputs remain available for debugging and independent scoring. They must not leak
into observed execution or be represented as real robot sensors.

## Baseline and disposition

- Main `c9dd742f` contains #177, #178, and the official Sourccey Mk.V model (#192).
- Isolated integration branch: `feat/tamp-native-integration`. The placement,
  containment, and articulation stack merged onto main without textual conflicts.
- #179 remains blocked by live placement acceptance. #186 follows its failure
  containment gate. #187 can merge its paired accounting independently of score
  improvements. #188 has standalone assisted open/close evidence, not placement.
- #189 media, #191 generic rendering/floor repairs, and #193 Stretch recordings
  must be reconciled with #192; obsolete Sourccey assembly changes must not return.
- #190 is an independent measured-telemetry review dependency. Navigation #181–185
  is separately owned; do not rewrite or blindly merge those branches.

The previously queued `20261009_175205_1cc9bd` is terminal. Open/close passed.
Placement found paths, completed grasp, then returned `placement_stale_observation`
with measured-state receive age 4.277 s after the bounded 2 s wait. No release was
attempted. RBY1's purported table control actually selected the blue cube; Innate
Mars failed grasp reachability. Preserve these failures; corrected fixtures need
explicit target/support assertions and new identities.

## Execution order

1. Reproduce and repair telemetry starvation; validate old and new heads on the
   same live scenario without loosening freshness/collision/tracking thresholds.
2. Repair fixture selection; finish placement and failure-containment gates;
   restack/review/merge eligible layers individually and update their evidence.
3. Introduce explicit physical execution and observed/privileged input contracts,
   robot adapters, and additive JSON provenance. No fallback to latch/teleport.
4. Connect native Stretch execution through ZMQ and public agent tools using the
   proven block/cylinder fixtures, then reference the independent oracle score.
5. Add Sourccey jaw/robot contact geometry and native adapters on #192's model;
   validate left-arm manipulation and both-arm bindings. Preserve provenance.
6. Connect observed instance/support discovery, geometry, grasp candidates, and
   visual held/released evidence. Unknown/occluded space is not free. Missing
   sensing fails explicitly. No simulator metadata/contact/pose reads in this path.
7. Add read-only manipulation inspection and validate LLM plan/execute/recovery
   using the same tools as scripted tests. Start observed tests stationary; gate
   transport on localization and swept arm/payload validation.
8. Freeze paired development/holdout sets and produce videos, paper figures,
   JSON provenance, and exact run-agent examples alongside each promoted layer.

## Telemetry investigation

Concurrency regressions reproduce physics-lock retention during renderer calls
and state-lock retention during image decoding. The candidate renders copied
MuJoCo state, preserving RGB/depth and camera-pose coherence, and decodes client
images outside the state receiver's lock. These prove starvation mechanisms, not
that they alone explain the old live failure. A fresh live rerun is required.

Validation so far: render/camera/API group 48 passed; client observation group
4 passed; integrated motion/TAMP/model group 170 passed with local sockets
permitted. Groups overlap and are not a unique aggregate count. No new live
placement pass or observed/native agent pass has been established.

## Acceptance and remaining scope

For each robot/input combination: two native table fixtures with three fresh
repeats, then two-transfer composition, obstacle invalidation, and safe stopping
on uncertain grasp. Independently audit contact, retention, release, distractor
motion, and forbidden assistance. Keep the existing placement CHAT/admission/
repeat gates and failed denominators. Actual LLM runs are separately scored from
scripted calls with a pinned model/configuration.

Observed inputs use declared robot feedback and camera-derived estimates. For
Sourccey, rendered depth and simulated base coordinates are not hardware depth or
wheel odometry. Metric depth/localization availability must be validated; missing
capabilities stay unsupported. Physical door opening, automatic clutter clearing,
and real-hardware commissioning are outside this increment.

### Input-contract implementation

`feat/tamp-native-contracts` begins the input boundary: immutable, source-tagged,
fresh object poses; provider-based retention/lift checks; explicit privileged
selection by legacy physical evaluations; no physical-mode fallback into legacy
controllers or teleport; no GT task discovery while observed adapters are absent.
The focused input/physical/API/config suite passes 105 tests. This is a guarded
integration layer, not completed native CHAT execution or an observed-scene
provider. Adapter, geometry, perception, inspection, and live acceptance work in
the execution order above remains open.

### Live telemetry rerun — still failing

Diagnostic job `20261010_122429_d89b27`, implementation `5dc29884`, terminated
with exit 1. Pickup/lift passed and placement found three candidate paths, but
after base motion state age reached 6.132 s and execution stopped with
`placement_stale_observation` before release. Artifact directory:
`/tmp/tamp-render-state-fix-20261010/` (JSON, log, and video). This was a CPU/Mesa
run; it is not GPU validation or a paired benchmark (the articulation command
sequence differs from the earlier run). The renderer/client lock fixes are
necessary concurrency repairs, not a demonstrated fix for the live failure.

A local microprofile on the same merged scene measured articulation descriptors
at 1.4 ms, moved-bowl geometry at 10 ms, and refreshing all articulated bodies
at 202 ms with one BLAS thread. These costs alone do not explain the six-second
gap. The next diagnostic must capture server and client thread stacks/timings
at the failed transition before further changes or threshold adjustments.

### Review branches

- #194 `fix/tamp-render-telemetry`: independent on current main, 42 focused tests.
- #195 `feat/tamp-native-input-boundary`: stacked on #194; 95 tests at its first
  head, followed by 45 passing API/input tests after adding stored-plan execution
  guards and simulation-only privileged-mode cases. A stored assisted handle may
  not bypass a subsequent physical-mode request.
- Neither PR promotes native execution or unblocks #179's live acceptance gate.

### Physics-batch lock correction

Traced diagnostic `20261010_123558_40ba47` reproduced the placement failure with
6.004 s receive age. At the failed transition, the state publisher waited on
`_mj_lock` while physics stepped and snapped attachments. The server held that
lock across every substep in a tick. The correction releases it between complete
substeps, yielding to waiting publishers; integrator timestep, step count,
collision checks, and acceptance limits remain unchanged. Headless and passive
viewer paths share the same step helper. Its regression failed on the previous
code and passes now; the standalone telemetry suite passes 43 tests.

Exact integration head `5beb3ebe` is under live diagnostic
`20261010_124407_851058` in `/tmp/tamp-physics-substep-fix/`. Do not interpret the
unit regression as a live placement pass. Geometry refresh is also expensive
under load, but a trial loop-vectorization showed no useful microprofile gain
and was removed.

### Fast feedback and attachment follow-up

The substep-only retest `20261010_124407_851058` failed earlier at pregrasp:
feedback was fresh, but frequent scene refreshes reduced physics progress and
tracking missed its unchanged deadline. The next candidate removed scene metadata
refresh from fast state/servo publication; only full observations publish it.
That run (`20261010_124831_9c33c2`, `ddb30ff3`) passed pickup/lift and found three
placement paths, then failed on a command acknowledgement after base transport.
Neither result is a placement pass.

The terminal stacks also exposed two attachment issues. `set_free_body_pose`
used `jnt_qposadr` to clear `qvel`; free joints have different position/velocity
addresses. A three-freejoint regression failed before the correction: it left
one target velocity nonzero and zeroed a different object's velocity. Attachment
updates also ran full dynamics twice per physics step. They now refresh transforms
without redundant solver calls; explicit pose-setting retains its final dynamics
update. Standalone telemetry/manipulation coverage passes 64 tests. Integration
head `398eb7dc` is running as `20261010_125538_3fddc5`.

### Stack reconciliation completed

Non-rewriting restacks on main/#192 were pushed for #179, #186, #187, #188,
#189, #191, and #193. #191 now preserves the official Sourccey model, generator,
and robot tests exactly, removing its obsolete arm fragment. Its remaining
changes concern generic conversion, floors, and recording. Validation: 198 tests
on the restacked #193 chain; 30 model/converter checks on #191 (overlapping);
8 paired-metric tests on #187. These do not replace live placement or native-agent
acceptance. No production merge is justified by unit-test totals alone.

### Native boundary validation and corrected fixtures

The final combined input/physical/API/config/protocol/client suite passes 124
tests on #195's current ancestry; Ruff passes for the changed runtime files.
Stretch and generic clients expose immutable joint-position evidence with an
atomic receive timestamp. Physical-mode requests are guarded during discovery,
planning, direct pick/place, stored-plan execution, and environment overrides.
The native execution adapters and observed geometry/perception are still absent;
this does not enable `run agent` physical manipulation.

#189 now explicitly requests `object2` (red cylinder) and `table` in table media
controls, and rejects a mismatched receptacle before manipulation. Corrected
cases use `*_cylinder_to_table_v2` fixture identities in the JSON manifest.
Eight focused runner tests pass. No new live score is attached to these fixtures;
old object-on-cube failures remain unchanged.
