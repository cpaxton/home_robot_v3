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
