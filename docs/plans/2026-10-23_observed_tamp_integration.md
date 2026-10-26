# Observed agent TAMP integration

Branch: `feat/tamp-observed-integration`, based on main `ef4c1bfa` (merged #213).
Target: one actual `run agent` episode that discovers requested objects, navigates,
refreshes grounding, executes a multi-step task and independently verifies final
goals. Use Stretch first; qualify Sourccey against the same controls afterward.
Caliban hosts inference; keep the local GPU available for other work.

## Acceptance sequence

1. Preserve the two-goal and buffered-swap privileged native controls. These are
   execution regressions, not observed acceptance or a pooled success rate.
2. Bind observed task evidence to acquisition identities and motion epochs. Cached
   image publications must not renew freshness. In-flight segmentation and stored
   task/primitive plans must not survive externally dispatched motion.
3. Add qualified observed task grounding and native execution: tracked object
   pose/uncertainty, conservative object and arm geometry, support/occupancy,
   visual retention and final-goal evidence. Keep evaluator truth separate.
4. Exercise explore/find -> navigation -> reground -> plan -> execute through
   CHAT. Include missing evidence, moved obstacles, occupied targets and stale
   plan rejection. Recovery starts with explicit rediscovery/replanning; unknown
   physical outcomes stop. Do not add automatic held-object retries.
5. Freeze a small RoboCasa/MolmoSpaces population and compare greedy/beam/MCTS
   with the same agent, inputs and controllers. Qualify before paper claims.

## First implementation slice

Observed proposals now retain capture identity through perception, voxel ingestion
and scene-graph processing. The provider remembers the first local receive time
of a capture; rebroadcasting cached pixels cannot renew its age. Older captures
or changed clock domains are rejected. No server timestamp is subtracted from a
client clock: capture stamps provide identity/order, not cross-host clock sync.

CHAT navigation/view-motion invalidates task evidence and stored plan handles
before and after dispatch, including exceptions. This rejects in-flight results
from the earlier epoch and proposals generated during motion. A new capture is
required afterward. Legacy untimed proposals remain read-only before motion but
cannot satisfy the post-motion replacement requirement. Reconnection/new clocks
require rebuilding the provider. The guard does not lock unrelated controllers.

A different capture ID alone does not certify capture-after-motion, RGB/depth/pose
synchronization, visibility, complete object volume or physical feasibility.
Observed discovery stays `discovery_only`; neither task nor primitive observed
execution is enabled by this change. The existing RGBD provider estimates object
positions and heuristic support patches, not tracked full object pose, complete
payload geometry or measured contact. Never fill these gaps with simulator body
poses or pretend that simulator contact sensors exist on hardware.

## Remaining implementation

The observed native provider is still missing; do not equate frame freshness with
execution qualification. Next connect an explicit calibrated RGBD pose/geometry
provider and a robot-only kinematic model to the native motion seam, then add
visual effect verification and post-navigation acquisition admission. Missing or
occluded evidence must remain unknown. Simulator contact/state stays available
only to independent diagnostics/scoring. Shared whole-agent budget enforcement
and native articulation remain separate follow-ups.

## Grounding initialization diagnostic

The [recorded-frame candidate comparison](../experiments/observed_surface_grounding_20261023.md)
finds substantially cleaner object support than guessed point regions. Bounded
RGBD search padding is now available as an opt-in existing-API setting. This is
initialization work: remote selection still takes longer than the freshness limit.
Fresh-frame association and observed execution remain outstanding; no timestamp
renewal or simulator-pose fallback is authorized by this result.

## Local reacquisition slice (October 25)

[Recorded results and failures](../experiments/observed_reacquisition_20261025.md):
normal-split cylinder and unobstructed cube references each support ten current
RGBD updates with only two remote initialization calls. Local updates take about
9 ms; the full stationary privileged `run agent` regression still completes both
goals. Reset settling, mixed support/object components and arm self-occlusion
were concrete failure causes. They remain separate conditions in the records.

`RgbdTaskProvider.update_reacquired` is now a tested read-only provider seam with
capture-age and motion-epoch checks. Connect the actual acquisition worker next;
this is not yet a continuous CHAT camera binding or native observed provider.
Retain the original 2s current-geometry freshness gate and require a fresh semantic
reference after external navigation/view-motion. Add clear-view/stow admission
without turning occluded geometry into an executable placement claim.

## Live capture binding and main update (October 26)

The acquisition binding now uses the robot's existing stream on demand: normal
segmentation seeds a bounded semantic reference, and `scene_tasks` locally
reassociates against current measured RGBD. It preserves capture age, rejects
missing/ambiguous evidence and invalidates references across dispatched motion.
Custom providers keep their own acquisition path. No extra inference worker was
added. Negative segmentation discards the old semantic reference.

The live ZMQ test exposed and fixed head-camera RGB/depth registration and render
intrinsics. Unsampled depth remains unknown. Post-merge validation on main
`61da55cb` passed five sensor refreshes, the separate two-goal privileged native
CHAT control (529 audited commands, zero forbidden), and 369 regression checks.
See [the complete record](../experiments/observed_capture_binding_20261026.md).

Next qualify an observed execution provider with object-frame uncertainty,
conservative payload/arm/unknown-space geometry and visual effects. The camera
binding closes the acquisition seam, not the complete observed task path. Its
single-object sensor smoke has no support seed or task-success denominator.

## Observed execution implementation design

The [observed-native execution plan](2026-10-26_observed_native_execution.md)
identifies body-based payload attachment and privileged effect verification as
additional coupling beyond pose-provider injection. It orders the next work as
evidence/attachment separation, observed 3D free/occupied/unknown geometry,
uncertainty-aware reference tracking, visual effects, then the same agent's
held-out multi-step trials. Do not enable observed execution by changing a
provenance string or manufacturing an object orientation/complete volume.
