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
