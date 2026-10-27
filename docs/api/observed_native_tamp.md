# Observed native TAMP integration

The production CHAT tools can now use an explicitly injected
`ObservedNativeAdapter`. It shares `scene_tasks`, `plan_pick_place`,
`execute_pick_place_plan`, `pick_place`, and optionally `plan_task` /
`execute_task_plan` with the privileged control. No simulator body names, object
pose writes, scene snapshots or contacts are used by this adapter.

**Status:** implementation and synthetic integration regressions pass. A complete
live RGBD source is not assembled or qualified by default. The usual RGBD
`scene_tasks` provider remains read-only. This is not an observed-task benchmark,
a contact-execution result, or a hardware sensor-availability claim.

## Components

| Component | Responsibility | Rejects / limits |
| --- | --- | --- |
| `RigidRgbdTracker` | RGB feature correspondences, forward/backward optical flow, measured RGBD landmarks, RANSAC rigid fitting against a fixed reference | Lost masks/depth, repeated or insufficient texture, poor feature spread, inconsistent motion, replay, reference age >30 s; no global re-identification |
| `PayloadGeometryEvidence` | Complete conservative volume in an explicit tracked reference frame | No visible-point AABB promoted to a complete shape; geometry priors remain declared inputs |
| `ObservedNativeCollisionChecker` | Robot-only FK, complete link/payload boxes, self collision and whole-volume known-free queries | Missing workspace, stale geometry, scene revisions, unresolved payload geometry |
| `VisualEffectFrame` | One tracked pose with capture interval, error bounds, bracketed robot feedback, full calibrated gripper volumes | Receive time alone, matching joint endpoints alone, missing alignment bounds, changing calibration, excessive uncertainty |
| `ObservedVisualVerifier` | Fresh repeated visual lift and post-retreat placement effects | Height-only lift, static/slipped object, occlusion, replay, unsupported/outside/unsettled placement, gripper volume still near object |
| `ObservedRegionTransferProvider` | Complete observed role/region predicates and the existing task search/session interface | Missing roles, unsupported/held objects, ambiguous occupancy, unknown destination space |

The tracker supplies measured relative rigid motion from material features; it
does not infer a canonical object yaw from a symmetric/untextured shape. Such
objects return unknown in this implementation. Registration residuals are
diagnostics, not uncertainty estimates: position/orientation admission bounds
must come from independent calibration. Automatic current-instance mask
production and long-term identity recovery are not supplied here.

## Provider assembly

`ObservedNativeAdapter(robot, source=..., refine=..., tasks=...)` is deliberately
explicit. `source.input_mode` must be `observed` and it provides:

| Method | Contract |
| --- | --- |
| `snapshot(object_ref, support_ref)` | A fresh `ObservedNativeSnapshot`: temporally qualified `VisualEffectFrame`, background `PlacementScene`, coherent scene revision, observed support bounds |
| `current_revision()` | Revision of the current background geometry; changes during a segment stop execution |
| `collect(object_ref, support_ref, after=...)` | A bounded batch of at least two distinct fresh frames captured strictly after the requested client-monotonic barrier, spanning the required settling interval |
| `invalidate()` | Clear old/in-flight tracking and geometry evidence after external base/head/view motion |

A source must separate robot and target occupancy by provenance, without erasing
nearby obstacles or declaring occluded space free. `ObservedWorkspace.from_depth`
provides conservative 3D depth carving, but **does not** perform that separation,
fuse views, bound the whole object, or qualify sensor timing. These are remaining
source-assembly requirements, not inferred behavior of the adapter.

`refine(task, snapshot, placement_region=...)` is planning-only and returns a
robot-specific `PhysicalPickPlaceExecutor` with a robot-only calibrated model,
measured-joint synchronization, normal actuator commands, open-gripper geometry,
and three grasp / three placement pose targets. `placement_region` is passed only
when requested. It must search/calibrate grasp targets independently of object
names. No controller calls may occur in this callback.

The adapter certifies those targets through shared IK/dense motion planning,
binds plans to the robot model, object/reference/geometry and scene revision,
and rechecks before execution. It expands the supplied payload envelope by the
initial admitted pose-error bounds; an increase in those bounds stops execution.
Scene refresh uses original capture receive times. New captures do not excuse a
changed scene during a path. Controller/gripper calls mark the actuation boundary;
pre-command observation failures remain `execution_started=false`, while failures
after a possible command return `partial`. Plans are single-use.

`VisualEffectFrame` explicitly requires conservative `alignment_error_m` and
`alignment_error_rad` from qualified camera/encoder clock and motion bounds.
Two equal joint samples do not establish motion between those samples. A new
image ID or fresh receive time is insufficient. `CaptureInterval.calibration_id`
identifies the qualified clock/calibration bundle, not an arbitrary availability
flag. Current generic sensor observations do not automatically supply these
bounds; no adapter constructs them by substituting the local current time.

## CHAT and multi-step binding

Programmatic CHAT startup accepts
`run_agent_with_robot(..., observed_native_factory=factory)`. The factory receives
`(robot_client, executor)` and returns the explicitly configured adapter. Startup
requires physical mode with observed inputs and validates the returned robot
identity; errors close the client. There is no new CLI flag that fabricates an
observed source from the existing discovery provider.

A factory can call:

```python
adapter.bind_regions(
    regions,  # tuple of TransferRegion values grounded on one observed support
    robot_profile_id=calibrated_profile_id,
    reset_id=episode_reset_id,
)
```

This binds a `TaskSession` to the same adapter. CHAT installs it alongside the
primitive tools. Region role names come from the configured semantic task names;
all roles must be freshly observed and visually supported/settled. A destination
is free only when its entire required 3D volume is known free. Missing objects
cannot become `outside`, and missing map cells cannot become `free`. The existing
greedy/beam/MCTS implementations, protected-goal checks and failure stop policy
remain shared. Native articulation and autonomous physical recovery remain
unsupported.

Direct tool-registry callers can supply `context['observed_native_adapter']` and,
for multi-step tools, its `task_session`. No adapter is constructed merely by
setting an input label or providing a discovery proposal.

## Evidence and next gate

See [integration regressions](../experiments/observed_execution_integration_20261026.md)
and the [execution plan](../plans/2026-10-26_observed_native_execution.md).
The next gate is a concrete live source with independent geometry/clock
qualification, followed by held-out stationary Stretch single transfers,
multi-step plans and explicit recovery-boundary failures using Caliban inference.
Only afterward should navigation/exploration or another robot enter the scored
observed-task denominator.
