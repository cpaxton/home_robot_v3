# Placement planning

`emet.motion.placement.plan_placement_paths` searches multiple base endpoints,
object-center targets, pose-IK seeds and arm paths. It plans on a private MuJoCo
state and returns alternatives without commanding the robot. The arm retains the
measured EE orientation, so an offset held object has an explicit, consistent
attachment transform throughout each path.

## Geometry contract

All scene geometry, support bounds and base poses are in the same world frame, in
metres. `HeldObject.vertices_ee` is expressed in the end-effector frame.

- **Observed geometry:** `PlacementScene.from_voxels(centers, resolution=...,
  workspace=..., known_free=...)` uses occupied **3D** cell volumes. The resolution
  is the cell width. `from_pointcloud(voxel_map, ...)` adapts `SparseVoxelMap` and
  expands centroids conservatively using its 3D `voxel_resolution`.
- **Ground truth:** `PlacementScene.from_placements(placements,
  held_object=...)` uses conservative world AABBs. It requires bounds and excludes
  only the explicitly identified held object. It does not exempt the receptacle.
- **Held object:** use `HeldObject.from_world_bounds(...)` with observed object
  bounds and the measured EE pose, or supply conservative EE-local vertices
  directly. Include uncertainty in the supplied volume. Missing geometry fails;
  it is not replaced by a point or a guessed category-specific size.
- **Support:** supply explicitly grounded horizontal support patches to
  `free_surface_centers`. Candidates fit the object's full footprint and
  stay above the support. A receptacle's center is not an interior shelf.

Observed occupancy must have robot and held-object points segmented out by the
caller. The planner does **not** erase a box around the gripper, since that can
also erase furniture. Unobserved surfaces remain uncertain: occupied voxels alone
are not a free-space certificate. Supply `known_free(bounds) -> bool` to reject
volumes outside observed free space; out-of-workspace volumes always fail.
The callback must account for the full queried volume, not just its center.
Snapshots must be coherent and refreshed from perception before execution.

```python
from emet.motion.placement import plan_placement_paths
from emet.motion.placement_surfaces import free_surface_centers
from emet.motion.placement_geometry import HeldObject, PlacementScene

scene = PlacementScene.from_voxels(
    scene_voxel_centers, resolution=0.02,
    workspace=observed_workspace_bounds, known_free=is_known_free_volume,
)
payload = HeldObject.from_world_bounds(
    observed_object_bounds, ee_position=measured_ee_position,
    ee_rotation=measured_ee_rotation,
)
surface_search = free_surface_centers(
    observed_support_patches, scene=scene, payload=payload,
    ee_rotation=measured_ee_rotation,
)
result = plan_placement_paths(
    robot_model, measured_data, joint_names=joint_names,
    robot_body=base_body, ee_body=ee_body, scene=scene, payload=payload,
    object_centers=surface_search.centers, base_candidates=candidate_base_poses,
    set_base=write_private_model_base_pose, contact_bodies=grasp_contact_bodies,
    max_solutions=3,
)
```

The base writer receives `(model, private_data, xyt)`, preserves measured base
height, and returns `True` only on success. No callback should command live motion.

## Checks and result

Each alternative contains a base endpoint, desired object center, preplace/place
EE targets and rotation, and two densely sampled joint paths. `rejections`
records why candidates failed. `geometry_source` records observed voxels or GT.
Search budgets bound base candidates, IK attempts, total IK calls and RRT work.
No path means failure, with no unchecked fallback.

Defaults are three solutions, three IK attempts per target, up to 64 base
candidates, at most 64 supplied target centers, 96 total IK calls, 12 IK calls
per base and 400 RRT iterations per path attempt. Preplace height is 0.12 m and
collision margin is 0.005 m. These are work limits, not wall-clock guarantees.
`seed` controls a private generator shared by IK sampling, RRT, and shortcutting.
Seeded placement does not consume process-global NumPy or Python random state. Surface `budget_exhausted` currently reports region truncation,
not truncation at the final candidate cap. Neither an empty result nor an unset
budget flag proves infeasibility.

Collision checks use conservative world boxes enclosing complete declared robot
geoms (including meshes), transformed payload volume, and MuJoCo robot self
collision. Payload/robot intersections are permitted only at explicitly supplied
grasp-contact body subtrees. Every accepted arm edge is checked at joint-space
intervals no larger than 0.025 in Euclidean joint-coordinate distance, including
its exact endpoints. The result is **sampled**, not continuous collision detection;
mixed revolute/prismatic units and small obstacles require appropriate refinement.
Conservative boxes can reject feasible motions, especially in tight shelves.
The current RRT integration rejects coupled-joint groups; those need planning in
actuator coordinates before use with this entry point.

`collision_scope` is `sampled_arm_and_payload;base_endpoint_only`.
A collision-free base endpoint does **not** certify transport to it. The navigation
controller must separately validate the swept robot and payload along its route.
This API does not certify support stability, release dynamics, gripper opening,
post-release retreat, or real hardware readiness.

## Simulator integration

`KinematicPickPlaceExecutor.place_only` uses this search before base movement,
then searches again from measured state and fresh geometry after transport.
Before each arm segment it refreshes obstacles, rejects changed support or
attachment, and validates the connector from measured joints to the planned path.
Measured position and orientation arrival are required before continuing.

The existing assisted release remains explicit: this executor attaches and snaps
objects in simulation. Retraction uses a static placed-object volume rather than
moving the released object with the arm. Verification targets the selected
placement point, which can be offset from the receptacle center.

For observed geometry, inject `placement_geometry_provider(executor, object_id,
receptacle_id) -> (scene, payload, support_bounds)`. The provider is called after
state synchronization. Selecting `manip_collision="voxel"` without this provider
fails instead of falling back to GT. The kinematic task-grounding/release wrapper
still requires simulator placements; the reusable planner above does not.

The default simulator adapter requires `support_surfaces` for the selected
receptacle: horizontal top faces of axis-aligned collision boxes, preserved through
server serialization and the client reader. It never substitutes a semantic or
visual AABB top. Rotated/mesh supports and interior shelves require an explicit
geometry provider. Disjoint support patches are not joined across unsupported gaps.
Full-scene GT meshes and physical release execution remain separate integrations.

### Refresh and simulator geometry details

Ground-truth placements may provide `collision_bounds`: one world box per
MuJoCo collision geom, including explicitly paired geoms. Those components are
used in preference to the whole semantic/visual `bounds`; visual-only geometry
must not fill free interiors of fixtures. Legacy entries without components retain
the conservative whole-box fallback. Observed voxel handling is unchanged.

Each refreshed scene has a digest of its occupancy boxes and workspace. A changed
obstacle snapshot is logged, then the entire segment (including the measured-start
connector) is validated against a **new checker using the new snapshot**. Harmless
obstacle motion need not abort; intersecting motion invalidates the path. Support
or attachment changes still abort. This is a between-segment refresh contract,
not continuous dynamic-obstacle monitoring. Known-free predicates are supplied
anew by the provider and are not encoded by the occupancy digest.

Collision checks use checker-owned scratch `MjData`; caller qpos, FK and contact
buffers remain unchanged. The executor's model is local, separate from the server,
and query/transport runtime errors and timeouts return explicit failure results.

The simulator's base teleport snaps attachments immediately using their existing
EE-local offsets. Placement verifies that the observed center offset remains
within 1 cm after transport. It refreshes the payload volume after transport and
does not re-register the attachment to conceal a changed offset. The assisted
simulator latch preserves the positional offset; it does not enforce a rigid
object orientation. Physical attachment/orientation estimation remains the
responsibility of an observed-geometry provider and physical execution controller.

The executor uses `free_surface_centers` to subtract clutter footprints expanded
by the payload footprint and margin from each explicit support patch. It checks
the whole vertical preplace-to-place corridor, ranks remaining regions by area,
and samples their centers and interior points. This can find off-grid slots missed
by the earlier 5×5 lattice. It retains bounded region/candidate budgets and reports
truncation separately from a complete search with no accepted candidates.

`last_surface_search` exposes candidate centers, blocking geometry bounds and the
budget flag to the caller. These are evidence for a future rearrangement planner,
not a minimal blocker set or a claim that an obstacle is movable. Unknown-space
predicates, full arm paths, measured state and attachment checks remain mandatory.
A fixed-grid helper remains available for controlled comparisons. Pose orientation
is still fixed to the measured orientation; failure is not a proof of infeasibility.

IK effort is capped both globally (96 calls by default) and per base endpoint
(12 calls). An unreachable current base must not exhaust the budget before other
approaches are considered. Reaching either budget is reported explicitly and
never treated as proof that the scene requires rearrangement.

This is a Python API used by the existing TAMP tools, not an additional CHAT
tool. The [agent API](tamp.md) describes the public JSON interface. The
[integration review](../plans/2026-10-09_tamp_merge_and_metrics.md) tracks current
acceptance blockers; the latest full simulator gate failed.
