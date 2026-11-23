# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Contact-based execution through shared robot controllers.

The caller supplies observed state/geometry (GT only in the oracle evaluation).
This executor never attaches objects or sets live simulator poses. It is usable
through the same ``execute_task_plan`` seam as symbolic and latch controls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from emet.controller.manipulation.native_inputs import ObservationUnavailable, read_object_pose
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker
from emet.core.zmq_protocol import motion_wait_timeout_scale, read_sim_to_real_ratio
from emet.motion.arm_rrt import ContactClearancePath, DeparturePath, joint_limits_from_model, plan_arm_joint_path
from emet.motion.mujoco_arm_ik import joint_qpos_addrs, solve_pose_ik

logger = logging.getLogger(__name__)


def coalesce_collinear_waypoints(path, maximum_step):
    """Remove only redundant points on the same bounded joint-space segment."""
    if len(path) < 3:
        return path
    result = [path[0]]
    for index in range(1, len(path)-1):
        start, middle, end = map(np.asarray, (result[-1], path[index], path[index+1]))
        delta = end-start
        norm = float(delta @ delta)
        fraction = float((middle-start) @ delta)/norm if norm else 0.
        redundant = 0. <= fraction <= 1. and np.max(np.abs(middle-start-fraction*delta)) <= 1e-10
        if not redundant or np.any(np.abs(delta) > maximum_step):
            result.append(path[index])
    result.append(path[-1])
    return result


@dataclass
class PhysicalMotionResult:
    success: bool
    message: str
    phase: str = ""
    residual: float | None = None


class PhysicalPickPlaceExecutor:
    # Maximum joint motion during the stricter endpoint stillness window.
    ENDPOINT_STILLNESS_TOLERANCE = 0.002
    GRIPPER_TRACKING_TOLERANCE = 0.001

    execution_mode = "physical"
    release_refiner: Callable[[], PhysicalMotionResult] | None = None
    release_verifier: Callable[[], PhysicalMotionResult] | None = None
    released_collision = None
    gripper_close_command: Callable[[], bool] | None = None
    gripper_hold_command: Callable[[], bool] | None = None

    def __init__(
        self,
        robot,
        *,
        model,
        data,
        ee_body,
        joint_names,
        collision,
        synchronize: Callable,
        command_joints: Callable,
        coupled_groups=(),
        event: Callable = lambda **kwargs: None,
        joint_tolerance: float = 0.03,
        linear_joint_tolerance_m: float | None = None,
        joint_settle_timeout_s: float = 5.0,
        endpoint_settle_s: float = 0.0,
        use_sim_time: bool = False,
        settling_wall_timeout_s: float = 30.0,
        position_tolerance_m: float = 0.01,
        orientation_tolerance_rad: float = 0.1,
        base_body: str | None = None,
        joint_limit_margins=None,
        gripper_open_configuration=(),
        gripper_close_configuration=(),
        command_gripper_joints=None,
        gripper_close_command=None,
        input_mode: str = "observed",
        object_pose_provider=None,
        visual_verifier=None,
        ik_seed_configurations=(),
        approach_joint_routes=(),
        joint_planning_timeout_s=10.0,
        ik_random_starts=0,
        preplace_staging_offsets_m=(),
        pregrasp_staging_offsets_m=(),
        cartesian_transport=False,
        coalesce_execution_waypoints=False,
        reuse_validated_grasp_paths=False,
    ):
        if input_mode not in {"observed", "privileged"}:
            raise ValueError("invalid_tamp_inputs")
        if input_mode == "observed":
            if object_pose_provider is None or object_pose_provider.input_mode != "observed":
                raise ValueError("observed_object_provider_required")
            if getattr(collision, "geometry_source", None) != "observed_voxels":
                raise ValueError("observed_collision_provider_required")
            if isinstance(collision, ObservedNativeCollisionChecker) and object_pose_provider is not collision:
                raise ValueError("observed_object_provider_mismatch")
        elif object_pose_provider is not None and object_pose_provider.input_mode != "privileged":
            raise ValueError("input_provider_mismatch")
        self.input_mode, self.object_pose_provider = input_mode, object_pose_provider
        self.visual_verifier = visual_verifier
        self.robot, self.model, self.data = robot, model, data
        self.ee_body, self.joint_names = ee_body, tuple(joint_names)
        self.collision = collision
        self.synchronize, self.command_joints = synchronize, command_joints
        self.coupled_groups, self.event = coupled_groups, event
        self.joint_tolerance = joint_tolerance
        if linear_joint_tolerance_m is not None and (
            not np.isfinite(linear_joint_tolerance_m) or linear_joint_tolerance_m <= 0
        ):
            raise ValueError("invalid_linear_joint_tolerance")
        self.reuse_validated_grasp_paths = bool(reuse_validated_grasp_paths)
        self.coalesce_execution_waypoints = bool(coalesce_execution_waypoints)
        self.linear_joint_tolerance_m = linear_joint_tolerance_m
        if not np.isfinite(joint_settle_timeout_s) or joint_settle_timeout_s <= 0:
            raise ValueError("Positive finite joint settling timeout required")
        self.joint_settle_timeout_s = float(joint_settle_timeout_s)
        if not np.isfinite(endpoint_settle_s) or not 0 <= endpoint_settle_s < joint_settle_timeout_s:
            raise ValueError("Endpoint settling must fit within the joint settling timeout")
        self.endpoint_settle_s = float(endpoint_settle_s)
        if not np.isfinite(settling_wall_timeout_s) or settling_wall_timeout_s <= 0:
            raise ValueError("Positive finite settling wall timeout required")
        self.use_sim_time = bool(use_sim_time)
        self.settling_wall_timeout_s = float(settling_wall_timeout_s)
        self.position_tolerance_m, self.orientation_tolerance_rad = position_tolerance_m, orientation_tolerance_rad
        self.qadr = joint_qpos_addrs(model, self.joint_names)
        self.ik_seed_configurations = tuple(np.asarray(seed, float) for seed in ik_seed_configurations)
        if not np.isfinite(joint_planning_timeout_s) or not 0 < joint_planning_timeout_s <= 120:
            raise ValueError("invalid_joint_planning_timeout")
        self.joint_planning_timeout_s = float(joint_planning_timeout_s)
        self.approach_joint_routes = tuple(np.asarray(route, float) for route in approach_joint_routes)
        if len(self.approach_joint_routes) > 4 or any(
            route.ndim != 2 or route.shape[1] != len(self.joint_names)
            or not 1 <= len(route) <= 32 or not np.isfinite(route).all()
            for route in self.approach_joint_routes
        ):
            raise ValueError("invalid_approach_joint_route")
        if type(ik_random_starts) is not int or not 0 <= ik_random_starts <= 32 or (ik_random_starts and coupled_groups):
            raise ValueError("invalid_ik_random_starts")
        self.ik_random_starts = ik_random_starts
        self.preplace_staging_offsets_m = tuple(np.asarray(offset, float) for offset in preplace_staging_offsets_m)
        if len(self.preplace_staging_offsets_m) > 4 or any(
            offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .5
            for offset in self.preplace_staging_offsets_m
        ):
            raise ValueError('invalid_preplace_staging_offsets')
        self.pregrasp_staging_offsets_m = tuple(np.asarray(offset, float) for offset in pregrasp_staging_offsets_m)
        if len(self.pregrasp_staging_offsets_m) > 4 or any(
            offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .5
            for offset in self.pregrasp_staging_offsets_m
        ):
            raise ValueError('invalid_pregrasp_staging_offsets')
        self.ik_solver_calls = 0
        self.ik_call_budget = None
        if any(seed.shape != (len(self.joint_names),) or not np.isfinite(seed).all()
               for seed in self.ik_seed_configurations):
            raise ValueError("invalid_native_ik_seed")
        self.base_body = base_body
        self.joint_limit_margins = dict(joint_limit_margins or {})
        self.gripper_open_configuration = dict(gripper_open_configuration)
        self.gripper_close_configuration = dict(gripper_close_configuration)
        self.command_gripper_joints = command_gripper_joints
        self.gripper_close_command = gripper_close_command
        if type(cartesian_transport) is not bool:
            raise ValueError("invalid_cartesian_transport")
        self.cartesian_transport = cartesian_transport
        self._contact_clearance_scope = None
        self.grasp_paths: list = []
        self.place_paths: list = []
        self.grasp_targets: list = []
        self.grasp_target_options: list = []
        self.place_targets: list = []
        self.transfer_candidates = ()
        self.transport: Callable | None = None
        self.payload_body: str | None = None

    def _motion_time(self):
        return float(self.data.time) if self.use_sim_time else time.monotonic()

    def _collision_phase(self, phase):
        if isinstance(self.collision, ObservedNativeCollisionChecker) and self.collision.target_contact is not None:
            return self.collision.motion_phase(phase)
        return nullcontext()

    def _settling_window(self, duration):
        """Physics duration with a capped, simulation-rate-scaled wall bound."""
        started, wall = self._motion_time(), time.monotonic()
        wall_budget = self.settling_wall_timeout_s
        if self.use_sim_time:
            wall_budget *= motion_wait_timeout_scale(read_sim_to_real_ratio(getattr(self.robot, "_state", None)))

        def active():
            elapsed = self._motion_time() - started
            return (
                np.isfinite(elapsed)
                and 0 <= elapsed < duration
                and (not self.use_sim_time or time.monotonic() - wall < wall_budget)
            )

        return active

    def plan_pose(self, position, rotation, *, ordered_groups=False):
        previous = getattr(self, '_pose_planning_deadline', None)
        self._pose_planning_deadline = time.monotonic()+self.joint_planning_timeout_s
        try:
            return self._plan_pose(position, rotation, ordered_groups=ordered_groups)
        finally:
            self._pose_planning_deadline = previous

    def _plan_pose(self, position, rotation, *, ordered_groups=False):
        """Plan from the offline current state; caller preserves candidate state."""
        start = self.data.qpos.copy()
        error = 'pose_ik_failed'
        phase = getattr(self.collision, 'phase', None)
        if self.cartesian_transport and phase in {'pregrasp', 'grasp', 'lift', 'preplace', 'place', 'retreat'}:
            offsets = self.preplace_staging_offsets_m if phase == 'preplace' else ()
            routes = [((np.asarray(position)+offset), np.asarray(position)) for offset in offsets]
            routes.append((np.asarray(position),))
            for route in routes:
                if time.monotonic() >= self._pose_planning_deadline:
                    return None, "no_plan_within_budget"
                path, error = self._plan_cartesian_route(route, rotation)
                logger.info("Cartesian %s proposal: %s (%s waypoints; collision=%s geom=%s admission=%s)",
                            phase, error or "ok", len(path) if path is not None else 0,
                            self.collision.last_reason, getattr(self.collision, 'last_blocked_geom', None),
                            getattr(getattr(self.collision, 'last_admission', None), 'reason', None))
                if path is not None:
                    return path, None
                self.data.qpos[:] = start
                mujoco.mj_kinematics(self.model, self.data)
        route_seeds = tuple(route[-1] for route in self.approach_joint_routes) if (
            getattr(self.collision, 'phase', None) == 'pregrasp'
        ) else ()
        seeds = [*route_seeds, None, *self.ik_seed_configurations]
        if self.ik_random_starts:
            low, high = joint_limits_from_model(self.model, self.joint_names)
            rng = np.random.default_rng(0)
            # Redundant arms need distinct elbow/torso solutions for one tool
            # pose. These are proposals only: IK, joint bounds, goal collision
            # and the complete swept path still have to pass for every seed.
            for index in range(self.ik_random_starts):
                anchor = start[self.qadr] if index % 2 == 0 else (low+high)/2
                seeds.append(np.clip(anchor+rng.normal(size=len(low))*(high-low)*.25, low, high))
        staged = []
        phase = getattr(self.collision, 'phase', None)
        offsets = self.pregrasp_staging_offsets_m if phase == 'pregrasp' else self.preplace_staging_offsets_m if phase == 'preplace' else ()
        if offsets:
            for offset in offsets:
                if self.ik_call_budget is not None and self.ik_solver_calls >= self.ik_call_budget:
                    break
                self.data.qpos[:] = start
                self.ik_solver_calls += 1
                via = solve_pose_ik(self.model, self.data, ee_body=self.ee_body,
                    joint_names=self.joint_names, target_pos=np.asarray(position)+offset,
                    target_rotation=rotation, coupled_groups=self.coupled_groups,
                    joint_limit_margins=self.joint_limit_margins, tol_m=self.position_tolerance_m,
                    tol_rad=self.orientation_tolerance_rad)
                if via.success:
                    staged.append(self.data.qpos[self.qadr].copy())
        proposals = [(seed, (seed,)) for seed in staged]+[(seed, ()) for seed in seeds]
        for seed_index, (seed, via) in enumerate(proposals):
            if time.monotonic() >= self._pose_planning_deadline:
                self.data.qpos[:] = start
                mujoco.mj_kinematics(self.model, self.data)
                return None, "no_plan_within_budget"
            if self.ik_call_budget is not None and self.ik_solver_calls >= self.ik_call_budget:
                error = 'no_plan_within_budget'
                break
            self.ik_solver_calls += 1
            self.data.qpos[:] = start
            if seed is not None:
                self.data.qpos[self.qadr] = seed
            result = solve_pose_ik(
                self.model,
                self.data,
                ee_body=self.ee_body,
                joint_names=self.joint_names,
                target_pos=position,
                target_rotation=rotation,
                coupled_groups=self.coupled_groups,
                joint_limit_margins=self.joint_limit_margins,
                tol_m=self.position_tolerance_m,
                tol_rad=self.orientation_tolerance_rad,
            )
            if not result.success:
                error = f"pose_ik_failed:position={result.pos_error_m:.4f},rotation={result.orientation_error_rad:.4f}"
                continue
            goal = self.data.qpos[self.qadr].copy()
            self.data.qpos[:] = start
            mujoco.mj_kinematics(self.model, self.data)
            path, error = self.plan_joint_target(goal, ordered_groups=ordered_groups, via_configurations=via)
            if path is not None:
                return path, None
            logger.warning(
                'Native pose seed %s failed path certification: %s (geometry=%s, admission=%s, contacts=%s, surface=%s, reason=%s, departure=%s)',
                seed_index, error, getattr(self.collision, 'last_blocked_geom', None),
                getattr(getattr(self.collision, 'last_admission', None), 'reason', None),
                getattr(self.collision, 'last_contacts', ()),
                getattr(getattr(self.collision, 'local_scene', None), 'last_collision', None),
                getattr(self.collision, 'last_reason', None),
                getattr(self.collision, 'last_departure_diagnostics', None),
            )
        self.data.qpos[:] = start
        mujoco.mj_kinematics(self.model, self.data)
        return None, error

    def _plan_cartesian_route(self, positions, rotation):
        """Continue native IK through a proposed tool route; check every joint edge."""
        start = self.data.qpos.copy()
        path = [start[self.qadr].copy()]
        complete = False
        low, high = joint_limits_from_model(self.model, self.joint_names)
        resolution = np.array([.002 if self.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE
                               else .025 for name in self.joint_names])
        scope = self._contact_clearance_scope or nullcontext
        try:
            with scope():
                for destination in positions:
                    ee = self.data.body(self.ee_body)
                    origin, initial_rotation = ee.xpos.copy(), ee.xmat.reshape(3, 3).copy()
                    rotations = Rotation.from_matrix([initial_rotation, rotation])
                    sweep = Slerp([0., 1.], rotations)
                    angle = (rotations[0].inv()*rotations[1]).magnitude()
                    count = max(1, int(np.ceil(np.linalg.norm(destination-origin)/.02)), int(np.ceil(angle/.1)))
                    for fraction in np.linspace(0., 1., count+1)[1:]:
                        if self.ik_call_budget is not None and self.ik_solver_calls >= self.ik_call_budget:
                            return None, 'no_plan_within_budget'
                        before = self.data.qpos.copy()
                        self.ik_solver_calls += 1
                        pose = solve_pose_ik(self.model, self.data, ee_body=self.ee_body,
                            joint_names=self.joint_names, target_pos=origin+(destination-origin)*fraction,
                            target_rotation=sweep(fraction).as_matrix(), coupled_groups=self.coupled_groups,
                            joint_limit_margins=self.joint_limit_margins, tol_m=self.position_tolerance_m,
                            tol_rad=self.orientation_tolerance_rad)
                        if not pose.success:
                            return None, 'cartesian_ik_failed'
                        goal = self.data.qpos[self.qadr].copy()
                        if np.any(goal < low) or np.any(goal > high):
                            return None, 'cartesian_joint_bounds'
                        self.data.qpos[:] = before
                        mujoco.mj_kinematics(self.model, self.data)
                        samples = max(2, int(np.ceil(np.max(np.abs(goal-before[self.qadr])/resolution)))+1)
                        segment = plan_arm_joint_path(self.model, self.data, joint_names=self.joint_names,
                            q_start=before[self.qadr], q_goal=goal, collision=self.collision, planner='linear',
                            linear_steps=samples, goal_tolerance=0., departure=False, shortcut=False,
                            max_time_s=min(self.joint_planning_timeout_s, 10.))
                        if not segment.success:
                            return None, 'cartesian_path_failed:'+str(segment.reason)
                        path.extend(segment.waypoints[1:])
                        self.data.qpos[self.qadr] = goal
                        mujoco.mj_kinematics(self.model, self.data)
            complete = True
            return (ContactClearancePath(path, self._contact_clearance_scope)
                    if self._contact_clearance_scope else path), None
        finally:
            # Preserve the endpoint only on a complete route; failed proposals
            # cannot seed the next attempt from an unexecuted intermediate pose.
            if not complete:
                self.data.qpos[:] = start
                mujoco.mj_kinematics(self.model, self.data)

    def plan_joint_target(self, goal, *, ordered_groups=False, via_configurations=()):
        """Keep original contact clearance across this transfer's phase plans."""
        previous = self._contact_clearance_scope
        if (previous is None and isinstance(self.collision, ObservedNativeCollisionChecker)
                and self.collision.allow_contact_departure and self.collision.local_scene is not None):
            with self.collision.motion_scope(self.model, self.data, self.joint_names, self.data.qpos[self.qadr]):
                policy = self.collision.make_departure(self.model, self.data, self.joint_names,
                                                      self.data.qpos[self.qadr], clearance_only=True)
                if policy is not None:
                    with policy.clearance_scope():
                        if not self.collision.configuration_collides(self.model, self.data):
                            self._contact_clearance_scope = policy.clearance_scope
        scope = self._contact_clearance_scope or nullcontext
        with scope():
            result = self._plan_joint_target(goal, ordered_groups=ordered_groups, via_configurations=via_configurations)
        if result[0] is None:
            self._contact_clearance_scope = previous
        return result

    def _plan_joint_target(self, goal, *, ordered_groups=False, via_configurations=()):
        """Certify a profile posture transition using the same arm collision path."""
        start = self.data.qpos.copy()
        q0 = start[self.qadr].copy()
        deadline = getattr(self, '_pose_planning_deadline', None)
        timeout = min(self.joint_planning_timeout_s, 30.)
        if deadline is not None:
            timeout = max(.001, min(timeout, deadline-time.monotonic()))
        path = plan_arm_joint_path(
            self.model,
            self.data,
            joint_names=self.joint_names,
            q_start=q0,
            q_goal=goal,
            collision=self.collision,
            via_configurations=(*via_configurations, *((*self.approach_joint_routes, *self.ik_seed_configurations) if getattr(self.collision, 'phase', None) == 'pregrasp'
                                and getattr(self.collision, 'payload', None) is None else ())),
            step_size=0.025,
            goal_tolerance=0.01,
            max_iter=400,
            shortcut=False,
            linear_fallback=True,
            planner="linear" if self.coupled_groups else "rrt_connect",
            max_time_s=timeout if getattr(self.collision, 'local_scene', None) is not None else None,
            rng=np.random.default_rng(0) if getattr(self.collision, 'local_scene', None) is not None else None,
        )
        # Coupled actuators cannot use independent-joint RRT samples. When a
        # simultaneous move cuts through furniture, try two bounded group orders
        # using the same segment checker (e.g. raise/orient, then extend).
        if (
            self.coupled_groups
            and (ordered_groups or not path.success)
            and path.reason not in ("invalid_start", "invalid_goal")
        ):
            coupled = {name for group in self.coupled_groups for name in group}
            groups = [[name] for name in self.joint_names if name not in coupled] + list(self.coupled_groups)
            # Pregrasp uses the chain's lift/orient/extend ordering even when
            # simultaneous interpolation is nominally clear: unequal controller
            # lag can sweep low fingertips into the support during that motion.
            if ordered_groups:
                path = replace(path, success=False, reason="ordered_path_failed")
            for order in (groups,) if ordered_groups else (groups, list(reversed(groups))):
                self.data.qpos[:] = start
                current = q0.copy()
                waypoints = [current.copy()]
                for group in order:
                    target = current.copy()
                    for name in group:
                        index = self.joint_names.index(name)
                        target[index] = goal[index]
                    segment = plan_arm_joint_path(
                        self.model,
                        self.data,
                        joint_names=self.joint_names,
                        q_start=current,
                        q_goal=target,
                        collision=self.collision,
                        planner="linear",
                        step_size=0.025,
                    )
                    if not segment.success:
                        if ordered_groups:
                            detail = segment.reason
                            if segment.detail:
                                detail = f"{detail}:{segment.detail}"
                            path = replace(path, detail=detail)
                        break
                    waypoints.extend(segment.waypoints[1:])
                    current = target
                else:
                    path = replace(path, success=True, waypoints=waypoints, planner="linear_group_order", reason=None)
                    break
        self.data.qpos[:] = start
        if path.success:
            self.data.qpos[self.qadr] = goal
            mujoco.mj_kinematics(self.model, self.data)
            if self.collision.payload_body is not None and self.collision.configuration_collides(self.model, self.data):
                return None, "arm_goal_revalidation_failed"
            waypoints = path.waypoints
            clearance = getattr(waypoints, 'clearance_scope', None) or self._contact_clearance_scope
            if clearance is not None:
                self._contact_clearance_scope = clearance
                if not callable(getattr(waypoints, 'clearance_scope', None)):
                    waypoints = ContactClearancePath(waypoints, clearance)
            return waypoints, None
        return None, (
            f"arm_path_failed:{path.reason}:{path.detail}" if path.detail else f"arm_path_failed:{path.reason}"
        )

    def prepare_for_navigation(self, path):
        return self._execute_path("navigation_posture", path)

    def plan_gripper_open(self):
        """Certify opening at the current arm pose; leave private data open on success."""
        if not self.gripper_open_configuration:
            return None, "unsupported_gripper_open_geometry"
        names = tuple(self.gripper_open_configuration)
        addresses = joint_qpos_addrs(self.model, names)
        before = self.data.qpos.copy()
        goal = np.array(list(self.gripper_open_configuration.values()))
        if np.max(np.abs(before[addresses]-goal)) <= self.GRIPPER_TRACKING_TOLERANCE:
            # Retain the measured open geometry. A redundant opening cannot
            # resolve nearby arm clearance; the subsequent arm sweep still has
            # to certify its own departure from the measured starting state.
            return [before[addresses].copy()], None
        try:
            result = plan_arm_joint_path(
                self.model,
                self.data,
                joint_names=names,
                q_start=before[addresses],
                q_goal=goal,
                collision=self.collision,
                planner="linear",
                step_size=0.025,
                goal_tolerance=0.001,
            )
        finally:
            self.data.qpos[:] = before
            mujoco.mj_kinematics(self.model, self.data)
        if not result.success:
            logger.warning('Gripper open rejected: %s; geometry=%s; reason=%s', result.reason,
                           getattr(self.collision, 'last_blocked_geom', None),
                           getattr(self.collision, 'last_reason', None))
            detail = f":{result.detail}" if result.detail else ""
            return None, f"gripper_open_path_failed:{result.reason}{detail}"
        self.data.qpos[addresses] = goal
        mujoco.mj_kinematics(self.model, self.data)
        return result.waypoints, None

    def plan_gripper_close(self):
        """Certify a finite motor target instead of the unbounded close preset.

        The refiner supplies a target and an ordinary measured-motor command
        mapping. Passive geometry still needs the checker's outer bounds. This
        is a sampled path certificate, like arm planning, not force sensing.
        """
        if not self.gripper_close_configuration or not callable(self.command_gripper_joints):
            return None, 'gripper_close_geometry_unavailable'
        names = tuple(self.gripper_close_configuration)
        addresses = joint_qpos_addrs(self.model, names)
        before = self.data.qpos.copy()
        goal = np.array(list(self.gripper_close_configuration.values()))
        try:
            with self._collision_phase('close'):
                probe = getattr(self.collision, 'simulation_contact_terminated_close', False)
                previous = getattr(self.collision, '_planning_gripper_contact_corridor', False)
                try:
                    if probe:
                        self.collision._planning_gripper_contact_corridor = True
                    result = plan_arm_joint_path(self.model, self.data, joint_names=names,
                        q_start=before[addresses], q_goal=goal, collision=self.collision,
                        planner='linear', step_size=.001, goal_tolerance=.0001)
                finally:
                    self.collision._planning_gripper_contact_corridor = previous
                if result.success and probe:
                    # Certify the rest of the task with a bounded contact-stop
                    # hypothesis, never with fingers passing through the object.
                    prefix = []
                    contact = self.collision.target_contact
                    try:
                        # Predict first nominal contact, not the deepest state
                        # admitted by the execution penetration allowance. That
                        # allowance is needed for real contact/model error; it
                        # is not a prediction that fingers compress the object.
                        self.collision.target_contact = replace(contact, max_penetration_m=0.)
                        with self.collision.motion_scope(self.model, self.data, names, before[addresses]):
                            for q in np.linspace(before[addresses], goal, 65):
                                self.data.qpos[addresses] = q
                                if self.collision.configuration_collides(self.model, self.data):
                                    break
                                prefix.append(q.copy())
                    finally:
                        self.collision.target_contact = contact
                    if not prefix:
                        return None, 'gripper_contact_initial_state_invalid'
                    goal = prefix[-1]
                    result = type(result)(True, prefix, 'contact_terminated_closure')
        finally:
            self.data.qpos[:] = before
            mujoco.mj_kinematics(self.model, self.data)
        if not result.success:
            logger.warning('Gripper close rejected: %s; geometry=%s; reason=%s', result.reason,
                           getattr(self.collision, 'last_blocked_geom', None),
                           getattr(self.collision, 'last_reason', None))
            detail = f':{result.detail}' if result.detail else ''
            return None, f'gripper_close_path_failed:{result.reason}{detail}'
        self.data.qpos[addresses] = goal
        mujoco.mj_kinematics(self.model, self.data)
        return result.waypoints, None

    def _close_bounded_gripper(self, timeout_s):
        position = getattr(self.robot, 'get_gripper_position', None)
        if not callable(position):
            return PhysicalMotionResult(False, 'unsupported_gripper_feedback', 'grasp')
        self.synchronize(self.data)
        before = self.data.qpos.copy()
        try:
            path, error = self.plan_gripper_close()
        finally:
            self.data.qpos[:] = before
            mujoco.mj_kinematics(self.model, self.data)
        if error:
            return PhysicalMotionResult(False, error, 'grasp')
        if getattr(self.collision, 'simulation_contact_terminated_close', False):
            target = np.asarray(list(self.gripper_close_configuration.values()), float)
            response = self.command_gripper_joints(target)
            if not isinstance(response, (bool, np.bool_)) or not response:
                return PhysicalMotionResult(False, 'gripper_close_command_failed', 'grasp')
            # Motor settling short of the finite endpoint is only possible
            # contact. The existing measured lift/retention checks prove pickup.
            names = tuple(self.gripper_close_configuration)
            addresses = joint_qpos_addrs(self.model, names)
            with self.collision.motion_scope(self.model, self.data, names, before[addresses]):
                result = self._wait_gripper_still(position, timeout_s)
            if not result.success:
                self.synchronize(self.data)
                addresses = joint_qpos_addrs(self.model, tuple(self.gripper_close_configuration))
                self.command_gripper_joints(self.data.qpos[addresses].copy())
            return result
        names = tuple(self.gripper_close_configuration)
        addresses = joint_qpos_addrs(self.model, names)
        active = self._settling_window(timeout_s)
        started = self._motion_time()
        for target in path[1:]:
            self.synchronize(self.data)
            if not active():
                return PhysicalMotionResult(False, 'gripper_settling_timeout', 'grasp')
            measured_before = self.data.qpos.copy()
            check = plan_arm_joint_path(self.model, self.data, joint_names=names,
                q_start=self.data.qpos[addresses].copy(), q_goal=target, collision=self.collision,
                planner='linear', step_size=.001, goal_tolerance=.0001)
            # Planning modifies private data. Restore measured state before
            # entering transport, even on a rejected segment.
            self.synchronize(self.data)
            if not check.success:
                return PhysicalMotionResult(False, 'gripper_contact_collision', 'grasp')
            if (not np.isfinite(self.data.qpos).all()
                    or np.max(np.abs(self.data.qpos-measured_before)) > self.GRIPPER_TRACKING_TOLERANCE):
                return PhysicalMotionResult(False, 'stale_gripper_plan', 'grasp')
            if not active():
                return PhysicalMotionResult(False, 'gripper_settling_timeout', 'grasp')
            response = self.command_gripper_joints(np.asarray(target).copy())
            if not isinstance(response, (bool, np.bool_)) or not response:
                return PhysicalMotionResult(False, 'gripper_close_command_failed', 'grasp')
            self.synchronize(self.data)
            while True:
                if self.collision.configuration_collides(self.model, self.data):
                    return PhysicalMotionResult(False, 'gripper_contact_collision', 'grasp')
                measured = self.data.qpos[addresses]
                if not np.isfinite(measured).all():
                    return PhysicalMotionResult(False, 'invalid_gripper_feedback', 'grasp')
                if np.max(np.abs(measured-target)) <= self.GRIPPER_TRACKING_TOLERANCE:
                    break
                if not active():
                    return PhysicalMotionResult(False, 'gripper_tracking_failed', 'grasp')
                time.sleep(.01)
                self.synchronize(self.data)
        remaining = timeout_s-(self._motion_time()-started)
        if remaining <= 0:
            return PhysicalMotionResult(False, 'gripper_settling_timeout', 'grasp')
        return self._wait_gripper_still(position, remaining, target_joints=self.gripper_close_configuration)

    def _object_pose(self, object_ref):
        # Native height checks need the geometry origin, not material yaw. The
        # typed checker validates the complete prior and retains shape symmetry.
        if self.input_mode == "observed" and isinstance(self.collision, ObservedNativeCollisionChecker):
            return self.collision.observed_geometry_pose(object_ref)
        if self.object_pose_provider is not None:
            return read_object_pose(self.object_pose_provider, object_ref, inputs=self.input_mode)
        # Explicit oracle evaluation/debug path only. Observed construction
        # rejects a missing provider and cannot read simulator body poses here.
        if self.input_mode != "privileged":
            raise ObservationUnavailable("observed_object_provider_required")
        body = self.data.body(object_ref)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        return pose

    def payload_retained(self):
        """Check the freshly synchronized payload pose, before hypothetical attachment."""
        if self.payload_body is None:
            return True
        expected = self.collision.payload_transform
        if expected is None:
            self.event(phase="payload_retention", accepted=False, reason="missing_payload_reference")
            return False
        try:
            if self.input_mode == "observed" and isinstance(self.collision, ObservedNativeCollisionChecker):
                displacement, angle = self.collision.payload_retention_residual(
                    self.model, self.data, self.payload_body
                )
            else:
                object_pose = self._object_pose(self.payload_body)
                ee = self.data.body(self.ee_body)
                rotation = ee.xmat.reshape(3, 3).T
                position = rotation @ (object_pose[:3, 3] - ee.xpos)
                relative_rotation = rotation @ object_pose[:3, :3]
                angle = np.arccos(np.clip((np.sum(relative_rotation * expected[:3, :3]) - 1) / 2, -1, 1))
                displacement = float(np.linalg.norm(position - expected[:3, 3]))
        except ObservationUnavailable as exc:
            self.event(phase="payload_retention", accepted=False, reason=str(exc))
            return False
        retained = bool(displacement <= 0.02 and angle <= 0.1)
        if not retained:
            self.event(
                phase="payload_retention",
                accepted=False,
                reason="payload_not_retained",
                translation_residual_m=displacement,
                rotation_residual_rad=float(angle),
                translation_tolerance_m=0.02,
                rotation_tolerance_rad=0.1,
            )
        return retained

    def _close_gripper_until_still(self, timeout_s=10.0):
        with self._collision_phase('close'):
            if isinstance(self.collision, ObservedNativeCollisionChecker) and self.collision.target_contact is not None:
                return self._close_bounded_gripper(timeout_s)
            return self._close_gripper_until_still_in_phase(timeout_s)

    def _close_gripper_until_still_in_phase(self, timeout_s):
        """A grasp stops short of the empty-jaw endpoint; verify motor settling.

        Settling is not pickup success. The subsequent measured lift/retention
        checks must establish that the object actually follows the gripper.
        """
        position = getattr(self.robot, "get_gripper_position", None)
        if not callable(position):
            return PhysicalMotionResult(False, "unsupported_gripper_feedback", "grasp")
        if isinstance(self.collision, ObservedNativeCollisionChecker) and self.collision.target_contact is not None:
            self.synchronize(self.data)
            if self.collision.configuration_collides(self.model, self.data):
                return PhysicalMotionResult(False, 'gripper_contact_collision', 'grasp')
        response = (
            self.gripper_close_command()
            if self.gripper_close_command is not None
            else self.robot.close_gripper(blocking=False)
        )
        if not isinstance(response, (bool, np.bool_)) or not response:
            return PhysicalMotionResult(False, "gripper_close_command_failed", "grasp")
        self.synchronize(self.data)
        started = self._motion_time()
        result = self._wait_gripper_still(
            position,
            timeout_s,
            event_phase="gripper_motor_settled" if self.gripper_hold_command is not None else "gripper_settled",
        )
        if not result.success or self.gripper_hold_command is None:
            return result
        response = self.gripper_hold_command()
        if not isinstance(response, (bool, np.bool_)) or not response:
            return PhysicalMotionResult(False, "gripper_hold_command_failed", "grasp")
        self.synchronize(self.data)
        remaining = timeout_s - (self._motion_time() - started)
        if remaining <= 0:
            return PhysicalMotionResult(False, "gripper_settling_timeout", "grasp")
        return self._wait_gripper_still(position, remaining)

    def _wait_gripper_still(self, position, timeout_s, *, event_phase="gripper_settled", target_joints=None):
        start_sim_time = float(self.data.time)
        try:
            previous = float(position())
        except (TypeError, ValueError):
            return PhysicalMotionResult(False, "invalid_gripper_feedback", "grasp")
        active = self._settling_window(timeout_s)
        stable_since = self._motion_time()
        if target_joints is not None:
            addresses = joint_qpos_addrs(self.model, tuple(target_joints))
            target = np.array(list(target_joints.values()))
            model_anchor = self.data.qpos[addresses].copy()
        while active():
            time.sleep(0.05)
            self.synchronize(self.data)
            if (isinstance(self.collision, ObservedNativeCollisionChecker)
                    and self.collision.target_contact is not None
                    and self.collision.configuration_collides(self.model, self.data)):
                return PhysicalMotionResult(False, 'gripper_contact_collision', 'grasp')
            if target_joints is not None:
                measured = self.data.qpos[addresses]
                if not np.isfinite(measured).all():
                    return PhysicalMotionResult(False, 'invalid_gripper_feedback', 'grasp')
                if np.max(np.abs(measured-target)) > self.GRIPPER_TRACKING_TOLERANCE:
                    return PhysicalMotionResult(False, 'gripper_tracking_failed', 'grasp')
                if np.max(np.abs(measured-model_anchor)) > self.GRIPPER_TRACKING_TOLERANCE:
                    stable_since = self._motion_time()
                    model_anchor = measured.copy()
            try:
                current = float(position())
            except (TypeError, ValueError):
                return PhysicalMotionResult(False, "invalid_gripper_feedback", "grasp")
            if not np.isfinite([current, previous]).all():
                return PhysicalMotionResult(False, "invalid_gripper_feedback", "grasp")
            # Accumulate small drift relative to the stable-interval anchor.
            if abs(current - previous) > 0.01:
                stable_since = self._motion_time()
                previous = current
            if self._motion_time() - stable_since >= 0.3 and (
                not self.use_sim_time or self.data.time - start_sim_time >= 0.25
            ):
                self.event(phase=event_phase, measured=current, controller_success=True)
                return PhysicalMotionResult(True, "gripper_settled", "grasp")
        return PhysicalMotionResult(False, "gripper_settling_timeout", "grasp")

    def _replan_at_measured_pose(self, phases, targets, *, object_body, grasp, open_gripper_geometry=True):
        """Arrival tolerance is not an IK certificate: rebuild world-pose paths."""
        snapshot = getattr(self.collision, 'planning_snapshot', nullcontext)
        with snapshot():
            return self._replan_snapshot(phases, targets, object_body=object_body, grasp=grasp,
                                         open_gripper_geometry=open_gripper_geometry)

    def _replan_snapshot(self, phases, targets, *, object_body, grasp, open_gripper_geometry):
        self.synchronize(self.data)
        before = self.data.qpos.copy()
        previous_payload = self.collision.capture_payload()
        paths = []
        phase = phases[0] if phases else "planning"
        try:
            if grasp and open_gripper_geometry and self.gripper_open_configuration:
                addresses = joint_qpos_addrs(self.model, tuple(self.gripper_open_configuration))
                self.data.qpos[addresses] = list(self.gripper_open_configuration.values())
            for phase, (point, rotation) in zip(phases, targets, strict=True):
                with self._collision_phase(phase):
                    if grasp and phase == "lift":
                        self.collision.set_payload(self.model, self.data, object_body, self.ee_body)
                    elif not grasp and phase == "retreat":
                        self.collision.set_payload(self.model, self.data, None)
                        if self.gripper_open_configuration:
                            with self._collision_phase('release'):
                                _, error = self.plan_gripper_open()
                            if error:
                                return None, PhysicalMotionResult(False, error, "release")
                    options = {"ordered_groups": True} if phase == "pregrasp" and self.coupled_groups else {}
                    path, error = self.plan_pose(np.asarray(point), np.asarray(rotation), **options)
                    if (error is None and grasp and phase == 'grasp'
                            and isinstance(self.collision, ObservedNativeCollisionChecker)
                            and self.collision.target_contact is not None):
                        _, error = self.plan_gripper_close()
                if error:
                    return None, PhysicalMotionResult(False, f"measured_pose_replan_failed:{error}", phase)
                paths.append(path)
            return paths, None
        except ObservationUnavailable as exc:
            return None, PhysicalMotionResult(False, str(exc), phase)
        finally:
            self.data.qpos[:] = before
            self.collision.restore_payload(previous_payload)
            mujoco.mj_kinematics(self.model, self.data)

    def _execute_path(self, phase, path):
        clearance = getattr(path, 'clearance_scope', None)
        if callable(clearance):
            self._contact_clearance_scope = clearance
        escape_end = getattr(path, 'escape_end_index', None)
        if escape_end is not None:
            prefix = DeparturePath(path.departure.execution_waypoints(path[:escape_end + 1]), path.departure)
            result = self._execute_path(phase, prefix)
            if not result.success or escape_end == len(path) - 1:
                return result
            suffix = path[escape_end:]
            clearance_scope = getattr(path, 'clearance_scope', None)
            if clearance_scope is not None:
                suffix = ContactClearancePath(suffix, clearance_scope)
            return self._execute_path(phase, suffix)
        with self._collision_phase(phase):
            scope = (self.collision.motion_scope(self.model, self.data, self.joint_names, path[0])
                     if hasattr(self.collision, 'motion_scope') and path else nullcontext())
            clearance_scope = getattr(path, 'clearance_scope', nullcontext)
            with scope, clearance_scope():
                previous = getattr(self.collision, 'active_departure', None)
                departure = getattr(path, 'departure', None)
                if departure is not None:
                    self.collision.active_departure = departure
                try:
                    return self._execute_path_in_phase(phase, path)
                finally:
                    if departure is not None:
                        self.collision.active_departure = previous

    def _execute_path_in_phase(self, phase, path):
        departure = getattr(path, 'departure', None)
        departure_deadline = time.monotonic() + min(30., max(5., 2.*len(path))) if departure is not None else None
        joint_tolerance = min(self.joint_tolerance, .001) if departure is not None else self.joint_tolerance
        scales = np.ones(len(self.joint_names))
        if self.linear_joint_tolerance_m is not None:
            for i, name in enumerate(self.joint_names):
                if self.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE:
                    scales[i] = joint_tolerance / min(joint_tolerance, self.linear_joint_tolerance_m)

        def joint_residual(target):
            delta = self.data.qpos[self.qadr] - target
            if not np.isfinite(delta).all():
                return float("inf")
            errors = [float(np.max(np.abs(delta)*scales))]
            errors.extend(
                abs(float(sum(delta[self.joint_names.index(n)] for n in group))) for group in self.coupled_groups
            )
            return max(errors)

        self.synchronize(self.data)
        base_reference = None
        if self.base_body is not None:
            base = self.data.body(self.base_body)
            base_reference = (base.xpos.copy(), base.xmat.copy())

        def base_drifted():
            if base_reference is None:
                return False
            base = self.data.body(self.base_body)
            distance = float(np.linalg.norm(base.xpos - base_reference[0]))
            angle = float(np.arccos(np.clip((np.sum(base.xmat * base_reference[1]) - 1) / 2, -1, 1)))
            if not np.isfinite([distance, angle]).all() or distance > 0.01 or angle > 0.02:
                self.event(phase=phase, reason="base_drift_during_arm", base_distance_m=distance, base_angle_rad=angle)
                return True
            return False

        if not self.payload_retained():
            return PhysicalMotionResult(False, "payload_not_retained", phase)
        residual = joint_residual(path[0])
        if residual > joint_tolerance:
            return PhysicalMotionResult(False, "stale_arm_plan", phase, residual)
        waypoints = path
        if self.coalesce_execution_waypoints and departure is None:
            maximum = np.array([.005 if self.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE else .1
                                for name in self.joint_names])
            waypoints = coalesce_collinear_waypoints(path, maximum)
        for target in waypoints[1:]:
            if departure_deadline is not None and time.monotonic() >= departure_deadline:
                return PhysicalMotionResult(False, "contact_departure_timeout", phase)
            command_started = time.monotonic()
            # Recheck the next segment against fresh state, including the payload.
            self.synchronize(self.data)
            if base_drifted():
                return PhysicalMotionResult(False, "base_drift_during_arm", phase)
            if not self.payload_retained():
                return PhysicalMotionResult(False, "payload_not_retained", phase)
            current = self.data.qpos[self.qadr].copy()
            departure = getattr(path, 'departure', None)
            if departure is not None and not departure.validate_edge(current, target):
                return PhysicalMotionResult(False, 'contact_departure_invalidated', phase)
            check = plan_arm_joint_path(
                self.model,
                self.data,
                joint_names=self.joint_names,
                q_start=current,
                q_goal=target,
                collision=self.collision,
                planner="linear",
                step_size=0.025,
                goal_tolerance=0.,
                linear_steps=max(15, int(np.ceil(max(
                    abs(float(b-a))/(.002 if self.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE else .025)
                    for name, a, b in zip(self.joint_names, current, target, strict=True)
                )))+1),
            )
            if not check.success:
                self.event(
                    phase=phase,
                    reason=check.reason,
                    detail=check.detail,
                    measured=current.tolist(),
                    command=np.asarray(target).tolist(),
                    contacts=list(self.collision.last_contacts) if check.detail == "collision" else [],
                )
                reason = f"revalidation_failed:{check.reason}"
                if check.detail:
                    reason += f":{check.detail}"
                return PhysicalMotionResult(False, reason, phase)
            response = self.command_joints(np.asarray(target))
            ok = isinstance(response, (bool, np.bool_)) and bool(response)
            self.synchronize(self.data)
            if base_drifted():
                return PhysicalMotionResult(False, "base_drift_during_arm", phase)
            if not self.payload_retained():
                return PhysicalMotionResult(False, "payload_not_retained", phase)
            residual = joint_residual(target)
            # A client may acknowledge within a looser tolerance before the
            # simulator has applied/settled the command. Wait for measured
            # convergence, retaining the same certificate tolerance.
            active = self._settling_window(self.joint_settle_timeout_s)
            while ok and np.isfinite(residual) and residual > joint_tolerance and active():
                time.sleep(0.05)
                if departure_deadline is not None and time.monotonic() >= departure_deadline:
                    return PhysicalMotionResult(False, "contact_departure_timeout", phase)
                self.synchronize(self.data)
                if base_drifted():
                    return PhysicalMotionResult(False, "base_drift_during_arm", phase)
                if not self.payload_retained():
                    return PhysicalMotionResult(False, "payload_not_retained", phase)
                residual = joint_residual(target)
            if not np.isfinite(residual):
                return PhysicalMotionResult(False, "invalid_measured_joint_state", phase)
            self.event(
                phase=phase,
                command=np.asarray(target).tolist(),
                measured=self.data.qpos[self.qadr].tolist(),
                residual=residual,
                controller_success=bool(ok),
            )
            if departure is not None:
                time.sleep(max(0., .05 - (time.monotonic() - command_started)))
            if not ok or residual > joint_tolerance:
                return PhysicalMotionResult(False, "arm_tracking_failed", phase, residual)
        if self.endpoint_settle_s:
            # A waypoint tolerance can be reached while the arm is still moving.
            # Contact transitions need a settled endpoint, not merely an ACK.
            active = self._settling_window(self.joint_settle_timeout_s)
            stable_since = self._motion_time()
            anchor = self.data.qpos[self.qadr].copy()
            while active():
                time.sleep(0.05)
                self.synchronize(self.data)
                if base_drifted():
                    return PhysicalMotionResult(False, "base_drift_during_arm", phase)
                if not self.payload_retained():
                    return PhysicalMotionResult(False, "payload_not_retained", phase)
                residual = joint_residual(path[-1])
                if not np.isfinite(residual):
                    return PhysicalMotionResult(False, "invalid_measured_joint_state", phase)
                if residual > joint_tolerance or joint_residual(anchor) > self.ENDPOINT_STILLNESS_TOLERANCE:
                    anchor = self.data.qpos[self.qadr].copy()
                    stable_since = self._motion_time()
                elif self._motion_time() - stable_since >= self.endpoint_settle_s:
                    break
            else:
                return PhysicalMotionResult(False, "arm_tracking_failed", phase, residual)
        self.event(phase=f"{phase}_complete", controller_success=True, residual=residual)
        return PhysicalMotionResult(True, "ok", phase, residual)

    def _visual_check(self, method):
        try:
            effect = getattr(self.visual_verifier, method)(after=time.monotonic())
        except ObservationUnavailable as exc:
            self.event(phase=method, accepted=False, reason=str(exc), evidence_state="unknown")
            return PhysicalMotionResult(False, "visual_effect_unknown", method)
        self.event(phase=method, accepted=effect.satisfied, effect=effect.as_dict())
        if not effect.satisfied:
            reason = "visual_effect_unknown" if effect.state == "unknown" else effect.reason
            return PhysicalMotionResult(False, reason, method)
        return None

    def grasp_only(self, object_query, *, object_gt_body=None, grasp_T_world=None):
        if self.input_mode == "observed" and self.visual_verifier is None:
            return PhysicalMotionResult(False, "visual_verification_unavailable", "preflight")
        if len(self.grasp_paths) != 3 or len(self.grasp_targets) != 3 or object_gt_body is None:
            return PhysicalMotionResult(False, "missing_validated_grasp", "grasp")
        already_open = False
        if self.gripper_open_configuration:
            self.synchronize(self.data)
            before = self.data.qpos.copy()
            try:
                opening, error = self.plan_gripper_open()
                already_open = error is None and len(opening) == 1
            finally:
                self.data.qpos[:] = before
                mujoco.mj_kinematics(self.model, self.data)
            if error:
                return PhysicalMotionResult(False, error, "gripper_open")

        def replan(phase, *, open_geometry):
            for option, targets in enumerate(self.grasp_target_options or [self.grasp_targets]):
                paths, error = self._replan_at_measured_pose(
                    ("pregrasp", "grasp", "lift"),
                    targets,
                    object_body=object_gt_body,
                    grasp=True,
                    open_gripper_geometry=open_geometry,
                )
                self.event(
                    phase=phase,
                    option=option,
                    accepted=error is None,
                    reason=None if error is None else error.message,
                    targets=[[np.asarray(p).tolist(), np.asarray(r).tolist()] for p, r in targets],
                )
                if error is None:
                    self.grasp_targets = targets
                    break
            return paths, error

        limits = np.array([self.linear_joint_tolerance_m
                           if self.linear_joint_tolerance_m is not None and self.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE
                           else self.joint_tolerance for name in self.joint_names])
        reusable = self.reuse_validated_grasp_paths and already_open and self.grasp_paths[0] and np.all(
            np.abs(self.data.qpos[self.qadr]-self.grasp_paths[0][0]) <= limits)
        if reusable:
            paths = self.grasp_paths
            self.event(phase="grasp_plan_reused", controller_success=True)
        else:
            paths, error = replan("grasp_replan", open_geometry=True)
            if error is not None:
                return error
            if not self.robot.open_gripper(blocking=True):
                return PhysicalMotionResult(False, "gripper_open_failed", "pregrasp")
            # Opening changes collision geometry and may slightly disturb the arm.
            # Rebuild paths from the measured open state before any arm motion.
            paths, error = replan("grasp_replan_after_open", open_geometry=False)
            if error is not None:
                return error
        self.grasp_paths = paths
        for phase, path in zip(("pregrasp", "grasp"), self.grasp_paths[:2], strict=True):
            result = self._execute_path(phase, path)
            if not result.success:
                return result
        if self.input_mode == "observed":
            error = self._visual_check("begin_grasp")
            if error is not None:
                return error
        closing = self._close_gripper_until_still()
        if not closing.success:
            return closing
        self.synchronize(self.data)
        try:
            initial_object_height = float(self._object_pose(object_gt_body)[2, 3])
            self.collision.set_payload(self.model, self.data, object_gt_body, self.ee_body)
        except ObservationUnavailable as exc:
            return PhysicalMotionResult(False, str(exc), "grasp")
        self.payload_body = object_gt_body
        result = self._execute_path("lift", self.grasp_paths[2])
        if result.success:
            try:
                height = self._object_pose(object_gt_body)[2, 3]
            except ObservationUnavailable as exc:
                return PhysicalMotionResult(False, str(exc), "lift")
            if height - initial_object_height < 0.05:
                return PhysicalMotionResult(False, "object_not_lifted", "lift")
            if self.input_mode == "observed":
                error = self._visual_check("verify_lift")
                if error is not None:
                    return error
        return result

    def place_only(self, receptacle_query, *, object_gt_body=None, receptacle_gt_body=None):
        if self.input_mode == "observed" and self.visual_verifier is None:
            return PhysicalMotionResult(False, "visual_verification_unavailable", "preflight")
        if self.input_mode == "observed" and self.visual_verifier.lift_reference is None:
            return PhysicalMotionResult(False, "visual_effect_unknown", "preflight")
        if (
            self.payload_body != object_gt_body
            or len(self.place_paths) != 3
            or len(self.place_targets) != 3
            or self.transport is None
        ):
            return PhysicalMotionResult(False, "missing_validated_place", "place")
        result = self.transport()
        if not result.success:
            reason = getattr(result, "reason", None) or getattr(result, "message", "transport_failed")
            return PhysicalMotionResult(False, reason, "transport")
        paths, error = self._replan_at_measured_pose(
            ("preplace", "place", "retreat"),
            self.place_targets,
            object_body=object_gt_body,
            grasp=False,
        )
        if error is not None:
            return error
        self.place_paths = paths
        for phase, path in zip(("preplace", "place"), self.place_paths[:2], strict=True):
            result = self._execute_path(phase, path)
            if not result.success:
                return result
        if self.release_refiner is not None:
            result = self.release_refiner()
            if not result.success:
                return result
        # Certify release and open-finger retreat from the actual placement pose.
        # Planning preserves measured state and the retention reference on failure.
        paths, error = self._replan_at_measured_pose(
            ("retreat",),
            self.place_targets[2:],
            object_body=object_gt_body,
            grasp=False,
        )
        if error is not None:
            return error
        if not self.robot.open_gripper(blocking=True):
            return PhysicalMotionResult(False, "gripper_open_failed", "release")
        self.event(phase="release", controller_success=True)
        self.collision.set_payload(self.model, self.data, None)
        self.payload_body = None
        if self.release_verifier is not None:
            result = self.release_verifier()
            if not result.success:
                return result
        if self.released_collision is not None:
            self.collision = self.released_collision
        paths, error = self._replan_at_measured_pose(
            ("retreat",),
            self.place_targets[2:],
            object_body=object_gt_body,
            grasp=False,
        )
        if error is not None:
            return error
        self.place_paths[2] = paths[0]
        result = self._execute_path("retreat", paths[0])
        if result.success and self.input_mode == "observed":
            error = self._visual_check("verify_place")
            if error is not None:
                return error
        return result
