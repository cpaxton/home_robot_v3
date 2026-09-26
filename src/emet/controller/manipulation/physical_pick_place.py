# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Contact-based execution through shared robot controllers.

The caller supplies observed state/geometry (GT only in the oracle evaluation).
This executor never attaches objects or sets live simulator poses. It is usable
through the same ``execute_task_plan`` seam as symbolic and latch controls.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.mujoco_arm_ik import joint_qpos_addrs, solve_pose_ik


@dataclass
class PhysicalMotionResult:
    success: bool
    message: str
    phase: str = ""
    residual: float | None = None


class PhysicalPickPlaceExecutor:
    execution_mode = "physical"

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
        position_tolerance_m: float = 0.01,
        orientation_tolerance_rad: float = 0.1,
    ):
        self.robot, self.model, self.data = robot, model, data
        self.ee_body, self.joint_names = ee_body, tuple(joint_names)
        self.collision = collision
        self.synchronize, self.command_joints = synchronize, command_joints
        self.coupled_groups, self.event = coupled_groups, event
        self.joint_tolerance = joint_tolerance
        self.position_tolerance_m, self.orientation_tolerance_rad = position_tolerance_m, orientation_tolerance_rad
        self.qadr = joint_qpos_addrs(model, self.joint_names)
        self.grasp_paths: list = []
        self.place_paths: list = []
        self.grasp_targets: list = []
        self.place_targets: list = []
        self.transport: Callable | None = None
        self.payload_body: str | None = None

    def plan_pose(self, position, rotation):
        """Plan from the offline current state; caller preserves candidate state."""
        start = self.data.qpos.copy()
        result = solve_pose_ik(
            self.model,
            self.data,
            ee_body=self.ee_body,
            joint_names=self.joint_names,
            target_pos=position,
            target_rotation=rotation,
            coupled_groups=self.coupled_groups,
            tol_m=self.position_tolerance_m,
            tol_rad=self.orientation_tolerance_rad,
        )
        if not result.success:
            self.data.qpos[:] = start
            return None, f"pose_ik_failed:position={result.pos_error_m:.4f},rotation={result.orientation_error_rad:.4f}"
        goal = self.data.qpos[self.qadr].copy()
        self.data.qpos[:] = start
        return self.plan_joint_target(goal)

    def plan_joint_target(self, goal):
        """Certify a profile posture transition using the same arm collision path."""
        start = self.data.qpos.copy()
        q0 = start[self.qadr].copy()
        path = plan_arm_joint_path(
            self.model,
            self.data,
            joint_names=self.joint_names,
            q_start=q0,
            q_goal=goal,
            collision=self.collision,
            step_size=0.025,
            goal_tolerance=0.01,
            max_iter=400,
            shortcut=False,
            linear_fallback=True,
            planner="linear" if self.coupled_groups else "rrt_connect",
        )
        # Coupled actuators cannot use independent-joint RRT samples. When a
        # simultaneous move cuts through furniture, try two bounded group orders
        # using the same segment checker (e.g. raise/orient, then extend).
        if self.coupled_groups and not path.success and path.reason not in ("invalid_start", "invalid_goal"):
            coupled = {name for group in self.coupled_groups for name in group}
            groups = [[name] for name in self.joint_names if name not in coupled] + list(self.coupled_groups)
            for order in (groups, list(reversed(groups))):
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
            return path.waypoints, None
        return None, f"arm_path_failed:{path.reason}"

    def prepare_for_navigation(self, path):
        return self._execute_path("navigation_posture", path)

    def payload_retained(self):
        """Check the freshly synchronized payload pose, before hypothetical attachment."""
        if self.payload_body is None:
            return True
        expected = self.collision.payload_transform
        if expected is None:
            return False
        ee, obj = self.data.body(self.ee_body), self.data.body(self.payload_body)
        rotation = ee.xmat.reshape(3, 3).T
        position = rotation @ (obj.xpos - ee.xpos)
        relative_rotation = rotation @ obj.xmat.reshape(3, 3)
        angle = np.arccos(np.clip((np.sum(relative_rotation * expected[:3, :3]) - 1) / 2, -1, 1))
        return bool(np.linalg.norm(position - expected[:3, 3]) <= 0.02 and angle <= 0.1)

    def _close_gripper_until_still(self, timeout_s=10.0):
        """A grasp stops short of the empty-jaw endpoint; verify motor settling.

        Settling is not pickup success. The subsequent measured lift/retention
        checks must establish that the object actually follows the gripper.
        """
        position = getattr(self.robot, "get_gripper_position", None)
        if not callable(position):
            return PhysicalMotionResult(False, "unsupported_gripper_feedback", "grasp")
        response = self.robot.close_gripper(blocking=False)
        if not isinstance(response, (bool, np.bool_)) or not response:
            return PhysicalMotionResult(False, "gripper_close_command_failed", "grasp")
        self.synchronize(self.data)
        start_sim_time = float(self.data.time)
        previous = float(position())
        started = stable_since = time.monotonic()
        while time.monotonic() - started < timeout_s:
            time.sleep(0.05)
            self.synchronize(self.data)
            current = float(position())
            if not np.isfinite([current, previous]).all():
                return PhysicalMotionResult(False, "invalid_gripper_feedback", "grasp")
            if abs(current - previous) > 0.01:
                stable_since = time.monotonic()
            if time.monotonic() - stable_since >= 0.3 and self.data.time - start_sim_time >= 0.25:
                self.event(phase="gripper_settled", measured=current, controller_success=True)
                return PhysicalMotionResult(True, "gripper_settled", "grasp")
            previous = current
        return PhysicalMotionResult(False, "gripper_settling_timeout", "grasp")

    def _replan_at_measured_pose(self, phases, targets, *, object_body, grasp):
        """Arrival tolerance is not an IK certificate: rebuild world-pose paths."""
        self.synchronize(self.data)
        before = self.data.qpos.copy()
        previous_payload = (
            self.collision.payload_body,
            self.collision.payload_parent,
            None if self.collision.payload_transform is None else self.collision.payload_transform.copy(),
        )
        paths = []
        try:
            for phase, (point, rotation) in zip(phases, targets, strict=True):
                if grasp and phase == "lift":
                    self.collision.set_payload(self.model, self.data, object_body, self.ee_body)
                elif not grasp and phase == "retreat":
                    self.collision.set_payload(self.model, self.data, None)
                path, error = self.plan_pose(np.asarray(point), np.asarray(rotation))
                if error:
                    return None, PhysicalMotionResult(False, f"measured_pose_replan_failed:{error}", phase)
                paths.append(path)
            return paths, None
        finally:
            self.data.qpos[:] = before
            self.collision.payload_body, self.collision.payload_parent, self.collision.payload_transform = (
                previous_payload
            )
            mujoco.mj_kinematics(self.model, self.data)

    def _execute_path(self, phase, path):
        def joint_residual(target):
            delta = self.data.qpos[self.qadr] - target
            errors = [float(np.max(np.abs(delta)))]
            errors.extend(
                abs(float(sum(delta[self.joint_names.index(n)] for n in group))) for group in self.coupled_groups
            )
            return max(errors)

        self.synchronize(self.data)
        if not self.payload_retained():
            return PhysicalMotionResult(False, "payload_not_retained", phase)
        residual = joint_residual(path[0])
        if residual > self.joint_tolerance:
            return PhysicalMotionResult(False, "stale_arm_plan", phase, residual)
        for target in path[1:]:
            # Recheck the next segment against fresh state, including the payload.
            self.synchronize(self.data)
            if not self.payload_retained():
                return PhysicalMotionResult(False, "payload_not_retained", phase)
            current = self.data.qpos[self.qadr].copy()
            check = plan_arm_joint_path(
                self.model,
                self.data,
                joint_names=self.joint_names,
                q_start=current,
                q_goal=target,
                collision=self.collision,
                planner="linear",
                step_size=0.025,
                goal_tolerance=0.001,
            )
            if not check.success:
                return PhysicalMotionResult(False, f"revalidation_failed:{check.reason}", phase)
            response = self.command_joints(np.asarray(target))
            ok = isinstance(response, (bool, np.bool_)) and bool(response)
            self.synchronize(self.data)
            if not self.payload_retained():
                return PhysicalMotionResult(False, "payload_not_retained", phase)
            residual = joint_residual(target)
            self.event(
                phase=phase,
                command=np.asarray(target).tolist(),
                measured=self.data.qpos[self.qadr].tolist(),
                residual=residual,
                controller_success=bool(ok),
            )
            if not ok or residual > self.joint_tolerance:
                return PhysicalMotionResult(False, "arm_tracking_failed", phase, residual)
        self.event(phase=f"{phase}_complete", controller_success=True, residual=residual)
        return PhysicalMotionResult(True, "ok", phase, residual)

    def grasp_only(self, object_query, *, object_gt_body=None, grasp_T_world=None):
        if len(self.grasp_paths) != 3 or len(self.grasp_targets) != 3 or object_gt_body is None:
            return PhysicalMotionResult(False, "missing_validated_grasp", "grasp")
        paths, error = self._replan_at_measured_pose(
            ("pregrasp", "grasp", "lift"),
            self.grasp_targets,
            object_body=object_gt_body,
            grasp=True,
        )
        if error is not None:
            return error
        self.grasp_paths = paths
        if not self.robot.open_gripper(blocking=True):
            return PhysicalMotionResult(False, "gripper_open_failed", "pregrasp")
        for phase, path in zip(("pregrasp", "grasp"), self.grasp_paths[:2], strict=True):
            result = self._execute_path(phase, path)
            if not result.success:
                return result
        closing = self._close_gripper_until_still()
        if not closing.success:
            return closing
        self.synchronize(self.data)
        initial_object_height = float(self.data.body(object_gt_body).xpos[2])
        self.collision.set_payload(self.model, self.data, object_gt_body, self.ee_body)
        self.payload_body = object_gt_body
        result = self._execute_path("lift", self.grasp_paths[2])
        if result.success and self.data.body(object_gt_body).xpos[2] - initial_object_height < 0.05:
            return PhysicalMotionResult(False, "object_not_lifted", "lift")
        return result

    def place_only(self, receptacle_query, *, object_gt_body=None, receptacle_gt_body=None):
        if (
            self.payload_body != object_gt_body
            or len(self.place_paths) != 3
            or len(self.place_targets) != 3
            or self.transport is None
        ):
            return PhysicalMotionResult(False, "missing_validated_place", "place")
        result = self.transport()
        if not result.success:
            return PhysicalMotionResult(False, getattr(result, "reason", "transport_failed"), "transport")
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
        if not self.robot.open_gripper(blocking=True):
            return PhysicalMotionResult(False, "gripper_open_failed", "release")
        self.event(phase="release", controller_success=True)
        self.collision.set_payload(self.model, self.data, None)
        self.payload_body = None
        return self._execute_path("retreat", self.place_paths[2])
