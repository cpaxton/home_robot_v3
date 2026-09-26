# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Contact-based execution through shared robot controllers.

The caller supplies observed state/geometry (GT only in the oracle evaluation).
This executor never attaches objects or sets live simulator poses. It is usable
through the same ``execute_task_plan`` seam as symbolic and latch controls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
        self.transport: Callable | None = None
        self.payload_body: str | None = None

    def plan_pose(self, position, rotation):
        """Plan from the offline current state; caller preserves candidate state."""
        start = self.data.qpos.copy()
        q0 = start[self.qadr].copy()
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
        self.data.qpos[:] = start
        if path.success:
            self.data.qpos[self.qadr] = goal
            mujoco.mj_forward(self.model, self.data)
            return path.waypoints, None
        return None, f"arm_path_failed:{path.reason}"

    def _execute_path(self, phase, path):
        self.synchronize(self.data)
        residual = float(np.max(np.abs(self.data.qpos[self.qadr] - path[0])))
        if residual > self.joint_tolerance:
            return PhysicalMotionResult(False, "stale_arm_plan", phase, residual)
        for target in path[1:]:
            # Recheck the next segment against fresh state, including the payload.
            self.synchronize(self.data)
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
            ok = self.command_joints(np.asarray(target))
            self.synchronize(self.data)
            residual = float(np.max(np.abs(self.data.qpos[self.qadr] - target)))
            self.event(
                phase=phase,
                command=np.asarray(target).tolist(),
                measured=self.data.qpos[self.qadr].tolist(),
                residual=residual,
                controller_success=bool(ok),
            )
            if not ok or residual > self.joint_tolerance:
                return PhysicalMotionResult(False, "arm_tracking_failed", phase, residual)
        return PhysicalMotionResult(True, "ok", phase, residual)

    def grasp_only(self, object_query, *, object_gt_body=None, grasp_T_world=None):
        if len(self.grasp_paths) != 3 or object_gt_body is None:
            return PhysicalMotionResult(False, "missing_validated_grasp", "grasp")
        if not self.robot.open_gripper(blocking=True):
            return PhysicalMotionResult(False, "gripper_open_failed", "pregrasp")
        for phase, path in zip(("pregrasp", "grasp"), self.grasp_paths[:2], strict=True):
            result = self._execute_path(phase, path)
            if not result.success:
                return result
        if not self.robot.close_gripper(blocking=True):
            return PhysicalMotionResult(False, "gripper_close_failed", "grasp")
        self.synchronize(self.data)
        self.collision.set_payload(self.model, self.data, object_gt_body, self.ee_body)
        self.payload_body = object_gt_body
        result = self._execute_path("lift", self.grasp_paths[2])
        self.event(phase="lift_complete", controller_success=result.success)
        return result

    def place_only(self, receptacle_query, *, object_gt_body=None, receptacle_gt_body=None):
        if self.payload_body != object_gt_body or len(self.place_paths) != 3 or self.transport is None:
            return PhysicalMotionResult(False, "missing_validated_place", "place")
        result = self.transport()
        if not result.success:
            return PhysicalMotionResult(False, getattr(result, "reason", "transport_failed"), "transport")
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
