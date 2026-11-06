# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded native release-pose search on private data.

This first provider uses privileged rigid geometry. Collision tests sample joint
edges and vertical post-release positions; they are not a dynamics certificate.
No live object pose, contact tolerance or retention reference is changed.
"""

from __future__ import annotations

import time
from copy import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker, body_subtree
from emet.simulation.sim_object_placements import placements_from_mujoco_model


@dataclass(frozen=True)
class ReleaseLimits:
    max_raise_m: float
    max_drop_m: float
    clearance_m: float = 0.003
    offsets_m: tuple[float, ...] = (0.003, 0.006, 0.01)
    max_adjustments: int = 2
    height_joint: str = "joint_lift"

    def __post_init__(self):
        values = [self.max_raise_m, self.max_drop_m, self.clearance_m, *self.offsets_m]
        if (
            not np.isfinite(values).all()
            or min(values) < 0
            or type(self.max_adjustments) is not int
            or self.max_adjustments < 0
            or not self.offsets_m
            or tuple(sorted(set(self.offsets_m))) != self.offsets_m
        ):
            raise ValueError("invalid_release_limits")


def body_bounds(model, data, ids):
    boxes = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in ids or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        rotation = data.geom_xmat[gid].reshape(3, 3)
        center = data.geom_xpos[gid] + rotation @ model.geom_aabb[gid, :3]
        half = np.abs(rotation) @ model.geom_aabb[gid, 3:]
        boxes.append((center - half, center + half))
    if not boxes:
        raise ValueError("missing_payload_collision_volume")
    return np.stack((np.min(np.array(boxes)[:, 0], axis=0), np.max(np.array(boxes)[:, 1], axis=0)))


def relative_pose(data, body, parent):
    obj, ee = data.body(body), data.body(parent)
    result = np.eye(4)
    result[:3, :3] = ee.xmat.reshape(3, 3).T @ obj.xmat.reshape(3, 3)
    result[:3, 3] = ee.xmat.reshape(3, 3).T @ (obj.xpos - ee.xpos)
    return result


def set_private_payload(model, data, body, parent, transform):
    root = model.body(body)
    joint = int(root.jntadr[0])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("release_requires_free_rigid_payload")
    ee = data.body(parent)
    rotation = ee.xmat.reshape(3, 3)
    addr = int(model.jnt_qposadr[joint])
    data.qpos[addr : addr + 3] = ee.xpos + rotation @ transform[:3, 3]
    mujoco.mju_mat2Quat(data.qpos[addr + 3 : addr + 7], (rotation @ transform[:3, :3]).ravel())
    mujoco.mj_kinematics(model, data)


class HeldPoseChecker(MujocoSceneCollisionChecker):
    """Check the retained planning pose and the measured held pose at each sample."""

    def configuration_collides(self, model, data):
        original = self.payload_transform
        try:
            for pose in (original, self.measured_transform) if self.payload_body else (None,):
                self.payload_transform = pose
                if super().configuration_collides(model, data):
                    return True
            return False
        finally:
            self.payload_transform = original


def held_checker(base, measured):
    checker = HeldPoseChecker.__new__(HeldPoseChecker)
    checker.__dict__ = base.__dict__.copy()
    checker.payload_transform = base.payload_transform.copy()
    checker.measured_transform = measured.copy()
    checker.last_contacts = []
    return checker


class ReleasedFallChecker(MujocoSceneCollisionChecker):
    """Watch the released object too, with no gripper/object collision exemption.

    Samples a fixed-orientation vertical descent from fresh measured position to
    the declared support. Actual post-release rotation/dynamics remain measured
    execution/scoring responsibilities, not a guarantee made by this probe.
    """

    def __init__(self, model, base, obj, support, patch, region, limits):
        self.__dict__ = base.__dict__.copy()
        self.payload_body = self.payload_parent = self.payload_transform = None
        self.ids = body_subtree(model, obj)
        names = {model.body(b).name for b in self.ids}
        self.robot_ids = base.robot_ids | self.ids
        self.allowed_pairs = {p for p in base.allowed_pairs if not names.intersection(p)}
        self.allowed_pairs |= {frozenset((name, support)) for name in names}
        self.obj, self.patch, self.region, self.limits = obj, patch, region, limits
        self._probe = mujoco.MjData(model)
        self.last_contacts = []

    def configuration_collides(self, model, data):
        mujoco.mj_copyData(self._probe, model, data)
        probe = self._probe
        mujoco.mj_kinematics(model, probe)
        bounds = body_bounds(model, probe, self.ids)
        gap = float(bounds[0, 2] - self.patch[1, 2])
        if not bounds_fit(bounds, self.patch, self.region, self.limits.max_drop_m):
            self.last_contacts = [{"bodies": [self.obj], "reason": "release_bounds_invalid"}]
            return True
        addr = int(model.jnt_qposadr[model.body(self.obj).jntadr[0]])
        z = float(probe.qpos[addr + 2])
        for shift in np.linspace(0, max(0.0, gap), max(2, int(np.ceil(max(0.0, gap) / 0.002)) + 1)):
            probe.qpos[addr + 2] = z - shift
            if MujocoSceneCollisionChecker.configuration_collides(self, model, probe):
                return True
        return False


def bounds_fit(bounds, patch, region, max_drop):
    gap = bounds[0, 2] - patch[1, 2]
    return bool(
        -0.001 <= gap <= max_drop
        and np.all(bounds[0, :2] >= patch[0, :2])
        and np.all(bounds[1, :2] <= patch[1, :2])
        and (
            region is None
            or (np.all(bounds[0, :2] >= region[0] - 0.005) and np.all(bounds[1, :2] <= region[1] + 0.005))
        )
    )


@dataclass
class ReleasePlan:
    prefix: list
    checker: HeldPoseChecker
    released_checker: ReleasedFallChecker
    target_z: float
    gap_m: float


def plan_release(executor, *, support, region, limits, initial_z, attempted=()):
    """Certify closed-arm motion, opening (including reserve) and retreat; no IO."""
    ex = executor
    if (
        ex.input_mode != "privileged"
        or type(ex.collision) not in (MujocoSceneCollisionChecker, HeldPoseChecker)
        or ex.payload_body is None
        or ex.collision.payload_transform is None
    ):
        return None, "native_execution_unavailable"
    model = ex.model
    cursor = int(model.body(support).id)
    while cursor:
        if model.body_jntnum[cursor]:
            return None, "native_execution_unavailable"
        cursor = int(model.body_parentid[cursor])
    root = int(model.body(ex.payload_body).id)
    if any(model.body_jntnum[b] for b in body_subtree(model, ex.payload_body) if b != root):
        return None, "native_execution_unavailable"
    probe = copy(ex)
    probe.data = mujoco.MjData(model)
    mujoco.mj_copyData(probe.data, model, ex.data)
    mujoco.mj_kinematics(model, probe.data)
    measured = relative_pose(probe.data, ex.payload_body, ex.ee_body)
    probe.collision = held_checker(ex.collision, measured)
    original = probe.data.qpos.copy()
    start = original[ex.qadr].copy()
    current = probe.data.body(ex.ee_body)
    current_point, current_rotation = current.xpos.copy(), current.xmat.reshape(3, 3).copy()
    height = model.joint(limits.height_joint)
    height_z = float(probe.data.xaxis[height.id, 2])
    if height.type[0] != mujoco.mjtJoint.mjJNT_SLIDE or height_z < 0.99:
        return None, "native_execution_unavailable"
    surfaces = placements_from_mujoco_model(model, probe.data)[support]["support_surfaces"]
    region = None if region is None else np.asarray(region)
    targets = [(current_point, current_rotation)]
    nominal, rotation = ex.place_targets[1]
    targets += [(np.asarray(nominal) + [0, 0, delta], rotation) for delta in limits.offsets_m]
    last = "no_plan_within_budget"
    for index, (point, rotation) in enumerate(targets):
        target_z = float(point[2])
        if target_z > initial_z + limits.max_raise_m or (
            index and (target_z < current_point[2] or any(abs(target_z - old) < 1e-6 for old in attempted))
        ):
            continue
        probe.data.qpos[:] = original
        probe.collision = held_checker(ex.collision, measured)
        mujoco.mj_kinematics(model, probe.data)
        if index:
            path, error = probe.plan_pose(np.asarray(point), np.asarray(rotation))
            if error:
                last = error
                continue
            goal = probe.data.qpos[ex.qadr].copy()
            probe.data.qpos[:] = original
            direct = plan_arm_joint_path(
                model,
                probe.data,
                joint_names=ex.joint_names,
                q_start=start,
                q_goal=goal,
                collision=probe.collision,
                planner="linear",
                step_size=0.01,
            )
            if not direct.success:
                last = f"release_adjustment_failed:{direct.reason}:{direct.detail}"
                continue
            path = [start.copy(), goal.copy()]
            probe.data.qpos[ex.qadr] = goal
            mujoco.mj_kinematics(model, probe.data)
        else:
            path = [start.copy()]
        set_private_payload(model, probe.data, ex.payload_body, ex.ee_body, measured)
        bounds = body_bounds(model, probe.data, body_subtree(model, ex.payload_body))
        patches = [np.asarray(p) for p in surfaces if bounds_fit(bounds, np.asarray(p), region, limits.max_drop_m)]
        if not patches:
            continue
        patch = max(patches, key=lambda p: p[1, 2])
        closed = probe.data.qpos.copy()
        probe.collision.set_payload(model, probe.data, None)
        probe.collision.robot_ids = probe.collision.robot_ids | body_subtree(model, ex.payload_body)
        # Reserve vertical clearance without waiving a contact: also certify
        # opening with the lifting joint lower than the proposed arm pose.
        # Native base compliance can tilt the lift slightly. Certify the actual
        # lift-axis displacement that lowers the fingers by the declared reserve.
        probe.data.qpos[int(height.qposadr[0])] -= limits.clearance_m / height_z
        _, error = probe.plan_gripper_open()
        if error:
            last = error
            continue
        probe.data.qpos[:] = closed
        for joint, value in ex.gripper_open_configuration.items():
            probe.data.qpos[int(model.joint(joint).qposadr[0])] = value
        mujoco.mj_kinematics(model, probe.data)
        dropped = ReleasedFallChecker(model, ex.collision, ex.payload_body, support, patch, region, limits)
        probe.collision = dropped
        _, error = probe.plan_pose(*ex.place_targets[2])
        if error:
            last = error
            continue
        return ReleasePlan(
            path, held_checker(ex.collision, measured), dropped, target_z, float(bounds[0, 2] - patch[1, 2])
        ), None
    return None, last


def refine_release(executor, *, support, region, limits):
    """Bounded measured-state correction; opening remains owned by place_only."""
    ex = executor
    ex.synchronize(ex.data)
    initial_z = float(ex.data.body(ex.ee_body).xpos[2])
    attempted = []
    for _ in range(limits.max_adjustments + 1):
        ex.synchronize(ex.data)
        if not ex.payload_retained():
            return PhysicalMotionResult(False, "payload_not_retained", "release")
        plan, error = plan_release(
            ex, support=support, region=region, limits=limits, initial_z=initial_z, attempted=attempted
        )
        ex.event(
            phase="release_plan",
            accepted=plan is not None,
            reason=error,
            target_z=None if plan is None else plan.target_z,
            max_drop_m=limits.max_drop_m,
        )
        if plan is None:
            return PhysicalMotionResult(False, error, "release")
        if len(plan.prefix) == 1:
            ex.collision = plan.checker
            ex.released_collision = plan.released_checker
            ex.release_verifier = lambda: wait_native_open(ex)
            return PhysicalMotionResult(True, "release_geometry_verified", "release")
        if len(attempted) >= limits.max_adjustments:
            break
        attempted.append(plan.target_z)
        ex.collision = plan.checker
        result = ex._execute_path("release_adjustment", plan.prefix)
        if not result.success:
            return result
    return PhysicalMotionResult(False, "no_plan_within_budget", "release")


def wait_native_open(executor):
    """Simulation-only measured opening, not an ACK or a hardware sensor claim."""
    ex = executor
    if ex.input_mode != "privileged":
        return PhysicalMotionResult(False, "native_execution_unavailable", "release")
    scratch = mujoco.MjData(ex.model)
    active = ex._settling_window(3.0)
    released = ex.released_collision
    while active():
        ex.synchronize(ex.data)
        mujoco.mj_copyData(scratch, ex.model, ex.data)
        mujoco.mj_kinematics(ex.model, scratch)
        mujoco.mj_collision(ex.model, scratch)
        obj = ex.model.body(released.obj).id
        tips = {ex.model.body(n).id for n in ("rubber_tip_left", "rubber_tip_right")}
        touching = any(
            obj in ex.model.geom_bodyid[c.geom] and tips.intersection(ex.model.geom_bodyid[c.geom])
            for c in scratch.contact
        )
        arrived = all(
            abs(float(ex.data.qpos[ex.model.joint(name).qposadr[0]]) - target)
            <= (0.002 if ex.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE else 0.02)
            for name, target in ex.gripper_open_configuration.items()
        )
        if arrived and not touching:
            ex.event(phase="release_open_verified", controller_success=True, source="simulator_ground_truth_debug")
            return PhysicalMotionResult(True, "gripper_open_verified", "release")
        time.sleep(0.05)
    return PhysicalMotionResult(False, "gripper_open_tracking_failed", "release")
