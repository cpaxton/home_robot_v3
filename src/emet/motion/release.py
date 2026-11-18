# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded native release-pose search on private data.

This first provider uses privileged rigid geometry. Collision tests sample joint
edges and vertical post-release positions; they are not a dynamics certificate.
No live object pose, contact tolerance or retention reference is changed.
"""

from __future__ import annotations

import time
from copy import copy, deepcopy
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
    offsets_m: tuple[float, ...] = (0.003, 0.01, 0.02)
    max_adjustments: int = 2
    height_joint: str = "joint_lift"
    open_settle_timeout_s: float = 3.0
    descent_step_m: float = 0.002
    height_axis_min_z: float = 0.99
    target_epsilon_m: float = 1e-6
    stable_interval_s: float = 0.1
    poll_interval_s: float = 0.05

    def __post_init__(self):
        values = [self.max_raise_m, self.max_drop_m, self.clearance_m, self.open_settle_timeout_s,
                  self.descent_step_m, self.height_axis_min_z, self.target_epsilon_m,
                  self.stable_interval_s, self.poll_interval_s, *self.offsets_m]
        if (
            not np.isfinite(values).all()
            or min(values) < 0
            or self.open_settle_timeout_s <= 0
            or min(self.descent_step_m, self.height_axis_min_z, self.target_epsilon_m,
                   self.stable_interval_s, self.poll_interval_s) <= 0
            or self.height_axis_min_z > 1
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
        kind, size = model.geom_type[gid], model.geom_size[gid]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            center, half = data.geom_xpos[gid], np.full(3, size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis = rotation[:, 2]
            center = data.geom_xpos[gid]
            half = np.abs(axis) * size[1] + size[0] * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            center, half = data.geom_xpos[gid], size[0] + size[1] * np.abs(rotation[:, 2])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            center, half = data.geom_xpos[gid], np.sqrt(np.sum((rotation * size) ** 2, axis=1))
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[gid])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start : start + count] @ rotation.T + data.geom_xpos[gid]
            boxes.append((vertices.min(axis=0), vertices.max(axis=0)))
            continue
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
    checker.robot_ids = set(base.robot_ids)
    checker.allowed_pairs = set(base.allowed_pairs)
    checker.allowed_geom_pairs = set(base.allowed_geom_pairs)
    checker._support_departure = deepcopy(base._support_departure)
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
        self.allowed_geom_pairs = {p for p in base.allowed_geom_pairs
                                   if not any(int(model.geom_bodyid[g]) in self.ids for g in p)}
        self._support_departure = {}
        # Ordinary support contact is below the existing penetration tolerance.
        # Watching the released payload must not exempt deep table penetration.
        self.obj, self.patch, self.region, self.limits = obj, patch, region, limits
        self.support = support
        self._probe = mujoco.MjData(model)
        self.last_contacts = []

    def configuration_collides(self, model, data):
        mujoco.mj_copyData(self._probe, model, data)
        probe = self._probe
        mujoco.mj_kinematics(model, probe)
        bounds = body_bounds(model, probe, self.ids)
        gap = float(bounds[0, 2] - self.patch[1, 2])
        if not bounds_fit(bounds, self.patch, self.region, self.limits.max_drop_m):
            self.last_contacts = [{"bodies": [self.obj], "reason":
                "support_penetration" if gap < -.001 else "release_bounds_invalid"}]
            return True
        addr = int(model.jnt_qposadr[model.body(self.obj).jntadr[0]])
        z = float(probe.qpos[addr + 2])
        for shift in np.linspace(0, max(0.0, gap),
                                 max(2, int(np.ceil(max(0.0, gap) / self.limits.descent_step_m)) + 1)):
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


def plan_release(executor, *, support, region, limits, initial_z, attempted=(), tracking_reserve_m=0.0,
                 allow_adjustment=True):
    """Certify closed-arm motion, opening (including reserve) and retreat; no IO."""
    ex = executor
    if (
        ex.input_mode != "privileged"
        or not np.isfinite([initial_z, tracking_reserve_m]).all()
        or tracking_reserve_m < 0
        or type(ex.collision) not in (MujocoSceneCollisionChecker, HeldPoseChecker)
        or ex.payload_body is None
        or ex.collision.payload_transform is None
        or ex.collision.payload_body != ex.payload_body
        or ex.collision.payload_parent != ex.ee_body
        or len(ex.place_targets) != 3
        or not ex.gripper_open_configuration
    ):
        return None, "native_execution_unavailable"
    model = ex.model
    for target in ex.place_targets:
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return None, 'native_execution_unavailable'
        try:
            point, rotation = (np.asarray(value, dtype=float) for value in target)
        except (TypeError, ValueError):
            return None, 'native_execution_unavailable'
        if (point.shape != (3,) or rotation.shape != (3, 3)
                or not np.isfinite(point).all() or not np.isfinite(rotation).all()):
            return None, 'native_execution_unavailable'
    if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
           for name in ex.gripper_open_configuration):
        return None, 'native_execution_unavailable'
    cursor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, support)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ex.payload_body)
    height_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, limits.height_joint)
    if cursor < 0 or root < 0 or height_id < 0:
        return None, 'native_execution_unavailable'
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ex.ee_body)
    ancestors = set()
    while ee > 0:
        ancestors.add(ee)
        ee = int(model.body_parentid[ee])
    if limits.height_joint not in ex.joint_names or int(model.jnt_bodyid[height_id]) not in ancestors:
        return None, 'native_execution_unavailable'
    joint = int(model.body_jntadr[root])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        return None, 'native_execution_unavailable'
    while cursor:
        if model.body_jntnum[cursor]:
            return None, "native_execution_unavailable"
        cursor = int(model.body_parentid[cursor])
    payload_ids = body_subtree(model, ex.payload_body)
    if any(model.body_jntnum[b] for b in payload_ids if b != root):
        return None, "native_execution_unavailable"
    if not any(int(model.geom_bodyid[g]) in payload_ids
               and (model.geom_contype[g] or model.geom_conaffinity[g]) for g in range(model.ngeom)):
        return None, 'native_execution_unavailable'
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
    if height.type[0] != mujoco.mjtJoint.mjJNT_SLIDE or height_z < limits.height_axis_min_z:
        return None, "native_execution_unavailable"
    entry = placements_from_mujoco_model(model, probe.data).get(support, {})
    surfaces = entry.get('support_surfaces', ())
    if len(surfaces) == 0:
        return None, 'native_execution_unavailable'
    region = None if region is None else np.asarray(region)
    targets = [(current_point, current_rotation)]
    if allow_adjustment:
        # Start from measured geometry, while reserving the observed command/FK
        # shortfall in both candidate construction and the lowered probe. The
        # original global max_raise still bounds requested and sampled FK poses.
        targets += [(current_point + [0, 0, tracking_reserve_m+delta], current_rotation)
                    for delta in limits.offsets_m]
    last = "no_plan_within_budget"
    best, best_score, best_index = None, -float("inf"), None

    def decision(index, stage, **detail):
        ex.event(phase="release_candidate", candidate=index, stage=stage,
                 initial_z=float(initial_z), tracking_reserve_m=float(tracking_reserve_m), **detail)

    for index, (point, rotation) in enumerate(targets):
        target_z = float(point[2])
        if target_z > initial_z + limits.max_raise_m or (
            index and (target_z < current_point[2]
                       or any(abs(target_z - old) < limits.target_epsilon_m for old in attempted))
        ):
            decision(index, "target_limit_or_repeat", requested_z=target_z)
            continue
        probe.data.qpos[:] = original
        probe.collision = held_checker(ex.collision, measured)
        mujoco.mj_kinematics(model, probe.data)
        if index:
            path, error = probe.plan_pose(np.asarray(point), np.asarray(rotation))
            if error:
                last = error
                decision(index, "adjustment_ik", reason=error)
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
                decision(index, "adjustment_path", reason=last)
                continue
            path = [start.copy(), goal.copy()]
            probe.data.qpos[ex.qadr] = goal
            mujoco.mj_kinematics(model, probe.data)
            # IK tolerance cannot widen the physical raise limit. Check actual
            # FK along the already sampled direct path and its delivered goal.
            heights_valid = True
            for waypoint in direct.waypoints:
                probe.data.qpos[ex.qadr] = waypoint
                mujoco.mj_kinematics(model, probe.data)
                z = float(probe.data.body(ex.ee_body).xpos[2])
                if not current_point[2] <= z <= initial_z+limits.max_raise_m:
                    heights_valid = False
                    break
            if not heights_valid:
                last = 'release_raise_limit'
                decision(index, 'adjustment_height', reason=last)
                continue
            probe.data.qpos[ex.qadr] = goal
            mujoco.mj_kinematics(model, probe.data)
            target_z = float(probe.data.body(ex.ee_body).xpos[2])
        else:
            path = [start.copy()]
        set_private_payload(model, probe.data, ex.payload_body, ex.ee_body, measured)
        bounds = body_bounds(model, probe.data, body_subtree(model, ex.payload_body))
        patches = [np.asarray(p) for p in surfaces if bounds_fit(bounds, np.asarray(p), region, limits.max_drop_m)]
        if not patches:
            decision(index, "support_bounds", bounds=bounds.tolist(),
                     surfaces=[np.asarray(p).tolist() for p in surfaces],
                     region=None if region is None else region.tolist(), max_drop_m=limits.max_drop_m)
            continue
        patch = max(patches, key=lambda p: p[1, 2])
        closed = probe.data.qpos.copy()
        # Both the commanded configuration and a lower delivered configuration
        # must permit opening. The measured command/FK offset is a diagnostic
        # reserve, not a guarantee that future tracking error is bounded by it.
        error = None
        reserve = limits.clearance_m + (tracking_reserve_m if index else 0.0)
        for lower_by in (0.0, reserve):
            probe.data.qpos[:] = closed
            probe.data.qpos[int(height.qposadr[0])] -= lower_by / height_z
            mujoco.mj_kinematics(model, probe.data)
            set_private_payload(model, probe.data, ex.payload_body, ex.ee_body, measured)
            if not bounds_fit(
                body_bounds(model, probe.data, body_subtree(model, ex.payload_body)), patch, region, limits.max_drop_m
            ):
                error = "no_plan_within_budget"
                decision(index, "lowered_support_bounds", lower_by_m=float(lower_by),
                         bounds=body_bounds(model, probe.data, payload_ids).tolist(), patch=patch.tolist())
                break
            probe.collision = held_checker(ex.collision, measured)
            probe.collision.set_payload(model, probe.data, None)
            probe.collision.robot_ids |= body_subtree(model, ex.payload_body)
            _, error = probe.plan_gripper_open()
            if error:
                decision(index, "opening", lower_by_m=float(lower_by), reason=error,
                         contacts=list(probe.collision.last_contacts))
                break
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
            decision(index, "retreat", reason=error, contacts=list(probe.collision.last_contacts))
            continue
        gap = float(bounds[0, 2] - patch[1, 2])
        candidate = ReleasePlan(path, held_checker(ex.collision, measured), dropped, target_z, gap)
        if index == 0:
            # A certified measured pose needs no adjustment at all.
            decision(index, "accepted", target_z=target_z, gap_m=gap)
            return candidate, None
        # Compare the already certified delivery interval, rather than taking
        # the first pose with nearly zero lower-support margin. Both endpoint
        # constraints and the existing raise/drop limits remain unchanged.
        score = min(gap - reserve + 0.001, limits.max_drop_m - gap)
        decision(index, "feasible", target_z=target_z, gap_m=gap, support_slack_m=score)
        if score > best_score:
            best, best_score, best_index = candidate, score, index
    if best is not None:
        decision(best_index, "accepted", target_z=best.target_z, gap_m=best.gap_m, support_slack_m=best_score)
        return best, None
    return None, last


def refine_release(executor, *, support, region, limits):
    """Bounded measured-state correction; opening remains owned by place_only."""
    ex = executor
    ex.synchronize(ex.data)
    initial_z = float(ex.data.body(ex.ee_body).xpos[2])
    attempted = []
    command = ex.place_paths[1][-1] if ex.place_paths else None
    scratch = mujoco.MjData(ex.model)
    for _ in range(limits.max_adjustments + 1):
        ex.synchronize(ex.data)
        if not ex.payload_retained():
            return PhysicalMotionResult(False, "payload_not_retained", "release")
        tracking_reserve = 0.0
        if command is not None:
            mujoco.mj_copyData(scratch, ex.model, ex.data)
            scratch.qpos[ex.qadr] = command
            mujoco.mj_kinematics(ex.model, scratch)
            tracking_reserve = max(0.0, float(scratch.body(ex.ee_body).xpos[2] - ex.data.body(ex.ee_body).xpos[2]))
        plan, error = plan_release(
            ex,
            support=support,
            region=region,
            limits=limits,
            initial_z=initial_z,
            attempted=attempted,
            tracking_reserve_m=tracking_reserve,
            allow_adjustment=len(attempted) < limits.max_adjustments,
        )
        ex.event(
            phase="release_plan",
            accepted=plan is not None,
            reason=error,
            target_z=None if plan is None else plan.target_z,
            max_drop_m=limits.max_drop_m,
            tracking_reserve_m=tracking_reserve,
            measured_ee=ex.data.body(ex.ee_body).xpos.tolist(),
            adjustment_count=len(attempted),
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
        command = plan.prefix[-1]
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
    released = ex.released_collision
    if released is None or not ex.gripper_open_configuration:
        return PhysicalMotionResult(False, 'native_execution_unavailable', 'release')
    # Planning and the opening ACK may advance native time. Anchor the phase
    # budget to fresh post-command feedback, not the saved placement snapshot.
    ex.synchronize(ex.data)
    active = ex._settling_window(released.limits.open_settle_timeout_s)
    stable_since = None
    last_reason = "gripper_open_tracking_failed"
    evidence = {}
    while active():
        ex.synchronize(ex.data)
        mujoco.mj_copyData(scratch, ex.model, ex.data)
        mujoco.mj_kinematics(ex.model, scratch)
        mujoco.mj_collision(ex.model, scratch)
        objects = body_subtree(ex.model, released.obj)
        tips = {ex.model.body(n).id for n in ("rubber_tip_left", "rubber_tip_right")}
        touching = any(
            objects.intersection(ex.model.geom_bodyid[c.geom]) and tips.intersection(ex.model.geom_bodyid[c.geom])
            for c in scratch.contact
        )
        arrived = all(
            abs(float(ex.data.qpos[ex.model.joint(name).qposadr[0]]) - target)
            <= (0.002 if ex.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE else 0.02)
            for name, target in ex.gripper_open_configuration.items()
        )
        support = body_subtree(ex.model, released.support)
        supported = any(
            objects.intersection(ex.model.geom_bodyid[c.geom]) and support.intersection(ex.model.geom_bodyid[c.geom])
            for c in scratch.contact
        )
        dof = int(ex.model.jnt_dofadr[ex.model.body(released.obj).jntadr[0]])
        settled = (
            np.linalg.norm(scratch.qvel[dof : dof + 3]) <= 0.01
            and np.linalg.norm(scratch.qvel[dof + 3 : dof + 6]) <= 0.1
        )
        geometry_clear = (
            not released.configuration_collides(ex.model, scratch)
            if arrived and not touching and supported and settled
            else None
        )
        evidence = {
            "arrived": arrived,
            "fingertip_contact": bool(touching),
            "supported": bool(supported),
            "settled": bool(settled),
            "geometry_clear": geometry_clear,
            "geometry_rejections": list(getattr(released, 'last_contacts', ())),
            "linear_speed_m_s": float(np.linalg.norm(scratch.qvel[dof : dof + 3])),
            "angular_speed_rad_s": float(np.linalg.norm(scratch.qvel[dof + 3 : dof + 6])),
            "timeout_s": released.limits.open_settle_timeout_s,
        }
        if geometry_clear:
            if stable_since is None:
                stable_since = ex._motion_time()
            if ex._motion_time() - stable_since >= released.limits.stable_interval_s:
                ex.event(
                    phase="release_open_verified",
                    controller_success=True,
                    source="simulator_ground_truth_debug",
                    supported=True,
                    settled=True,
                    stable_interval_s=released.limits.stable_interval_s,
                )
                return PhysicalMotionResult(True, "gripper_open_verified", "release")
        else:
            stable_since = None
        if arrived:
            last_reason = "release_verify_failed"
        time.sleep(released.limits.poll_interval_s)
    ex.event(phase="release_check", reason=last_reason, **evidence)
    return PhysicalMotionResult(False, last_reason, "release")
