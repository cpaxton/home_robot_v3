# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""GT collision validation on an offline scene, never a live state writer.

Uses the scene's declared collision geometry and exclusions. Visual-only links
are reported as unsupported; this is a sampled static check, not a dynamics proof.
"""

from __future__ import annotations

import mujoco
import numpy as np


def body_subtree(model, root: str) -> set[int]:
    found = {int(model.body(root).id)}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in found:
            found.add(body)
    return found


class MujocoSceneCollisionChecker:
    def __init__(
        self, model, *, robot_body: str, allowed_pairs=(), allowed_geom_pairs=(), penetration_tolerance_m: float = 0.001
    ):
        self.robot_ids = body_subtree(model, robot_body)
        self.allowed_pairs = {frozenset(pair) for pair in allowed_pairs}
        self.allowed_geom_pairs = {frozenset(pair) for pair in allowed_geom_pairs}
        self.penetration_tolerance_m = float(penetration_tolerance_m)
        if not np.isfinite(self.penetration_tolerance_m) or self.penetration_tolerance_m < 0:
            raise ValueError("Invalid collision tolerance")
        self.payload_body: str | None = None
        self.payload_parent: str | None = None
        self.payload_transform: np.ndarray | None = None
        self.last_contacts: list[dict] = []

    def unsupported_links(self, model, link_bodies) -> list[str]:
        result = []
        for name in link_bodies:
            bodies = body_subtree(model, name)
            if not any(
                int(model.geom_bodyid[g]) in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])
                for g in range(model.ngeom)
            ):
                result.append(name)
        return result

    def set_payload(self, model, data, body: str | None, parent: str | None = None):
        self.payload_body, self.payload_parent = body, parent
        self.payload_transform = None
        if body is not None:
            if parent is None:
                raise ValueError("Payload needs a parent frame")
            mujoco.mj_fwdPosition(model, data)
            ee, obj = data.body(parent), data.body(body)
            transform = np.eye(4)
            transform[:3, :3] = ee.xmat.reshape(3, 3).T @ obj.xmat.reshape(3, 3)
            transform[:3, 3] = ee.xmat.reshape(3, 3).T @ (obj.xpos - ee.xpos)
            self.payload_transform = transform

    def configuration_collides(self, model, data) -> bool:
        mujoco.mj_fwdPosition(model, data)
        watched = set(self.robot_ids)
        if self.payload_body is not None:
            body = model.body(self.payload_body)
            joint = int(body.jntadr[0])
            if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("Payload collision requires a free-joint object")
            parent = data.body(self.payload_parent)
            rotation = parent.xmat.reshape(3, 3)
            address = int(model.jnt_qposadr[joint])
            data.qpos[address : address + 3] = parent.xpos + rotation @ self.payload_transform[:3, 3]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (rotation @ self.payload_transform[:3, :3]).ravel())
            data.qpos[address + 3 : address + 7] = quat
            mujoco.mj_fwdPosition(model, data)
            watched |= body_subtree(model, self.payload_body)
        self.last_contacts = []
        for contact in data.contact[: data.ncon]:
            a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
            if not ({a, b} & watched):
                continue
            pair = [model.body(a).name, model.body(b).name]
            if (
                frozenset(pair) in self.allowed_pairs
                or frozenset(int(g) for g in contact.geom) in self.allowed_geom_pairs
            ):
                continue
            if contact.dist < -self.penetration_tolerance_m:
                self.last_contacts.append({"bodies": pair, "distance_m": float(contact.dist)})
        return bool(self.last_contacts)

    def trajectory_collides(self, model, data, *, joint_names, arm_waypoints):
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        addresses = joint_qpos_addrs(model, joint_names)
        for index, point in enumerate(arm_waypoints):
            data.qpos[addresses] = point
            if self.configuration_collides(model, data):
                return index
        return None
