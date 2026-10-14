"""Physics-step contact, support and retention evidence, separate from actor inputs."""

import mujoco
import numpy as np


def relative_pose(data, ee, payload):
    rotation = data.xmat[ee].reshape(3, 3).T
    return rotation @ (data.xpos[payload] - data.xpos[ee]), rotation @ data.xmat[payload].reshape(3, 3)


def retention_error(reference, current):
    if reference is None:
        return None, None
    translation = float(np.linalg.norm(current[0] - reference[0]))
    angle = float(np.arccos(np.clip((np.trace(reference[1].T @ current[1]) - 1) / 2, -1, 1)))
    return translation, angle


class ContactMonitor:
    def __init__(self, model, profile):
        self.model, self.profile = model, profile
        self.base = model.body(profile.base_body).id
        self.robot_bodies = {i for i in range(1, model.nbody) if self._descendant(i, self.base)}
        self.robot_geoms = {i for i, body in enumerate(model.geom_bodyid) if body in self.robot_bodies}
        self.floor = {i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
        self.wheels = [model.body(name).id for name in profile.wheel_bodies]
        self.wheel_geoms = [
            set(
                np.flatnonzero(
                    (model.geom_bodyid == body) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
                )
            )
            for body in self.wheels
        ]
        # The Stretch caster is a sphere on the base body, rather than a separate body.
        self.support_geoms = set().union(*self.wheel_geoms) if self.wheel_geoms else set()
        self.support_geoms.update(
            i
            for i in self.robot_geoms
            if model.geom_bodyid[i] == self.base and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        )
        self.fingers = [model.body(name).id for name in profile.finger_bodies]
        self.reset()

    def _descendant(self, body, parent):
        while body and body != parent:
            body = self.model.body_parentid[body]
        return body == parent

    def reset(self):
        self.peak_force = self.impulse = self.max_penetration = 0.0
        self.wheel_lift = False
        self.contact = False
        self.max_wheel_gap = np.zeros(len(self.wheels))
        self.min_wheel_load = np.full(len(self.wheels), np.inf)

    def update(self, data, payload=None):
        model = self.model
        loads = np.zeros(len(self.wheels))
        force_sum, force_peak = 0.0, 0.0
        pair_forces = {}
        fingers = set()
        contacts = []
        wrench = np.empty(6)
        for index, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if (
                a not in self.robot_geoms
                and b not in self.robot_geoms
                and payload not in (model.geom_bodyid[a], model.geom_bodyid[b])
            ):
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            normal = max(0.0, float(wrench[0]))
            force = float(np.linalg.norm(wrench[:3]))
            for wheel, geoms in enumerate(self.wheel_geoms):
                if (a in geoms and b in self.floor) or (b in geoms and a in self.floor):
                    loads[wheel] += normal
            supporting = (a in self.support_geoms and b in self.floor) or (b in self.support_geoms and a in self.floor)
            if supporting:
                continue
            if payload is not None:
                for finger in self.fingers:
                    if (self._descendant(model.geom_bodyid[a], finger) and model.geom_bodyid[b] == payload) or (
                        self._descendant(model.geom_bodyid[b], finger) and model.geom_bodyid[a] == payload
                    ):
                        if normal > 0.01:
                            fingers.add(finger)
            force_sum += force
            pair = tuple(sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b]))))
            pair_forces[pair] = pair_forces.get(pair, 0.0) + force
            force_peak = max(force_peak, pair_forces[pair])
            self.contact |= force > 0.01
            self.max_penetration = max(self.max_penetration, -float(contact.dist))
            contacts.append((a, b, float(contact.dist), force))
        gaps = []
        for geoms in self.wheel_geoms:
            gaps.append(
                min(
                    mujoco.mj_geomDistance(model, data, wheel, floor, 1, None)
                    for wheel in geoms
                    for floor in self.floor
                )
            )
        gaps = np.asarray(gaps)
        self.max_wheel_gap = np.maximum(self.max_wheel_gap, gaps)
        self.min_wheel_load = np.minimum(self.min_wheel_load, loads)
        # Sample every physics step; no debounce that could hide a short lift.
        self.wheel_lift |= bool(np.any((gaps > 0.0001) & (loads < 0.01)))
        self.peak_force = max(self.peak_force, force_peak)
        self.impulse += force_sum * model.opt.timestep
        return {
            "force": force_peak,
            "force_sum": force_sum,
            "wheel_gap": gaps,
            "wheel_load": loads,
            "bilateral_contact": len(fingers) == 2,
            "contacts": contacts,
        }

    def metrics(self):
        return {
            "peak_contact_force": self.peak_force,
            "contact_impulse": self.impulse,
            "contact": self.contact,
            "max_penetration": self.max_penetration,
            "wheel_lift": self.wheel_lift if self.wheels else None,
            "wheel_support_evaluated": bool(self.wheels),
            "max_wheel_gap": self.max_wheel_gap.tolist(),
            "min_wheel_load": np.where(np.isfinite(self.min_wheel_load), self.min_wheel_load, 0).tolist(),
        }
