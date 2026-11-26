# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Measured wrist wrench and finger contact state for WBC observations.

The sensor reads MuJoCo force/torque sensors and solved contact loads. It never
looks up object names, object identities, or task labels: a contact here is a
measured plant event, not a GT label. The actor receives the measurement; a
teacher/critic may separately consume privileged contact labels that never
enter this module's output.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.wbc_contract import CONTACT_WRENCH_DIM, ContactObservation
from emet.rl.contact import solved_contact_force


@dataclass(frozen=True)
class ContactReading:
    """One measured contact snapshot for an arm and its gripper."""

    contact: ContactObservation
    finger_force: tuple[float, ...]
    collision_force: float
    hand_force: float
    self_collision: bool
    external_contact_count: int

    def to_info(self) -> dict[str, object]:
        return {
            "contact_force": float(self.collision_force),
            "hand_contact_force": float(self.hand_force),
            "self_collision": bool(self.self_collision),
            "external_contact_count": int(self.external_contact_count),
            "contact_flag": bool(self.contact.in_contact),
            "contact_wrench": self.contact.wrench,
            "finger_force": self.finger_force,
            "contact_source": self.contact.source,
        }


class MuJoCoContactSensor:
    """Read a wrist F/T sensor plus external and self contacts.

    ``base_body`` defines the robot subtree. ``ee_body`` defines the hand
    subtree; contacts between the hand and a body outside the robot are
    external, while contacts between two robot bodies are self-collisions.
    Hand-hand contacts are ignored because a closing gripper's fingers can
    touch each other intentionally.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        base_body: str,
        ee_body: str,
        finger_bodies: tuple[str, ...] = (),
        support_bodies: tuple[str, ...] = (),
        force_sensor: str | None = None,
        torque_sensor: str | None = None,
    ) -> None:
        if (force_sensor is None) != (torque_sensor is None):
            raise ValueError("force_sensor and torque_sensor must be supplied together")
        self.model = model
        self._body_parent = np.asarray(model.body_parentid, dtype=np.int32).copy()
        self._geom_body = np.asarray(model.geom_bodyid, dtype=np.int32).copy()
        self._body_mass = np.asarray(model.body_mass, dtype=np.float64).copy()
        self.base_body = int(model.body(base_body).id)
        self.ee_body = int(model.body(ee_body).id)
        self.robot_bodies = self._descendants(self.base_body)
        self.hand_bodies = self._descendants(self.ee_body)
        self.finger_bodies = tuple(int(model.body(name).id) for name in finger_bodies)
        self.finger_bodies_set = tuple(frozenset(self._descendants(body)) for body in self.finger_bodies)
        self.support_bodies = frozenset(int(model.body(name).id) for name in support_bodies)

        self._force_adr: int | None = None
        self._torque_adr: int | None = None
        self._ft_site: int | None = None
        if force_sensor is not None and torque_sensor is not None:
            force = model.sensor(force_sensor)
            torque = model.sensor(torque_sensor)
            if int(model.sensor_type[force.id]) != int(mujoco.mjtSensor.mjSENS_FORCE):
                raise ValueError(f"{force_sensor} is not a force sensor")
            if int(model.sensor_type[torque.id]) != int(mujoco.mjtSensor.mjSENS_TORQUE):
                raise ValueError(f"{torque_sensor} is not a torque sensor")
            if int(model.sensor_objtype[force.id]) != int(mujoco.mjtObj.mjOBJ_SITE):
                raise ValueError(f"{force_sensor} is not attached to a site")
            site = int(model.sensor_objid[force.id])
            if int(model.sensor_objid[torque.id]) != site:
                raise ValueError("force and torque sensors must share a site")
            if int(model.sensor_dim[force.id]) != 3 or int(model.sensor_dim[torque.id]) != 3:
                raise ValueError("wrist force/torque sensors must be three-axis")
            self._ft_site = site
            self._force_adr = int(model.sensor_adr[force.id])
            self._torque_adr = int(model.sensor_adr[torque.id])

    def _descendants(self, root: int) -> frozenset[int]:
        result = {int(root)}
        for body in range(1, len(self._body_parent)):
            parent = int(self._body_parent[body])
            for _ in range(len(self._body_parent)):
                if parent == root:
                    result.add(body)
                    break
                if parent == 0:
                    break
                parent = int(self._body_parent[parent])
        return frozenset(result)

    def read(self, data: mujoco.MjData) -> ContactReading:
        model = self.model
        count = min(int(data.ncon), len(data.contact))
        geom1 = np.asarray(data.contact.geom1[:count], dtype=np.int32).copy()
        geom2 = np.asarray(data.contact.geom2[:count], dtype=np.int32).copy()
        addresses = np.asarray(data.contact.efc_address[:count], dtype=np.int32).copy()
        dimensions = np.asarray(data.contact.dim[:count], dtype=np.int32).copy()
        friction = np.asarray(data.contact.friction[:count], dtype=np.float64).copy()
        excludes = np.asarray(data.contact.exclude[:count], dtype=np.int32).copy()
        frames = np.asarray(data.contact.frame[:count], dtype=np.float64).reshape(-1, 9).copy()
        positions = np.asarray(data.contact.pos[:count], dtype=np.float64).reshape(-1, 3).copy()
        efc_force = np.asarray(data.efc_force, dtype=np.float64).copy()

        ee_position = np.asarray(data.xpos[self.ee_body], dtype=np.float64).copy()
        ee_rotation = np.asarray(data.xmat[self.ee_body], dtype=np.float64).reshape(3, 3).copy()
        finger_load = np.zeros(len(self.finger_bodies_set), dtype=np.float64)
        external_count = 0
        hand_force = 0.0
        collision_force = 0.0
        self_collision = False
        contact_force_world = np.zeros(3, dtype=np.float64)
        contact_torque_world = np.zeros(3, dtype=np.float64)

        for index in range(count):
            first = int(geom1[index])
            second = int(geom2[index])
            if not (0 <= first < len(self._geom_body) and 0 <= second < len(self._geom_body)):
                continue
            first_body = int(self._geom_body[first])
            second_body = int(self._geom_body[second])
            first_hand = first_body in self.hand_bodies
            second_hand = second_body in self.hand_bodies
            first_robot = first_body in self.robot_bodies
            second_robot = second_body in self.robot_bodies
            if first_body in self.support_bodies or second_body in self.support_bodies:
                continue
            normal, bound = solved_contact_force(
                model,
                data,
                index,
                efc_address=addresses[index],
                dim=dimensions[index],
                exclude=excludes[index],
                efc_force=efc_force,
                friction=friction[index],
            )
            if first_robot and second_robot:
                if first_hand and second_hand:
                    # A closing gripper may touch its own fingers; that is not
                    # an external contact or a posture self-collision.
                    continue
                self_collision = True
                collision_force = max(collision_force, bound)
                continue
            if first_hand or second_hand:
                external_count += 1
                hand_force = max(hand_force, bound)
                if first_hand:
                    for finger, bodies in enumerate(self.finger_bodies_set):
                        if first_body in bodies:
                            finger_load[finger] += normal
                if second_hand:
                    for finger, bodies in enumerate(self.finger_bodies_set):
                        if second_body in bodies:
                            finger_load[finger] += normal
                # The contact frame's first row is the normal. Its positive
                # direction acts on geom2; flip it when the hand is geom1.
                sign = 1.0 if second_hand else -1.0
                force_world = sign * normal * frames[index, :3]
                contact_force_world += force_world
                contact_torque_world += np.cross(positions[index] - ee_position, force_world)
                continue
            if first_robot or second_robot:
                collision_force = max(collision_force, bound)
                continue

        if self._ft_site is not None:
            assert self._force_adr is not None and self._torque_adr is not None
            sensor_force = np.asarray(data.sensordata[self._force_adr : self._force_adr + 3], dtype=np.float64).copy()
            sensor_torque = np.asarray(
                data.sensordata[self._torque_adr : self._torque_adr + 3], dtype=np.float64
            ).copy()
            site_position = np.asarray(data.site_xpos[self._ft_site], dtype=np.float64).copy()
            site_rotation = np.asarray(data.site_xmat[self._ft_site], dtype=np.float64).reshape(3, 3).copy()
            # MuJoCo reports the reaction wrench at the sensor site in the site
            # frame. Negate it to obtain the external load on the distal chain,
            # then shift the torque from the site to the EE origin.
            force_world = -(site_rotation @ sensor_force)
            torque_world = -(site_rotation @ sensor_torque) + np.cross(site_position - ee_position, force_world)
            source = "ft_sensor"
        else:
            force_world = contact_force_world
            torque_world = contact_torque_world
            source = "contact_normal" if external_count else "none"

        wrench = np.r_[ee_rotation.T @ force_world, ee_rotation.T @ torque_world]
        if wrench.shape != (CONTACT_WRENCH_DIM,) or not np.isfinite(wrench).all():
            raise ValueError("contact sensor produced a non-finite wrench")
        return ContactReading(
            contact=ContactObservation(
                in_contact=bool(external_count),
                wrench=tuple(float(value) for value in wrench),
                source=source,
            ),
            finger_force=tuple(float(value) for value in finger_load),
            collision_force=float(collision_force),
            hand_force=float(hand_force),
            self_collision=bool(self_collision),
            external_contact_count=int(external_count),
        )


__all__ = ["ContactReading", "MuJoCoContactSensor"]
