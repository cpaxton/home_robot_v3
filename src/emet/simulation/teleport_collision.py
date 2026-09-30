"""Check a simulator teleport endpoint on private data before changing live state.

This is an endpoint guard for oracle navigation, not a swept-path certificate.
Normal shallow support contacts (up to 1 mm) are retained by the collision kernel.
"""

import mujoco
import numpy as np

from emet.motion.mujoco_collision import MujocoSceneCollisionChecker
from emet.simulation.spawn_planar import write_planar_base_xyt


def teleport_endpoint_contacts(model, data, spec, xyt):
    """Return penetrating robot contacts at a proposed base pose; never mutate data."""
    xyt = np.asarray(xyt, dtype=float).reshape(3)
    if not np.isfinite(xyt).all():
        raise ValueError("nonfinite_teleport_goal")
    probe = mujoco.MjData(model)
    mujoco.mj_copyData(probe, model, data)
    names = getattr(spec, "planar_base_joint_names", None)
    if names:
        if not write_planar_base_xyt(
            model, probe, joint_names=tuple(names), world_x=xyt[0], world_y=xyt[1],
            world_yaw=xyt[2], base_body_name=spec.base_link_name,
        ):
            raise ValueError("unsupported_teleport_base")
    else:
        base = model.body(spec.base_link_name)
        joint = int(base.jntadr[0])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("unsupported_teleport_base")
        address = int(model.jnt_qposadr[joint])
        probe.qpos[address:address + 2] = xyt[:2]
        probe.qpos[address + 3:address + 7] = [np.cos(xyt[2] / 2), 0, 0, np.sin(xyt[2] / 2)]
    checker = MujocoSceneCollisionChecker(model, robot_body=spec.base_link_name)
    checker.configuration_collides(model, probe)
    return checker.last_contacts
