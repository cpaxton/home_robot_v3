# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Shared scoring and input validation for privileged stationary table controls."""
import json

import mujoco
import numpy as np

from emet.simulation.sim_object_placements import placements_from_mujoco_model


def goal_predicates(robot, goals):
    """Independent privileged scoring: whole XY footprint, support, and stillness."""
    scratch = mujoco.MjData(robot.model)
    mujoco.mj_copyData(scratch, robot.model, robot.data)
    mujoco.mj_kinematics(robot.model, scratch)
    mujoco.mj_collision(robot.model, scratch)
    placements = placements_from_mujoco_model(robot.model, scratch)
    table = robot.model.body('table').id
    fingertips = {robot.model.body(name).id for name in ('rubber_tip_left', 'rubber_tip_right')}
    rows = {}
    for name, region in goals.items():
        body = robot.model.body(name)
        bounds = np.asarray(placements[name]['bounds'])[:, :2]
        region = np.asarray(region)
        inside = bool(np.all(bounds[0] >= region[0] - .005) and np.all(bounds[1] <= region[1] + .005))
        touched = {int(other) for c in scratch.contact if body.id in robot.model.geom_bodyid[c.geom]
                   for other in robot.model.geom_bodyid[c.geom] if other != body.id}
        supported = table in touched and not fingertips.intersection(touched)
        dof = robot.model.jnt_dofadr[body.jntadr[0]]
        settled = bool(np.linalg.norm(scratch.qvel[dof:dof+3]) <= .01
                       and np.linalg.norm(scratch.qvel[dof+3:dof+6]) <= .1)
        rows[name] = {'in_region': inside, 'supported': supported, 'settled': settled,
                      'satisfied': inside and supported and settled, 'bounds_xy': bounds.tolist()}
    return rows


def load_placement_regions(path):
    regions = np.asarray(json.loads(path.read_text()), dtype=float)
    if (regions.shape != (2, 2, 2) or not np.isfinite(regions).all()
            or not np.all(regions[:, 0] < regions[:, 1])):
        raise ValueError('Expected cylinder and box XY regions, each with finite increasing min/max bounds')
    return regions.tolist()
