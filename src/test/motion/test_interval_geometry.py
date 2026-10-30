# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""FK interval containment at rotated ancestors, joint pivots and references."""

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.interval_geometry import subtree_geometry_intervals
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.placement_geometry import box_corners


def test_multijoint_body_nonzero_pivots_refs_and_rotated_ancestors_enclose_fk():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body name="root" pos=".3 -.2 .6" euler=".4 -.3 .2">
      <body pos=".1 .03 -.02" euler="-.2 .3 .1">
        <joint name="hinge" axis="1 2 3" pos=".03 -.02 .01" range="-.7 .5" ref=".1"/>
        <joint name="slide" type="slide" axis="1 1 -1" range=".02 .08" ref=".04"/>
        <geom type="box" size=".01 .02 .03" pos=".04 -.02 .01" euler=".2 .4 .1"/>
        <body pos=".03 .02 .01" euler=".2 -.1 .3">
          <joint name="spin" axis="0 1 1" pos=".02 .01 -.01" limited="false"/>
          <geom type="capsule" size=".01 .04" pos=".01 -.01 .02" euler=".1 .2 .3"/>
        </body></body></body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    root = model.body("root").id
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.xmat[root].reshape(3, 3), data.xpos[root]
    intervals = {
        model.joint(n).id: lim
        for n, lim in [("hinge", (-0.7, 0.5)), ("slide", (0.02, 0.08)), ("spin", (-np.pi, np.pi))]
    }
    ids, boxes = subtree_geometry_intervals(model, root_id=root, world_from_root=pose, joint_intervals=intervals)
    rng = np.random.default_rng(29)
    # Actual MuJoCo FK is the independent oracle, including every interval end.
    for sample in range(252):
        for j, (lo, hi) in intervals.items():
            data.qpos[model.jnt_qposadr[j]] = lo if sample == 0 else hi if sample == 1 else rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, data)
        for g, box in zip(ids, boxes, strict=True):
            aabb = model.geom_aabb[g]
            points = box_corners([aabb[:3] - aabb[3:], aabb[:3] + aabb[3:]])
            points = points @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            assert np.all(points >= box[0]) and np.all(points <= box[1])
    # Degenerate intervals must reproduce exact FK, not merely contain it in a
    # large box that could conceal an incorrect pivot/reference composition.
    fixed = {j: (float(data.qpos[model.jnt_qposadr[j]]),) * 2 for j in intervals}
    ids, boxes = subtree_geometry_intervals(model, root_id=root, world_from_root=pose, joint_intervals=fixed)
    for g, box in zip(ids, boxes, strict=True):
        aabb = model.geom_aabb[g]
        points = box_corners([aabb[:3] - aabb[3:], aabb[:3] + aabb[3:]])
        points = points @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        np.testing.assert_allclose(box, [points.min(0), points.max(0)], atol=2e-9, rtol=0)


def test_unbounded_or_unreported_descendants_are_rejected():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="root">
      <body><joint name="unknown" type="ball"/><geom size=".02"/></body>
      </body></worldbody></mujoco>""")
    args = {"root_id": model.body("root").id, "world_from_root": np.eye(4)}
    with pytest.raises(ObservationUnavailable, match="unavailable"):
        subtree_geometry_intervals(model, **args, joint_intervals={})
    with pytest.raises(ObservationUnavailable, match="unsupported"):
        subtree_geometry_intervals(model, **args, joint_intervals={0: (-1.0, 1.0)})
