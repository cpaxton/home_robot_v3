# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import mujoco
import numpy as np

from emet.robots.sourccey import (
    SOURCCEY_CAMERA_NAMES,
    SOURCCEY_GRIPPER_ACTUATORS,
    SOURCCEY_GRIPPER_JOINTS,
    SOURCCEY_HOME_KEYFRAME,
    SOURCCEY_JOINT_NAMES,
    SourcceyBackend,
)


def _load():
    spec = SourcceyBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    return spec, model


def test_sourccey_mjcf_joints_and_actuators():
    spec, model = _load()
    assert model.nq == spec.dof == 16
    assert model.nu == len(spec.actuator_names) == 16
    for jname in SOURCCEY_JOINT_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname) >= 0
    for aname in spec.actuator_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname) >= 0
    # planar base + lift
    assert spec.planar_base_joint_names == ("base_x", "base_y", "base_yaw")
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift") >= 0


def test_sourccey_mjcf_cameras():
    spec, model = _load()
    assert spec.camera_names == SOURCCEY_CAMERA_NAMES == ["front_left", "front_right", "wrist_left", "wrist_right"]
    for cname in SOURCCEY_CAMERA_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cname) >= 0


def test_sourccey_mjcf_geometry_sane():
    spec, model = _load()
    data = mujoco.MjData(model)
    # tuck arms to the home keyframe so extents reflect the mobile footprint
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    # ~1 m tall mobile manipulator (real: 1030 mm)
    zs = [data.body(i).xpos[2] for i in range(model.nbody)]
    assert max(zs) > 0.8
    # body centers stay near the robot footprint (tucked home pose)
    xs = [data.body(i).xpos[0] for i in range(model.nbody)]
    ys = [data.body(i).xpos[1] for i in range(model.nbody)]
    assert max(abs(v) for v in xs) < 0.45
    assert max(abs(v) for v in ys) < 0.45
    # total mass plausible (~15.88 kg real; allow some tolerance for the simplified base)
    assert 5.0 < float(sum(model.body_mass)) < 40.0


def test_sourccey_home_keyframe_no_self_collision():
    spec, model = _load()
    data = mujoco.MjData(model)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, SOURCCEY_HOME_KEYFRAME) >= 0
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(20):
        mujoco.mj_step(model, data)
    pen = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -0.001:
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
            if b1 != "world" and b2 != "world":
                pen.append((b1, b2, round(float(c.dist), 4)))
    assert pen == [], f"self-penetrations at home pose: {pen}"


def test_sourccey_left_right_mirror_symmetry():
    spec, model = _load()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for link in ("Arm-Wrist-v1", "Gripper-Base-Back-v1", "Arm-Forearm-v1"):
        l = data.body(f"left_{link}").xpos
        r = data.body(f"right_{link}").xpos
        assert abs(float(l[0]) + float(r[0])) < 1e-3, f"{link} not mirror-symmetric in x"
        assert abs(float(l[1]) - float(r[1])) < 1e-3, f"{link} not mirror-symmetric in y"
        assert abs(float(l[2]) - float(r[2])) < 1e-3, f"{link} not mirror-symmetric in z"


def test_sourccey_gripper_mappings():
    spec, model = _load()
    for side in ("left", "right"):
        jname = SOURCCEY_GRIPPER_JOINTS[side]
        aname = SOURCCEY_GRIPPER_ACTUATORS[side]
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname) >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname) >= 0


def test_sourccey_mjcf_stable_sim_step():
    spec, model = _load()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qacc).all()
    assert float(data.body("head").xpos[2]) > 0.5


def test_sourccey_registry_and_assets():
    from emet.robots import get_robot_spec
    from emet.utils.assets import get_robot_mjcf_path

    assert get_robot_mjcf_path("sourccey") is not None
    spec = get_robot_spec("sourccey")
    assert spec is not None and spec.name == "sourccey"
