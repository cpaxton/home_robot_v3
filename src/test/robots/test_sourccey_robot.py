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
    # New official ArmLeft URDF arm chain (canonical left + code-side right mirror).
    for link in ("Arm-Base-Shoulder", "Arm-Bicep", "Arm-Forearm", "Arm-Wrist", "Gripper-Base"):
        l = data.body(f"left_{link}").xpos
        r = data.body(f"right_{link}").xpos
        assert abs(float(l[0]) + float(r[0])) < 1e-3, f"{link} not mirror-symmetric in x"
        assert abs(float(l[1]) - float(r[1])) < 1e-3, f"{link} not mirror-symmetric in y"
        assert abs(float(l[2]) - float(r[2])) < 1e-3, f"{link} not mirror-symmetric in z"


def test_sourccey_arm_links_connected():
    """Consecutive arm link meshes must overlap (no visible gaps at the joints)."""
    from scipy.spatial import cKDTree

    spec, model = _load()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def mesh_verts(body):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        out = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = int(model.geom_dataid[g])
            nv = model.mesh_vertnum[mid]
            vs = model.mesh_vert.reshape(-1, 3)[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + nv]
            out.append(vs @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        return np.concatenate(out) if out else np.zeros((0, 3))

    chain = [
        "right_Arm-Base-Shoulder",
        "right_Feetech_Servo_Motor_v1_2",
        "right_Arm-Bicep",
        "right_Bicep_Right_v1_1",
        "right_Feetech_Servo_Motor_v1_3",
        "right_Arm-Forearm",
        "right_Feetech_Servo_Motor_v1_4",
        "right_Arm-Wrist",
        "right_Feetech_Servo_Motor_v1_5",
        "right_Gripper-Base",
        "right_Feetech_Servo_Motor_v1_6",
        "right_Gripper-Finger",
    ]
    # The joint servo-motor meshes bridge the link-to-link gaps (they are the real
    # connective hardware between bicep/forearm/wrist), so allow a small slack.
    max_gap_m = 0.02
    for a, b in zip(chain, chain[1:], strict=False):
        va, vb = mesh_verts(a), mesh_verts(b)
        assert len(va) > 0 and len(vb) > 0
        tree = cKDTree(va)
        dist, _ = tree.query(vb)
        assert float(dist.min()) < max_gap_m, (
            f"arm links {a} and {b} separated by {dist.min():.3f} m — check align_urdf_meshes"
        )


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


def test_sourccey_vendored_official_urdf():
    """The updated official ArmLeft URDF is vendored and referenced by the backend."""
    from pathlib import Path

    spec, _ = _load()
    assert spec.urdf_path, "urdf_path must point at the vendored official arm URDF"
    urdf = Path(spec.urdf_path)
    assert urdf.is_file(), f"vendored URDF missing: {urdf}"
    text = urdf.read_text()
    # updated official arm chain
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
        assert f'name="{joint}" type="revolute"' in text
    # ArmLeft + ArmRight reference copies live next to it
    assert (urdf.parent.parent / "ArmRight" / "ArmRight.urdf").is_file()


def test_sourccey_declares_arm_chains_and_kinematic_manip():
    """Declarative left/right arm chains + advertised kinematic pick/place."""
    spec, model = _load()
    assert spec.advertise_kinematic_manip is True
    assert spec.arm_chains and "left" in spec.arm_chains and "right" in spec.arm_chains
    for side in ("left", "right"):
        chain = spec.arm_chains[side]
        assert len(chain.joint_names) == 6
        for jn in chain.joint_names:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0
        assert chain.ee_body == f"{side}_Gripper-Finger"
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, chain.ee_body) >= 0
    from emet.motion.arm_manip_profile import ArmManipProfile

    for side in ("left", "right"):
        profile = ArmManipProfile.for_robot("sourccey", arm=side)
        assert profile.joint_names == tuple(spec.arm_chains[side].joint_names)
        assert profile.ee_body == f"{side}_Gripper-Finger"
        assert profile.gripper_contact_bodies()


def test_sourccey_create_model():
    from emet.robots import get_robot_backend

    model = get_robot_backend("sourccey").create_model()
    assert model is not None and model.get_dof() == 16
