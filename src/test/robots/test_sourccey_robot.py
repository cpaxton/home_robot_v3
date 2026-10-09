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

import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.robots.sourccey import (
    SOURCCEY_CAMERA_NAMES,
    SOURCCEY_GRIPPER_ACTUATORS,
    SOURCCEY_GRIPPER_JOINTS,
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
    assert spec.tamp_approach == "front"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift") >= 0


def test_sourccey_mjcf_cameras():
    spec, model = _load()
    assert (
        spec.camera_names
        == SOURCCEY_CAMERA_NAMES
        == ["front_left", "front_right", "bottom", "wrist_left", "wrist_right"]
    )
    for cname in SOURCCEY_CAMERA_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cname) >= 0


def test_sourccey_front_camera_image_up_is_upright():
    """Production pixel transforms must preserve the asset's upright horizon."""
    from emet.utils.pinhole_intrinsics import chain_pinhole_K_pixel_ops

    spec, model = _load()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in ("front_left", "front_right"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        optical_rotation = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        k = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
        k, _, _ = chain_pinhole_K_pixel_ops(k, 480, 640, spec.robosuite_rgb_depth_ops)
        up = optical_rotation @ np.linalg.solve(k, [0.0, -1.0, 0.0])
        assert up[2] / np.linalg.norm(up) > 0.7


def test_sourccey_mjcf_geometry_sane():
    spec, model = _load()
    data = mujoco.MjData(model)
    # Use the raised-arm startup pose to check full assembly extents
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    # ~1 m tall mobile manipulator (real: 1030 mm)
    zs = [data.body(i).xpos[2] for i in range(model.nbody)]
    assert max(zs) > 0.8
    # Raised hands extend forward, with all body centers within 0.5 m
    xs = [data.body(i).xpos[0] for i in range(model.nbody)]
    ys = [data.body(i).xpos[1] for i in range(model.nbody)]
    assert max(abs(v) for v in xs) < 0.5
    assert max(abs(v) for v in ys) < 0.45
    # Upstream warns its CAD inertials total 169 kg and are not measured hardware mass.
    assert float(sum(model.body_mass)) == pytest.approx(169.042941, abs=1e-5)


@pytest.mark.parametrize("pose", ["home", "zero", "interior"])
def test_sourccey_matches_official_full_assembly_fk(pose):
    """Independent upstream oracle catches mirrored arms, frame and lift errors."""
    spec, model = _load()
    official = mujoco.MjModel.from_xml_path(str(Path(spec.mjcf_path).parent / "upstream/models/sourccey.xml"))
    data, expected = mujoco.MjData(model), mujoco.MjData(official)
    if pose == "home":
        mujoco.mj_resetDataKeyframe(model, data, 0)
    elif pose == "interior":
        for i in range(3, model.njnt):
            lo, hi = model.jnt_range[i]
            data.qpos[model.jnt_qposadr[i]] = lo * 0.35 + hi * 0.65
    for name in SOURCCEY_JOINT_NAMES[3:]:
        target = "linear_actuator" if name == "lift" else name
        expected.qpos[official.joint(target).qposadr[0]] = data.qpos[model.joint(name).qposadr[0]]
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(official, expected)
    for i in range(1, official.nbody):
        name = official.body(i).name
        np.testing.assert_allclose(data.body(name).xpos, expected.body(name).xpos, atol=1e-9)
        np.testing.assert_allclose(data.body(name).xmat, expected.body(name).xmat, atol=1e-9)
    for name in SOURCCEY_CAMERA_NAMES:
        np.testing.assert_allclose(data.camera(name).xpos, expected.camera(name).xpos, atol=1e-9)
        np.testing.assert_allclose(data.camera(name).xmat, expected.camera(name).xmat, atol=1e-9)


def test_sourccey_hardware_limits_and_home():
    spec, model = _load()
    source = ET.parse(spec.urdf_path)
    for name in SOURCCEY_JOINT_NAMES[3:]:
        official_name = "linear_actuator" if name == "lift" else name
        limit = source.find(f"joint[@name='{official_name}']/limit")
        jid, aid = model.joint(name).id, model.actuator(name + "_act").id
        lo, hi = model.jnt_range[jid]
        assert lo >= float(limit.get("lower")) - 1e-10
        assert hi <= float(limit.get("upper")) + 1e-10
        np.testing.assert_allclose(model.actuator_ctrlrange[aid], [lo, hi])
        assert lo <= model.key_qpos[0, model.jnt_qposadr[jid]] <= hi
        assert lo <= model.key_ctrl[0, aid] <= hi
    # A finger command must not move the IK target frame.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = data.body(spec.arm_chain.ee_body).xpos.copy()
    data.qpos[model.joint("left_gripper").qposadr[0]] = 1.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body(spec.arm_chain.ee_body).xpos, before)


def test_sourccey_generated_model_and_source_hashes():
    spec, _ = _load()
    assets = Path(spec.mjcf_path).parent
    manifest = json.loads((assets / "upstream/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((assets / "upstream" / name).read_bytes()).hexdigest() == expected
    path = Path(__file__).resolve().parents[3] / "scripts/robot_assets/assemble_sourccey.py"
    module_spec = importlib.util.spec_from_file_location("assemble_sourccey", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.build() == Path(spec.mjcf_path).read_text()


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
    for _ in range(2500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qacc).all()
    assert float(data.camera("front_left").xpos[2]) > 0.5


def test_sourccey_registry_and_assets():
    from emet.robots import get_robot_spec
    from emet.utils.assets import get_robot_mjcf_path

    assert get_robot_mjcf_path("sourccey") is not None
    spec = get_robot_spec("sourccey")
    assert spec is not None and spec.name == "sourccey"


def test_sourccey_vendored_official_urdf():
    spec, _ = _load()
    urdf = Path(spec.urdf_path)
    source = ET.parse(urdf)
    assert source.find("joint[@name='linear_actuator']") is not None
    for side in ("left", "right"):
        assert source.find(f"joint[@name='{side}_shoulder_pan']") is not None
    for mesh in source.iter("mesh"):
        assert (urdf.parent / mesh.get("filename")).is_file()
    assert not list(urdf.parent.rglob("*.meta"))


def test_sourccey_declares_arm_chains_and_kinematic_manip():
    """Declarative left/right arm chains + advertised kinematic pick/place."""
    spec, model = _load()
    assert spec.advertise_kinematic_manip is True
    assert spec.arm_chains and "left" in spec.arm_chains and "right" in spec.arm_chains
    for side in ("left", "right"):
        chain = spec.arm_chains[side]
        assert len(chain.joint_names) == 5
        assert not any("gripper" in jn for jn in chain.joint_names)
        assert any(a.endswith("gripper_act") for a in chain.actuator_names)
        for jn in chain.joint_names:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0
        assert chain.ee_body == f"Gripper_Base_v1_{1 if side == 'left' else 2}"
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, chain.ee_body) >= 0
    from emet.motion.arm_manip_profile import ArmManipProfile

    for side in ("left", "right"):
        profile = ArmManipProfile.for_robot("sourccey", arm=side)
        assert profile.joint_names == tuple(spec.arm_chains[side].joint_names)
        assert profile.ee_body == spec.arm_chains[side].ee_body
        assert profile.gripper_contact_bodies()
        from emet.motion.mujoco_arm_ik import pack_arm_into_actuator_dict

        q = np.linspace(0.1, 0.5, len(profile.joint_names))
        packed = pack_arm_into_actuator_dict(spec.actuator_names, profile.joint_names, q)
        assert len(packed) == len(profile.joint_names)
        assert all(f"{jn}_act" in packed for jn in profile.joint_names)


def test_sourccey_executor_actuator_and_gripper_aliases():
    from unittest.mock import MagicMock

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.motion.arm_manip_profile import ArmManipProfile

    spec, _ = _load()
    profile = ArmManipProfile.for_robot("sourccey", arm="left")
    robot = MagicMock()
    robot._spec = spec
    robot.get_joint_state.return_value = (np.zeros(len(spec.actuator_names)), None, None)
    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    exe.arm = "left"
    exe.profile = profile
    exe.joint_names = list(profile.joint_names)
    assert exe._actuator_to_joint_name("left_shoulder_pan_act") == "left_shoulder_pan"
    assert exe._actuator_to_joint_name("left_arm1") == "left_arm_joint1"
    exe._set_gripper(open_=True)
    sent = robot.set_actuator_positions.call_args[0][0]
    assert sent["left_gripper_act"] == pytest.approx(np.deg2rad(60))
    exe._set_gripper(open_=False)
    assert robot.set_actuator_positions.call_args[0][0]["left_gripper_act"] == pytest.approx(np.deg2rad(-5))


def test_wrap_recentered_on_joint_puts_parent_at_origin():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts" / "robot_assets" / "urdf_to_mjcf.py"
    spec = importlib.util.spec_from_file_location("urdf_to_mjcf", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    xml = (
        '<body name="base_link" pos="0 0 0">\n'
        '  <body name="servo" pos="0.3 0.2 -0.02">\n'
        '    <joint name="shoulder_pan" type="hinge" axis="0 0 1"/>\n'
        "  </body>\n"
        "</body>"
    )
    out = mod.wrap_recentered_on_joint(xml, joint_name="shoulder_pan")
    assert 'name="arm_root"' in out
    assert "-0.3" in out and "-0.2" in out and "0.02" in out


def test_sourccey_create_model():
    from emet.robots import get_robot_backend

    model = get_robot_backend("sourccey").create_model()
    assert model is not None and model.get_dof() == 16


def test_sourccey_merged_table_home_keeps_object_freejoints():
    """Scene include-first + sourccey_home must not drop the table cube/cylinder to the origin."""
    from emet.simulation.mujoco_server import _load_default_scene_with_robot
    from emet.simulation.robosuite_load_utils import apply_home_keyframe_preserving_planar_base

    spec = SourcceyBackend().get_spec()
    model = _load_default_scene_with_robot("sourccey")
    assert model is not None
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    o1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object1")
    o2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object2")
    assert o1 >= 0 and o2 >= 0
    obj1_before = np.array(data.xpos[o1], dtype=np.float64, copy=True)
    obj2_before = np.array(data.xpos[o2], dtype=np.float64, copy=True)
    np.testing.assert_allclose(obj1_before, [-0.02, -0.55, 0.6], atol=1e-4)
    np.testing.assert_allclose(obj2_before, [0.08, -0.55, 0.6], atol=1e-4)
    pan_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pan")
    pan_adr = int(model.jnt_qposadr[pan_jid])
    pan_before = float(data.qpos[pan_adr])
    planar = spec.planar_base_joint_names
    assert planar is not None
    assert apply_home_keyframe_preserving_planar_base(
        model,
        data,
        planar_joint_names=planar,
        base_body_name=spec.base_link_name,
        spec=spec,
    )
    np.testing.assert_allclose(data.xpos[o1], obj1_before, atol=1e-4)
    np.testing.assert_allclose(data.xpos[o2], obj2_before, atol=1e-4)
    pan_after = float(data.qpos[pan_adr])
    assert abs(pan_after - pan_before) > 0.2
    assert abs(pan_after - (-0.785)) < 0.05


@pytest.mark.parametrize("side", ["left", "right"])
def test_sourccey_client_gripper_commands_reach_only_selected_actuator(side):
    from unittest.mock import Mock

    from emet.controller.generic_zmq_client import GenericZmqClient
    from emet.simulation.gripper_action import apply_gripper_action_robosuite

    spec, model = _load()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    client = object.__new__(GenericZmqClient)
    client._spec = spec
    client.send_action = Mock()
    for method, expected in ((client.open_gripper, np.deg2rad(60)), (client.close_gripper, np.deg2rad(-5))):
        before = data.ctrl.copy()
        method(f"{side}_gripper")
        action = client.send_action.call_args.args[0]
        assert action[f"gripper_{side}"] == pytest.approx(expected)
        updated = apply_gripper_action_robosuite(spec, model, data, action)
        assert updated == [f"{side}_gripper_act"]
        aid = model.actuator(updated[0]).id
        assert data.ctrl[aid] == pytest.approx(expected)
        np.testing.assert_allclose(np.delete(data.ctrl, aid), np.delete(before, aid))
    with pytest.raises(ValueError):
        client.open_gripper("unknown")
    before = data.ctrl.copy()
    with pytest.raises(ValueError):
        apply_gripper_action_robosuite(spec, model, data, {"gripper_left": 0.0, "gripper_right": float("nan")})
    np.testing.assert_array_equal(data.ctrl, before)
