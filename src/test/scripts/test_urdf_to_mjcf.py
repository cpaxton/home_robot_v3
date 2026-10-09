"""Independent URDF FK parity catches misplaced pivots and moving parent links."""
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]


def converter():
    spec = importlib.util.spec_from_file_location("urdf_converter", ROOT / "scripts/robot_assets/urdf_to_mjcf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_sourccey_all_link_poses_match_urdf_at_nonzero_angles():
    source = ROOT / "src/emet/assets/robot/sourccey/urdf/ArmLeft/ArmLeft.urdf"
    conv = converter().UrdfToMjcf(source, mesh_map={})
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody>' + conv.build() + '</worldbody></mujoco>')
    data = mujoco.MjData(model)
    joints = ET.parse(source).getroot().findall("joint")
    for sign in (-1., 0., 1.):
        values = {j.get("name"): sign * .35 for j in joints if j.get("type") != "fixed"}
        for name, q in values.items():
            data.qpos[model.joint(name).qposadr[0]] = q
        mujoco.mj_forward(model, data)
        poses = {conv.root_body: np.eye(4)}
        pending = list(joints)
        while pending:
            progressed = False
            for joint in list(pending):
                parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
                if parent not in poses:
                    continue
                origin = joint.find("origin")
                local = np.eye(4)
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                motion = np.eye(4)
                if joint.get("type") != "fixed":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    motion[:3, :3] = Rotation.from_rotvec(axis * values[joint.get("name")]).as_matrix()
                    assert model.joint(joint.get("name")).bodyid[0] == model.body(child).id
                poses[child] = poses[parent] @ local @ motion
                pending.remove(joint)
                progressed = True
            assert progressed
        for name, pose in poses.items():
            np.testing.assert_allclose(data.body(name).xpos, pose[:3, 3], atol=1e-7, err_msg=name)
            np.testing.assert_allclose(data.body(name).xmat.reshape(3, 3), pose[:3, :3], atol=1e-7, err_msg=name)


def test_recenter_composes_rotated_ancestors():
    xml = '''<body name="root" pos="1 0 0" quat=".7071067811865476 0 0 .7071067811865476">
      <body name="child" pos="1 0 0"><joint name="pivot" pos=".2 0 0"/>
      <geom size=".1"/></body></body>'''
    text = converter().wrap_recentered_on_joint(xml, joint_name="pivot")
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>' + text + '</worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xanchor[model.joint("pivot").id], 0., atol=1e-7)
