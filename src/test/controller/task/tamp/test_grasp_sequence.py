"""Approach selection must admit the endpoints the executor will actually use."""
import mujoco
import numpy as np

from emet.controller.task.tamp.grasp_frames import top_down_grasp_T
from emet.controller.task.tamp.task_search import rank_grasps_by_ik


def rig():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="tool"><joint name="z" type="slide" axis="0 0 1" range="-.2 .2"/>
        <geom size=".01"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_reachable_grasp_with_unreachable_pregrasp_is_rejected():
    model, data = rig()
    # Both grasp points are reachable; the first pregrasp/lift is above the limit.
    grasps = [top_down_grasp_T([0, 0, .15], z_offset_m=0),
              top_down_grasp_T([0, 0, 0], z_offset_m=0)]
    before = data.qpos.copy()
    scores = rank_grasps_by_ik(model, data, ee_body="tool", joint_names=["z"], grasp_poses=grasps)
    assert scores[0][0] == 1 and scores[0][2]
    assert scores[1][0] == 0 and not scores[1][2]
    np.testing.assert_array_equal(data.qpos, before)


def test_lift_is_checked_even_with_reachable_side_pregrasp():
    model, data = rig()
    grasp = top_down_grasp_T([0, 0, .15], z_offset_m=0)
    # Downward approach stays reachable; lift still exceeds the upper limit.
    grasp[:3, :3] = np.eye(3)
    scores = rank_grasps_by_ik(model, data, ee_body="tool", joint_names=["z"], grasp_poses=[grasp])
    assert not scores[0][2]


def test_execution_tolerance_rejects_old_five_centimeter_endpoint_error():
    model, data = rig()
    grasp = top_down_grasp_T([.03, 0, 0], z_offset_m=0)
    scores = rank_grasps_by_ik(model, data, ee_body="tool", joint_names=["z"], grasp_poses=[grasp])
    assert not scores[0][2]
