"""Acceptance scorer never promotes incomplete or fabricated success."""

import json

import numpy as np
import pytest

from emet.eval.physical_tamp import score_physical_acceptance


def rows():
    result = []
    rotation = np.eye(3).ravel().tolist()
    for i in range(65):
        t = i * 0.1
        carried = 1.5 <= t < 4.5
        placed = t >= 4.5
        row = {
            "sim_time": t,
            "object_pos": [0, 0, 0.2 if carried else (0.1 if placed else 0)],
            "object_rot": rotation,
            "relative_pos": [0, 0, 0.03],
            "relative_rot": rotation,
            "gripper_contact": carried,
            "support_contact": placed,
            "other_contact": not (carried or placed),
            "scored_execution": True,
            "robot_collision_audited": True,
            "forbidden_robot_contacts": [],
            "release_command_audited": True,
            "release_open_command": placed,
        }
        result.append(row)
    return result


def score(tmp_path, records, *, complete=True, forbidden=False):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in [{"schema": 1, "config": {}}, *records]) + "\n")
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"accepted": not forbidden}) + "\n")
    return score_physical_acceptance(trace, audit, execution_completed=complete)


def test_physical_trace_requires_complete_retained_supported_task(tmp_path):
    result = score(tmp_path, rows())
    assert result["task_success"], result


@pytest.mark.parametrize(
    "fault",
    ["drop", "slip_rotation", "wrong_support", "collision", "missing_audit", "truncated", "incomplete", "teleport"],
)
def test_physical_negatives(tmp_path, fault):
    records = rows()
    if fault == "drop":
        records[35]["gripper_contact"] = False
    elif fault == "slip_rotation":
        records[35]["relative_rot"] = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    elif fault == "wrong_support":
        for row in records[45:]:
            row["support_contact"] = False
            row["other_contact"] = True
    elif fault == "collision":
        records[-1]["forbidden_robot_contacts"] = [{"bodies": ["base", "wall"]}]
    elif fault == "missing_audit":
        records[-1].pop("robot_collision_audited")
    elif fault == "truncated":
        records = records[:49]
    result = score(tmp_path, records, complete=fault != "incomplete", forbidden=fault == "teleport")
    assert not result["task_success"], (fault, result)


def test_route_budget_restores_state_and_leaves_execution_validation_enabled(monkeypatch):
    import mujoco

    import emet.eval.physical_tamp as module
    from emet.motion.mujoco_collision import MujocoSceneCollisionChecker

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="base_link"><freejoint/><geom size=".1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    space = module.SceneNavigationSpace(model, data, MujocoSceneCollisionChecker(model, robot_body="base_link"))
    before = data.qpos.copy()
    clock = iter([0.0, 11.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock, 11.0))
    assert space.plan_route(np.zeros(3), np.ones(3)) == []
    assert space.last_validity["reason"] == "route_planning_budget_exhausted"
    np.testing.assert_array_equal(data.qpos, before)
    assert space.is_valid(np.zeros(3))  # No lingering planning deadline in execution.


def test_approach_candidates_follow_actual_extension_line_and_preserve_state():
    import mujoco

    from emet.eval.physical_tamp import kinematic_base_candidates

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="base_link"><freejoint/><geom size=".1"/>
    <body name="ee" pos=".2 .07 0"><joint name="extension" type="slide" axis="1 0 0" range="0 .5"/>
    <geom size=".02"/></body></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    before = d.qpos.copy()
    target = np.array([1.0, 2.0])
    candidates = kinematic_base_candidates(m, d, target_xy=target, ee_body="ee", extension_joints=["extension"])
    assert len(candidates) == 40
    np.testing.assert_array_equal(d.qpos, before)
    for pose in candidates:
        c, sn = np.cos(pose[2]), np.sin(pose[2])
        local = np.array([[c, sn], [-sn, c]]) @ (target - pose[:2])
        assert abs(local[1] - 0.07) < 1e-10  # Preserve the lateral wrist offset.
        assert min(abs(local[0] - (0.2 + 0.5 * f)) for f in (0.25, 0.5, 0.75, 0.9, 0.98)) < 1e-10


def test_offline_base_pose_preserves_measured_tilt_height_and_velocity():
    import mujoco
    from scipy.spatial.transform import Rotation

    from emet.eval.physical_tamp import write_offline_base_pose

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="base_link"><freejoint/><geom size=".1"/></body>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qpos[2] = 0.2
    quat = Rotation.from_euler("xyz", [0.1, -0.05, 0.3]).as_quat()
    d.qpos[3:7] = quat[[3, 0, 1, 2]]
    d.qvel[:] = np.arange(6)
    assert write_offline_base_pose(m, d, base_body_name="base_link", x=1, y=2, theta=1.2)
    angles = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")
    np.testing.assert_allclose(angles, [0.1, -0.05, 1.2], atol=1e-9)
    np.testing.assert_allclose(d.qpos[:3], [1, 2, 0.2])
    np.testing.assert_array_equal(d.qvel, np.arange(6))


def test_commanded_release_allows_brief_settling_before_stable_support(tmp_path):
    records = rows()
    for row in records[43:45]:
        row["release_open_command"] = True
        row["gripper_contact"] = False
    assert score(tmp_path, records)["task_success"]


def test_uncommanded_drop_onto_correct_support_is_not_release(tmp_path):
    records = rows()
    for row in records:
        row["release_open_command"] = False
    assert not score(tmp_path, records)["task_success"]


def test_support_surface_ignores_tiny_parked_geoms_without_height_cutoff():
    import mujoco

    from emet.eval.physical_tamp import support_release_points

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="ee" pos="0 0 1"><geom size=".01"/></body>
    <body name="payload" pos="0 0 1"><freejoint/><geom type="box" size=".04 .04 .05"/></body>
    <body name="support" pos="2 0 0">
      <geom type="box" pos="0 0 .9" size=".4 .4 .05"/>
      <geom type="box" pos="0 0 10" size=".01 .01 .01"/>
    </body></worldbody></mujoco>""")
    points = support_release_points(m, mujoco.MjData(m), support_body="support", payload_body="payload", ee_body="ee")
    assert len(points) == 9
    np.testing.assert_allclose(points[0], [2, 0, 1.015])
    assert all(abs(p[0] - 2) + .04 < .4 and abs(p[1]) + .04 < .4 for p in points)
    assert all(abs(p[2] - 1.015) < 1e-8 for p in points)
    # An obstacle at the center need not rule out the whole named support.
    assert any(np.linalg.norm(p[:2] - [2, 0]) > .25 for p in points)
