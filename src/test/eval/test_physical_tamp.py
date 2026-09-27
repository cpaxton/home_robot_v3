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


@pytest.mark.parametrize("explicit_pair", [False, True])
def test_payload_tracking_envelope_rejects_nominally_clear_extended_load(explicit_pair):
    import mujoco

    from emet.eval.physical_tamp import SceneNavigationSpace
    from emet.motion.mujoco_collision import MujocoSceneCollisionChecker

    masks = 'contype="0" conaffinity="0"' if explicit_pair else ''
    pair = '<contact><pair geom1="load_geom" geom2="obstacle_geom"/></contact>' if explicit_pair else ''
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="base_link" pos="0 0 .3"><freejoint/><geom size=".05"/>
        <body name="hand" pos="1 0 .5"/>
      </body>
      <body name="load" pos="1 0 .8"><freejoint/><geom name="load_geom" size=".02" {masks}/></body>
      <body name="obstacle" pos="1 .06 .8"><geom name="obstacle_geom" size=".02" {masks}/></body>
    </worldbody>{pair}</mujoco>''')
    data = mujoco.MjData(model)
    checker = MujocoSceneCollisionChecker(model, robot_body="base_link")
    checker.set_payload(model, data, "load", "hand")
    nominal = SceneNavigationSpace(model, data, checker)
    robust = SceneNavigationSpace(model, data, checker, check_payload_tracking_envelope=True)
    assert nominal.is_valid([0, 0, 0])
    before = data.qpos.copy()
    assert not robust.is_valid([0, 0, 0])
    assert robust.last_validity["reason"] == "payload_tracking_envelope:scene_collision"
    np.testing.assert_array_equal(data.qpos, before)
    assert checker.payload_body == "load"
    # The same extended load has sufficient room when moved away from the wall.
    assert robust.is_valid([0, -.2, 0])
    if not explicit_pair:
        # The existing center margin is applied once, at the nominal pose.
        # Perturbed geometry is clear here; requiring another 22 cm at each
        # offset would silently inflate the nominal gate to 24 cm.
        model.body("obstacle").pos[:] = [.25, 0, .4]
        assert nominal.is_valid([0, 0, 0])
        assert robust.is_valid([0, 0, 0])
        assert not robust.is_valid([.02, 0, 0])
        assert robust.last_validity["reason"] == "below_clearance"


@pytest.mark.parametrize("reachable", [False, True])
def test_measured_placement_search_tries_alternatives_and_restores_payload(monkeypatch, reachable):
    from types import SimpleNamespace

    import mujoco

    import emet.eval.physical_tamp as module
    from emet.motion.mujoco_collision import MujocoSceneCollisionChecker

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base_link"><freejoint/><geom size=".05"/><body name="hand" pos="0 0 1"/></body>
      <body name="load" pos="0 0 1"><freejoint/><geom size=".02"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    checker = MujocoSceneCollisionChecker(model, robot_body="base_link")
    checker.set_payload(model, data, "load", "hand")
    before, transform = data.qpos.copy(), checker.payload_transform.copy()
    monkeypatch.setattr(module, "support_release_points", lambda *args, **kwargs: [np.array([3., 0., 1.])])
    monkeypatch.setattr(module, "kinematic_base_candidates", lambda *args, **kwargs: [np.array([2., 0., 0.])])
    attempts, arm_payloads = [], []

    def route(start, goal):
        np.testing.assert_array_equal(data.qpos, before)
        assert checker.payload_body == "load"
        attempts.append(goal[0])
        data.qpos[0] = 9.  # Hypothetical search state must not leak into another candidate.
        return [goal.tolist()] if reachable and goal[0] == 2 else []

    def plan_pose(point, rotation):
        if data.qpos[0] < 1.5:
            return None, "pose_ik_failed"
        arm_payloads.append(checker.payload_body)
        data.qpos[0] += .1
        return [np.zeros(1)], None

    executor = SimpleNamespace(model=model, data=data, collision=checker, coupled_groups=(), plan_pose=plan_pose)
    space = SimpleNamespace(base_body="base_link", plan_route=route, is_valid=lambda pose: pose[0] != 1,
                            last_validity={"reason": "scene_collision"})
    rejections, events = [], []
    result, error = module.plan_payload_placement(
        executor, space, {"object_body": "load", "ee_body": "hand", "support_body": "support"},
        approach=np.zeros(3), preferred_pose=[1., 0., 0.], rejections=rejections,
        event=lambda **row: events.append(row),
    )
    assert attempts == [2]  # Endpoint collision and failed arm IK never spend RRT budget.
    assert events[-1]["phase"] == "placement_route" and events[-1]["accepted"] == reachable
    assert {row["phase"] for row in rejections} >= {"transport", "preplace"}
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_array_equal(checker.payload_transform, transform)
    assert checker.payload_body == "load" and checker.payload_parent == "hand"
    if reachable:
        assert error is None and result["place_pose"] == [2, 0, 0]
        assert len(result["place_paths"]) == 3
        assert arm_payloads == ["load", "load", None]
    else:
        assert result is None and error == "no_place_witness_within_budget"


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


def test_direct_route_does_not_command_duplicate_final_pose():
    import mujoco

    from emet.eval.physical_tamp import SceneNavigationSpace
    from emet.motion.mujoco_collision import MujocoSceneCollisionChecker

    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="base_link"><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    space = SceneNavigationSpace(m, d, MujocoSceneCollisionChecker(m, robot_body='base_link'))
    route = np.asarray(space.plan_route(np.zeros(3), np.array([.8, 0, 0])))
    assert len(route) == 4
    assert np.all(np.linalg.norm(np.diff(route, axis=0), axis=1) > 0)
    np.testing.assert_allclose(route[-1], [.8, 0, 0])


def test_arrival_samples_reject_reach_boundary_and_restore_nominal_state():
    from types import SimpleNamespace

    import mujoco

    from emet.eval.physical_tamp import check_arrival_ik_samples

    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base"><freejoint/><geom size=".1"/>
        <body><joint name="x" type="slide" axis="1 0 0" range="0 1"/><geom size=".02"/>
          <body name="ee"><joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
          <geom size=".02"/></body>
        </body>
      </body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    e = SimpleNamespace(model=m, data=d, base_body='base', ee_body='ee',
                        joint_names=('x','y'), coupled_groups=(), joint_limit_margins={},
                        position_tolerance_m=.01, orientation_tolerance_rad=.1)
    before = d.qpos.copy()
    assert check_arrival_ik_samples(e, before, [([.5,0,0], np.eye(3))]) is None
    failure = check_arrival_ik_samples(e, before, [([.995,0,0], np.eye(3))])
    assert failure['reason'] == 'arrival_sample_ik_failed'
    assert failure['position_error_m'] > .01
    np.testing.assert_array_equal(d.qpos, before)
    # This arm cannot rotate its wrist. Only the explicitly supplied alternate
    # frame can satisfy orientation at the sampled arrivals.
    c, sn = np.cos(.3), np.sin(.3)
    unreachable = [([.5,0,0], [[c,-sn,0],[sn,c,0],[0,0,1]])]
    assert check_arrival_ik_samples(e, before, unreachable) is not None
    assert check_arrival_ik_samples(e, before, unreachable,
                                   target_options=[unreachable, [([.5,0,0], np.eye(3))]]) is None
    np.testing.assert_array_equal(d.qpos, before)


def test_rrt_steering_never_translates_sideways():
    import mujoco

    from emet.eval.physical_tamp import SceneNavigationSpace
    from emet.motion.mujoco_collision import MujocoSceneCollisionChecker

    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="base_link"><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    space = SceneNavigationSpace(m, d, MujocoSceneCollisionChecker(m, robot_body='base_link'), allow_reverse=True)
    previous = np.zeros(3)
    for pose in space.extend(previous, [-.4,.3,1.]):
        delta = pose[:2] - previous[:2]
        if np.linalg.norm(delta) > 1e-8:
            forward = np.array([np.cos(pose[2]),np.sin(pose[2])])
            assert abs(delta[0]*forward[1]-delta[1]*forward[0]) < 1e-8
            assert abs(pose[2]-previous[2]) < 1e-8
        previous = pose
    np.testing.assert_allclose(previous, [-.4,.3,1.])


def test_grasp_yaw_options_preserve_positions_and_bound_orientation_family():
    from emet.eval.physical_tamp import grasp_yaw_options

    target = [([1,2,3], np.eye(3))]
    options = grasp_yaw_options(target)
    assert len(options) == 3
    for expected, option in zip([0.,-.08,.08], options, strict=True):
        point, rotation = option[0]
        np.testing.assert_array_equal(point, target[0][0])
        assert np.arctan2(rotation[1][0],rotation[0][0]) == pytest.approx(expected)
        np.testing.assert_allclose(np.array(rotation).T @ rotation, np.eye(3), atol=1e-12)
