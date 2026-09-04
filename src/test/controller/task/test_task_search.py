# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for deterministic TAMP task search (no sim)."""

from __future__ import annotations

import numpy as np

from emet.controller.task.tamp.smoke_grasps import plant_mixed_grasp_poses
from emet.controller.task.tamp.task_search import (
    TaskPlan,
    TaskPlanStep,
    approach_pose_for_object_xy,
    execute_task_plan,
    plan_pick_place,
    rank_grasps_by_ik,
)


class _FakeRobot:
    def __init__(self, placements: dict):
        self._pl = placements
        self.moved: list[np.ndarray] = []

    def get_emet_session(self):
        return {
            "is_simulation": True,
            "emet_robot_id": "rby1",
            "sim_object_placements": self._pl,
            "capabilities": {"kinematic_manip": True, "sim_set_body_pose": True},
        }

    def move_base_to(self, xyt, blocking=True, world_frame=True):
        self.moved.append(np.asarray(xyt, dtype=np.float64).copy())


class _FakeGrasp:
    def __init__(self, xyz):
        T = np.eye(4)
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        self.T_world = T
        self.position = T[:3, 3].copy()


class _FakeVideo:
    def __init__(self):
        self.dumps: list[str] = []
        self.captures = 0

    def set_status(self, action: str = "", *, goal=None, detail=None):
        return None

    def capture_once(self):
        self.captures += 1
        return True

    def dump_paper_stills(self, tag: str):
        self.dumps.append(str(tag))
        return {}


def test_approach_pose_faces_minus_y():
    p = approach_pose_for_object_xy([1.0, 2.0], standoff=0.4)
    assert abs(p[0] - 1.0) < 1e-9
    assert abs(p[1] - 2.4) < 1e-9
    assert abs(p[2] + np.pi / 2) < 1e-9


def test_approach_pose_side_left_yaw_plus_half_pi():
    p = approach_pose_for_object_xy([0.08, -0.55], standoff=0.55, mode="side", arm="left")
    assert abs(p[0] - 0.08) < 1e-9
    assert abs(p[1] - 0.0) < 1e-9
    assert abs(p[2] - np.pi / 2) < 1e-9


def test_approach_pose_side_right_yaw_minus_half_pi():
    p = approach_pose_for_object_xy([0.08, -0.55], standoff=0.55, mode="side", arm="right")
    assert abs(p[2] + np.pi / 2) < 1e-9


def test_plant_mixed_grasps_puts_decoys_first():
    poses = plant_mixed_grasp_poses([0.1, -0.5, 0.8], n_infeasible=2)
    assert len(poses) == 3
    assert poses[0].asset_id.startswith("decoy")
    assert poses[-1].asset_id == "reachable_topdown"
    # top_down_grasp_T adds the 0.02 m approach z-offset (grasp frame +Z into object).
    np.testing.assert_allclose(poses[-1].position, np.array([0.1, -0.5, 0.82]), atol=1e-9)


def test_rank_grasps_skips_infeasible_picks_reachable():
    """Offline IK: planted far grasps fail; near-EE grasp succeeds and is chosen."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS, RBY1_LEFT_EE_BODY
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    for name in RBY1_LEFT_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        data.qpos[int(model.jnt_qposadr[jid])] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RBY1_LEFT_EE_BODY)
    ee = np.asarray(data.body(ee_id).xpos, dtype=np.float64).copy()
    poses = plant_mixed_grasp_poses(ee + np.array([0.0, 0.0, 0.05]), n_infeasible=2)
    scores = rank_grasps_by_ik(
        model,
        data,
        ee_body=RBY1_LEFT_EE_BODY,
        joint_names=RBY1_LEFT_ARM_JOINTS,
        grasp_poses=poses,
        top_k=8,
    )
    assert any(ok for _i, _e, ok in scores), scores
    assert any(not ok for _i, _e, ok in scores), scores
    chosen = next(i for i, _e, ok in scores if ok)
    assert chosen == len(poses) - 1


def test_plan_pick_place_chooses_reachable_when_executor_ranks():
    """plan_pick_place with a stub executor that ranks via real IK model."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS, RBY1_LEFT_EE_BODY
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    for name in RBY1_LEFT_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        data.qpos[int(model.jnt_qposadr[jid])] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RBY1_LEFT_EE_BODY)
    ee = np.asarray(data.body(ee_id).xpos, dtype=np.float64).copy()
    good = ee + np.array([0.0, 0.0, 0.04])
    poses = plant_mixed_grasp_poses(good, n_infeasible=2)

    class _Exe:
        ee_body = RBY1_LEFT_EE_BODY
        joint_names = RBY1_LEFT_ARM_JOINTS
        _model = model
        _data = data

        def _ensure_model(self):
            return True

    pl = {
        "obj_a": {"cat": "red cylinder", "pos": good.tolist()},
        "cube_b": {"cat": "blue cube", "pos": [0.2, -0.4, 0.75]},
    }
    robot = _FakeRobot(pl)
    plan = plan_pick_place(
        robot,
        object_query="red cylinder",
        receptacle_query="blue cube",
        grasp_poses=poses,
        object_gt_body="obj_a",
        receptacle_gt_body="cube_b",
        executor=_Exe(),
    )
    assert plan.success
    assert plan.chosen_grasp_index == len(poses) - 1
    assert any(ok for _i, _e, ok in plan.grasp_scores)
    assert any(not ok for _i, _e, ok in plan.grasp_scores)


def test_plan_pick_place_without_executor():
    pl = {
        "obj_a": {"cat": "red cylinder", "pos": [0.1, -0.5, 0.8]},
        "cube_b": {"cat": "blue cube", "pos": [0.2, -0.4, 0.75]},
    }
    robot = _FakeRobot(pl)
    grasps = [_FakeGrasp([0.1, -0.5, 0.85]), _FakeGrasp([0.12, -0.5, 0.85])]
    plan = plan_pick_place(
        robot,
        object_query="red cylinder",
        receptacle_query="blue cube",
        grasp_poses=grasps,
        object_gt_body="obj_a",
        receptacle_gt_body="cube_b",
        executor=None,
    )
    assert plan.success
    assert plan.chosen_grasp_index == 0
    assert [s.op for s in plan.steps] == ["approach", "grasp", "place"]
    assert any("approach@" in n for n in plan.expanded_nodes)


def test_plan_pick_place_uses_spec_side_approach():
    from types import SimpleNamespace

    pl = {
        "obj_a": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6]},
        "cube_b": {"cat": "blue cube", "pos": [-0.02, -0.55, 0.6]},
    }
    robot = _FakeRobot(pl)
    robot._spec = SimpleNamespace(tamp_approach="side")
    grasps = [_FakeGrasp([0.08, -0.55, 0.62])]
    plan = plan_pick_place(
        robot,
        object_query="red cylinder",
        receptacle_query="blue cube",
        grasp_poses=grasps,
        object_gt_body="obj_a",
        receptacle_gt_body="cube_b",
        executor=None,
    )
    assert plan.success
    xyt = plan.steps[0].args["xyt"]
    assert abs(xyt[2] - np.pi / 2) < 1e-9
    assert any("mode=side" in n for n in plan.expanded_nodes)


def test_plan_missing_object():
    robot = _FakeRobot({})
    plan = plan_pick_place(
        robot,
        object_query="x",
        receptacle_query="y",
        grasp_poses=[],
        object_gt_body="missing",
        executor=None,
    )
    assert not plan.success
    assert plan.message == "object_not_in_gt"


def test_task_plan_dataclass():
    p = TaskPlan(steps=[TaskPlanStep("approach", {"xyt": [0, 0, 0]})], object_body="a", receptacle_body=None)
    assert p.steps[0].op == "approach"


def test_execute_task_plan_dumps_stills_per_op():
    robot = _FakeRobot({})
    video = _FakeVideo()
    plan = TaskPlan(
        steps=[TaskPlanStep("approach", {"xyt": [0.08, 0.0, 1.57], "world_frame": True})],
        object_body="obj_a",
        receptacle_body="cube_b",
    )
    out = execute_task_plan(robot, plan, executor=None, grasp_poses=[], manip_mode="teleport", video_recorder=video)
    assert out.success
    assert out.completed_ops == ["approach"]
    assert video.captures == 1
    assert video.dumps == ["approach"]
    assert len(robot.moved) == 1


def test_sourccey_side_approach_reaches_default_table_cylinder():
    """CPU: left-arm IK to the default-table red cylinder from the side standoff.

    The rby1 front pose (yaw=−π/2 at y=+0.55) misses this workspace by >1 m.
    """
    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import write_offline_mjcf_base_xyt
    from emet.motion.arm_manip_profile import ArmManipProfile
    from emet.motion.mujoco_arm_ik import solve_position_ik_multiseed
    from emet.robots.sourccey import SourcceyBackend
    from emet.simulation.sim_object_placements import DEFAULT_TABLE_SCENE_PLACEMENTS

    spec = SourcceyBackend().get_spec()
    assert spec.tamp_approach == "side"
    obj = np.asarray(DEFAULT_TABLE_SCENE_PLACEMENTS["object2"]["pos"], dtype=np.float64)
    approach = approach_pose_for_object_xy(obj[:2], mode="side", arm="left")
    assert abs(approach[2] - np.pi / 2) < 1e-9

    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    write_offline_mjcf_base_xyt(model, data, approach, planar_joint_names=spec.planar_base_joint_names)
    mujoco.mj_forward(model, data)
    prof = ArmManipProfile.for_robot("sourccey", arm="left")
    res = solve_position_ik_multiseed(
        model,
        data,
        ee_body=prof.ee_body,
        joint_names=prof.joint_names,
        target_pos=obj,
        max_iters=80,
        tol_m=0.05,
    )
    assert res.success, f"side approach IK err={res.pos_error_m:.3f} m (approach={approach.tolist()})"
    assert res.pos_error_m < 0.05

    front = approach_pose_for_object_xy(obj[:2], mode="front", arm="left")
    write_offline_mjcf_base_xyt(model, data, front, planar_joint_names=spec.planar_base_joint_names)
    if model.nkey:
        data.qpos[4:] = model.key_qpos[0][4:]
    mujoco.mj_forward(model, data)
    front_res = solve_position_ik_multiseed(
        model,
        data,
        ee_body=prof.ee_body,
        joint_names=prof.joint_names,
        target_pos=obj,
        max_iters=80,
        tol_m=0.05,
    )
    assert not front_res.success
    assert front_res.pos_error_m > 0.5
