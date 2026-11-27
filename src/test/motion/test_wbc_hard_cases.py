# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Hard-case extraction and Panda failure-posture replay."""

from __future__ import annotations

import numpy as np
import pytest

from emet.rl.manipulation.wbc_hard_cases import (
    WbcHardCase,
    hard_cases_from_failure_records,
    load_hard_cases,
    save_hard_cases,
)


def _record(initial, target, *, robot="panda_omron", mode="tracking_failed"):
    names = [f"robot0_joint{i}" for i in range(1, 8)]
    return {
        "attempt": {
            "joint_names": names,
            "trace": [{"qpos": list(initial)}],
            "planned_waypoints": [list(target)],
            "failure_mode": mode,
            "reached": False,
            "op": "grasp",
        },
        "context": {
            "robot_id": robot,
            "object_query": "mug",
            "receptacle_query": "cab",
            "job_id": "job-1",
        },
    }


def test_hard_cases_extract_contract_joint_order():
    initial = (0.1, 0.2, 0.3, -2.0, 0.0, 2.5, 0.7)
    target = (0.2, 0.1, 0.4, -1.9, 0.0, 2.4, 0.8)
    cases = hard_cases_from_failure_records([_record(initial, target)], robot="panda_omron")
    assert len(cases) == 1
    assert cases[0].initial_q == initial
    assert cases[0].target_q == target
    assert cases[0].arm == "right"
    assert cases[0].failure_mode == "tracking_failed"
    assert cases[0].source["op"] == "grasp"


def test_hard_cases_infer_dual_arm_side_and_contract_order():
    initial = (0.1, 0.2, 0.3, -2.0, 0.0, 2.5)
    target = (0.2, 0.1, 0.4, -1.9, 0.0, 2.4)
    record = _record(initial, target, robot="rb_y1")
    record["attempt"]["joint_names"] = [f"left_arm_joint{i}" for i in range(1, 7)]
    cases = hard_cases_from_failure_records([record], robot="rb_y1")
    assert len(cases) == 1
    assert cases[0].arm == "left"
    assert cases[0].initial_q == initial
    assert cases[0].target_q == target


def test_hard_cases_do_not_match_unrelated_numeric_suffixes():
    initial = (0.1, 0.2, 0.3, -2.0, 0.0, 2.5)
    target = (0.2, 0.1, 0.4, -1.9, 0.0, 2.4)
    record = _record(initial, target, robot="rb_y1")
    record["attempt"]["joint_names"] = [
        "left_elbow_1",
        "left_elbow_2",
        "left_elbow_3",
        "left_elbow_4",
        "left_elbow_5",
        "left_elbow_6",
    ]
    assert hard_cases_from_failure_records([record], robot="rb_y1") == []


def test_hard_cases_skip_other_robots_and_noop_targets():
    q = (0.1, 0.2, 0.3, -2.0, 0.0, 2.5, 0.7)
    cases = hard_cases_from_failure_records(
        [_record(q, q), _record(q, q, robot="rby1")],
        robot="panda_omron",
    )
    assert cases == []


def test_hard_cases_skip_successful_last_attempts():
    initial = (0.1, 0.2, 0.3, -2.0, 0.0, 2.5, 0.7)
    target = (0.2, 0.1, 0.4, -1.9, 0.0, 2.4, 0.8)
    record = _record(initial, target)
    record["attempt"]["reached"] = True
    record["attempt"]["failure_mode"] = None
    assert hard_cases_from_failure_records([record], robot="pandaomron") == []


def test_hard_cases_normalize_robot_aliases():
    initial = (0.1, 0.2, 0.3, -2.0, 0.0, 2.5, 0.7)
    target = (0.2, 0.1, 0.4, -1.9, 0.0, 2.4, 0.8)
    for requested in ("pandaomron", "panda_omron", "PandaOmron"):
        cases = hard_cases_from_failure_records([_record(initial, target)], robot=requested)
        assert len(cases) == 1, requested
    rby1 = _record(initial[:6], target[:6], robot="rby1")
    rby1["attempt"]["joint_names"] = [f"right_arm_joint{i}" for i in range(1, 7)]
    assert hard_cases_from_failure_records([rby1], robot="rb_y1") == hard_cases_from_failure_records(
        [rby1], robot="rby1"
    )


def test_hard_cases_capture_scene_context():
    initial = (0.1, 0.2, 0.3, -2.0, 0.0, 2.5, 0.7)
    target = (0.2, 0.1, 0.4, -1.9, 0.0, 2.4, 0.8)
    record = _record(initial, target)
    record["attempt"]["op"] = "grasp"
    record["context"].update(
        {
            "object_body": "obj_main",
            "receptacle_body": "cab_main",
            "layout": 1,
            "style": 1,
        }
    )
    case = hard_cases_from_failure_records([record], robot="pandaomron")[0]
    assert case.source["op"] == "grasp"
    assert case.scene["op"] == "grasp"
    assert case.scene["object_body"] == "obj_main"
    assert case.scene["receptacle_body"] == "cab_main"
    assert case.scene["layout"] == 1
    assert case.scene["style"] == 1


def test_hard_cases_json_roundtrip(tmp_path):
    case = WbcHardCase(
        name="c1",
        robot="panda_omron",
        arm="right",
        initial_q=(0.0,) * 7,
        target_q=(0.1,) * 7,
        failure_mode="tracking_failed",
        scene={"env": "PickPlaceCounterToCabinet", "layout": 1, "style": 1},
    )
    path = save_hard_cases(tmp_path / "cases.json", [case])
    loaded = load_hard_cases(path)
    assert loaded == [case]


@pytest.mark.sim
def test_pandaomron_replays_a_hard_case_posture():
    from emet.rl.manipulation.wbc_pandaomron import PandaOmronWbcBackend

    backend = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0)
    case = WbcHardCase(
        name="sink-reach",
        robot="panda_omron",
        arm="right",
        initial_q=(0.02, -0.51, 0.04, -1.93, 0.0, 2.60, 0.66),
        target_q=(-0.01, -0.20, 0.01, -2.29, 0.0, 2.48, 0.65),
        failure_mode="tracking_failed",
    )
    observation = backend.load_hard_case(case)
    assert observation is not None
    assert np.allclose(observation.arms[0].joint_position, case.initial_q)
    assert observation.arms[0].ee_target is not None
    assert observation.arms[0].target_mask == (True, True)
    action = backend.teacher_action()
    assert action.shape == (13,)
    assert np.isfinite(action).all()
    backend.close()
