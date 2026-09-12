# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from emet.controller.task.dynamem.dynamem_task import DynamemTaskExecutor
from emet.memory.grounded_target import GroundedTarget
from emet.memory.query_candidates import QueryCandidates


@pytest.mark.parametrize("backend,expects_detector", [("vlm", False), ("yoloe", True)])
def test_visual_servo_constructor_respects_grounding_backend(backend, expects_detector):
    from emet.core import AbstractRobotClient

    parameters = {
        "query_driven_memory": True,
        "query_memory": {"grounding_backend": backend},
        "detection": {},
        "encoder": "siglip",
    }
    module = "emet.controller.task.dynamem.dynamem_task"
    with (
        patch(f"{module}.create_semantic_sensor") as detector,
        patch.object(DynamemTaskExecutor, "_build_agent", return_value=Mock()),
        patch(f"{module}.GraspObjectOperation") as grasp,
        patch(f"{module}.EmoteTask"),
    ):
        task = DynamemTaskExecutor(Mock(spec=AbstractRobotClient), parameters, visual_servo=True, cpu_only=True)
    assert detector.called is expects_detector
    assert task.grasp_object is grasp.return_value
    assert parameters["encoder"] == "siglip"
    if not expects_detector:
        assert task.semantic_sensor is None
        assert parameters["detection"] == {}


def executor():
    task = object.__new__(DynamemTaskExecutor)
    store = QueryCandidates()
    record = store.propose("mug", 1, 0, [1, 1, 1])
    store.ground(record.handle, instance_id=7, observation_revision=2)
    target = GroundedTarget(record.handle, 7, 2, np.ones((16, 3)))
    task.agent = SimpleNamespace(
        query_driven_memory=True,
        current_object=None,
        current_receptacle=None,
        query_candidates=store,
        voxel_map=SimpleNamespace(observations=[0, 1]),
        prepare_query_target=Mock(return_value=target),
        graph_memory=Mock(),
    )
    task.agent.update = Mock(side_effect=lambda **kw: task.agent.voxel_map.observations.append(2))
    task.visual_servo = True
    task.grasp_object = Mock(return_value=True)
    return task, target


@pytest.mark.parametrize("point,expected", [(None, False), (np.ones(3), True)])
def test_find_reports_task_outcome_without_quitting(point, expected, monkeypatch):
    monkeypatch.delenv("EMET_BASE_ROTATE_ONLY", raising=False)
    task = object.__new__(DynamemTaskExecutor)
    task._find = Mock(return_value=point)
    assert task([("find", "cup")]) is True
    assert task._last_exec_ok is expected


@pytest.mark.parametrize("status,expected", [(False, False), (None, False), (True, True)])
def test_navigation_only_returns_confirmed_target(status, expected):
    from emet.controller.dynamem.navigation import navigate

    point = np.ones(3)
    agent = SimpleNamespace(maybe_save_rerun_recording=Mock(), execute_action=Mock(return_value=(status, point)))
    result = navigate(agent, "cup", max_step=1)
    assert (result is point) is expected


def test_pick_handoff_passes_geometry_and_revokes_all_aliases():
    task, target = executor()
    alias = task.agent.query_candidates.propose("cup", 1, 0, [1, 1, 1])
    task.agent.query_candidates.ground(alias.handle, instance_id=7, observation_revision=2)
    assert task._pickup("mug")
    assert task.grasp_object.call_args.kwargs["grounded_target"] is target
    assert task._held_query_instance.global_id == 7
    assert task.agent.current_object is None
    assert task.last_query_manipulation["observed_after_action"]
    for record in task.agent.query_candidates.records.values():
        with pytest.raises(ValueError, match="fresh"):
            record.require_grounding(2)


def test_query_mode_never_falls_back_to_oracle_or_text_only():
    task, _ = executor()
    task.visual_servo = False
    with patch("emet.simulation.sim_manipulation.sim_teleport_pickup") as oracle:
        assert not task._pickup("mug")
        oracle.assert_not_called()
    task.grasp_object.assert_not_called()


@pytest.mark.parametrize("failure", ["grounding", "operation", "capture"])
def test_failures_are_not_success(failure):
    task, target = executor()
    if failure == "grounding":
        task.agent.prepare_query_target.side_effect = ValueError("ambiguous")
    elif failure == "operation":
        task.grasp_object.return_value = False
    else:
        task.agent.update.side_effect = None
    assert not task._pickup("mug")
    if failure == "grounding":
        task.grasp_object.assert_not_called()
    else:
        with pytest.raises(ValueError, match="fresh"):
            task.agent.query_candidates.records[target.candidate_id].require_grounding(2)


def test_place_consumes_fresh_receptacle_points_and_observes_after():
    task, target = executor()
    task._held_query_instance = SimpleNamespace(global_id=99)
    with patch("emet.controller.operations.place_object.PlaceObjectOperation") as operation:
        operation.return_value.return_value = True

        def place():
            assert task.agent.current_object.global_id == 99
            assert np.allclose(task.agent.current_receptacle.point_cloud.numpy(), target.points)
            return True

        operation.return_value.side_effect = place
        assert task._place("table", None)
    assert task._held_query_instance is None
    assert task.last_query_manipulation["observed_after_action"]


def test_tracking_rejects_missing_or_ambiguous_geometry():
    _, target = executor()
    xyz = np.ones((8, 8, 3))
    masks = np.zeros((8, 8), dtype=int)
    classes = np.ones((8, 8), dtype=bool)
    assert target.select_mask(masks, classes, xyz).all()
    masks[4:] = 1
    with pytest.raises(ValueError, match="ambiguous"):
        target.select_mask(masks, classes, xyz)
    xyz[4:] = 10
    selected = target.select_mask(masks, classes, xyz)
    assert selected[:4].all() and not selected[4:].any()
    with pytest.raises(ValueError, match="depth"):
        target.select_mask(masks, classes, None)


def test_visual_servo_operation_uses_geometry_not_centered_distractor():
    from emet.controller.operations.grasp_object import GraspObjectOperation

    _, target = executor()
    operation = object.__new__(GraspObjectOperation)
    operation.grounded_target = target
    operation.get_class_mask = Mock(return_value=np.ones((8, 8), dtype=bool))
    masks = np.zeros((8, 8), dtype=int)
    masks[4:] = 1
    world = np.ones((8, 8, 3))
    world[4:] = 10
    servo = SimpleNamespace(instance=masks, get_ee_xyz_in_world_frame=lambda: world)
    chosen = operation.get_target_mask(servo, center=(5, 5))
    assert chosen[:4].all() and not chosen[4:].any()
    world[:] = 10
    with pytest.raises(ValueError, match="absent"):
        operation.get_target_mask(servo, center=(5, 5))


def test_vlm_wrist_tracking_requires_shared_semantics_and_world_association():
    from emet.controller.operations.grasp_object import GraspObjectOperation

    operation = object.__new__(GraspObjectOperation)
    operation.grounded_target = GroundedTarget(1, 7, 2, np.ones((30, 3)), "vlm_selected_depth_surface")
    operation.target_object = "mug"
    masks = np.full((10, 10), -1, dtype=int)
    masks[:5] = 0
    operation.agent = SimpleNamespace(
        ground_vlm_frame=Mock(return_value=(SimpleNamespace(instance=masks), [], [0], {"valid": True}))
    )
    operation.get_class_mask = Mock(side_effect=AssertionError("detector must not gate VLM surface"))
    world = np.ones((10, 10, 3))
    world[5:] = 10
    servo = SimpleNamespace(
        ee_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        ee_depth=np.ones((10, 10)),
        get_ee_xyz_in_world_frame=lambda: world,
    )
    selected = operation.get_target_mask(servo, center=(8, 8))
    assert selected[:5].all() and not selected[5:].any()
    assert operation.agent.ground_vlm_frame.call_args.kwargs == {"min_depth": 0.0}
    world[:] = 10
    with pytest.raises(ValueError, match="absent"):
        operation.get_target_mask(servo, center=(8, 8))
    world[:] = 1
    operation.agent.ground_vlm_frame.return_value = (SimpleNamespace(instance=masks), [], [], {"valid": False})
    with pytest.raises(ValueError, match="identity absent"):
        operation.get_target_mask(servo, center=(8, 8))


def test_placement_sampling_handles_zero_horizontal_offset():
    import torch

    from emet.controller.operations.place_object import PlaceObjectOperation
    from emet.mapping.instance import Instance

    operation = object.__new__(PlaceObjectOperation)
    instance = Instance(point_cloud=torch.ones((16, 3)))
    operation.agent = SimpleNamespace(current_receptacle=instance)
    operation.verbose = False
    assert np.isfinite(operation.sample_placement_position(np.array([0, 0, 0]))).all()
