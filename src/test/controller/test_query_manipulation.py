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
