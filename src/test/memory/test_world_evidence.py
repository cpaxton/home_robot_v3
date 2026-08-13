# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emet.memory.adapters import GraphEQABackend
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.lifelong import apply_se2_to_graph


def _memory() -> GraphEQAMemory:
    memory = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={
            "eqa": {
                "graph_evidence_mode": "shadow",
                "attempt_ledger": True,
            }
        },
    )
    memory.spatial_merge_m = 0.5
    return memory


def test_spatial_merge_keeps_one_entity_and_two_immutable_views():
    memory = _memory()
    red = np.zeros((6, 6, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((6, 6, 3), dtype=np.uint8)
    blue[..., 2] = 255

    obs_id = memory.add_observation(
        red,
        np.array([1.0, 2.0, 0.5]),
        ["mug"],
        viewer_xyz=np.array([0.0, 2.0, 0.0]),
    )
    same_obs_id = memory.add_observation(
        blue,
        np.array([1.1, 2.0, 0.5]),
        ["mug"],
        viewer_xyz=np.array([0.2, 2.0, 0.0]),
    )

    assert same_obs_id == obs_id
    assert len(memory.world_evidence.entities) == 1
    assert len(memory.world_evidence.places) == 1
    assert len(memory.world_evidence.views) == 2
    views = sorted(memory.world_evidence.views.values(), key=lambda item: item.revision)
    assert np.array_equal(views[0].rgb, red)
    assert np.array_equal(views[1].rgb, blue)
    assert np.array_equal(views[0].rgb, red)
    assert views[0].view_id != views[1].view_id


def test_dense_node_renumber_does_not_change_entity_identity():
    memory = _memory()
    memory.spatial_merge_m = 0.0
    first = memory.add_observation(np.zeros((2, 2, 3), dtype=np.uint8), [0, 0, 0], ["chair"])
    second = memory.add_observation(np.zeros((2, 2, 3), dtype=np.uint8), [4, 0, 0], ["lamp"])
    second_node = next(node for node in memory.get_nodes() if node.obs_id == second)
    entity_before = memory.world_evidence.entity_for_node(second_node.node_id)
    assert entity_before is not None

    memory._nodes = [node for node in memory._nodes if node.obs_id != first]
    memory._observations = [obs for obs in memory._observations if obs.obs_id != first]
    memory._nodes = [replace(node, node_id=i) for i, node in enumerate(memory._nodes, start=1)]
    memory._reindex_world_entities()

    entity_after = memory.world_evidence.entity_for_node(1)
    assert entity_after is not None
    assert entity_after.entity_id == entity_before.entity_id


def test_world_evidence_checkpoint_round_trip_preserves_ids_and_runtime(tmp_path):
    memory = _memory()
    obs_id = memory.add_observation(
        np.full((4, 4, 3), 17, dtype=np.uint8),
        [1.0, 2.0, 0.5],
        ["clock"],
        viewer_xyz=[0.0, 2.0, 0.0],
    )
    memory.record_attempt(
        action_kind="investigate",
        outcome="ok",
        status_code="ok",
        obs_id=obs_id,
        source="eqa",
    )
    memory.record_room_event(room="kitchen", kind="stamp", obs_id=obs_id, step=3)
    memory._next_obs_id = 19
    entity_ids = set(memory.world_evidence.entities)
    view_ids = set(memory.world_evidence.views)

    path = tmp_path / "checkpoint"
    GraphEQABackend(memory).save(str(path), final_step=7)
    restored = _memory()
    backend = GraphEQABackend(restored)
    backend.load(str(path))

    assert set(restored.world_evidence.entities) == entity_ids
    assert set(restored.world_evidence.views) == view_ids
    assert restored._next_obs_id == 19
    assert restored.obs_revision(obs_id) == 1
    assert restored.get_attempt_records()[0].view_id
    assert restored.get_room_events()[0]["room"] == "kitchen"
    assert np.array_equal(restored.get_observations()[0].rgb, np.full((4, 4, 3), 17, dtype=np.uint8))


def test_se2_transforms_entities_places_and_views():
    memory = _memory()
    obs_id = memory.add_observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        [1.0, 2.0, 0.5],
        ["plant"],
        viewer_xyz=[0.0, 2.0, 0.0],
    )
    transform = np.eye(4)
    transform[0, 3] = 3.0

    apply_se2_to_graph(memory, transform)

    entity = next(iter(memory.world_evidence.entities.values()))
    place = next(iter(memory.world_evidence.places.values()))
    view = memory.world_evidence.view_for_obs(obs_id)
    assert entity.xyz[0] == 4.0
    assert place.anchor_xyz[0] == 4.0
    assert view is not None
    assert view.object_xyz[0] == 4.0
    assert view.base_pose_world is not None and view.base_pose_world[0] == 3.0
