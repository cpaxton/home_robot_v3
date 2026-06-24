# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Lifelong checkpoint resume: GraphEQABackend save/load must round-trip staleness and
# fusion state (last_seen, support_count, is_viewpoint, extent_half, bounds_3d) and the
# controller step counter (manifest final_step) so maintain() keeps reloaded nodes.

import numpy as np

from emet.memory.adapters import GraphEQABackend
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

PARAMS = {
    "dynagraph_merge_xy_m": 0.0,
    "dynagraph_staleness_horizon": 8,
}


def _make_memory() -> GraphEQAMemory:
    return GraphEQAMemory(parameters=dict(PARAMS), defer_llm_clients=True)


def _rgb() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_checkpoint_roundtrip_preserves_staleness_state(tmp_path):
    mem = _make_memory()
    mem.set_graph_timestep(40)
    mem.add_observation(
        _rgb(),
        np.array([1.0, 2.0, 0.5]),
        ["mug"],
        viewer_xyz=np.array([0.0, 0.0, 1.2]),
        extent_half=np.array([0.05, 0.05, 0.08]),
    )
    mem.add_observation(_rgb(), np.array([3.0, 1.0, 0.9]), ["apple"])
    # Give the apple node fused 3D bounds (GraphObjectFusion path).
    for i, n in enumerate(mem._nodes):
        if n.labels and n.labels[0] == "apple":
            from dataclasses import replace

            mem._nodes[i] = replace(
                n,
                support_count=3,
                bounds_3d={
                    "min": [2.9, 0.9, 0.8],
                    "max": [3.1, 1.1, 1.0],
                    "center": [3.0, 1.0, 0.9],
                    "size": [0.2, 0.2, 0.2],
                },
            )

    path = tmp_path / "ckpt"
    GraphEQABackend(mem).save(str(path), final_step=42)

    mem2 = _make_memory()
    backend2 = GraphEQABackend(mem2)
    backend2.load(str(path))

    assert backend2.loaded_final_step == 42
    assert mem2._graph_timestep == 42

    nodes = {n.labels[0]: n for n in mem2.get_nodes() if not n.is_viewpoint}
    assert set(nodes) == {"mug", "apple"}
    assert nodes["mug"].last_seen == 40
    assert nodes["apple"].last_seen == 40
    assert nodes["apple"].support_count == 3
    assert nodes["mug"].extent_half is not None
    np.testing.assert_allclose(nodes["mug"].extent_half, [0.05, 0.05, 0.08])
    assert nodes["apple"].bounds_3d is not None
    np.testing.assert_allclose(nodes["apple"].bounds_3d["min"], [2.9, 0.9, 0.8])
    np.testing.assert_allclose(nodes["apple"].bounds_3d["size"], [0.2, 0.2, 0.2])

    viewpoints = [n for n in mem2.get_nodes() if n.is_viewpoint]
    assert len(viewpoints) == 1
    assert mem2._viewpoint_by_obs_id == {int(viewpoints[0].obs_id): int(viewpoints[0].node_id)}


def test_checkpoint_roundtrip_survives_maintain(tmp_path):
    mem = _make_memory()
    mem.set_graph_timestep(40)
    mem.add_observation(_rgb(), np.array([1.0, 2.0, 0.5]), ["mug"])

    path = tmp_path / "ckpt"
    GraphEQABackend(mem).save(str(path), final_step=42)

    mem2 = _make_memory()
    GraphEQABackend(mem2).load(str(path))

    # Resume at step 45: 45 - 40 = 5 <= horizon 8, node must survive. Before
    # last_seen/final_step persistence, last_seen reset to 0 and this pruned everything.
    removed = mem2.maintain(45)
    assert removed == 0
    assert len([n for n in mem2.get_nodes() if not n.is_viewpoint]) == 1

    # Far past the horizon the node is pruned as usual.
    removed = mem2.maintain(40 + PARAMS["dynagraph_staleness_horizon"] + 1)
    assert removed == 1


def test_checkpoint_next_obs_id_continues(tmp_path):
    mem = _make_memory()
    mem.set_graph_timestep(10)
    mem.add_observation(_rgb(), np.array([1.0, 2.0, 0.5]), ["mug"])
    mem.add_observation(_rgb(), np.array([3.0, 1.0, 0.9]), ["apple"])
    next_before = mem._next_obs_id

    path = tmp_path / "ckpt"
    GraphEQABackend(mem).save(str(path), final_step=12)

    mem2 = _make_memory()
    GraphEQABackend(mem2).load(str(path))
    assert mem2._next_obs_id >= next_before
    new_obs = mem2.add_observation(_rgb(), np.array([0.0, 0.0, 0.3]), ["banana"])
    assert new_obs >= next_before
