# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Dynagraph graph lifecycle: staleness prunes disappeared object nodes."""

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory


def test_maintain_prunes_stale_object_node():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.staleness_horizon = 10
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.set_graph_timestep(1)
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["red_cylinder"])
    assert len([n for n in mem.get_nodes() if not n.is_viewpoint]) == 1

    mem.set_graph_timestep(20)
    removed = mem.maintain(20)
    assert removed == 1
    assert len([n for n in mem.get_nodes() if not n.is_viewpoint]) == 0


def test_maintain_keeps_recently_seen_node():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.staleness_horizon = 10
    mem.spatial_merge_m = 0.5
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.set_graph_timestep(5)
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["blue_cube"])
    mem.set_graph_timestep(12)
    mem.add_observation(rgb, np.array([1.02, 0.01, 0.51]), ["blue_cube"])
    mem.set_graph_timestep(14)
    removed = mem.maintain(14)
    assert removed == 0
    assert len([n for n in mem.get_nodes() if not n.is_viewpoint]) == 1


def test_disappearance_episode_simulated_steps():
    """Simulate see object -> stop observing -> prune (unit-level disappearance)."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.staleness_horizon = 5
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    for step in range(1, 4):
        mem.set_graph_timestep(step)
        mem.add_observation(rgb, np.array([0.5, 0.0, 0.8]), ["mug"])

    assert len(mem.get_nodes()) >= 1

    for step in range(4, 12):
        mem.set_graph_timestep(step)
        mem.maintain(step)

    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 0, "stale mug node should be pruned after disappearance"


def test_invalidate_nodes_near_prunes_without_waiting_horizon():
    """World-change: known relocate must drop the old-pose node immediately."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.staleness_horizon = 256
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.set_graph_timestep(10)
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["obj_main"])
    mem.add_observation(rgb, np.array([5.0, 5.0, 0.5]), ["sink"])
    assert len([n for n in mem.get_nodes() if not n.is_viewpoint]) == 2

    aged, pruned = mem.invalidate_nodes_near(
        [1.0, 0.0, 0.5],
        radius_m=0.75,
        current_step=12,
        prune=True,
    )
    assert aged >= 1
    assert pruned >= 1
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    labels = [" ".join(n.labels) for n in objs]
    assert any("sink" in lb for lb in labels)
    assert not any("obj_main" in lb for lb in labels)


def test_clear_eqa_working_memory_resets_confirmed_cache():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.last_eqa_obs_ids = [1, 2]
    mem.last_eqa_raw = "Answer: A"
    mem.last_eqa_model_confident = True
    mem.last_eqa_prompt_node_count = 9
    mem.clear_eqa_working_memory()
    assert mem.last_eqa_obs_ids == []
    assert mem.last_eqa_raw == ""
    assert mem.last_eqa_model_confident is False
    assert mem.last_eqa_prompt_node_count == 0
