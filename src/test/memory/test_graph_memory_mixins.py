# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU-only guards for the GraphEQAMemory facade (size + public methods)."""

import importlib
import sys
from pathlib import Path

import numpy as np

from emet.memory.graph_eqa import graph_memory
from emet.memory.graph_eqa.graph_memory import (
    SIGLIP_PRESENT_THRESHOLD,
    GraphEQAMemory,
    GraphNode,
    GraphStore,
    NavHypothesis,
    _near,
    replace,
)
from emet.memory.graph_eqa.store import GraphStore as StoreType

# Keep the facade ingestible: loading a 7k-line module crashes the agent host.
_FACADE_MAX_LINES = 250


def test_facade_still_reexports_split_symbols() -> None:
    assert GraphNode.__name__ == "GraphNode"
    assert NavHypothesis.__name__ == "NavHypothesis"
    assert GraphStore is StoreType
    assert SIGLIP_PRESENT_THRESHOLD > 0
    assert callable(_near)
    assert replace is not None
    assert hasattr(GraphEQAMemory, "add_observation")
    assert hasattr(GraphEQAMemory, "query_answer")
    assert hasattr(GraphEQAMemory, "hypothesize_nav_targets")
    assert hasattr(GraphEQAMemory, "stamp_vlm_room_at_robot")
    assert hasattr(GraphEQAMemory, "sync_frontier_nodes")
    assert hasattr(GraphEQAMemory, "maintain")
    assert hasattr(GraphEQAMemory, "record_nav_attempt")


def test_graph_memory_facade_stays_small() -> None:
    n = sum(1 for _ in Path(graph_memory.__file__).open(encoding="utf-8"))
    assert n <= _FACADE_MAX_LINES, f"graph_memory.py is {n} lines; keep it a thin facade"


def test_memory_init_owns_graph_store() -> None:
    mem = GraphEQAMemory(defer_llm_clients=True)
    assert isinstance(mem.store, GraphStore)
    assert mem._nodes is mem.store.nodes
    mem._nodes.append(GraphNode(node_id=1, labels=["cup"], xyz=np.zeros(3), obs_id=1))
    assert mem.store.nodes[0].labels == ["cup"]


def test_compat_shims_share_add_observation() -> None:
    from emet.memory.graph_eqa.graph_mutate import add_observation as old
    from emet.memory.graph_eqa.ingest.graph_mutate import add_observation as new

    assert old is new


_SHIM_MODULES = (
    ("emet.memory.graph_eqa.graph_mutate", "emet.memory.graph_eqa.ingest.graph_mutate"),
    ("emet.memory.graph_eqa.graph_observation_pipeline", "emet.memory.graph_eqa.ingest.graph_observation_pipeline"),
    ("emet.memory.graph_eqa.instance_items", "emet.memory.graph_eqa.ingest.instance_items"),
    ("emet.memory.graph_eqa.instance_observations", "emet.memory.graph_eqa.ingest.instance_observations"),
    ("emet.memory.graph_eqa.dynamem_graph_hooks", "emet.memory.graph_eqa.ingest.dynamem_graph_hooks"),
    ("emet.memory.graph_eqa.sensor_graph_builder", "emet.memory.graph_eqa.ingest.sensor_graph_builder"),
    ("emet.memory.graph_eqa.lazy_graph_commit", "emet.memory.graph_eqa.ingest.lazy_graph_commit"),
    ("emet.memory.graph_eqa.graph_rooms", "emet.memory.graph_eqa.spatial.graph_rooms"),
    ("emet.memory.graph_eqa.room_clusters", "emet.memory.graph_eqa.spatial.room_clusters"),
    ("emet.memory.graph_eqa.room_labels", "emet.memory.graph_eqa.spatial.room_labels"),
    ("emet.memory.graph_eqa.frontier_nodes", "emet.memory.graph_eqa.spatial.frontier_nodes"),
    ("emet.memory.graph_eqa.frontier_regions", "emet.memory.graph_eqa.spatial.frontier_regions"),
    ("emet.memory.graph_eqa.spatial_rag", "emet.memory.graph_eqa.spatial.spatial_rag"),
    ("emet.memory.graph_eqa.place_approaches", "emet.memory.graph_eqa.spatial.place_approaches"),
    ("emet.memory.graph_eqa.graph_answer", "emet.memory.graph_eqa.eqa.graph_answer"),
    ("emet.memory.graph_eqa.graph_prompt", "emet.memory.graph_eqa.eqa.graph_prompt"),
    ("emet.memory.graph_eqa.graph_hypotheses", "emet.memory.graph_eqa.eqa.graph_hypotheses"),
    ("emet.memory.graph_eqa.graph_nav", "emet.memory.graph_eqa.eqa.graph_nav"),
    ("emet.memory.graph_eqa.graph_eqa_obs", "emet.memory.graph_eqa.eqa.graph_eqa_obs"),
    ("emet.memory.graph_eqa.eqa_views", "emet.memory.graph_eqa.eqa.eqa_views"),
    ("emet.memory.graph_eqa.query_images", "emet.memory.graph_eqa.eqa.query_images"),
    ("emet.memory.graph_eqa.human_answer", "emet.memory.graph_eqa.eqa.human_answer"),
    ("emet.memory.graph_eqa.mcq_debias", "emet.memory.graph_eqa.eqa.mcq_debias"),
    ("emet.memory.graph_eqa.graph_eqa_siglip", "emet.memory.graph_eqa.eqa.graph_eqa_siglip"),
    ("emet.memory.graph_eqa.agentic_init", "emet.memory.graph_eqa.agentic.executor_init"),
    ("emet.memory.graph_eqa.agentic_run", "emet.memory.graph_eqa.agentic.run"),
    ("emet.memory.graph_eqa.agentic_router", "emet.memory.graph_eqa.agentic.router"),
    ("emet.memory.graph_eqa.agentic_answer", "emet.memory.graph_eqa.agentic.answer"),
    ("emet.memory.graph_eqa.agentic_verify", "emet.memory.graph_eqa.agentic.verify"),
    ("emet.memory.graph_eqa.agentic_assess", "emet.memory.graph_eqa.agentic.assess"),
    ("emet.memory.graph_eqa.agentic_capture", "emet.memory.graph_eqa.agentic.capture"),
    ("emet.memory.graph_eqa.agentic_investigate", "emet.memory.graph_eqa.agentic.investigate"),
    ("emet.memory.graph_eqa.agentic_place", "emet.memory.graph_eqa.agentic.place"),
    ("emet.memory.graph_eqa.agentic_explore", "emet.memory.graph_eqa.agentic.explore"),
    ("emet.memory.graph_eqa.agentic_action", "emet.memory.graph_eqa.agentic.action"),
    ("emet.memory.graph_eqa.agentic_tools", "emet.memory.graph_eqa.agentic.tools"),
    ("emet.memory.graph_eqa.agentic_config", "emet.memory.graph_eqa.agentic.config"),
    ("emet.memory.graph_eqa.agentic_policy", "emet.memory.graph_eqa.agentic.policy"),
    ("emet.memory.graph_eqa.agentic_state", "emet.memory.graph_eqa.agentic.state"),
    ("emet.memory.graph_eqa.agentic_types", "emet.memory.graph_eqa.agentic.types"),
    ("emet.memory.graph_eqa.agentic_session", "emet.memory.graph_eqa.agentic.session"),
    ("emet.memory.graph_eqa.action_history", "emet.memory.graph_eqa.agentic.action_history"),
    ("emet.memory.graph_eqa.attempt_metrics", "emet.memory.graph_eqa.eval.attempt_metrics"),
    ("emet.memory.graph_eqa.calibration_export", "emet.memory.graph_eqa.eval.calibration_export"),
    ("emet.memory.graph_eqa.dynagraph_eval", "emet.memory.graph_eqa.eval.dynagraph_eval"),
    ("emet.memory.graph_eqa.mujoco_align", "emet.memory.graph_eqa.eval.mujoco_align"),
    ("emet.memory.graph_eqa.nav_benchmark", "emet.memory.graph_eqa.eval.nav_benchmark"),
    ("emet.memory.graph_eqa.question_bank", "emet.memory.graph_eqa.eval.question_bank"),
    ("emet.memory.graph_eqa.sim_ground_truth_graph", "emet.memory.graph_eqa.eval.sim_ground_truth_graph"),
    ("emet.memory.graph_eqa.graph_object_fusion.setup", "emet.memory.graph_eqa.graph_object_fusion.attach"),
)


def test_compat_shims_alias_implementation_modules() -> None:
    for old_name, new_name in _SHIM_MODULES:
        old = importlib.import_module(old_name)
        new = importlib.import_module(new_name)
        assert sys.modules[old_name] is new, old_name
        assert old is new, old_name
