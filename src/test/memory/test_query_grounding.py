# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Exercise the lazy controller's real promotion boundary without model downloads."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from emet.controller.controller_lazy_graph import LazyGraphController
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory


def controller():
    agent = object.__new__(LazyGraphController)
    agent.parameters = {"graph_object_fusion": {"enabled": True, "instance_min_mask_points": 10}}
    agent.graph_memory = GraphEQAMemory(parameters={}, defer_llm_clients=True)
    frame = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8)),
        full_world_xyz=np.ones((8, 8, 3)),
        instance=None,
    )
    agent.voxel_map = SimpleNamespace(observations=[frame, frame], min_depth=0.1, max_depth=5.0)
    agent.detection_model = Mock(class_list=["mug"])
    agent.detection_model.predict.return_value = (
        None,
        np.zeros((8, 8), dtype=int),
        {"instance_classes": np.array([0]), "instance_scores": np.array([0.9])},
    )
    return agent


def test_retrieval_is_not_an_instance_and_fresh_mask_promotes():
    agent = controller()
    candidate = agent.propose_query_candidate("mug", [9, 9, 9], {"source_obs_id": 1})
    assert not agent.graph_memory.get_nodes()
    result = agent.ground_query_candidate(candidate.handle, after_observation=1)
    assert result["ok"], result
    node = next(n for n in agent.graph_memory.get_nodes() if n.node_id == result["instance_id"])
    assert np.allclose(node.xyz, [1, 1, 1])  # mask geometry, never retrieval anchor
    assert candidate.require_grounding(2) == node.node_id
    agent.detection_model.predict.assert_called_once()
    assert not agent.ground_query_candidate(candidate.handle, after_observation=2)["ok"]
    with pytest.raises(ValueError, match="fresh"):
        candidate.require_grounding(2)


@pytest.mark.parametrize("failure", ["depth", "absent", "ambiguous", "disabled", "attribute"])
def test_failed_admission_never_creates_instance(failure):
    agent = controller()
    candidate = agent.propose_query_candidate("mug", [9, 9, 9], {"source_obs_id": 1})
    if failure == "depth":
        agent.voxel_map.observations[-1].depth[:] = np.nan
    elif failure == "absent":
        agent.detection_model.class_list = ["chair"]
    elif failure == "ambiguous":
        masks = np.zeros((8, 8), dtype=int)
        masks[4:] = 1
        agent.detection_model.predict.return_value = (
            None,
            masks,
            {"instance_classes": np.array([0, 0]), "instance_scores": np.array([0.9, 0.9])},
        )
    elif failure == "attribute":
        candidate.query = "red mug"
    else:
        agent.parameters["graph_object_fusion"]["use_instance_nodes"] = False
    assert not agent.ground_query_candidate(candidate.handle, after_observation=1)["ok"]
    assert not agent.graph_memory.get_nodes()
    with pytest.raises(ValueError, match="fresh"):
        candidate.require_grounding(2)


def test_graph_checkpoint_preserves_candidates_but_revokes_geometry(tmp_path):
    from emet.memory.adapters import GraphEQABackend

    agent = controller()
    candidate = agent.propose_query_candidate("mug", [9, 9, 9], {"source_obs_id": 1})
    assert agent.ground_query_candidate(candidate.handle, after_observation=1)["ok"]
    GraphEQABackend(agent.graph_memory).save(str(tmp_path / "checkpoint"))
    restored = controller()
    GraphEQABackend(restored.graph_memory).load(str(tmp_path / "checkpoint"))
    record = restored.query_candidates.records[candidate.handle]
    assert record.source_obs_id == 1
    assert record.instance_id == candidate.instance_id
    with pytest.raises(ValueError, match="fresh"):
        record.require_grounding(2)


@pytest.mark.parametrize("question", ["Where is the mug?", "What color is the mug? A) red B) blue"])
def test_shared_executor_reuses_provenance_handles(question):
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    grounding_agent = controller()
    agent = MagicMock()
    agent.parameters = {}
    agent.query_driven_memory = True
    agent.propose_query_candidate = grounding_agent.propose_query_candidate
    ex = AgenticEQAExecutor(agent, question, router=False)
    ex._target_boost_phrases = lambda: ["mug"]
    ex._siglip_phrase = lambda: "mug"
    with patch(
        "emet.memory.graph_eqa.agentic.capture.localize_text_xyz",
        return_value=(np.array([9, 9, 9]), {"source_obs_id": 1, "max_cosine": 0.4}),
    ):
        first = ex._voxel_localize_hypotheses()
        second = ex._voxel_localize_hypotheses()
    assert first[0].obs_id == second[0].obs_id
    assert first[0].obs_id in grounding_agent.query_candidates.records
    assert not grounding_agent.graph_memory.get_nodes()
