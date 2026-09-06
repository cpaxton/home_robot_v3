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
    node = next(n for n in agent.graph_memory.get_nodes() if n.obs_id == result["instance_id"] and n.countable_instance)
    assert np.allclose(node.xyz, [1, 1, 1])  # mask geometry, never retrieval anchor
    assert candidate.require_grounding(2) == node.obs_id
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
        agent.detection_model.predict.return_value = (None, -np.ones((8, 8), dtype=int), {})
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
    agent.retrieve_query_candidate = Mock(return_value=(np.array([9, 9, 9]), {"source_obs_id": 1, "max_cosine": 0.4}))
    ex = AgenticEQAExecutor(agent, question, router=False)
    ex._target_boost_phrases = lambda: ["mug"]
    ex._siglip_phrase = lambda: "mug"
    with patch(
        "emet.memory.graph_eqa.agentic.capture.localize_text_xyz",
        side_effect=AssertionError("query candidates must not use the confirmed-localization gate"),
    ):
        first = ex._voxel_localize_hypotheses()
        second = ex._voxel_localize_hypotheses()
    assert first[0].obs_id == second[0].obs_id
    assert first[0].obs_id in grounding_agent.query_candidates.records
    assert not grounding_agent.graph_memory.get_nodes()


def test_lazy_grounding_initializes_detector_only_on_demand():
    agent = controller()
    detector = agent.detection_model
    agent.detection_model = None
    agent.device = "cpu"
    with patch("emet.perception.detection.yoloe.get_shared_yoloe_perception", return_value=detector) as factory:
        record = agent.propose_query_candidate("mug", [1, 1, 1], {"source_obs_id": 1})
        factory.assert_not_called()
        assert agent.ground_query_candidate(record.handle, after_observation=1)["ok"]
        factory.assert_called_once()


def test_fresh_grounding_ignores_background_vocabulary():
    agent = controller()
    agent.detection_model.class_list = ["chair"]
    agent.voxel_map.observations[-1].instance = np.zeros((8, 8), dtype=int)
    record = agent.propose_query_candidate("mug", [9, 9, 9], {"source_obs_id": 1})
    assert agent.ground_query_candidate(record.handle, after_observation=1)["ok"]
    assert agent.detection_model.predict.call_args.kwargs["vocabulary"] == ["mug"]
    assert agent.detection_model.class_list == ["chair"]


def test_weak_recovery_cached_but_never_promotes_and_new_frame_rechecks():
    agent = controller()
    agent.voxel_map.retrieve_text_candidate = Mock(
        side_effect=lambda *args, **kwargs: (
            np.array([9, 9, 9]) if "minimum_similarity" in kwargs else None,
            {"source_obs_id": 1, "max_cosine": 0.1},
        )
    )
    for _ in range(2):
        point, stats = agent.retrieve_query_candidate("mug")
        assert np.allclose(point, [1, 1, 1])
        assert stats["recovery_source"] == "query_detector" and not stats["yoloe_hit"]
    agent.detection_model.predict.assert_called_once()
    assert not agent.graph_memory.get_nodes()
    assert not agent.query_candidates.records
    agent.voxel_map.observations[0] = SimpleNamespace(**vars(agent.voxel_map.observations[0]))
    agent.retrieve_query_candidate("mug")
    assert agent.detection_model.predict.call_count == 2


def test_detector_query_vocabulary_is_per_call():
    from emet.perception.detection.yoloe import YoloEPerception

    detector = object.__new__(YoloEPerception)
    detector.class_list = ["chair"]
    detector.verbose = False
    detector.confidence = 0.05
    detector.model = Mock(return_value=[SimpleNamespace(boxes=None)])
    with patch("emet.perception.detection.yoloe._text_pe_for_classes", return_value=None):
        detector.predict(np.zeros((8, 8, 3), dtype=np.uint8), vocabulary=["red cylinder"])
        assert detector.model.set_classes.call_args.args[0] == ["red cylinder"]
        assert detector.class_list == ["chair"]
        detector.predict(np.zeros((8, 8, 3), dtype=np.uint8))
        assert detector.model.set_classes.call_args.args[0] == ["chair"]
        with pytest.raises(ValueError, match="empty"):
            detector.predict(np.zeros((8, 8, 3), dtype=np.uint8), vocabulary=[])


def test_weak_voxel_hit_is_search_evidence_not_localization():
    import torch

    from emet.mapping.voxel.dynamem_localize import DynamemVoxelLocalizeMixin

    vm = DynamemVoxelLocalizeMixin()
    vm.observations = [object()]
    vm.semantic_memory = SimpleNamespace(
        get_pointcloud=lambda: (torch.ones((1, 3)), None, None, None),
        _obs_counts=torch.tensor([1]),
    )
    vm.find_alignment_over_model = lambda text: torch.tensor([0.16])
    point, stats = vm.retrieve_text_candidate("mug")
    assert np.allclose(point, [1, 1, 1])
    assert stats["source_obs_id"] == 1 and not stats["yoloe_hit"]
    assert not hasattr(vm, "_last_localize_stats")
    assert vm.retrieve_text_candidate("mug", excluded_obs_ids={1})[0] is None
    vm.find_alignment_over_model = lambda text: torch.tensor([0.10])
    assert vm.retrieve_text_candidate("mug")[0] is None


def test_failed_grounding_excludes_only_that_query_source():
    agent = controller()
    candidate = agent.propose_query_candidate("mug", [1, 1, 1], {"source_obs_id": 1})
    agent.detection_model.predict.return_value = (None, -np.ones((8, 8), dtype=int), {})
    assert not agent.ground_query_candidate(candidate.handle, after_observation=1)["ok"]
    agent.voxel_map.retrieve_text_candidate = Mock(return_value=(None, {}))
    agent.retrieve_query_candidate("mug")
    agent.voxel_map.retrieve_text_candidate.assert_called_with("mug", minimum_similarity=0.0, excluded_obs_ids={1})
    agent.retrieve_query_candidate("chair")
    agent.voxel_map.retrieve_text_candidate.assert_called_with("chair", minimum_similarity=0.0, excluded_obs_ids=set())


def test_rejected_candidate_cannot_be_approached_again():
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = controller()
    agent.parameters = {}
    agent.query_driven_memory = True
    agent.navigate_to_target_pose = Mock()
    agent.robot = Mock()
    record = agent.propose_query_candidate("mug", [1, 1, 1], {"source_obs_id": 1})
    agent.query_candidates.reject(record.handle, observation_revision=2, reason="absent")
    ex = AgenticEQAExecutor(agent, "Where is the mug?", router=False)
    ex._hypotheses = [NavHypothesis(phrase="mug", obs_id=record.handle, xyz=np.ones(3), score=1, source="voxel")]
    assert not ex._investigate_hypotheses()
    assert ex._tool_investigate(record.handle)["status"] == "CANDIDATE_REJECTED"
    agent.navigate_to_target_pose.assert_not_called()


def test_retiring_object_does_not_reassign_surviving_candidate_identity():
    agent = controller()
    gm = agent.graph_memory
    first = gm.add_observation(np.zeros((8, 8, 3), dtype=np.uint8), np.array([1, 1, 1]), ["mug"])
    second = gm.add_observation(np.zeros((8, 8, 3), dtype=np.uint8), np.array([5, 5, 1]), ["table"])
    mug = agent.query_candidates.propose("mug", 1, 0, [1, 1, 1])
    table = agent.query_candidates.propose("table", 2, 0, [5, 5, 1])
    agent.query_candidates.ground(mug.handle, instance_id=first, observation_revision=2)
    agent.query_candidates.ground(table.handle, instance_id=second, observation_revision=2)
    assert gm.retire_object_observations({first}) == 1
    with pytest.raises(ValueError, match="fresh"):
        mug.require_grounding(2)
    assert table.require_grounding(2) == second
    assert any(n.obs_id == second for n in gm.get_nodes())
