# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Tests for GraphEQA memory (graph-based EQA). No code copied from closed-source repos.

import tempfile
from pathlib import Path

import numpy as np

from emet.memory.adapters import GraphEQABackend
from emet.memory.graph_eqa import GraphEQAMemory, labels_are_semantic_graph_hypothesis
from emet.memory.graph_eqa.graph_memory import _near, _on_floor


def test_graph_memory_add_observation():
    """Adding observations creates nodes and updates edges."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "reasoning: r\nanswer: yes\nconfidence: true\naction:\nconfidence_reasoning: ok",
        image_description_client=lambda x: "table, cup",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    xyz1 = np.array([0.0, 0.0, 0.5])
    id1 = mem.add_observation(rgb, xyz1, ["table"])
    assert id1 == 1
    assert len(mem.get_nodes()) == 1
    assert len(mem.get_observations()) == 1

    id2 = mem.add_observation(rgb, np.array([0.3, 0.0, 0.5]), ["cup"])
    assert id2 == 2
    assert len(mem.get_nodes()) == 2
    edges = mem.get_edges()
    # near(table, cup) should exist
    assert any((1, 2, "near") == e or (2, 1, "near") == e for e in edges)


def test_graph_memory_to_string():
    """Scene graph serializes to a string for prompts."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.0]), ["floor", "carpet"])
    mem.add_observation(rgb, np.array([0.5, 0.0, 0.8]), ["table"])
    s = mem.to_string()
    assert "SCENE_GRAPH" in s
    assert "floor" in s or "carpet" in s
    assert "table" in s
    assert "Node 1" in s
    assert "Node 2" in s


def test_parse_answer():
    """parse_answer extracts reasoning, answer, confidence, action, confidence_reasoning."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    raw = "reasoning: I see a table.\nanswer: Yes\nconfidence: True\naction: \nconfidence_reasoning: I am sure."
    r, a, c, act, cr = mem.parse_answer(raw)
    assert "table" in r
    assert a.strip().lower() == "yes"
    assert c is True
    assert "sure" in cr.lower()


def test_parse_answer_not_confident():
    """When confidence is False, action can be an image id."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    raw = (
        "reasoning: Need to look more.\n"
        "answer: Unknown\n"
        "confidence: FALSE\n"
        "action: 3\n"
        "confidence_reasoning: Not seen yet."
    )
    r, a, c, act, cr = mem.parse_answer(raw)
    assert c is False
    assert act.strip() == "3"


def test_near_heuristic():
    """_near returns True when 2D distance <= max_dist."""
    assert _near(np.array([0, 0, 0]), np.array([0.5, 0, 0]), max_dist=1.0) is True
    assert _near(np.array([0, 0, 0]), np.array([2, 0, 0]), max_dist=1.0) is False


def test_add_observation_stores_bbox_xyxy_on_node():
    """Instance graph nodes keep a pixel crop for Dynagraph Rerun thumbnails."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    mem.add_observation(
        rgb,
        np.array([1.0, 2.0, 0.5]),
        ["mug"],
        bbox_xyxy=(10, 20, 40, 55),
    )
    node = mem.get_nodes()[0]
    assert node.bbox_xyxy == (10, 20, 40, 55)


def test_seen_from_edge_links_object_to_viewpoint_node():
    """``seen_from`` targets a viewpoint graph node at ``viewer_xyz``."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    viewer = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    mem.add_observation(
        rgb,
        np.array([3.0, 4.0, 0.9]),
        ["mug"],
        viewer_xyz=viewer,
    )
    nodes = mem.get_nodes()
    objects = [n for n in nodes if not n.is_viewpoint]
    viewpoints = [n for n in nodes if n.is_viewpoint]
    assert len(objects) == 1
    assert len(viewpoints) == 1
    np.testing.assert_allclose(viewpoints[0].xyz, viewer)
    edges = mem.get_edges()
    assert (objects[0].node_id, viewpoints[0].node_id, "seen_from") in edges
    obs = mem.get_observations()[0]
    assert obs.viewer_xyz is not None
    np.testing.assert_allclose(obs.viewer_xyz, viewer)
    s = mem.to_string()
    assert "seen_from" in s
    assert "View " in s


def test_on_floor_heuristic():
    """_on_floor returns True when z <= threshold."""
    assert _on_floor(np.array([0, 0, 0.02])) is True
    assert _on_floor(np.array([0, 0, 0.2])) is False


def test_query_answer_returns_tuple_with_mock_client():
    """query_answer returns (reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images)."""

    def mock_eqa(commands):
        return "reasoning: I see a table.\nanswer: Yes\nconfidence: true\naction: \nconfidence_reasoning: Sure."

    mem = GraphEQAMemory(
        eqa_client=mock_eqa,
        image_description_client=lambda x: "table",
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.5]),
        ["table"],
    )
    out = mem.query_answer("Is there a table?", None, None)
    assert len(out) == 6
    reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images = out
    assert isinstance(reasoning, str)
    assert isinstance(answer, str)
    assert isinstance(confidence, bool)
    assert isinstance(confidence_reasoning, str)
    assert target_point is None  # confident, so no exploration
    assert isinstance(relevant_images, list)


def test_color_question_answer_contains_red_and_blue():
    """
    Default MuJoCo scene has a red cylinder and blue cube; GraphEQA answer should name both colors.

    Uses mocks (no GPU LLM). Sim coverage: test_graph_eqa_color_question_default_mujoco_scene.
    """

    def mock_eqa(commands):
        return (
            "reasoning: the scene graph lists a red cylinder and a blue cube on the table.\n"
            "answer: red and blue (red cylinder, blue cube).\n"
            "confidence: true\n"
            "action: \n"
            "confidence_reasoning: both objects are in the graph labels.\n"
        )

    mem = GraphEQAMemory(
        eqa_client=mock_eqa,
        image_description_client=lambda cmd: "red cylinder, blue cube",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    # Same layout as default scene: object2 / object1 positions (see test_red_cylinder_in_sim)
    mem.add_observation(rgb, np.array([0.08, -0.55, 0.6]), ["red cylinder"])
    mem.add_observation(rgb, np.array([-0.02, -0.55, 0.6]), ["blue cube"])
    graph_str = mem.to_string().lower()
    assert "red" in graph_str
    assert "blue" in graph_str

    _reasoning, answer, confidence, _cr, _tp, _imgs = mem.query_answer("Which color objects can you see?", None, None)
    assert confidence is True
    al = answer.lower()
    assert "red" in al, f"expected 'red' in answer, got: {answer!r}"
    assert "blue" in al, f"expected 'blue' in answer, got: {answer!r}"


def test_to_tree_string():
    """to_tree_string formats the scene graph as an indented tree with Floor and relations."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.0]), ["carpet"])
    mem.add_observation(rgb, np.array([0.5, 0.0, 0.8]), ["table"])
    mem.add_observation(rgb, np.array([0.5, 0.0, 0.85]), ["cup"], description="A red cup on the table")
    tree = mem.to_tree_string()
    assert "Scene (3D spatial graph)" in tree
    assert "Floor" in tree
    assert "carpet" in tree
    assert "table" in tree
    assert "cup" in tree
    assert "0.50" in tree
    assert "A red cup" in tree or "red cup" in tree


def test_print_memory():
    """print_memory returns the same tree as to_tree_string."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.1]),
        ["sofa"],
    )
    out = mem.print_memory()
    assert "Scene" in out
    assert "sofa" in out


def test_graph_eqa_backend_print_memory():
    """GraphEQABackend.print_memory delegates to graph and returns tree text."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.0]),
        ["lamp"],
    )
    backend = GraphEQABackend(mem)
    text = backend.print_memory()
    assert "Scene" in text
    assert "lamp" in text


def test_graph_eqa_save_load_roundtrip():
    """Save and load graph memory restores nodes, edges, observations (xyz and labels)."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([1.0, 2.0, 0.5]), ["chair"], description="A wooden chair")
    mem.add_observation(rgb, np.array([1.1, 2.1, 0.5]), ["desk"])
    backend = GraphEQABackend(mem)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph_mem"
        backend.save(str(path))
        assert (path / "manifest.json").exists()
        assert (path / "graph.json").exists()
        mem2 = GraphEQAMemory(eqa_client=lambda x: "", image_description_client=lambda x: "")
        backend2 = GraphEQABackend(mem2)
        backend2.load(str(path))
        nodes = mem2.get_nodes()
        obs = mem2.get_observations()
        assert len(nodes) == 2
        assert len(obs) == 2
        n1 = next(n for n in nodes if n.node_id == 1)
        assert "chair" in n1.labels
        assert n1.description == "A wooden chair"
        o1 = next(o for o in obs if o.obs_id == 1)
        assert list(o1.xyz) == [1.0, 2.0, 0.5]
        assert o1.labels == ["chair"]
        assert o1.description == "A wooden chair"


def test_labels_are_semantic_graph_hypothesis():
    assert labels_are_semantic_graph_hypothesis(["cup"]) is True
    assert labels_are_semantic_graph_hypothesis(["object"]) is False
    assert labels_are_semantic_graph_hypothesis(["OBJECT"]) is False
    assert labels_are_semantic_graph_hypothesis(["object", "mug"]) is True
    assert labels_are_semantic_graph_hypothesis([]) is False


def test_record_navigation_sample_adds_viewpoint_not_object_node():
    mem = GraphEQAMemory(eqa_client=lambda x: "", image_description_client=lambda x: "")
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    base = np.array([1.1, 2.0, 0.0])
    mem.record_navigation_sample(rgb, np.array([1.0, 2.0, 0.1]), base_xyz=base)
    nodes = mem.get_nodes()
    assert len(mem.get_observations()) == 0
    assert len(mem.get_navigation_samples()) == 1
    assert len(nodes) == 1
    assert nodes[0].is_viewpoint
    np.testing.assert_allclose(nodes[0].xyz, base)


def test_record_navigation_respects_graph_eqa_record_navigation_false():
    mem = GraphEQAMemory(
        parameters={"graph_eqa_record_navigation": False},
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mem.record_navigation_sample(rgb, np.array([0.0, 0.0, 0.0]))
    assert mem.get_navigation_samples() == []


def test_query_answer_navigation_fallback_images_and_target():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "reasoning: r\nanswer: no\nconfidence: false\naction: 1\nconfidence_reasoning: look",
        image_description_client=lambda q: "mug",
    )
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    mem.record_navigation_sample(rgb, np.array([0.5, -0.25, 0.12]), base_xyz=np.array([0.0, 0.0, 0.0]))
    _r, _a, _c, _cr, target_point, imgs = mem.query_answer("Is there a mug?", None, None)
    assert len(imgs) == 1
    assert target_point is not None
    assert abs(float(target_point[0]) - 0.5) < 1e-5
    assert abs(float(target_point[1]) - (-0.25)) < 1e-5


def test_dynagraph_spatial_merge_keeps_detection_bbox():
    mem = GraphEQAMemory(
        parameters={"dynagraph_merge_xy_m": 1.0},
        defer_llm_clients=True,
    )
    rgb = np.zeros((80, 60, 3), dtype=np.uint8)
    mem.set_graph_timestep(1)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["cup"], bbox_xyxy=(5, 5, 25, 30))
    mem.set_graph_timestep(2)
    mem.add_observation(rgb, np.array([0.15, 0.0, 0.5]), ["cup"], bbox_xyxy=(30, 10, 50, 35))
    node = mem.get_nodes()[0]
    assert not node.is_viewpoint
    assert node.bbox_xyxy == (30, 10, 50, 35)


def test_dynagraph_spatial_merge_same_obs_id():
    mem = GraphEQAMemory(
        parameters={"dynagraph_merge_xy_m": 1.0},
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.set_graph_timestep(1)
    id1 = mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["cup"])
    mem.set_graph_timestep(2)
    id2 = mem.add_observation(rgb, np.array([0.2, 0.0, 0.5]), ["cup"])
    assert id1 == id2
    assert len(mem.get_nodes()) == 1
    assert mem.get_nodes()[0].support_count == 2


def test_dynagraph_maintain_prunes_stale_nodes():
    mem = GraphEQAMemory(
        parameters={"dynagraph_staleness_horizon": 5},
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.set_graph_timestep(1)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.1]), ["table"])
    mem.set_graph_timestep(16)
    mem.add_observation(rgb, np.array([2.0, 0.0, 0.1]), ["chair"])
    removed = mem.maintain(current_step=20)
    assert removed == 1
    assert len(mem.get_nodes()) == 1
    assert mem.get_nodes()[0].node_id == 1
    assert "chair" in mem.get_nodes()[0].labels
