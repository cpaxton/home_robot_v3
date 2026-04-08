# Copyright (c) Hello Robot, Inc.
#
# Unit tests for the OpenVocabSceneGraph: node creation, deduplication,
# edge computation, text localization, and save/load.

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Direct import to avoid heavy emet.mapping.__init__ chain (pinocchio etc.)
_sg_path = Path(__file__).resolve().parents[2] / "emet" / "mapping" / "scene_graph" / "open_vocab_scene_graph.py"
_spec = importlib.util.spec_from_file_location("_ovsg", str(_sg_path))
_mod = importlib.util.module_from_spec(_spec)
_mod.__package__ = "emet.mapping.scene_graph"
sys.modules["_ovsg"] = _mod
_spec.loader.exec_module(_mod)

ObjectObservation = _mod.ObjectObservation
OpenVocabSceneGraph = _mod.OpenVocabSceneGraph
SceneGraphNode = _mod.SceneGraphNode
_bbox3d_iou = _mod._bbox3d_iou


class MockTextEncoder:
    """Mock encoder mapping known labels to orthogonal feature vectors."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self._map = {
            "red cylinder": 0,
            "blue cube": 1,
            "cup": 2,
            "table": 3,
            "chair": 4,
            "bowl": 5,
        }

    def encode_text(self, text: str) -> torch.Tensor:
        idx = self._map.get(text.lower().strip(), hash(text) % self.dim)
        feat = torch.zeros(1, self.dim)
        feat[0, idx % self.dim] = 1.0
        return feat

    def encode_image(self, image) -> torch.Tensor:
        return torch.randn(1, self.dim)


def _make_observation(
    label: str,
    center: np.ndarray,
    timestep: int = 1,
    n_points: int = 50,
    siglip_idx: int = 0,
    dinov3_seed: int = 42,
    dim: int = 16,
) -> ObjectObservation:
    """Create a synthetic ObjectObservation."""
    pts = torch.randn(n_points, 3) * 0.1 + torch.tensor(center, dtype=torch.float32)
    rgb_crop = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:60, 30:60] = True

    siglip = torch.zeros(dim)
    siglip[siglip_idx % dim] = 1.0
    siglip = F.normalize(siglip.unsqueeze(0), dim=-1).squeeze(0)

    torch.manual_seed(dinov3_seed)
    dinov3 = F.normalize(torch.randn(dim).unsqueeze(0), dim=-1).squeeze(0)

    return ObjectObservation(
        mask=mask,
        bbox_xyxy=np.array([30, 30, 60, 60]),
        rgb_crop=rgb_crop,
        points_3d=pts,
        points_rgb=torch.rand(n_points, 3),
        camera_pose=np.eye(4),
        label=label,
        score=0.9,
        timestep=timestep,
        siglip_embedding=siglip,
        dinov3_embedding=dinov3,
    )


class TestOpenVocabSceneGraph:
    def test_add_single_object(self):
        sg = OpenVocabSceneGraph()
        obs = _make_observation("cup", [1.0, 0.0, 0.5])
        nid = sg.add_observation(obs)
        assert nid == 0
        assert sg.num_objects == 1
        node = sg.nodes[nid]
        assert node.primary_label == "cup"
        assert node.observation_count == 1
        assert node.center is not None

    def test_add_two_different_objects(self):
        sg = OpenVocabSceneGraph()
        obs1 = _make_observation("cup", [1.0, 0.0, 0.5], siglip_idx=0, dinov3_seed=1)
        obs2 = _make_observation("table", [5.0, 5.0, 0.3], siglip_idx=3, dinov3_seed=99)
        sg.add_observation(obs1)
        sg.add_observation(obs2)
        assert sg.num_objects == 2

    def test_dedup_same_object_same_location(self):
        """Two observations of the same object at the same location should merge."""
        sg = OpenVocabSceneGraph(dedup_iou_threshold=0.1, dedup_visual_threshold=0.5)
        obs1 = _make_observation("cup", [1.0, 0.0, 0.5], timestep=1, siglip_idx=2, dinov3_seed=42)
        obs2 = _make_observation("cup", [1.0, 0.0, 0.5], timestep=2, siglip_idx=2, dinov3_seed=42)
        nid1 = sg.add_observation(obs1)
        nid2 = sg.add_observation(obs2)
        assert nid1 == nid2, "Same object should be merged"
        assert sg.num_objects == 1
        assert sg.nodes[nid1].observation_count == 2

    def test_stability(self):
        sg = OpenVocabSceneGraph(min_observations_stable=2)
        obs1 = _make_observation("cup", [1.0, 0.0, 0.5], timestep=1, siglip_idx=2, dinov3_seed=42)
        nid = sg.add_observation(obs1)
        assert not sg.nodes[nid].is_stable

        obs2 = _make_observation("cup", [1.0, 0.0, 0.5], timestep=2, siglip_idx=2, dinov3_seed=42)
        sg.add_observation(obs2)
        assert sg.nodes[nid].is_stable

    def test_edges_near(self):
        sg = OpenVocabSceneGraph(max_near_distance=2.0)
        obs1 = _make_observation("cup", [0.0, 0.0, 0.5], siglip_idx=0, dinov3_seed=1)
        obs2 = _make_observation("bowl", [1.0, 0.0, 0.5], siglip_idx=5, dinov3_seed=2)
        sg.add_observation(obs1)
        sg.add_observation(obs2)
        sg.update_edges()
        near_edges = [e for e in sg.edges if e.relation == "near"]
        assert len(near_edges) >= 1

    def test_edges_on(self):
        sg = OpenVocabSceneGraph(min_on_height=0.02, max_on_height=0.3)
        obs_table = _make_observation("table", [1.0, 0.0, 0.3], siglip_idx=3, dinov3_seed=10)
        obs_cup = _make_observation("cup", [1.0, 0.0, 0.5], siglip_idx=2, dinov3_seed=20)
        sg.add_observation(obs_table)
        sg.add_observation(obs_cup)
        sg.update_edges()
        on_edges = [e for e in sg.edges if e.relation == "on"]
        assert len(on_edges) >= 1

    def test_localize_text(self):
        sg = OpenVocabSceneGraph()
        encoder = MockTextEncoder(dim=16)
        obs = _make_observation("red cylinder", [0.08, -0.55, 0.6], siglip_idx=0)
        sg.add_observation(obs)
        center = sg.localize_text("red cylinder", encoder)
        assert center is not None
        np.testing.assert_allclose(center, [0.08, -0.55, 0.6], atol=0.15)

    def test_check_for_object(self):
        sg = OpenVocabSceneGraph()
        encoder = MockTextEncoder(dim=16)
        obs = _make_observation("blue cube", [-0.02, -0.55, 0.6], siglip_idx=1)
        sg.add_observation(obs)
        conf, loc = sg.check_for_object("blue cube", encoder)
        assert conf > 0
        assert loc is not None

    def test_list_objects(self):
        sg = OpenVocabSceneGraph()
        # Need 2 observations per object to be "stable"
        obs1a = _make_observation("cup", [1.0, 0.0, 0.5], timestep=1, siglip_idx=2, dinov3_seed=1)
        obs1b = _make_observation("cup", [1.0, 0.0, 0.5], timestep=2, siglip_idx=2, dinov3_seed=1)
        obs2a = _make_observation("table", [5.0, 5.0, 0.3], timestep=1, siglip_idx=3, dinov3_seed=2)
        obs2b = _make_observation("table", [5.0, 5.0, 0.3], timestep=2, siglip_idx=3, dinov3_seed=2)
        sg.add_observation(obs1a)
        sg.add_observation(obs1b)
        sg.add_observation(obs2a)
        sg.add_observation(obs2b)
        labels = sg.list_objects()
        assert "cup" in labels
        assert "table" in labels

    def test_prune_small(self):
        sg = OpenVocabSceneGraph(min_points_per_object=100)
        obs = _make_observation("tiny", [0.0, 0.0, 0.0], n_points=5)
        sg.add_observation(obs)
        assert sg.num_objects == 1
        removed = sg.prune_small()
        assert len(removed) == 1
        assert sg.num_objects == 0

    def test_prune_stale(self):
        sg = OpenVocabSceneGraph(staleness_horizon=5)
        obs = _make_observation("old", [0.0, 0.0, 0.0], timestep=1, siglip_idx=0, dinov3_seed=1)
        sg.add_observation(obs)
        sg._current_step = 10
        removed = sg.prune_stale()
        assert len(removed) == 1
        assert sg.num_objects == 0

    def test_merge_duplicates(self):
        sg = OpenVocabSceneGraph(dedup_visual_threshold=0.5)
        # Force two nodes that are very similar
        obs1 = _make_observation("cup", [1.0, 0.0, 0.5], siglip_idx=2, dinov3_seed=42)
        obs2 = _make_observation("mug", [5.0, 5.0, 0.5], siglip_idx=7, dinov3_seed=99)
        sg.add_observation(obs1)
        sg.add_observation(obs2)
        assert sg.num_objects == 2
        # Manually make them overlap
        sg.nodes[0].point_cloud = sg.nodes[1].point_cloud.clone()
        sg.nodes[0].bounds = sg.nodes[1].bounds.clone()
        merges = sg.merge_duplicates()
        assert merges >= 1
        assert sg.num_objects == 1

    def test_save_and_load(self):
        sg = OpenVocabSceneGraph()
        obs1 = _make_observation("cup", [1.0, 0.0, 0.5], siglip_idx=2, dinov3_seed=42)
        obs2 = _make_observation("table", [3.0, 0.0, 0.3], siglip_idx=3, dinov3_seed=99)
        sg.add_observation(obs1)
        sg.add_observation(obs2)
        sg.update_edges()

        with tempfile.TemporaryDirectory() as tmpdir:
            sg.save(tmpdir)
            loaded = OpenVocabSceneGraph.load(tmpdir)

        assert loaded.num_objects == 2
        labels = [n.primary_label for n in loaded.nodes.values()]
        assert "cup" in labels
        assert "table" in labels

        for nid in loaded.nodes:
            node = loaded.nodes[nid]
            assert node.point_cloud is not None
            assert node.siglip_embedding is not None
            assert node.dinov3_embedding is not None

    def test_to_string(self):
        sg = OpenVocabSceneGraph()
        obs = _make_observation("cup", [1.0, 0.0, 0.5])
        sg.add_observation(obs)
        text = sg.to_string()
        assert "cup" in text
        assert "1 objects" in text

    def test_to_dict(self):
        sg = OpenVocabSceneGraph()
        obs = _make_observation("cup", [1.0, 0.0, 0.5])
        sg.add_observation(obs)
        d = sg.to_dict()
        assert d["num_objects"] == 1
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["primary_label"] == "cup"


class TestSceneGraphBackend:
    """Backend tests are skipped if emet.memory.adapters can't be imported (pinocchio chain)."""

    @pytest.fixture(autouse=True)
    def _try_import_backend(self):
        try:
            from emet.memory.adapters import SceneGraphBackend
            self.SceneGraphBackend = SceneGraphBackend
        except ImportError:
            pytest.skip("emet.memory.adapters not importable (missing pinocchio/hppfcl)")

    def test_backend_check_and_localize(self):
        sg = OpenVocabSceneGraph()
        encoder = MockTextEncoder(dim=16)
        obs = _make_observation("red cylinder", [0.08, -0.55, 0.6], siglip_idx=0)
        sg.add_observation(obs)

        backend = self.SceneGraphBackend(sg, text_encoder=encoder)

        check = backend.check_memory_for_object("red cylinder")
        assert check.confidence > 0
        assert check.location_xyz is not None

        loc = backend.localize_text("red cylinder")
        assert loc.success
        assert loc.point_xyz is not None

    def test_backend_list_objects(self):
        sg = OpenVocabSceneGraph()
        obs1 = _make_observation("cup", [1.0, 0.0, 0.5], timestep=1, siglip_idx=2, dinov3_seed=42)
        obs2 = _make_observation("cup", [1.0, 0.0, 0.5], timestep=2, siglip_idx=2, dinov3_seed=42)
        sg.add_observation(obs1)
        sg.add_observation(obs2)

        backend = self.SceneGraphBackend(sg)
        labels = backend.list_objects()
        assert "cup" in labels
