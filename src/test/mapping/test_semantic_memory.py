# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for Dynamem semantic memory (feature-based voxel memory)."""

import numpy as np
import pytest
import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


class MockEncoder:
    """Mock encoder that returns deterministic features for text queries."""

    def __init__(self, feature_dim: int = 8):
        self.feature_dim = feature_dim
        # Map text -> feature vector (normalized); includes default MuJoCo scene objects
        self._text_to_feature = {
            "apple": torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
            "bottle": torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
            "cup": torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0], dtype=torch.float32),
            "red cylinder": torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0], dtype=torch.float32),
            "blue cube": torch.tensor([0, 0, 0, 0, 1.0, 0, 0, 0], dtype=torch.float32),
        }
        # Pad or truncate to feature_dim
        for k, v in list(self._text_to_feature.items()):
            if len(v) < feature_dim:
                v = torch.nn.functional.pad(v, (0, feature_dim - len(v)))
            else:
                v = v[:feature_dim]
            self._text_to_feature[k] = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1)

    def encode_text(self, text: str) -> torch.Tensor:
        text_lower = text.lower().strip()
        if text_lower in self._text_to_feature:
            return self._text_to_feature[text_lower]
        # Default: random but deterministic per text
        torch.manual_seed(hash(text_lower) % (2**32))
        feat = torch.randn(1, self.feature_dim)
        return torch.nn.functional.normalize(feat, p=2, dim=1)


def _make_semantic_memory_map() -> SparseVoxelMap:
    """Create a Dynamem SparseVoxelMap with semantic memory and synthetic data."""
    feature_dim = 8
    encoder = MockEncoder(feature_dim=feature_dim)

    voxel_map = SparseVoxelMap(
        resolution=0.05,
        semantic_memory_resolution=0.05,
        feature_dim=feature_dim,
        use_instance_memory=False,
        encoder=encoder,
        device="cpu",
        map_2d_device="cpu",
    )

    # Add points with known features: apple at (1,0,0.5), bottle at (2,0,0.5), cup at (3,0,0.5)
    apple_feat = encoder.encode_text("apple").squeeze(0)
    bottle_feat = encoder.encode_text("bottle").squeeze(0)
    cup_feat = encoder.encode_text("cup").squeeze(0)

    points = torch.tensor(
        [[1.0, 0.0, 0.5], [2.0, 0.0, 0.5], [3.0, 0.0, 0.5]], dtype=torch.float32
    )
    features = torch.stack([apple_feat, bottle_feat, cup_feat])
    rgb = torch.tensor(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=torch.float32
    ) / 255.0

    voxel_map.semantic_memory.add(
        points=points,
        features=features,
        rgb=rgb,
        obs_count=0,
    )
    voxel_map.obs_count = 1

    return voxel_map


def test_find_alignment_over_model():
    """find_alignment_over_model returns cosine similarities between text and points."""
    voxel_map = _make_semantic_memory_map()

    alignments = voxel_map.find_alignment_over_model("apple")
    assert alignments is not None
    assert alignments.shape == (1, 3)  # 1 query, 3 points

    # Apple query should match apple point (index 0) best
    scores = alignments[0]
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > 0.9  # High similarity for exact match


def test_find_alignment_for_text():
    """find_alignment_for_text returns the best-matching 3D point."""
    voxel_map = _make_semantic_memory_map()

    point = voxel_map.find_alignment_for_text("bottle")
    assert point is not None
    point = point.squeeze()
    assert point.shape == (3,)
    # Bottle is at (2, 0, 0.5)
    np.testing.assert_array_almost_equal(point.numpy(), [2.0, 0.0, 0.5])


def test_find_obs_id_for_text():
    """find_obs_id_for_text returns obs_count for best-matching point."""
    voxel_map = _make_semantic_memory_map()

    obs_id = voxel_map.find_obs_id_for_text("cup")
    assert obs_id is not None
    assert obs_id.shape == (1,)
    assert obs_id.item() == 0  # We used obs_count=0


def test_verify_point():
    """verify_point returns True when point is near high-similarity points."""
    voxel_map = _make_semantic_memory_map()

    # Apple point at (1, 0, 0.5) - verify with "apple" query
    assert bool(voxel_map.verify_point("apple", np.array([1.0, 0.0, 0.5])))
    assert bool(voxel_map.verify_point("apple", np.array([1.05, 0.0, 0.5])))  # Within 0.1m

    # Wrong text for that point
    assert not bool(voxel_map.verify_point("bottle", np.array([1.0, 0.0, 0.5])))

    # Point far from any memory
    assert not bool(voxel_map.verify_point("apple", np.array([10.0, 10.0, 10.0])))


def test_find_alignment_empty_memory():
    """find_alignment_over_model returns None when semantic memory is empty."""
    feature_dim = 8
    encoder = MockEncoder(feature_dim=feature_dim)
    voxel_map = SparseVoxelMap(
        resolution=0.05,
        semantic_memory_resolution=0.05,
        feature_dim=feature_dim,
        use_instance_memory=False,
        encoder=encoder,
        device="cpu",
        map_2d_device="cpu",
    )
    # Don't add any points

    alignments = voxel_map.find_alignment_over_model("apple")
    assert alignments is None


def test_add_to_semantic_memory():
    """add_to_semantic_memory correctly adds points with features."""
    voxel_map = _make_semantic_memory_map()

    points, features, weights, rgb = voxel_map.semantic_memory.get_pointcloud()
    assert points is not None
    assert points.shape[0] >= 1  # May be voxelized/reduced
    assert features is not None
    assert features.shape[1] == 8


def _make_red_cylinder_map() -> SparseVoxelMap:
    """Map with 'red cylinder' in semantic memory (default MuJoCo scene object)."""
    feature_dim = 8
    encoder = MockEncoder(feature_dim=feature_dim)

    voxel_map = SparseVoxelMap(
        resolution=0.05,
        semantic_memory_resolution=0.05,
        feature_dim=feature_dim,
        use_instance_memory=False,
        encoder=encoder,
        device="cpu",
        map_2d_device="cpu",
        log="test",  # avoid writing to missing dir if code path touches it
    )

    # Red cylinder at table height (default scene: object2 at ~(.08, -0.55, .6))
    red_cylinder_feat = encoder.encode_text("red cylinder").squeeze(0)
    points = torch.tensor([[0.08, -0.55, 0.6]], dtype=torch.float32)
    features = red_cylinder_feat.unsqueeze(0)
    rgb = torch.tensor([[255, 0, 0]], dtype=torch.float32) / 255.0

    voxel_map.semantic_memory.add(
        points=points,
        features=features,
        rgb=rgb,
        obs_count=0,
    )
    voxel_map.obs_count = 1
    return voxel_map


def test_localize_red_cylinder():
    """Dynamem/SVM can find 'red cylinder' when it is in semantic memory (default MuJoCo scene)."""
    voxel_map = _make_red_cylinder_map()

    result = voxel_map.localize_text("red cylinder", debug=True, return_debug=True)
    target_point, debug_text = result[0], result[1]

    assert target_point is not None, "localize_text('red cylinder') should return a target point"
    target = target_point.squeeze()
    assert target.shape == (3,), "target should be 3D (x, y, z)"

    # Should be near the red cylinder we placed at (0.08, -0.55, 0.6)
    np.testing.assert_allclose(
        target.numpy(),
        [0.08, -0.55, 0.6],
        atol=0.15,
        err_msg="Target point should be near red cylinder position",
    )
    assert "😃" in debug_text or "identified" in debug_text.lower(), (
        "Debug text should indicate successful localization"
    )
