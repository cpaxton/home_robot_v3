# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Smoke tests for all three memory backends: SVM (instance memory), DynaMem (voxel
# semantic memory), GraphEQA (graph-based EQA). Run with:
#   pytest src/test/memory/test_memory_backends_smoke.py -v
# See docs/plans/TESTING_BACKENDS.md for full test matrix and sim integration test.

import pytest

from emet.controller import RobotAgent
from emet.core import Parameters
from emet.utils.config import Config
from emet.utils.dummy_stretch_client import DummyStretchClient


def test_svm_backend_smoke():
    """SVM (InstanceMemoryController / RobotAgent) can be created and exposes voxel map and navigation space."""
    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent / "mapping" / "planner.yaml"
    if not config_path.exists():
        pytest.skip("SVM test requires src/test/mapping/planner.yaml")
    config = Config()
    config.merge_from_file(str(config_path))
    config.freeze()
    parameters = Parameters(**config)
    dummy_robot = DummyStretchClient()
    agent = RobotAgent(
        dummy_robot,
        parameters,
        semantic_sensor=None,
        voxel_map=None,
        use_instance_memory=True,
        create_semantic_sensor=False,
    )
    voxel_map = agent.get_voxel_map()
    assert voxel_map is not None
    space = agent.get_navigation_space()
    assert space is None or callable(getattr(space, "is_valid", None)) or hasattr(space, "sample_frontier")


def _make_red_cylinder_map():
    """Minimal Dynamem map with red cylinder at default scene position (for smoke test)."""
    import torch
    from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap

    class _MockEnc:
        def __init__(self):
            self._dim = 8
            self._text_to_feature = {"red cylinder": torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0], dtype=torch.float32)}
            for k, v in list(self._text_to_feature.items()):
                v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1)
                self._text_to_feature[k] = v
        def encode_text(self, text):
            return self._text_to_feature.get(text.lower().strip(), self._text_to_feature["red cylinder"])
    encoder = _MockEnc()
    voxel_map = SparseVoxelMap(
        resolution=0.05,
        semantic_memory_resolution=0.05,
        feature_dim=8,
        use_instance_memory=False,
        encoder=encoder,
        device="cpu",
        map_2d_device="cpu",
        log="test",
    )
    red_cylinder_feat = encoder.encode_text("red cylinder").squeeze(0)
    points = torch.tensor([[0.08, -0.55, 0.6]], dtype=torch.float32)
    features = red_cylinder_feat.unsqueeze(0)
    rgb = torch.tensor([[255, 0, 0]], dtype=torch.float32) / 255.0
    voxel_map.semantic_memory.add(points=points, features=features, rgb=rgb, obs_count=0)
    voxel_map.obs_count = 1
    return voxel_map


def test_dynamem_backend_smoke():
    """DynaMem (SparseVoxelMap with semantic memory) can localize 'red cylinder' when it is in the map."""
    import numpy as np
    voxel_map = _make_red_cylinder_map()
    result = voxel_map.localize_text("red cylinder", debug=True, return_debug=True)
    target_point, _ = result[0], result[1]
    assert target_point is not None
    target = target_point.squeeze()
    assert target.shape == (3,)
    np.testing.assert_allclose(target.numpy(), [0.08, -0.55, 0.6], atol=0.15)


def test_graph_eqa_backend_smoke():
    """GraphEQA memory can add observations and run a query_answer with mock clients."""
    import numpy as np
    from emet.memory.graph_eqa import GraphEQAMemory

    def mock_eqa(_):
        return (
            "reasoning: I see a table.\n"
            "answer: Yes\n"
            "confidence: true\n"
            "action: \n"
            "confidence_reasoning: Sure."
        )
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
    reasoning, answer, confidence, _, target_point, relevant_images = out
    assert isinstance(reasoning, str) and isinstance(answer, str)
    assert confidence is True
    assert isinstance(relevant_images, list)
