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
    voxel_map, _ = _make_red_cylinder_and_blue_cube_map()
    return voxel_map


def _make_red_cylinder_and_blue_cube_map():
    """Dynamem map with red cylinder and blue cube at default MuJoCo scene positions."""
    import torch
    from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap

    class _MockEnc:
        def __init__(self):
            self._dim = 8
            self._text_to_feature = {
                "red cylinder": torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0], dtype=torch.float32),
                "blue cube": torch.tensor([0, 0, 0, 0, 1.0, 0, 0, 0], dtype=torch.float32),
            }
            for k, v in list(self._text_to_feature.items()):
                v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1)
                self._text_to_feature[k] = v

        def encode_text(self, text):
            return self._text_to_feature.get(
                text.lower().strip(), self._text_to_feature["red cylinder"]
            )

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
    # Red cylinder at (0.08, -0.55, 0.6), blue cube at (-0.02, -0.55, 0.6) (scene.xml object1/object2)
    red_feat = encoder.encode_text("red cylinder").squeeze(0)
    blue_feat = encoder.encode_text("blue cube").squeeze(0)
    points = torch.tensor(
        [[0.08, -0.55, 0.6], [-0.02, -0.55, 0.6]], dtype=torch.float32
    )
    features = torch.stack([red_feat, blue_feat])
    rgb = torch.tensor(
        [[255, 0, 0], [0, 0, 255]], dtype=torch.float32
    ) / 255.0
    voxel_map.semantic_memory.add(
        points=points,
        features=features,
        rgb=rgb,
        obs_count=0,
    )
    voxel_map.obs_count = 1
    return voxel_map, encoder


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


# ---- Unified MemoryBackend interface (check_memory_for_object / localize_text) ----


def test_unified_backend_dynamem():
    """Unified MemoryBackend (DynaMem): check_memory_for_object and localize_text for red cylinder and blue cube."""
    import numpy as np
    from emet.memory.backend import get_memory_backend

    voxel_map, _ = _make_red_cylinder_and_blue_cube_map()
    backend = get_memory_backend("dynamem", voxel_map=voxel_map)

    # Red cylinder
    check_red = backend.check_memory_for_object("red cylinder")
    assert check_red.confidence > 0, "red cylinder should be in memory"
    assert check_red.location_xyz is not None
    np.testing.assert_allclose(
        check_red.location_xyz,
        [0.08, -0.55, 0.6],
        atol=0.15,
        err_msg="red cylinder position",
    )
    loc_red = backend.localize_text("red cylinder")
    assert loc_red.success and loc_red.point_xyz is not None
    np.testing.assert_allclose(loc_red.point_xyz, [0.08, -0.55, 0.6], atol=0.15)

    # Blue cube
    check_blue = backend.check_memory_for_object("blue cube")
    assert check_blue.confidence > 0, "blue cube should be in memory"
    assert check_blue.location_xyz is not None
    np.testing.assert_allclose(
        check_blue.location_xyz,
        [-0.02, -0.55, 0.6],
        atol=0.15,
        err_msg="blue cube position",
    )
    loc_blue = backend.localize_text("blue cube")
    assert loc_blue.success and loc_blue.point_xyz is not None

    # Unknown object
    check_unknown = backend.check_memory_for_object("purple sphere")
    assert check_unknown.confidence >= 0  # may be 0 or low from default encoder fallback
    # save/load: DynaMem backend supports save; round-trip tested in integration test
    assert backend.supports_save_load()
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        backend.save(path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_unified_backend_graph_eqa():
    """Unified MemoryBackend (GraphEQA): check_memory_for_object and localize_text via graph nodes."""
    import numpy as np
    from emet.memory.backend import get_memory_backend
    from emet.memory.graph_eqa import GraphEQAMemory

    def mock_eqa(_):
        return (
            "reasoning: ok.\nanswer: Yes\nconfidence: true\naction: \nconfidence_reasoning: ok."
        )

    mem = GraphEQAMemory(
        eqa_client=mock_eqa,
        image_description_client=lambda x: "table",
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([0.08, -0.55, 0.5]),
        ["red cylinder"],
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([-0.02, -0.55, 0.5]),
        ["blue cube"],
    )
    backend = get_memory_backend("graph_eqa", graph_memory=mem)

    check_red = backend.check_memory_for_object("red cylinder")
    assert check_red.confidence > 0
    assert check_red.location_xyz is not None
    loc_red = backend.localize_text("red cylinder")
    assert loc_red.success and loc_red.point_xyz is not None

    check_blue = backend.check_memory_for_object("blue cube")
    assert check_blue.confidence > 0
    list_objs = backend.list_objects()
    assert "red cylinder" in list_objs or any("red" in o for o in list_objs)
    assert "blue cube" in list_objs or any("blue" in o for o in list_objs)


def test_unified_backend_svm_empty():
    """Unified MemoryBackend (SVM): empty instance memory returns confidence 0 and localize fails."""
    from pathlib import Path
    from emet.memory.backend import get_memory_backend

    config_path = Path(__file__).resolve().parent.parent / "mapping" / "planner.yaml"
    if not config_path.exists():
        pytest.skip("SVM test requires src/test/mapping/planner.yaml")
    from emet.utils.config import Config
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
    backend = get_memory_backend("svm", agent=agent)
    check = backend.check_memory_for_object("red cylinder")
    assert check.confidence == 0.0
    assert check.location_xyz is None
    loc = backend.localize_text("red cylinder")
    assert not loc.success
    assert loc.point_xyz is None
