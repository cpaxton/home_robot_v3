# Copyright (c) Hello Robot, Inc. All rights reserved.

import numpy as np

from emet.app.reprocess_graph_eqa_cache import reprocess_memory_directory
from emet.memory.format import FrameBlob, MemoryManifest, MemoryState, load_memory, save_memory


def test_reprocess_builds_graph_from_detections_json(tmp_path):
    fr = FrameBlob(
        camera_pose=np.eye(4, dtype=np.float64),
        camera_K=np.eye(3, dtype=np.float64),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        world_xyz=np.zeros((8, 8, 3), dtype=np.float32),
        detections=[
            {
                "instance_id": 0,
                "category_id": 1,
                "label_short": "mug",
                "xyz": [0.1, 0.2, 0.3],
            }
        ],
    )
    st = MemoryState(
        frames=[fr],
        manifest=MemoryManifest(backend="graph_eqa", has_frames=True, has_point_cloud=False, has_graph=False),
    )
    inp = tmp_path / "in"
    save_memory(st, str(inp))
    out = tmp_path / "out"
    gm, report = reprocess_memory_directory(
        str(inp),
        str(out),
        parameters={"graph_instance_dedup_xy_m": 0.0},
        require_cache=False,
        min_depth=0.01,
        max_depth=10.0,
    )
    assert len(gm.get_nodes()) == 1
    assert "mug" in report
    st3 = load_memory(str(out))
    assert st3.graph is not None
    assert len(st3.graph.nodes) == 1
