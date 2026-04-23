# Copyright (c) Hello Robot, Inc. All rights reserved.

import numpy as np

from emet.memory.format import FrameBlob, MemoryManifest, MemoryState, load_memory, save_memory


def test_save_load_instance_masks_and_detections_roundtrip(tmp_path):
    h, w = 8, 8
    inst = np.full((h, w), -1, dtype=np.int64)
    inst[0:4, 0:4] = 0
    wx = np.zeros((h, w, 3), dtype=np.float32)
    wx[0:4, 0:4] = [1.0, 2.0, 3.0]
    depth = np.ones((h, w), dtype=np.float32) * 0.5
    pose = np.eye(4, dtype=np.float64)
    K = np.eye(3, dtype=np.float64)

    dets = [
        {
            "instance_id": 0,
            "category_id": 2,
            "label_short": "cls_2",
            "xyz": [1.0, 2.0, 3.0],
        }
    ]
    fr = FrameBlob(
        camera_pose=pose,
        camera_K=K,
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=depth,
        world_xyz=wx,
        instance=inst,
        instance_classes=np.array([2], dtype=np.int64),
        instance_scores=np.array([0.9], dtype=np.float32),
        detections=dets,
    )
    st = MemoryState(
        frames=[fr],
        manifest=MemoryManifest(
            backend="graph_eqa",
            has_frames=True,
            has_point_cloud=False,
            has_graph=False,
        ),
    )
    root = tmp_path / "mem"
    save_memory(st, str(root))
    st2 = load_memory(str(root))
    assert len(st2.frames) == 1
    f2 = st2.frames[0]
    assert f2.instance is not None and f2.instance.shape == (h, w)
    assert f2.world_xyz is not None and f2.world_xyz.shape == (h, w, 3)
    assert f2.detections is not None and len(f2.detections) == 1
    assert f2.detections[0]["label_short"] == "cls_2"
