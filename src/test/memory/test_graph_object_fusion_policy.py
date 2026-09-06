# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from emet.memory.graph_eqa.dynamem_graph_hooks import update_graph_memory_from_dynamem_observation
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.config import (
    GraphObjectFusionConfig,
    decode_graph_object_fusion_config,
)
from emet.memory.graph_eqa.graph_object_fusion.fusion import (
    GraphDetectionCandidate,
    GraphObjectFusion,
)


def _fake_frame_with_instances(n_instances: int = 2, xy_spacing: float = 2.0):
    h, w = 8, 8
    inst = torch.full((h, w), -1, dtype=torch.long)
    depth = torch.ones((h, w), dtype=torch.float32) * 1.0
    fw = torch.zeros((h, w, 3), dtype=torch.float32)
    classes = torch.arange(n_instances, dtype=torch.long)
    for i in range(n_instances):
        x0, y0 = (i * 3) % (w - 2), (i * 2) % (h - 2)
        inst[y0 : y0 + 2, x0 : x0 + 2] = i
        fw[y0 : y0 + 2, x0 : x0 + 2, 0] = float(i) * xy_spacing
        fw[y0 : y0 + 2, x0 : x0 + 2, 1] = 0.0
        fw[y0 : y0 + 2, x0 : x0 + 2, 2] = 0.9

    class Frame:
        pass

    f = Frame()
    f.instance = inst
    f.depth = depth
    f.full_world_xyz = fw
    f.instance_classes = classes
    f.instance_scores = torch.ones(n_instances, dtype=torch.float32) * 0.8
    return f


def _fake_obs():
    obs = MagicMock()
    obs.rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    obs.camera_pose = np.eye(4)
    obs.camera_pose[:3, 3] = [1.0, 2.0, 3.0]
    obs.semantic = None
    obs.depth = None
    obs.gps = None
    obs.compass = None
    obs.navigation_origin_xyt = None
    return obs


# --------------------------------------------------------------------------- config decode


def test_config_nested_decode():
    raw = {
        "enabled": True,
        "use_instance_nodes": False,
        "gates": {"bounds": {"iou_merge_min": 0.25}},
        "growth": {"max_object_nodes": 120, "temporal_window_steps": 8},
        "labels": {"synonyms": [["cab", "cabinet"]], "incompatible": [["person", "lamp"]]},
    }
    c = decode_graph_object_fusion_config(raw)
    assert c.enabled is True
    assert c.use_instance_nodes is False
    assert c.gates.bounds.iou_merge_min == 0.25
    assert c.growth.max_object_nodes == 120
    assert c.growth.temporal_window_steps == 8
    assert c.labels.synonyms == [["cab", "cabinet"]]
    assert c.labels.incompatible == [["person", "lamp"]]


def test_config_legacy_flat_decode():
    raw = {
        "enabled": True,
        "spatial_merge_xy_m": 0.5,
        "bounds_3d_iou_merge_min": 0.45,
        "fallback_spatial_merge_xy_m": 0.5,
        "embedding_min_cosine": 0.0,
        "require_label_match_for_instances": True,
    }
    c = decode_graph_object_fusion_config(raw)
    assert c.gates.spatial.xy_m == 0.5
    assert c.gates.bounds.iou_merge_min == 0.45
    assert c.gates.spatial.fallback_xy_m == 0.5
    assert c.gates.embedding.min_cosine == 0.0
    assert c.labels.require_match_for_instances is True
    assert c.use_instance_nodes is True  # default preserved


def test_config_legacy_kwargs_construction():
    c = GraphObjectFusionConfig(enabled=True, spatial_merge_xy_m=0.55, bounds_3d_iou_merge_min=0.3)
    assert c.enabled is True
    assert c.gates.spatial.xy_m == 0.55
    assert c.gates.bounds.iou_merge_min == 0.3
    assert c.use_instance_nodes is True


def test_config_legacy_setattr_routes_to_nested():
    c = GraphObjectFusionConfig(enabled=True)
    c.bounds_3d_iou_merge_min = 0.45
    assert c.gates.bounds.iou_merge_min == 0.45


# --------------------------------------------------------------------------- policy behavior


def test_use_instance_nodes_false_skips_instance_graph_nodes():
    """use_instance_nodes=False keeps YoloE detections out of the scene graph entirely."""
    cfg = GraphObjectFusionConfig(enabled=True, use_instance_nodes=False)
    fusion = GraphObjectFusion(cfg)
    gm = GraphEQAMemory(parameters={}, defer_llm_clients=True)
    vm = MagicMock()
    vm.observations = [_fake_frame_with_instances(n_instances=3)]
    vm.min_depth = 0.1
    vm.max_depth = 5.0
    vm.image_descriptions = []
    vm.use_instance_memory = False
    det = MagicMock()
    det.class_list = ["mug", "cup", "bowl", "plate", "lamp"]

    update_graph_memory_from_dynamem_observation(
        graph_memory=gm,
        robot=MagicMock(get_base_pose=MagicMock(return_value=np.array([0.5, 1.0, 0.0]))),
        voxel_map=vm,
        detection_model=det,
        sensor_builder=MagicMock(),
        use_instance_graph=True,
        use_sensor_perception=False,
        dedup_skips=None,
        obs=_fake_obs(),
        frame_step=1,
        graph_object_fusion=fusion,
    )
    nodes = gm.get_nodes()
    object_nodes = [n for n in nodes if not n.is_viewpoint and not n.is_frontier]
    assert object_nodes == [], f"instance nodes must stay out of the graph, got {len(object_nodes)}"


def test_use_instance_nodes_false_skips_instance_memory_fallback():
    """The fallback is another instance source and must honor the same master switch."""
    cfg = GraphObjectFusionConfig(enabled=True, use_instance_nodes=False)
    fusion = GraphObjectFusion(cfg)
    gm = GraphEQAMemory(parameters={}, defer_llm_clients=True)
    vm = MagicMock()
    vm.observations = [_fake_frame_with_instances(n_instances=1)]
    vm.min_depth = 0.1
    vm.max_depth = 5.0
    vm.image_descriptions = []
    vm.use_instance_memory = True
    vm.get_instances.return_value = []
    update_graph_memory_from_dynamem_observation(
        graph_memory=gm,
        robot=MagicMock(get_base_pose=MagicMock(return_value=np.array([0.5, 1.0, 0.0]))),
        voxel_map=vm,
        detection_model=MagicMock(class_list=["mug"]),
        sensor_builder=MagicMock(),
        use_instance_graph=True,
        use_sensor_perception=False,
        dedup_skips=None,
        obs=_fake_obs(),
        frame_step=1,
        graph_object_fusion=fusion,
    )
    assert [n for n in gm.get_nodes() if not n.is_viewpoint and not n.is_frontier] == []


def test_growth_max_object_nodes_caps_flood():
    """growth.max_object_nodes caps the per-episode object-node count."""
    cfg = GraphObjectFusionConfig(enabled=True, spatial_merge_xy_m=0.1, growth={"max_object_nodes": 2})
    fusion = GraphObjectFusion(cfg)
    gm = GraphEQAMemory(parameters={}, defer_llm_clients=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(6):
        fusion.apply_detection(
            gm,
            rgb,
            GraphDetectionCandidate(
                label="cabinet",
                xyz=np.array([float(i) * 0.5, 0.0, 0.8]),
                bounds_3d={"min": [i * 0.5 - 0.2, -0.2, 0], "max": [i * 0.5 + 0.2, 0.2, 1.6]},
                countable_instance=True,
                detection_score=0.6,
                mask_point_count=100,
            ),
        )
        if i >= 2:
            break
    # The direct fusion path must enforce the same cap as stream ingestion.
    assert fusion.config.growth.max_object_nodes == 2
    objs = [n for n in gm.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 2
    # Verify the stream path cannot overshoot a cap within one dense frame either.
    vm = MagicMock()
    vm.observations = [_fake_frame_with_instances(n_instances=6)]
    vm.min_depth = 0.1
    vm.max_depth = 5.0
    vm.image_descriptions = []
    vm.use_instance_memory = False
    gm2 = GraphEQAMemory(parameters={}, defer_llm_clients=True)
    det = MagicMock()
    det.class_list = [f"obj_{i}" for i in range(20)]
    update_graph_memory_from_dynamem_observation(
        graph_memory=gm2,
        robot=MagicMock(get_base_pose=MagicMock(return_value=np.array([0.5, 1.0, 0.0]))),
        voxel_map=vm,
        detection_model=det,
        sensor_builder=MagicMock(),
        use_instance_graph=True,
        use_sensor_perception=False,
        dedup_skips=None,
        obs=_fake_obs(),
        frame_step=1,
        graph_object_fusion=fusion,
    )
    objs2 = [n for n in gm2.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs2) <= 2, f"instance nodes must be capped at max_object_nodes, got {len(objs2)}"


def test_labels_incompatible_blocks_merge():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        require_label_match=True,
        labels={"incompatible": [["person", "lamp"]]},
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="person", xyz=np.array([1.0, 1.0, 0.5])))
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="lamp", xyz=np.array([1.01, 1.01, 0.5])))
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 2, "incompatible labels must never merge"


def test_labels_synonyms_enable_cross_label_merge():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        require_label_match=True,
        labels={"synonyms": [["cab", "cabinet"]]},
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="cabinet", xyz=np.array([1.0, 1.0, 0.5])))
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="cab", xyz=np.array([1.02, 1.02, 0.5])))
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 1, "config synonym groups should allow cab/cabinet to merge"


def test_temporal_window_skips_stale_nodes():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        require_label_match=False,
        growth={"temporal_window_steps": 5},
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    mem.set_graph_timestep(1)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="cup", xyz=np.array([1.0, 1.0, 0.5])))
    mem.set_graph_timestep(20)
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="cup", xyz=np.array([1.01, 1.01, 0.5])))
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 2, "stale node (19 steps old) must not be merged into with window=5"


def test_keep_update_xyz_false_keeps_anchor():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        require_label_match=False,
        keep={"update_xyz": False},
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="cup", xyz=np.array([1.0, 1.0, 0.5])))
    fusion.apply_detection(mem, rgb, GraphDetectionCandidate(label="cup", xyz=np.array([1.02, 1.02, 0.5])))
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 1
    assert objs[0].support_count == 2
    assert np.allclose(objs[0].xyz[:2], [1.0, 1.0]), "update_xyz=False must keep the first anchor"


def test_appearance_merge_overrides_label_gate():
    """SigLIP appearance override: same-looking object merges across label drift."""
    cfg = GraphObjectFusionConfig(
        enabled=True,
        require_label_match_for_instances=True,
        spatial_merge_xy_m=0.5,
        embedding_min_cosine=0.6,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(label="cabinet", xyz=np.array([1.0, 1.0, 0.5]), embedding=emb, countable_instance=True),
    )
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(
            label="cupboard", xyz=np.array([1.02, 1.02, 0.5]), embedding=emb, countable_instance=True
        ),
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 1, "identical appearance must merge despite cabinet/cupboard label drift"


def test_appearance_does_not_merge_distinct_objects():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        require_label_match_for_instances=True,
        spatial_merge_xy_m=0.5,
        embedding_min_cosine=0.6,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(
            label="cabinet",
            xyz=np.array([1.0, 1.0, 0.5]),
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            countable_instance=True,
        ),
    )
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(
            label="cabinet",
            xyz=np.array([1.02, 1.02, 0.5]),
            embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            countable_instance=True,
        ),
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 2, "different appearance (cosine 0) must stay separate"


def test_attach_siglip_crop_embeddings(monkeypatch):
    from emet.memory.graph_eqa.ingest.dynamem_graph_hooks import _attach_siglip_crop_embeddings

    class StubEnc:
        def encode_image(self, crop):
            m = np.mean(np.asarray(crop, dtype=np.float32), axis=(0, 1))
            return np.array([float(m[0]), float(m[1]), float(m[2])], dtype=np.float32)

    monkeypatch.setattr(
        "emet.perception.encoders.siglip_encoder.get_shared_mask_siglip_encoder",
        lambda **kw: StubEnc(),
    )
    cfg = GraphObjectFusionConfig(enabled=True)
    frame_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_rgb[0:4, 0:4] = (200, 20, 20)
    dets = [{"bbox_xyxy": (0, 0, 4, 4)}, {"bbox_xyxy": (None, None, None, None)}]
    out = _attach_siglip_crop_embeddings(cfg, frame_rgb, dets)
    assert "embedding" in out[0]
    assert out[0]["embedding"].shape == (3,)
    assert "embedding" not in out[1], "detections without a usable bbox get no embedding"
