# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from emet.core import get_parameters
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.attach import attach_graph_object_fusion


def test_attach_fusion_loads_default_yaml_when_parameters_empty():
    mem = GraphEQAMemory(defer_llm_clients=True)
    fusion = attach_graph_object_fusion(mem, parameters={})
    assert fusion is not None
    assert fusion.config.enabled is True


def test_attach_fusion_loads_dynav_innate_mars_from_parameters_object():
    mem = GraphEQAMemory(defer_llm_clients=True)
    params = get_parameters("dynav_innate_mars.yaml")
    fusion = attach_graph_object_fusion(mem, params)
    assert fusion is not None
    assert fusion.config.spatial_merge_xy_m == 0.48
    assert fusion.config.embedding_min_cosine == 0.0
    assert fusion.config.fallback_spatial_merge_xy_m == 0.55
    assert fusion.config.bounds_3d_iou_merge_min == 0.40
