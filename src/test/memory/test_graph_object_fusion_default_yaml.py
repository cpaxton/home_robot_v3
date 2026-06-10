# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.setup import attach_graph_object_fusion


def test_attach_fusion_loads_default_yaml_when_parameters_empty():
    mem = GraphEQAMemory(defer_llm_clients=True)
    fusion = attach_graph_object_fusion(mem, parameters={})
    assert fusion is not None
    assert fusion.config.enabled is True
