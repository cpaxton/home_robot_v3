# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Live RGB-D ingest into graph memory.

- ``graph_mutate`` — ``add_observation`` / merge
- ``graph_observation_pipeline`` — instance rows → nodes
- ``instance_items`` — unpack detector/VLM records
- ``dynamem_graph_hooks`` — DynaMem/GraphEQA update tail
- ``sensor_graph_builder`` / ``lazy_graph_commit``
"""

from emet.memory.graph_eqa.ingest.instance_items import unpack_instance_item

__all__ = ["unpack_instance_item"]
