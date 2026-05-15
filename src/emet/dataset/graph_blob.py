# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Build :class:`emet.memory.format.GraphBlob` from simulator GT object dicts."""

from __future__ import annotations

from typing import Any

from emet.memory.format import GraphBlob, GraphNodeView


def gt_object_dicts_to_graph_blob(objects: list[dict[str, Any]]) -> GraphBlob:
    """Floor node + one node per GT object (labels = body name), no edges."""
    FLOOR_NODE_ID = 0
    nodes: list[GraphNodeView] = [
        GraphNodeView(
            node_id=FLOOR_NODE_ID,
            labels=["floor"],
            xyz=[0.0, 0.0, 0.0],
            obs_id=0,
            description=None,
        )
    ]
    for idx, obj in enumerate(objects):
        body = str(obj.get("body_name") or obj.get("name") or f"object_{idx}")
        pos = obj.get("pos_xyz") or [0.0, 0.0, 0.0]
        xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
        nodes.append(
            GraphNodeView(
                node_id=idx + 1,
                labels=[body],
                xyz=xyz,
                obs_id=idx + 1,
                description="sim_gt",
            )
        )
    return GraphBlob(nodes=nodes, edges=[])
