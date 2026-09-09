# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.memory.vlm_region_grounding import ground_vlm_region, region_depth_mask


def test_depth_support_excludes_background_and_disconnected_regions():
    depth = np.full((40, 40), 3.0)
    depth[10:30, 10:30] = 1.0
    depth[0:5, 0:5] = 1.0
    mask = region_depth_mask(depth, [0, 0, 1000, 1000], [500, 500], min_depth=0.2, max_depth=4)
    assert (mask == 0).sum() == 400
    assert mask[1, 1] == -1


@pytest.mark.parametrize(
    "box,point", [([0, 0, 1000, 1000], [2000, 500]), ([0, 0, 100, 100], [500, 500]), ([0, 0, 1000, 1000], [True, 500])]
)
def test_invalid_region_fails_closed(box, point):
    with pytest.raises(ValueError):
        region_depth_mask(np.ones((40, 40)), box, point, min_depth=0.2, max_depth=4)


def test_vlm_grounding_has_no_detector_dependency():
    frame = SimpleNamespace(
        rgb=np.zeros((40, 40, 3), dtype=np.uint8), depth=np.ones((40, 40)), full_world_xyz=np.ones((40, 40, 3))
    )
    client = Mock(return_value='{"verified":true,"box":[200,200,800,800],"point":[500,500]}')
    _, detections, ids, audit = ground_vlm_region(frame, "lamp", "lamp", client=client, min_depth=0.2, max_depth=4)
    assert ids == [0] and len(detections) == 1
    assert audit["geometry_source"] == "vlm_selected_depth_surface"
    client.return_value = '{"verified":false,"reason":"not visible"}'
    assert ground_vlm_region(frame, "lamp", "lamp", client=client, min_depth=0.2, max_depth=4)[2] == []


def test_invalid_depth_cannot_become_geometry():
    with pytest.raises(ValueError, match="invalid depth"):
        region_depth_mask(np.full((40, 40), np.nan), [0, 0, 1000, 1000], [500, 500], min_depth=0.2, max_depth=4)


def test_geometry_feedback_is_bounded_and_does_not_override_abstention():
    from emet.memory.vlm_region_grounding import select_supported_region

    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    depth = np.ones((40, 40))
    depth[0, 0] = np.nan
    client = Mock(
        side_effect=[
            '{"verified":true,"box":[0,0,1000,1000],"point":[0,0]}',
            '{"verified":true,"box":[0,0,1000,1000],"point":[500,500]}',
        ]
    )
    _, mask, audit = select_supported_region(rgb, depth, "mug", "mug", client=client, min_depth=0.2, max_depth=4)
    assert audit["valid"] and (mask == 0).sum() > 25
    assert client.call_count == 2
    assert len(client.call_args.args[0]) == 3
    assert "invalid depth" in client.call_args.args[0][0]
    client = Mock(return_value='{"verified":false}')
    assert not select_supported_region(rgb, depth, "mug", "mug", client=client, min_depth=0.2, max_depth=4)[2]["valid"]
    client.assert_called_once()
