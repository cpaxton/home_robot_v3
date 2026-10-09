# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import Mock

import numpy as np
import pytest

from emet.perception.detection.query_mask_proposals import refine_instance_proposals


def test_detector_proposes_boxes_but_segmenter_owns_masks():
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    instances = np.full((20, 30), -1)
    instances[3:8, 10:17] = 4
    segmenter = Mock()
    result = refine_instance_proposals(rgb, instances, segmenter)
    assert result is segmenter.segment.return_value
    np.testing.assert_array_equal(segmenter.segment.call_args.args[1], [[10, 3, 17, 8]])
    instances[:] = -1
    refine_instance_proposals(rgb, instances, segmenter)
    assert segmenter.segment.call_args.args[1].shape == (0, 4)
    with pytest.raises(ValueError, match="aligned integer"):
        refine_instance_proposals(rgb, instances.astype(float), segmenter)
