# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Low-confidence detector boxes refined by segmentation, never semantic acceptance."""

import numpy as np


def refine_instance_proposals(rgb, instances, segmenter):
    instances = np.asarray(instances)
    if instances.shape != rgb.shape[:2] or not np.issubdtype(instances.dtype, np.integer):
        raise ValueError("Instance proposals must be an aligned integer map")
    ids = np.unique(instances[instances >= 0])
    if len(ids) > 8:
        raise ValueError("Too many proposals; request another view")
    boxes = []
    for instance_id in ids:
        yy, xx = np.where(instances == instance_id)
        boxes.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
    return segmenter.segment(rgb, np.asarray(boxes).reshape(-1, 4))
