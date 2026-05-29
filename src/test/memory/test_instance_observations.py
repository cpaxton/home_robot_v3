# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emet.memory.graph_eqa.instance_observations import (
    frame_instances_to_labels_xyz,
    label_for_detection_category,
)


class _MockYoloEVocab:
    class_list = [f"cls_{i}" for i in range(200)]


def test_label_for_detection_category_uses_class_list():
    m = _MockYoloEVocab()
    assert label_for_detection_category(m, 3) == "cls_3"
    assert label_for_detection_category(m, 199) == "cls_199"


def test_label_for_detection_category_out_of_range():
    m = _MockYoloEVocab()
    assert label_for_detection_category(m, 500) == "class_500"


def test_frame_instances_to_labels_xyz_medians_and_labels():
    h, w = 8, 8
    inst = torch.full((h, w), -1, dtype=torch.long)
    inst[0:4, 0:4] = 0
    inst[4:8, 4:8] = 1
    fw = torch.zeros(h, w, 3, dtype=torch.float32)
    fw[0:4, 0:4] = torch.tensor([1.0, 2.0, 3.0])
    fw[4:8, 4:8] = torch.tensor([10.0, 20.0, 30.0])
    depth = torch.ones(h, w) * 0.5
    classes = torch.tensor([2, 5], dtype=torch.long)
    frame = SimpleNamespace(
        instance=inst,
        full_world_xyz=fw,
        depth=depth,
        instance_classes=classes,
    )
    out = frame_instances_to_labels_xyz(
        frame,
        min_depth=0.01,
        max_depth=10.0,
        detection_model=_MockYoloEVocab(),
        min_points=4,
    )
    assert len(out) == 2
    by_label = {label: xyz for label, xyz, _bbox in out}
    assert np.allclose(by_label["cls_2"], [1.0, 2.0, 3.0])
    assert np.allclose(by_label["cls_5"], [10.0, 20.0, 30.0])
    bboxes = {label: bbox for label, _xyz, bbox in out}
    assert bboxes["cls_2"] == (0, 0, 4, 4)
    assert bboxes["cls_5"] == (4, 4, 8, 8)


def test_frame_instances_to_labels_xyz_skips_small_mask():
    h, w = 4, 4
    inst = torch.zeros((h, w), dtype=torch.long)
    fw = torch.ones(h, w, 3)
    depth = torch.ones(h, w) * 0.5
    classes = torch.tensor([0], dtype=torch.long)
    frame = SimpleNamespace(instance=inst, full_world_xyz=fw, depth=depth, instance_classes=classes)
    out = frame_instances_to_labels_xyz(
        frame,
        min_depth=0.01,
        max_depth=10.0,
        detection_model=_MockYoloEVocab(),
        min_points=100,
    )
    assert out == []


@pytest.mark.parametrize("missing", ["instance", "full_world_xyz", "depth"])
def test_frame_instances_to_labels_xyz_missing_fields(missing):
    h, w = 4, 4
    inst = torch.zeros((h, w), dtype=torch.long)
    fw = torch.ones(h, w, 3)
    depth = torch.ones(h, w)
    d = {"instance": inst, "full_world_xyz": fw, "depth": depth}
    if missing == "instance":
        d["instance"] = None
    elif missing == "full_world_xyz":
        d["full_world_xyz"] = None
    else:
        d["depth"] = None
    frame = SimpleNamespace(**d)
    assert frame_instances_to_labels_xyz(
        frame, min_depth=0.01, max_depth=10.0, detection_model=_MockYoloEVocab()
    ) == []
