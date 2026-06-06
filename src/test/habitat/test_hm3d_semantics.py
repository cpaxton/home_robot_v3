# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.core.interfaces import Observations
from emet.habitat.hm3d_semantics import (
    Hm3dSemanticLabeler,
    _instance_index_from_object_id,
    hm3d_instance_items_from_obs,
)
import numpy as np


class _FakeCategory:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name

    def index(self) -> int:
        return 0


class _FakeObject:
    def __init__(self, object_id: str, category_name: str):
        self.id = object_id
        self.category = _FakeCategory(category_name)


class _FakeScene:
    def __init__(self, objects):
        self.objects = objects


def test_instance_index_from_object_id():
    assert _instance_index_from_object_id("lamp_42") == 42
    assert _instance_index_from_object_id("Unknown_0") == 0
    assert _instance_index_from_object_id("no_suffix") is None


def test_hm3d_semantic_labeler_skips_structural():
    scene = _FakeScene(
        [
            _FakeObject("lamp_5", "lamp"),
            _FakeObject("wall_6", "wall"),
            _FakeObject("bed_7", "bed"),
        ]
    )
    labeler = Hm3dSemanticLabeler.from_semantic_scene(scene)
    assert labeler is not None
    assert labeler.instance_to_label[5] == "lamp"
    assert labeler.instance_to_label[7] == "bed"
    assert 6 not in labeler.instance_to_label


def test_hm3d_instance_items_from_obs():
    scene = _FakeScene(
        [
            _FakeObject("lamp_5", "lamp"),
            _FakeObject("bed_7", "bed"),
        ]
    )
    labeler = Hm3dSemanticLabeler.from_semantic_scene(scene)
    sem = np.zeros((4, 4), dtype=np.uint32)
    sem[:, :2] = 5
    sem[:, 2:] = 7
    depth = np.ones((4, 4), dtype=np.float32) * 2.0
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        semantic=sem,
        camera_K=np.eye(3),
        camera_pose=np.eye(4),
    )
    items = hm3d_instance_items_from_obs(labeler, obs, min_pixels=1)
    labels = {lab for lab, _ in items}
    assert "lamp" in labels or "bed" in labels
