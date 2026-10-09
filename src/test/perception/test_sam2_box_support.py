# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("sam2")
from emet.perception.detection.sam2 import sam2_perception as sam


def test_constructor_shares_one_model_and_honors_device(monkeypatch):
    monkeypatch.setattr(sam.os.path, "exists", lambda _: True)
    monkeypatch.setattr(sam.torch.cuda, "is_available", lambda: False)
    builder = Mock()
    prompted = Mock()
    automatic = Mock()
    monkeypatch.setattr(sam, "build_sam2", builder)
    monkeypatch.setattr(sam, "SAM2ImagePredictor", prompted)
    monkeypatch.setattr(sam, "SAM2AutomaticMaskGenerator", automatic)
    sam.SAM2Perception(configuration="s")
    builder.assert_called_once()
    assert str(builder.call_args.kwargs["device"]) == "cpu"
    prompted.assert_called_once_with(builder.return_value)
    automatic.assert_called_once_with(builder.return_value)


def test_box_support_empty_invalid_and_best_mask():
    segmenter = sam.SAM2Perception.__new__(sam.SAM2Perception)
    segmenter.sam_predictor = Mock()
    rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    assert segmenter.segment(rgb, np.empty((0, 4))).shape == (0, 30, 40)
    segmenter.sam_predictor.set_image.assert_not_called()
    with pytest.raises(ValueError):
        segmenter.segment(rgb, np.array([[4, 0, 2, 3]]))
    masks = np.zeros((3, 30, 40), dtype=np.float32)
    masks[1, 5:20, 6:30] = True
    segmenter.sam_predictor.predict.return_value = (masks, np.array([0.1, 0.9, 0.2]), None)
    actual = segmenter.segment(rgb, np.array([[2, 3, 20, 25]]))
    assert actual.dtype == bool
    assert np.array_equal(actual[0], masks[1])
    masks[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="binary masks"):
        segmenter.segment(rgb, np.array([[2, 3, 20, 25]]))
