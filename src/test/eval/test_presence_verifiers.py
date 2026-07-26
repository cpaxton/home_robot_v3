# Copyright (c) Chris Paxton 2026

import numpy as np

from emet.eval.presence_verifiers import (
    OwlV2PresenceDetector,
    crop_bbox,
    detector_crop_evidence,
)
from emet.eval.verifier_bakeoff import best_operating_point, binary_metrics


def test_crop_bbox_clips_and_pads():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    crop = crop_bbox(rgb, (0, 1, 6, 8))
    assert crop is not None
    assert crop.shape[0] >= 7
    assert crop.shape[1] >= 6


def test_owl_detector_selects_max_box():
    class Model:
        def detect_object(self, rgb, phrase, confidence_threshold=0.0):
            return np.array([0.1, 0.7]), np.array([[0, 0, 2, 2], [2, 3, 8, 9]])

    detector = OwlV2PresenceDetector(model=Model())
    evidence = detector.score(np.zeros((10, 10, 3), dtype=np.uint8), "basket")
    assert evidence.score == 0.7
    assert evidence.bbox_xyxy == (2, 3, 8, 9)


def test_detector_crop_siglip_uses_crop():
    class Detector:
        def score(self, rgb, phrase):
            from emet.eval.presence_verifiers import DetectionEvidence

            return DetectionEvidence("fake", phrase, 0.8, (2, 2, 8, 8), 1.0)

    class TensorLike:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Encoder:
        def encode_image(self, rgb):
            assert rgb.shape[0] >= 6 and rgb.shape[1] >= 6
            return TensorLike([[1.0, 0.0]])

        def encode_text(self, phrase):
            return TensorLike([[1.0, 0.0]])

    evidence = detector_crop_evidence(
        Detector(),
        Encoder(),
        np.zeros((10, 10, 3), dtype=np.uint8),
        "basket",
    )
    assert evidence.crop_siglip_sim == 1.0


def test_high_recall_operating_point():
    rows = [
        {"score": 0.9, "gt_in_view": True},
        {"score": 0.8, "gt_in_view": True},
        {"score": 0.7, "gt_in_view": False},
        {"score": 0.1, "gt_in_view": False},
    ]
    low = binary_metrics(rows, score_key="score", threshold=0.5)
    assert low["recall"] == 1.0
    best = best_operating_point(
        [
            low,
            binary_metrics(rows, score_key="score", threshold=0.85),
        ],
        min_recall=0.8,
    )
    assert best is not None
    assert best["threshold"] == 0.5
