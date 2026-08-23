# Copyright (c) Chris Paxton 2026

import numpy as np
import torch

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


class _DenseSimEncoder:
    """Fake encoder whose dense sim peaks at a known patch so the crop is testable."""

    def __init__(self, peak_patch: int, n_patches: int):
        self._peak = peak_patch
        self._n = n_patches
        peak = peak_patch
        n = n_patches
        self.head_calls = 0

        class _VM:
            def __init__(self):
                self.head = type(
                    "_Head",
                    (),
                    {
                        "attention": object(),
                        "layernorm": torch.nn.Identity(),
                        "mlp": torch.nn.Identity(),
                    },
                )()
                self.embeddings = type(
                    "_Embeddings",
                    (),
                    {"patch_embedding": lambda _self, _pixels: torch.zeros(1, 8, int(n**0.5), int(n**0.5))},
                )()

            def __call__(self, pixel_values, output_hidden_states=False):
                return self.forward(pixel_values, output_hidden_states=output_hidden_states)

            def forward(self, pixel_values, output_hidden_states=False):
                sims = -torch.ones(1, n, 8)
                sims[0, peak, :] = 1.0
                out = self
                out.last_hidden_state = sims
                return out

        class _Model:
            def __init__(self):
                self.vision_model = _VM()

        self._model = _Model()

    def forward_one_block_(self, _attention, features):
        self.head_calls += 1
        return features

    def encode_text(self, phrase):
        return torch.ones(8)

    def _to_model_inputs(self, inputs):
        return inputs

    def processor(self, **kwargs):
        class _P(dict):
            pixel_values = None

        p = _P()
        p["pixel_values"] = None
        p.pixel_values = None
        return p

    @property
    def model(self):
        return self._model


def test_dense_siglip_argmax_crop_returns_crop():
    from emet.eval.presence_verifiers import dense_siglip_argmax_crop

    # SigLIP has 16 spatial patches on a 4x4 grid and no CLS token. Peak patch 0
    # ensures an accidental [1:] slice would discard the only positive match.
    encoder = _DenseSimEncoder(peak_patch=0, n_patches=16)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    crop = dense_siglip_argmax_crop(encoder, rgb, "clock", patch_frac=0.5)
    assert crop is not None
    crop_img, sim = crop
    assert crop_img.ndim == 3
    assert sim > 0.0
    assert encoder.head_calls == 1


def test_dense_siglip_argmax_crop_none_on_failure():
    from emet.eval.presence_verifiers import dense_siglip_argmax_crop

    class BadEncoder:
        def encode_text(self, phrase):
            raise RuntimeError("boom")

    assert dense_siglip_argmax_crop(BadEncoder(), np.zeros((16, 16, 3), dtype=np.uint8), "x") is None


def test_dense_siglip_patch_sims_reject_dimension_mismatch():
    """Image head features and text projection must share a dimension; mismatch -> None."""
    from emet.eval.presence_verifiers import dense_siglip_patch_similarities

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    encoder = _DenseSimEncoder(peak_patch=0, n_patches=16)
    aligned = dense_siglip_patch_similarities(encoder, rgb, "clock")
    assert aligned is not None  # 8-dim head features == 8-dim text projection

    wrong = _DenseSimEncoder(peak_patch=0, n_patches=16)
    wrong.encode_text = lambda _phrase: torch.ones(32)  # noqa: E731
    assert dense_siglip_patch_similarities(wrong, rgb, "clock") is None


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
