# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Detector selection, label alignment and CPU device contracts."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.mapping.scene_graph.processor import SceneGraphProcessor


@pytest.mark.parametrize('primary', ['owlsam', 'sam2'])
def test_requested_primary_is_not_replaced_by_fallback(monkeypatch, primary):
    processor = SceneGraphProcessor(config={'segmentation': {'primary': primary, 'fallback': 'sam2'}}, device='cpu')
    build = Mock(return_value='detector')
    monkeypatch.setattr(processor, '_build_fallback_segmenter', build)
    assert processor.segmenter == 'detector'
    assert build.call_args.args[0] == primary


def test_owl_class_ids_preserve_detection_order_and_background():
    instances = np.full((20, 20), -1)
    instances[:, :10] = 0
    instances[:, 10:] = 1
    detector = SimpleNamespace(texts=['a photo of a table', 'a photo of a cup'],
        predict=lambda *a, **kw: (np.zeros((20, 20)), instances,
            {'instance_classes': [1, 0], 'instance_scores': [.8, .7]}))
    processor = SceneGraphProcessor(config={}, device='cpu', segmenter=detector)
    result = processor._run_segmentation(np.zeros((20, 20, 3), np.uint8), np.ones((20, 20)), [])
    assert [d['label'] for d in result] == ['cup', 'table']
    assert [d['score'] for d in result] == [.8, .7]
    instances[:] = -1
    assert processor._run_segmentation(np.zeros((20, 20, 3), np.uint8), np.ones((20, 20)), []) == []


def test_sam2_builds_once_on_requested_cpu_even_with_cuda_available(monkeypatch):
    import emet.perception.detection.sam2.sam2_perception as module

    model = object()
    build = Mock(return_value=model)
    monkeypatch.setattr(module, 'build_sam2', build)
    monkeypatch.setattr(module.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: True)
    predictor = Mock(return_value='image-predictor')
    monkeypatch.setattr(module, 'SAM2ImagePredictor', predictor)
    monkeypatch.setattr(module, 'SAM2AutomaticMaskGenerator', lambda m: m)
    result = module.SAM2Perception(device='cpu')
    assert build.call_count == 1
    assert str(build.call_args.kwargs['device']) == 'cpu'
    assert result.mask_generator is model
    predictor.assert_called_once_with(model)


def test_no_owl_detections_never_create_a_whole_image_instance():
    import torch

    from emet.perception.detection.owl.owlsam_perception import OWLSAMProcessor

    class Inputs:
        image_processor = SimpleNamespace(post_process_object_detection=lambda *a, **kw: [{'boxes': torch.empty((0, 4))}])
        def __call__(self, **kwargs):
            return {}
    detector = object.__new__(OWLSAMProcessor)
    detector.texts = ['a photo of a cup']
    detector.processor = Inputs()
    detector.model = lambda **kwargs: None
    detector.device = 'cpu'
    detector.confidence_threshold = .1
    _, instances, metadata = detector.predict(np.zeros((20, 20, 3), np.uint8))
    assert (instances == -1).all()
    assert metadata == {}
