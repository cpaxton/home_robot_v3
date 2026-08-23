# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from emet.perception.encoders import encoders, get_encoder

# Paths relative to project root (pytest runs from project root)
_DOCS = Path(__file__).resolve().parent.parent.parent.parent / "docs"
images = [
    str(_DOCS / "object.png"),
    str(_DOCS / "receptacle.png"),
]


def test_get_encoder():
    encoder = get_encoder("clip", {})
    assert encoder is not None
    assert encoder.__class__.__name__ == "ClipEncoder"

    encoder = get_encoder("normalized_clip", {})
    assert encoder is not None
    assert encoder.__class__.__name__ == "NormalizedClipEncoder"

    encoder = get_encoder("siglip", {})
    assert encoder is not None
    assert encoder.__class__.__name__ == "SiglipEncoder"

    with pytest.raises(ValueError):
        get_encoder("invalid_encoder", {})


@pytest.mark.parametrize("encoder_name", encoders)
def test_get_encoder_all(encoder_name):
    print(f"Testing encoder: {encoder_name}")
    try:
        encoder = get_encoder(encoder_name, {})
    except Exception as e:
        # DINOv3 weights are gated on Hugging Face; skip when the env has no access.
        err = f"{type(e).__name__}: {e}"
        if encoder_name == "dinov3" and (
            "GatedRepo" in err or "403" in err or "gated" in err.lower() or "huggingface.co" in err.lower()
        ):
            pytest.skip(f"DINOv3 weights unavailable in this environment: {e}")
        raise
    assert encoder is not None

    with pytest.raises(ValueError):
        get_encoder("invalid_encoder", {})

    for image_path in images:
        if not Path(image_path).exists():
            pytest.skip(f"Test image not found: {image_path}")
        print(f"Testing encoder: {encoder_name} with image: {image_path}")
        encoder = get_encoder(encoder_name, {})
        assert encoder is not None

        image = Image.open(image_path)
        np_image = np.asarray(image)
        encoded = encoder.encode_image(np_image)
        assert encoded is not None
        assert isinstance(encoded, torch.Tensor)
        assert len(encoded) > 0
        print(f"Encoded: {encoded}")


if __name__ == "__main__":
    test_get_encoder()
    for enc in encoders:
        test_get_encoder_all(enc)
