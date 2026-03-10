# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from pathlib import Path

import pytest
from PIL import Image

from emet.perception.captioners import get_captioner

captioners = ["qwen", "blip"]

# Paths relative to project root
_DOCS = Path(__file__).resolve().parent.parent.parent.parent / "docs"
images = [str(_DOCS / "object.png"), str(_DOCS / "receptacle.png")]


def test_get_captioner():

    captioner = get_captioner("qwen", {})
    assert captioner is not None
    assert captioner.__class__.__name__ == "QwenCaptioner"

    captioner = get_captioner("blip", {})
    assert captioner is not None
    assert captioner.__class__.__name__ == "BlipCaptioner"

    with pytest.raises(ValueError):
        get_captioner("invalid_captioner", {})


@pytest.mark.parametrize("captioner_name", captioners)
def test_get_captioner_all(captioner_name):
    captioner = get_captioner(captioner_name, {})
    assert captioner is not None

    with pytest.raises(ValueError):
        get_captioner("invalid_captioner", {})

    # Test captioning on the two images
    for image_path in images:
        if not Path(image_path).exists():
            pytest.skip(f"Test image not found: {image_path}")
        captioner = get_captioner(captioner_name, {})
        assert captioner is not None

        image = Image.open(image_path)
        caption = captioner.caption_image(image)
        assert caption is not None
        assert isinstance(caption, str)
        assert len(caption) > 0
        print(f"Caption: {caption}")


if __name__ == "__main__":
    test_get_captioner()
    test_get_captioner_all("git")
    test_get_captioner_all("qwen")
    print("All tests passed!")
