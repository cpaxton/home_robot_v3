# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os

import torch
from numpy import ndarray
from PIL import Image, ImageDraw
from torch import Tensor

from emet.llms.eqa_qwen import get_shared_qwen35_vl_client

# pip install flash-attn

# Legacy size labels → Qwen3.5 HF ids (``Qwen/Qwen3.5-{size}``)
_LEGACY_VL_SIZE = {"3B": "2B", "7B": "9B", "8B": "9B", "32B": "27B", "72B": "27B"}


def _qwen35_vl_size(model_size: str) -> str:
    return _LEGACY_VL_SIZE.get(model_size, model_size)


class QwenCaptioner:
    """Image captioner using the same shared Qwen3.5 multimodal model as EQA when possible."""

    def __init__(
        self,
        model_size: str = "2B",
        max_length: int = 200,
        num_beams: int = 1,
        device: str | None = None,
        image_shape=None,
        draw_on_image=True,
    ):
        """Initialize the Qwen3.5 image captioner.

        Args:
            model_size: Qwen3.5 size (e.g. 2B, 4B, 9B). Legacy Qwen3-VL labels are mapped.
            max_length (int, optional): Maximum length of the generated caption. Defaults to 100.
            num_beams (int, optional): Number of beams for beam search. Defaults to 1.
            device (str, optional): Device to run the model on. Defaults to None (auto-detect).
        """
        self.max_length = max_length
        self.num_beams = num_beams
        self.image_shape = image_shape
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        if device is not None:
            td = torch.device(device)
            if td.type == "cuda":
                vl_device = "cuda"
            elif td.type == "mps":
                vl_device = "mps"
            else:
                raise RuntimeError(
                    "Qwen3.5 multimodal captioner requires CUDA or MPS.",
                )
        elif torch.cuda.is_available():
            vl_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            vl_device = "mps"
        else:
            raise RuntimeError(
                "Qwen3.5 multimodal captioner requires a GPU (CUDA or Apple MPS).",
            )

        self._caption_model_size = _qwen35_vl_size(model_size)
        self._vl_device = vl_device

        self.draw_on_image = draw_on_image

    def caption_image(
        self,
        image: ndarray | Tensor | Image.Image,
        bbox: list | Tensor | ndarray | None = None,
        verbose: bool = False,
    ) -> str:
        """Generate a caption for the given image.

        Args:
            image (Union[ndarray, Tensor, Image.Image]): The input image.
            bbox: Provide a bounding box if you just want to model to tell what is inside the box
            verbose: Set to True if you want to print some debug log

        Returns:
            str: The generated caption.
        """
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, Tensor):
                _image = image.cpu().numpy()
            else:
                _image = image
            pil_image = Image.fromarray(_image)

        if self.image_shape is not None:
            h, w = pil_image.size
            pil_image = pil_image.resize(self.image_shape)
            if bbox is not None:
                h1, w1 = self.image_shape
                bbox[0] = bbox[0] * h1 // h
                bbox[1] = bbox[1] * w1 // w
                bbox[2] = bbox[2] * h1 // h
                bbox[3] = bbox[3] * w1 // w
        if self.draw_on_image and bbox is not None:
            h, w = pil_image.size
            bbox[0] = max(1, bbox[0])
            bbox[1] = max(1, bbox[1])
            bbox[2] = min(h - 2, bbox[2])
            bbox[3] = min(w - 2, bbox[3])
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle(bbox, outline="red", width=1)

        if bbox is None:
            prompt = "Describe the image."
        elif self.draw_on_image:
            prompt = "Describe the object in the red bounding box."
        else:
            prompt = "Describe the object in the box " + str(bbox)

        messages = [
            pil_image,
            prompt,
            "Limit your answer in 10 words. E.G. a yellow banana; a white hand sanitizer",
        ]

        vl = get_shared_qwen35_vl_client(
            model_size=self._caption_model_size,
            device=self._vl_device,
            quantization="int4",
        )
        output_text = vl(
            messages,
            verbose=verbose,
            system_prompt=None,
            max_new_tokens=self.max_length,
        )

        if bbox is not None:
            if not self.draw_on_image:
                draw = ImageDraw.Draw(pil_image)
                draw.rectangle(bbox, outline="red", width=2)
            if not os.path.exists("test_caption/"):
                os.makedirs("test_caption")
            pil_image.save("test_caption/" + output_text + ".jpg")

        return output_text


if __name__ == "__main__":
    from pathlib import Path

    captioner = QwenCaptioner()
    example_path = Path(__file__).parent / "example.jpg"
    caption = captioner.caption_image(Image.open(example_path), verbose=True)
    print("caption for the image:", caption)
