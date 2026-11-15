# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Bound image attachments without rewriting dialogue or choosing robot actions.

The episode keeps its original messages and camera snapshots. This module makes
a provider-independent request copy: current attachments are mandatory, then
recent distinct historical pixels fill the remaining slots. Every omitted image
keeps a stable, recallable ID at its original place in the dialogue. Identical
pixels do not imply identical capture time, pose, or current object visibility.
"""

import hashlib
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from PIL import Image


@dataclass
class ImagePacket:
    messages: list[dict]
    audit: dict


class ImageContext:
    def __init__(self, max_images: int | None = 8):
        if max_images is not None and (type(max_images) is not int or max_images < 1):
            raise ValueError("max_images must be a positive integer or None (full-context ablation)")
        self.max_images = max_images
        self._scope = uuid4().hex[:12]
        self._by_pixels: dict[str, str] = {}
        self.images: dict[str, Image.Image | np.ndarray] = {}

    def _register(self, image):
        pixels = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError("Image context expects RGB uint8 camera images")
        digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if digest not in self._by_pixels:
            view_id = f"view_{self._scope}_{len(self.images) + 1:04d}"
            self._by_pixels[digest] = view_id
            self.images[view_id] = image
        return self._by_pixels[digest], digest, list(pixels.shape[:2])

    def pack(self, messages: list[dict]) -> ImagePacket:
        occurrences = []
        for turn, message in enumerate(messages):
            content = message["content"]
            if not isinstance(content, list):
                continue
            slot = 0
            for part_index, part in enumerate(content):
                if isinstance(part, (Image.Image, np.ndarray)):
                    slot += 1
                    view_id, digest, shape = self._register(part)
                    occurrences.append(
                        {
                            "message_index": turn,
                            "part_index": part_index,
                            "image_slot": slot,
                            "view_id": view_id,
                            "pixel_sha256": digest,
                            "shape_hw": shape,
                        }
                    )
        current = [o for o in occurrences if o["message_index"] == len(messages) - 1]
        if self.max_images is not None and len(current) > self.max_images:
            raise ValueError("Current/requested images exceed max_images; no current evidence was silently dropped")
        selected = {(o["message_index"], o["part_index"]) for o in current}
        seen = {o["view_id"] for o in current}
        for occurrence in reversed(occurrences):
            position = (occurrence["message_index"], occurrence["part_index"])
            if position in selected:
                continue
            if self.max_images is None or (occurrence["view_id"] not in seen and len(selected) < self.max_images):
                selected.add(position)
                seen.add(occurrence["view_id"])
        lookup = {(o["message_index"], o["part_index"]): o for o in occurrences}
        packed = []
        for turn, message in enumerate(messages):
            content = message["content"]
            if not isinstance(content, list):
                packed.append(dict(message))
                continue
            parts = []
            for index, part in enumerate(content):
                occurrence = lookup.get((turn, index))
                if occurrence is None:
                    parts.append(part)
                    continue
                attached = (turn, index) in selected
                occurrence["decision"] = (
                    "attached" if attached else ("repeated_pixels" if occurrence["view_id"] in seen else "image_budget")
                )
                label = (
                    f"[Image {occurrence['image_slot']} in this message; view_id={occurrence['view_id']}; "
                    f"{'latest request attachment' if turn == len(messages) - 1 else 'historical attachment'}; "
                )
                if attached:
                    label += "pixels attached. This is an image identity, not a claim of freshness or object geometry.]"
                    if parts and isinstance(parts[-1], str):
                        parts[-1] += "\n" + label
                    else:
                        parts.append(label)
                    parts.append(part)
                else:
                    parts.append(
                        label + "pixels omitted from this request. Use recall_view(view_id) to see them again.]"
                    )
            packed.append({**message, "content": parts})
        return ImagePacket(
            packed,
            {
                "policy": "recent_unique",
                "max_images": self.max_images,
                "input_images": len(occurrences),
                "unique_pixel_images": len({o["view_id"] for o in occurrences}),
                "attached_images": len(selected),
                "occurrences": occurrences,
                "attached_pixels": sum(
                    o["shape_hw"][0] * o["shape_hw"][1]
                    for o in occurrences
                    if (o["message_index"], o["part_index"]) in selected
                ),
                "text_and_tool_results": "preserved; image annotations added; original dialogue retained",
            },
        )
