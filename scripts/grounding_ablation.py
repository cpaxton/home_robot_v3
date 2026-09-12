# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline-only first-box interventions; production defaults are untouched."""

import json
import math
import time

import numpy as np
from PIL import Image

from emet.eval.agentic_vlm_assess import _parse_json_object
from emet.memory.vlm_region_grounding import region_annotation


def expand_box(box, fraction):
    if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int or not 0 <= v <= 1000 for v in box):
        raise ValueError("invalid box")
    if box[2] <= box[0] or box[3] <= box[1] or not math.isfinite(fraction) or fraction < 0:
        raise ValueError("invalid box extent or padding")
    dx, dy = (box[2] - box[0]) * fraction, (box[3] - box[1]) * fraction
    return [
        max(0, math.floor(box[0] - dx)),
        max(0, math.floor(box[1] - dy)),
        min(1000, math.ceil(box[2] + dx)),
        min(1000, math.ceil(box[3] + dy)),
    ]


class BoxAblation:
    def __init__(self, client, rgb, query, variant):
        self.client, self.rgb, self.query, self.variant = client, rgb, query, variant
        self.first = True
        self.requests = []
        self.verification_images = {}

    def call(self, command, **kwargs):
        start = time.monotonic()
        raw = self.client(command, **kwargs)
        self.requests.append(
            {
                "prompt": command[0],
                "system_prompt": kwargs.get("system_prompt"),
                "image_count": len(command) - 1,
                "raw": raw,
                "elapsed_s": time.monotonic() - start,
            }
        )
        return raw

    def __call__(self, command, **kwargs):
        if not self.first:
            return self.call(command, **kwargs)
        self.first = False
        command = list(command)
        if self.variant == "whole_object":
            command[0] += (
                " Enclose the ENTIRE visible extent of the requested object, including its "
                "top, bottom, left and right edges. Do not return only one face or a small surface patch."
            )
        raw = self.call(command, **kwargs)
        parsed = _parse_json_object(raw)
        if parsed.get("verified") is not True:
            return raw
        try:
            box = expand_box(parsed.get("box"), 0.0)
        except ValueError:
            return raw
        if self.variant == "expand_25":
            return json.dumps({**parsed, "box": expand_box(box, 0.25)})
        if self.variant not in ("verify", "repair"):
            return raw
        h, w = self.rgb.shape[:2]
        x0, y0, x1, y1 = np.array(box) * [w, h, w, h] / 1000
        crop = Image.fromarray(self.rgb).crop((math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)))
        self.verification_images = {"verify-box": region_annotation(self.rgb, parsed), "verify-crop": crop}
        prompt = (
            f"Check a proposed localization of {self.query!r}. Image 1 is the original, image 2 marks "
            "the proposed box in yellow, image 3 is its unmarked crop. The proposal may be wrong. "
            "Check that it selects the requested object rather than another object or background, "
            "and encloses its entire visible extent. Do not infer correctness from the annotation. "
            'Return JSON {"verified":true or false,"reason":"..."}.'
        )
        if self.variant == "repair":
            prompt += (
                " If the proposal is wrong but the target is unambiguous in image 1, return one corrected "
                'box with verified:true and "box":[xmin,ymin,xmax,ymax], normalized 0..1000 in the ORIGINAL '
                "image coordinates. Otherwise abstain."
            )
        verdict = _parse_json_object(
            self.call([prompt, Image.fromarray(self.rgb), region_annotation(self.rgb, parsed), crop], **kwargs)
        )
        if verdict.get("verified") is not True:
            return json.dumps({"verified": False, "reason": "box verification rejected", "verification": verdict})
        if self.variant == "repair" and "box" in verdict:
            try:
                box = expand_box(verdict["box"], 0)
            except ValueError:
                return json.dumps({"verified": False, "reason": "invalid correction"})
        return json.dumps({**parsed, "box": box, "verification": verdict})
