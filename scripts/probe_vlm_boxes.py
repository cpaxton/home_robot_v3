# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Probe VLM bounding-box localization for EQA objects.

Asks a VLM to localize a target object in one or more images and saves overlay
PNGs + a JSON report. Supports the two output conventions we care about:

* ``qwen3_vl`` / ``qwen2_5_vl``: native grounding — absolute-pixel JSON
  ``[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]``.
* ``gemma4``: prompted boxes, Gemini convention — normalized 0-1000
  ``[ymin, xmin, ymax, xmax]``.

Usage:
  python scripts/probe_vlm_boxes.py --family qwen3_vl --hf-id Qwen/Qwen3-VL-8B-Instruct \
      --target "woven basket" --out /tmp/bbox_probe image1.png image2.png
  python scripts/probe_vlm_boxes.py --self-test   # parser unit checks, no model
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click


def parse_qwen_bboxes(text: str) -> list[tuple[str, float, float, float, float]]:
    """Parse Qwen-VL grounding output: absolute-pixel ``bbox_2d`` JSON objects."""
    out: list[tuple[str, float, float, float, float]] = []
    # Qwen wraps JSON in ```json fences or emits bare lists; grab every bbox_2d object.
    for m in re.finditer(r'\{[^{}]*"bbox_2d"[^{}]*\}', text or ""):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        box = obj.get("bbox_2d")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = (float(v) for v in box)
            out.append((str(obj.get("label", "")), x1, y1, x2, y2))
    return out


def parse_gemma_bboxes(text: str, img_w: int, img_h: int) -> list[tuple[str, float, float, float, float]]:
    """Parse Gemini-convention boxes: normalized 0-1000 ``[ymin, xmin, ymax, xmax]``."""
    out: list[tuple[str, float, float, float, float]] = []
    for m in re.finditer(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text or ""):
        ymin, xmin, ymax, xmax = (float(v) for v in m.groups())
        if max(ymin, xmin, ymax, xmax) > 1000:
            continue
        out.append(
            (
                "",
                xmin / 1000.0 * img_w,
                ymin / 1000.0 * img_h,
                xmax / 1000.0 * img_w,
                ymax / 1000.0 * img_h,
            )
        )
    return out


def _self_test() -> int:
    qwen = '```json\n[{"bbox_2d": [12, 34, 200, 300], "label": "basket"}]\n```'
    boxes = parse_qwen_bboxes(qwen)
    assert boxes == [("basket", 12.0, 34.0, 200.0, 300.0)], boxes
    assert parse_qwen_bboxes("no boxes here") == []

    gemma = "The basket is at [100, 250, 400, 600]."
    boxes = parse_gemma_bboxes(gemma, img_w=1000, img_h=500)
    # ymin=100,xmin=250,ymax=400,xmax=600 over 1000x500 -> x1=250,y1=50,x2=600,y2=200
    assert boxes == [("", 250.0, 50.0, 600.0, 200.0)], boxes
    assert parse_gemma_bboxes("[9999, 1, 2, 3]", 100, 100) == []
    print("self-test OK")
    return 0


@click.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--family", default="qwen3_vl", type=click.Choice(["qwen3_vl", "qwen2_5_vl", "gemma4"]))
@click.option("--hf-id", default=None, help="HF checkpoint (family default when omitted)")
@click.option("--quant", default="int4")
@click.option("--target", required=False, help="Object to localize, e.g. 'woven basket'")
@click.option("--out", default="/tmp/bbox_probe", type=click.Path())
@click.option("--self-test", is_flag=True, default=False, help="Run parser checks only (no model)")
def main(images, family, hf_id, quant, target, out, self_test):
    if self_test:
        raise SystemExit(_self_test())
    if not images or not target:
        raise click.UsageError("IMAGES and --target are required unless --self-test")

    from PIL import Image, ImageDraw

    from emet.llms.vllm_factory import create_dynamem_vllm

    client = create_dynamem_vllm(
        family, hf_model_id=hf_id, vl_model_size="8B", max_tokens=256, device="cuda", quantization=quant
    )
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for path in images:
        img = Image.open(path).convert("RGB")
        if family.startswith("qwen"):
            prompt = (
                f"Locate the {target} in this image and output its bounding box in JSON "
                'format: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
                "If it is not visible, output []."
            )
        else:
            prompt = (
                f"Detect the {target} in this image. Output a bounding box as "
                "[ymin, xmin, ymax, xmax] with coordinates normalized to 0-1000. "
                "If it is not visible, output NONE."
            )
        reply = client.generate_multimodal([prompt, img], system_prompt=None, max_new_tokens=128, reset_context=True)
        if family.startswith("qwen"):
            boxes = parse_qwen_bboxes(reply)
        else:
            boxes = parse_gemma_bboxes(reply, img.width, img.height)
        draw = ImageDraw.Draw(img)
        for label, x1, y1, x2, y2 in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), label or target, fill=(255, 0, 0))
        overlay = out_dir / f"{Path(path).stem}_{family}_boxes.png"
        img.save(overlay)
        report.append({"image": str(path), "reply": reply[:500], "boxes": boxes, "overlay": str(overlay)})
        print(f"{path}: {len(boxes)} box(es) -> {overlay}")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
