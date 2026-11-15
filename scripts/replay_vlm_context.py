#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Replay a recorded client request through image budgeting, without model calls."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from emet.eval.eqa_decision_trace import record_vlm_request
from emet.llms.conversation import VLMConversation


def replay(request_path: Path, output: Path, *, max_images=8, max_characters=64000):
    if output.exists():
        raise ValueError("Use a fresh output directory; recorded evidence is never overwritten")
    source = json.loads(request_path.read_text())
    root = request_path.parent.resolve()

    def decode(blocks):
        parts = []
        for block in blocks:
            if block["type"] == "text":
                parts.append(block["text"])
            elif block["type"] == "image":
                path = (root / block["path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Recorded image path escapes its request directory")
                with Image.open(path) as image:
                    parts.append(image.convert("RGB").copy())
            else:
                raise ValueError("Unsupported recorded content type")
        return parts

    conversation = VLMConversation(max_images=max_images, max_characters=max_characters)
    for message in source.get("conversation_history", []):
        content = message["content"]
        conversation.messages.append({**message, "content": decode(content) if isinstance(content, list) else content})
    blocks = source.get("user_content") or [
        {"type": "text", "text": source["user_text"]},
        *({"type": "image", "path": entry["path"]} for entry in source["images"]),
    ]
    packet = conversation.preview(decode(blocks), system_prompt=source["system_prompt"])
    recorded = record_vlm_request(
        output,
        system_prompt=source["system_prompt"],
        user_text=source.get("source_user_text", source["user_text"]),
        images=[],
        history=[],
        packet=packet,
        metadata={
            "source_request": str(request_path.resolve()),
            "source_request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "offline_packet_replay": True,
            "model_calls": 0,
        },
    )
    # Show exactly the selected images, with their dialogue positions and IDs.
    rows = [o for o in packet.audit["occurrences"] if o["decision"] == "attached"]
    cols = min(4, max(1, len(rows)))
    sheet = Image.new("RGB", (cols * 320, 60 + ((len(rows) + cols - 1) // cols) * 290), "#111821")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (10, 10),
        f"Prepared packet: {packet.audit['input_images']} input attachments -> {len(rows)} attached",
        fill="white",
        font=font,
    )
    draw.text(
        (10, 30), "Historical images are not proof of current visibility or object geometry.", fill="white", font=font
    )
    for index, row in enumerate(rows):
        image = conversation.image_context.images[row["view_id"]].copy()
        image.thumbnail((320, 240))
        x, y = index % cols * 320, 60 + index // cols * 290
        sheet.paste(image, (x, y))
        label = f"{row['view_id']}\nmessage {row['message_index']}, image {row['image_slot']}"
        draw.text((x + 5, y + 243), label, fill="white", font=font)
    sheet.save(recorded.parent / "selected-images.png")
    return packet.audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-characters", type=int, default=64000)
    parser.add_argument("--full-context", action="store_true", help="Keep every attachment occurrence for ablation")
    args = parser.parse_args()
    audit = replay(
        args.request,
        args.output,
        max_images=None if args.full_context else args.max_images,
        max_characters=None if args.full_context else args.max_characters,
    )
    print(json.dumps({k: v for k, v in audit.items() if k != "occurrences"}, indent=2))
