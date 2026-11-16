#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Replay a recorded client request through image budgeting, without model calls."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from emet.eval.eqa_decision_trace import record_vlm_request
from emet.llms.conversation import VLMConversation


def decode_content(content, root):
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
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


def recorded_messages(request_path):
    """Read the exact recorded packet; do not reselect images or regenerate IDs."""
    source = json.loads(request_path.read_text())
    root = request_path.parent.resolve()
    messages = [{"role": "system", "content": source["system_prompt"]}]
    messages.extend(
        {"role": m["role"], "content": decode_content(m["content"], root)}
        for m in source.get("conversation_history", [])
    )
    blocks = source.get("user_content") or [
        {"type": "text", "text": source["user_text"]},
        *({"type": "image", "path": entry["path"]} for entry in source["images"]),
    ]
    messages.append({"role": "user", "content": decode_content(blocks, root)})
    return messages


def infer_recorded(request_path, output, *, endpoint, max_tokens=512, timeout_s=240):
    """One bounded model call, no tools or scoring; preserve wire body and failure."""
    import httpx
    from openai import OpenAI

    from emet.llms.openai_vllm_client import OpenaiVLLMClient, parse_openai_endpoint_spec

    if output.exists():
        raise ValueError("Use a fresh output directory")
    if max_tokens < 1 or timeout_s <= 0:
        raise ValueError("Positive output and timeout limits required")
    base_url, model = parse_openai_endpoint_spec(endpoint)
    if not model:
        raise ValueError("Endpoint must include #MODEL")
    messages = recorded_messages(request_path)
    output.mkdir(parents=True)
    row = {
        "source_request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "max_retries": 0,
        "tools_executed": 0,
        "precision": "unverified",
        "effective_decoding": "unverified",
        "scope": "Frozen next-response diagnostic, not task success; no scorer inputs",
        "transport": {"image_max_side": 512, "jpeg_quality": 85},
    }

    def capture_request(request):
        body = json.loads(request.content)
        (output / "wire-request.json").write_text(json.dumps(body))
        row["wire_messages_sha256"] = hashlib.sha256(json.dumps(body["messages"], sort_keys=True).encode()).hexdigest()

    def capture_response(response):
        response.read()
        (output / "wire-response.txt").write_text(response.text)
        row["http_status"] = response.status_code
        try:
            body = response.json()
            row["usage"] = body.get("usage")
            choice = (body.get("choices") or [{}])[0]
            row["finish_reason"] = choice.get("finish_reason")
            row["refusal"] = choice.get("message", {}).get("refusal")
        except ValueError:
            pass

    started = time.monotonic()
    try:
        client = OpenaiVLLMClient(base_url=base_url, model=model, max_tokens=max_tokens)
        client._openai.close()
        with httpx.Client(event_hooks={"request": [capture_request], "response": [capture_response]}) as transport:
            with OpenAI(
                base_url=base_url, api_key=client.api_key, timeout=timeout_s, max_retries=0, http_client=transport
            ) as api:
                client._openai = api
                row["reply"] = client.generate_messages(messages, max_new_tokens=max_tokens)
        row["status"] = "complete" if row.get("finish_reason") == "stop" and not row.get("refusal") else "incomplete"
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        row["wall_s"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def replay(request_path: Path, output: Path, *, max_images=8, max_characters=64000):
    if output.exists():
        raise ValueError("Use a fresh output directory; recorded evidence is never overwritten")
    source = json.loads(request_path.read_text())
    root = request_path.parent.resolve()

    conversation = VLMConversation(max_images=max_images, max_characters=max_characters)
    for message in source.get("conversation_history", []):
        content = message["content"]
        conversation.messages.append({**message, "content": decode_content(content, root)})
    blocks = source.get("user_content") or [
        {"type": "text", "text": source["user_text"]},
        *({"type": "image", "path": entry["path"]} for entry in source["images"]),
    ]
    packet = conversation.preview(decode_content(blocks, root), system_prompt=source["system_prompt"])
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
        image = conversation.image_context.images[row["image_id"]].copy()
        image.thumbnail((320, 240))
        x, y = index % cols * 320, 60 + index // cols * 290
        sheet.paste(image, (x, y))
        label = f"{row['image_id']}\nmessage {row['message_index']}, image {row['image_slot']}"
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
    parser.add_argument(
        "--vl-endpoint", help="Opt-in inference on the exact saved packet: URL#MODEL; no tool execution"
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=240)
    args = parser.parse_args()
    if args.vl_endpoint:
        if args.full_context:
            parser.error("Inference replays saved packets exactly; prepare any full-context ablation separately")
        row = infer_recorded(
            args.request,
            args.output,
            endpoint=args.vl_endpoint,
            max_tokens=args.max_output_tokens,
            timeout_s=args.timeout_s,
        )
        print(json.dumps(row, indent=2))
        raise SystemExit(0 if row["status"] == "complete" else 1)
    audit = replay(
        args.request,
        args.output,
        max_images=None if args.full_context else args.max_images,
        max_characters=None if args.full_context else args.max_characters,
    )
    print(json.dumps({k: v for k, v in audit.items() if k != "occurrences"}, indent=2))
