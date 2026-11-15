# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Episode-owned dialogue; loaded perception models never own agent memory."""

import copy
import json

from emet.llms.image_context import ImageContext


class VLMConversation:
    def __init__(self, *, max_images=8, max_characters=64000):
        if max_characters is not None and (type(max_characters) is not int or max_characters < 1):
            raise ValueError("max_characters must be a positive integer or None")
        self.messages = []
        self.max_characters = max_characters
        self.image_context = ImageContext(max_images)
        self._requested_views = []
        self.last_packet = None

    def reset(self):
        self.messages.clear()
        self.image_context = ImageContext(self.image_context.max_images)
        self._requested_views.clear()
        self.last_packet = None

    def recall_view(self, view_id):
        if view_id not in self.image_context.images:
            return {
                "ok": False,
                "status": "UNKNOWN_VIEW",
                "view_id": view_id,
                "note": "Use an exact view_id from this episode's image annotations.",
            }
        if view_id not in self._requested_views:
            self._requested_views.append(view_id)
        return {
            "ok": True,
            "status": "HISTORICAL_VIEW_QUEUED",
            "view_id": view_id,
            "note": "The next request will attach these retained pixels; no new capture or motion occurred.",
        }

    def history_id(self, index):
        return f"message_{self.image_context.scope_id}_{index:04d}"

    def read_history(self, message_id, *, offset=0, max_characters=4000):
        if type(offset) is not int or offset < 0 or type(max_characters) is not int or not 1 <= max_characters <= 8000:
            return {"ok": False, "status": "INVALID_HISTORY_RANGE", "note": "Use offset >= 0 and 1..8000 characters."}
        index = next((i for i in range(len(self.messages)) if self.history_id(i) == message_id), None)
        if index is None:
            return {"ok": False, "status": "UNKNOWN_HISTORY", "message_id": message_id}
        message = self.messages[index]
        content = message["content"]
        parts = [content] if isinstance(content, str) else content
        text = "\n".join(part for part in parts if isinstance(part, str))
        images = []
        for part in parts:
            if not isinstance(part, str):
                view_id, _, _ = self.image_context._register(part)
                images.append(view_id)
        end = min(len(text), offset + max_characters)
        return {
            "ok": True,
            "status": "HISTORICAL_TEXT",
            "message_id": message_id,
            "role": message["role"],
            "text": text[offset:end],
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "total_characters": len(text),
            "view_ids": images,
            "note": "Original historical text; use recall_view for its pixels. Not a new observation.",
        }

    def _user_message(self, content):
        parts = copy.deepcopy(content)
        if self._requested_views:
            parts = [parts] if isinstance(parts, str) else list(parts)
            for view_id in self._requested_views:
                parts.extend(
                    [
                        f"Requested historical view {view_id}; not a new camera capture.",
                        self.image_context.images[view_id],
                    ]
                )
        return {"role": "user", "content": parts}

    def preview(self, content, *, system_prompt):
        request = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return self._pack([*request, *self.messages, self._user_message(content)])

    def _pack(self, request):
        def text_size(messages):
            return sum(
                len(part)
                for message in messages
                for part in ([message["content"]] if isinstance(message["content"], str) else message["content"])
                if isinstance(part, str)
            )

        packet = self.image_context.pack(request)
        source_text_characters = text_size(packet.messages)
        archived = []
        offset = int(bool(request and request[0]["role"] == "system"))
        # Only older multimodal/user-state text is eligible. Assistant decisions,
        # plain-text instructions, tool receipts and the latest input stay exact.
        for position in range(offset, len(request) - 1):
            if self.max_characters is None or text_size(packet.messages) <= self.max_characters:
                break
            message = request[position]
            if message["role"] != "user" or not isinstance(message["content"], list):
                continue
            original = message["content"]
            length = sum(len(part) for part in original if isinstance(part, str))
            message_id = self.history_id(position - offset)
            marker = (
                f"[Historical state text archived: message_id={message_id}, {length} characters. "
                "Use read_history(message_id) for its exact text. Subsequent action/tool results remain in dialogue.]"
            )
            if length <= len(marker):
                continue
            request[position] = {**message, "content": [marker, *[p for p in original if not isinstance(p, str)]]}
            archived.append({"message_id": message_id, "original_characters": length})
            packet = self.image_context.pack(request)
        count = text_size(packet.messages)
        if self.max_characters is not None and count > self.max_characters:
            raise ValueError(
                "Required current/system/tool/action context exceeds max_characters; original dialogue retained"
            )
        packet.audit.update(
            text_characters=count,
            source_text_characters=source_text_characters,
            max_characters=self.max_characters,
            archived_state_text=archived,
            text_policy="Current/system text, assistant decisions and tool-result messages exact; older state text retrievable",
        )
        if archived:
            packet.audit["text_and_tool_results"] = (
                "Original dialogue retained; old state text archived by reference; assistant/tool/current text exact"
            )
        return packet

    def generate(self, client, content, *, system_prompt, **kwargs):
        # Own camera snapshots, not references to mutable observation buffers.
        user = self._user_message(content)
        request = [{"role": "system", "content": system_prompt}] if system_prompt else []
        request += [*self.messages, user]
        self.last_packet = self._pack(request)
        reply = client.generate_messages(self.last_packet.messages, **kwargs)
        # Failed inference must not leave a half-turn in episode history.
        self.messages.extend([user, {"role": "assistant", "content": str(reply or "")}])
        self._requested_views.clear()
        return reply

    def record_tool_result(self, name, arguments, result):
        self.messages.append(
            {
                "role": "user",
                "content": "TOOL_RESULT "
                + json.dumps({"tool": name, "arguments": arguments, "result": result}, default=str),
            }
        )
