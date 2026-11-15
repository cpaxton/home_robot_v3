# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Episode-owned dialogue; loaded perception models never own agent memory."""

import copy
import json

from emet.llms.image_context import ImageContext


class VLMConversation:
    def __init__(self, *, max_images=8):
        self.messages = []
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
        return self.image_context.pack([*request, *self.messages, self._user_message(content)])

    def generate(self, client, content, *, system_prompt, **kwargs):
        # Own camera snapshots, not references to mutable observation buffers.
        user = self._user_message(content)
        request = [{"role": "system", "content": system_prompt}] if system_prompt else []
        request += [*self.messages, user]
        self.last_packet = self.image_context.pack(request)
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
