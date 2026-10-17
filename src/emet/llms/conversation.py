# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Episode-owned dialogue; loaded perception models never own agent memory."""

import copy
import json


class VLMConversation:
    def __init__(self):
        self.messages = []

    def reset(self):
        self.messages.clear()

    def generate(self, client, content, *, system_prompt, **kwargs):
        # Own camera snapshots, not references to mutable observation buffers.
        user = {"role": "user", "content": copy.deepcopy(content)}
        request = [{"role": "system", "content": system_prompt}] if system_prompt else []
        request += [*self.messages, user]
        reply = client.generate_messages(request, **kwargs)
        # Failed inference must not leave a half-turn in episode history.
        self.messages.extend([user, {"role": "assistant", "content": str(reply or "")}])
        return reply

    def record_tool_result(self, name, arguments, result):
        self.messages.append(
            {
                "role": "user",
                "content": "TOOL_RESULT "
                + json.dumps({"tool": name, "arguments": arguments, "result": result}, default=str),
            }
        )
