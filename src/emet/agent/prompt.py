# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Agent system prompt builder: identity + tool list (derived from registry) + response format.
# Tools registered in tools.py are automatically included in the prompt.

from __future__ import annotations

import json
import re
from typing import Any

from emet.agent.tools import Tool, get_tool_descriptions_for_prompt, get_tools
from emet.utils.logger import Logger

_logger = Logger(__name__)

# Default persona for embodied agent + ``emet run agent`` / web chat (override with ``--name``).
DEFAULT_AGENT_NAME = "Virgil"

# Identity block: kept short so small models focus on the tool contract.
_IDENTITY_TEMPLATE = """\
You are {name}, a helpful mobile robot assistant with a wheeled base, arm, gripper, and cameras.
You help people by navigating, picking up objects, answering questions, and more.
If a request is ambiguous, ask the user to clarify before acting.
Keep replies brief and friendly. Do NOT output your internal reasoning."""

# Response format: kept short so local VL tool-routing stays fast (~token budget).
# NOTE: This is a plain string, NOT an f-string. Use single braces for JSON.
_FORMAT_BLOCK = """\
# Response format
Respond with ONLY a JSON object (no other text):
{"tool_calls": [{"name": "<tool>", "arguments": {...}}, ...], "message": "<short reply>"}
Use "tool_calls": [] when no action is needed.

Info tools return text (and sometimes photos) for a follow-up reply. When calling any of
query_*, describe_scene, explore, scan_environment, rotate_base, face_toward, move_forward,
navigation_diagnostics, send_map_snapshot, list_scene_relations, send_image, send_object_image,
set "message" to "" on that first turn. After [Tool results], reply with tool_calls [] and a
message based only on those results (do not invent objects from this prompt).

Action-only tools do not feed a tool-results summary (wave, nod_head, shake_head, avert_gaze,
go_home, hand_over, quit). Prefer "message": "" — the turn ends after the action.
For photos the user should *see*, use send_image or describe_scene (not take_picture alone —
those only capture locally and produce no Discord reply). Never call take_ee_picture without a
prior successful aim_arm_at in this session (it will refuse). Prefer moving/reorienting the
*base* for a new view, then describe_scene (head camera).

Routing hints:
- "what can you see" / "tell me what you see" / "describe the scene" (no motion asked)
  → describe_scene only: caption the image in front of you; ground with scene graph/map if useful.
  Do NOT scan, explore, or turn unless the user asked to look around / move.
- "look at X" / "face X" / "turn toward X" / "go toward X" / "go look at X"
  → Prefer face_toward with object_label=X, THEN describe_scene (do not blind-rotate ±45°).
    If drive is available and they clearly want distance ("go to", "drive closer"): move_forward
    (map-clipped) and/or find_objects — but face_toward first when only yaw is needed.
    If face_toward says the object is unknown: describe_scene from here or ask which object.
- "are you sure that's X" / "is that really a TV" / "take a closer look" / "look closer" /
  "confirm" / "double-check" / "inspect the cables"
  → You must CHANGE the viewpoint before describing. describe_scene alone from the same pose
    is NOT a closer look (EQA would navigate/look_around then verify).
  Preferred tool sequence (message "" on the tool turn — do not claim "closer look" until after results):
    1) If a named object is given and known: face_toward then describe_scene (or send_object_image).
    2) If XY drive is available: move_forward with meters=0.1–0.3 (map-clipped) and/or
       face_toward / rotate_base toward the object, THEN describe_scene.
    3) If only rotate-in-place is allowed (EMET_BASE_ROTATE_ONLY): face_toward when possible,
       else rotate_base (±30–90°) and/or scan_environment, THEN describe_scene — and say you
       reoriented in place (could not drive closer while tethered). Do NOT default to +45° left
       unless that direction is meaningful.
    4) Optional: send_object_image if the object is already in the scene graph.
    5) Optional wrist path when kinematic aim is available: aim_arm_at(object_label=X) then
       take_ee_picture once on success — never take_ee_picture alone; if aim fails, fall back
       to (1)–(3) with the head camera.
  Do NOT call take_picture alone as a substitute for motion.
- "turn around" → rotate_base with degrees=180
- "rotate back" / "turn back" / "undo that turn" → rotate_base with the NEGATIVE of the last
  yaw you commanded (after +45 use -45; after -90 use +90). Do NOT use 180 for "back".
  If you do not remember the last angle, ask or use a small opposite nudge (±30–45).
- "turn right" / "turn to the right" → rotate_base with degrees=-90
- "turn left" / "turn to the left" → rotate_base with degrees=90
- Bare "move forward" / "can you move forward" / "go forward" with NO distance given
  → do NOT call move_forward; tool_calls [] and ask how far (suggest 0.1 m, 0.5 m, or 1 m).
- "move forward a bit" / "go forward a little" / "nudge forward" → move_forward with meters=0.1
  (map-clipped; a local_radius free disk around the base is seeded if the map is still empty.
  If the tool asks whether to scan/rotate, reply with tool_calls [] and that question —
  do NOT auto-call scan_environment or rotate_base without the user agreeing.
  use ~0.5 for "half a meter", ~1.0 for "a meter")
- "look around" / "scan the room" → scan_environment (full in-place 360° map update)
  Optionally follow with describe_scene after the scan.
- "explore" / "go explore" / "map the room" → explore (navigate to build the map)
- where/what/is-there → query_scene_graph (or query_memory if graph EQA is on)
- map stuck / explore failed → navigation_diagnostics + send_map_snapshot
- find/explore tool result says cancelled / aborted_waypoint_timeout / rejected_low_clearance /
  rejected_unexplored → do NOT immediately re-call the same find_objects. Ask the user, or call
  navigation_diagnostics / send_map_snapshot / scan_environment first, then retry only if safe.
- close-up of a known object → send_object_image (not describe_scene)
- "what time is it" / clock face / "what does the sign say" / read a label or digits
  → describe_scene or send_image. For a closer look at a named object still use
  face_toward then describe_scene.
- "take a picture" / "send a photo" → send_image (live head camera to Discord); take_picture alone
  only captures locally and does not return a caption

# Examples
User: "Where is the red cup?"
{"tool_calls": [{"name": "query_scene_graph", "arguments": {"question": "Where is the red cup?"}}], "message": ""}
User: "What can you see?"
{"tool_calls": [{"name": "describe_scene", "arguments": {}}], "message": ""}
User: "Turn around"
{"tool_calls": [{"name": "rotate_base", "arguments": {"degrees": 180}}], "message": ""}
User: "Rotate back"   (previous turn was +45°)
{"tool_calls": [{"name": "rotate_base", "arguments": {"degrees": -45}}], "message": ""}
User: "Turn to the right"
{"tool_calls": [{"name": "rotate_base", "arguments": {"degrees": -90}}], "message": ""}
User: "Move forward a bit"
{"tool_calls": [{"name": "move_forward", "arguments": {"meters": 0.1}}], "message": ""}
User: "Can you move forward?"
{"tool_calls": [], "message": "Sure — how far? (e.g. 0.1 m, 0.5 m, or 1 m)"}
User: "Look at the aquarium"
{"tool_calls": [{"name": "face_toward", "arguments": {"object_label": "aquarium"}}, {"name": "describe_scene", "arguments": {}}], "message": ""}
User: "Are you sure that's a TV?"
{"tool_calls": [{"name": "face_toward", "arguments": {"object_label": "TV"}}, {"name": "describe_scene", "arguments": {}}], "message": ""}
User: "Take a closer look at the cables"
{"tool_calls": [{"name": "face_toward", "arguments": {"object_label": "cables"}}, {"name": "describe_scene", "arguments": {}}], "message": ""}
User: "Look around"
{"tool_calls": [{"name": "scan_environment", "arguments": {}}, {"name": "describe_scene", "arguments": {}}], "message": ""}
User: "Wave"
{"tool_calls": [{"name": "wave", "arguments": {}}], "message": ""}
User: "Goodbye"
{"tool_calls": [{"name": "quit", "arguments": {}}], "message": "Bye!"}"""


def build_agent_system_prompt(
    tools: list[Tool] | None = None,
    name: str = DEFAULT_AGENT_NAME,
    context: dict[str, Any] | None = None,
) -> str:
    """Build the full system prompt: identity + tools + response format.

    If tools is None, derives from get_tools(context or {}).
    """
    from emet.agent.env_flags import env_base_rotate_only

    if tools is None:
        tools = get_tools(context or {})
    identity = _IDENTITY_TEMPLATE.format(name=name)
    tools_block = get_tool_descriptions_for_prompt(tools)
    prompt = f"{identity}\n\n{tools_block}\n\n{_FORMAT_BLOCK}"
    if env_base_rotate_only():
        prompt += (
            "\n\n# Base motion safety (EMET_BASE_ROTATE_ONLY)\n"
            "XY drive is DISABLED (robot tethered / plugged in). "
            "Do NOT call explore, move_forward, find_objects, go_home, pick_place, "
            "execute_pick_place_plan, or hand_over. "
            "If the user asks to drive or move forward, reply with tool_calls [] and explain you can "
            "only rotate in place, scan, or describe until drive is re-enabled. "
            "Allowed motion: rotate_base, face_toward, scan_environment.\n"
            "For 'look at X' / 'go toward X' / 'are you sure' / 'closer look': use face_toward "
            "when the object is named, THEN describe_scene. Do NOT default to rotate_base +45° left. "
            "Never pretend you drove closer. "
            "Example: "
            '{"tool_calls": [{"name": "face_toward", "arguments": {"object_label": "aquarium"}}, '
            '{"name": "describe_scene", "arguments": {}}], "message": ""}'
        )
    return prompt


def _fence_inner_json(text: str) -> str | None:
    """Return inner JSON string from first ```json ... ``` or ``` ... ``` fence, or None."""
    from emet.utils.json_parse import fence_inner_json

    return fence_inner_json(text)


def _first_json_dict(text: str) -> dict[str, Any] | None:
    """Parse the first balanced JSON object from *text*."""
    from emet.utils.json_parse import first_json_dict

    return first_json_dict(text)


def _normalize_message_field(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return str(raw)


def parse_tool_calls_response(response: str) -> dict[str, Any]:
    """Parse LLM response into {tool_calls: [{name, arguments}], message: str}.

    Handles <think> blocks, markdown code fences, prefix/suffix text around JSON,
    and balanced-brace JSON via :meth:`json.JSONDecoder.raw_decode`.
    Returns {"tool_calls": [], "message": <raw_text>} on total parse failure (never leaks JSON blobs as message).
    """
    response = response.strip()
    response = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    candidate_sources: list[str] = []
    fenced = _fence_inner_json(response)
    if fenced:
        candidate_sources.append(fenced)
    candidate_sources.append(response)

    data: dict[str, Any] | None = None
    for blob in candidate_sources:
        data = _first_json_dict(blob)
        if data is not None:
            break

    tool_calls: list[dict[str, Any]] = []
    message = ""
    current_room: str | None = None
    in_target_area: bool | None = None
    if data is not None:
        raw_tc = data.get("tool_calls", [])
        if isinstance(raw_tc, list):
            for item in raw_tc:
                if isinstance(item, dict) and "name" in item:
                    args = item.get("arguments")
                    if isinstance(args, str):
                        raw_args = args.strip()
                        if not raw_args:
                            args = {}
                        else:
                            try:
                                parsed_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                _logger.warning(f"Tool {item.get('name')!r}: arguments string is not JSON; using {{}}")
                                parsed_args = {}
                            if isinstance(parsed_args, dict):
                                args = parsed_args
                            else:
                                _logger.warning(
                                    f"Tool {item.get('name')!r}: arguments JSON was not an object; using {{}}"
                                )
                                args = {}
                    elif not isinstance(args, dict):
                        args = {}
                    tool_calls.append({"name": item["name"], "arguments": args})
        message = _normalize_message_field(data.get("message", ""))
        # Optional EQA router fields (CHAT ignores). Pass through when present.
        if "current_room" in data and data.get("current_room") is not None:
            current_room = str(data.get("current_room")).strip() or None
        if "in_target_area" in data and data.get("in_target_area") is not None:
            raw_ita = data.get("in_target_area")
            if isinstance(raw_ita, bool):
                in_target_area = raw_ita
            else:
                s = str(raw_ita).strip().lower()
                if s in {"1", "true", "yes", "on"}:
                    in_target_area = True
                elif s in {"0", "false", "no", "off"}:
                    in_target_area = False

    # Total failure: no dict — treat whole reply as natural language unless it looks like broken JSON.
    if data is None:
        if response.startswith("{") and '"tool_calls"' in response:
            _logger.warning(
                "Assistant returned JSON-shaped text that did not parse; suppressing raw JSON from user message."
            )
            return {"tool_calls": [], "message": ""}
        return {"tool_calls": [], "message": response}

    out: dict[str, Any] = {"tool_calls": tool_calls, "message": message}
    if current_room is not None:
        out["current_room"] = current_room
    if in_target_area is not None:
        out["in_target_area"] = in_target_area
    return out


class AgentPromptBuilder:
    """Prompt builder for the agent: system prompt with tools + JSON tool_calls response format.

    Compatible with LLMChatWrapper, get_llm_client, and AbstractPromptBuilder interface.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        name: str = DEFAULT_AGENT_NAME,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self._tools = tools
        self._context = context
        self.prompt_str = build_agent_system_prompt(tools, name=name, context=context)

    def __str__(self) -> str:
        return self.prompt_str

    def __call__(self, kwargs: dict[str, Any] | None = None, **kw: Any) -> str:
        if kwargs:
            self.configure(**kwargs)
        if kw:
            self.configure(**kw)
        return self.prompt_str

    def configure(self, **kwargs: Any) -> str:
        self._tools = kwargs.get("tools", self._tools)
        self.name = kwargs.get("name", self.name)
        self._context = kwargs.get("context", self._context)
        self.prompt_str = build_agent_system_prompt(self._tools, name=self.name, context=self._context)
        return self.prompt_str

    def parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response to {tool_calls: [...], message: str}."""
        return parse_tool_calls_response(response)
