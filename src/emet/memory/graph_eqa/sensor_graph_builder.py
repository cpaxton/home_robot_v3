# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Build GraphEQA labels and 3D anchors from robot Observations (RGB-D + pose).
# Uses a vision-language model for open-vocabulary object names; world xyz from
# depth median in camera frame transformed by camera_pose (no sim ground truth).

from __future__ import annotations

import json
import re
from collections.abc import Callable

import numpy as np

from emet.core.interfaces import Observations
from emet.core.parameters import Parameters
from emet.utils.logger import Logger

logger = Logger(__name__)

# JSON-only contract for graph label extraction (Qwen / VLMs must output this shape only).
# Example: {"objects":[{"name":"red cylinder"},{"name":"wooden table"}]}
# Alternate: {"labels":["red cylinder","wooden table"]}
GRAPH_EXTRACT_JSON_SCHEMA = r'{"objects":[{"name":"<short noun phrase>"},...]} or {"labels":["..."]}'

_EXTRACT_SYSTEM_DEFAULT = (
    "You are a strict JSON label extractor for a robot vision system. "
    "Reply with ONLY a single JSON object, no markdown fences, no analysis, no text before or after. "
    f"Schema: {GRAPH_EXTRACT_JSON_SCHEMA}. "
    "Use 1–8 objects maximum; each name is at most 6 words, physical things only "
    "(furniture, props, appliances). Omit walls, ceiling, floor, lighting, darkness, "
    "'the image', 'the user', or scene meta-commentary."
)

_EXTRACT_USER_DEFAULT = (
    "Indoor robot head-camera: list distinct physical objects visible "
    "(furniture, props, appliances). "
    "Reply with a single JSON object only—no markdown fences, no analysis, no text before or after. "
    f"Shape: {GRAPH_EXTRACT_JSON_SCHEMA}. "
    "At most 8 entries; each name is a short noun phrase (≤6 words), physical things only."
)

_LABEL_REJECT = re.compile(
    r"(?i)^(the user|analyze the image|scan for|identify the|break down|visible distinct|"
    r"reply with|comma-separated|\*+|step\s*\d|^\d+\.\s*$|\d+\.\s*\*\*)"
)
_DIGITS_ONLY = re.compile(r"^\d+$")
# Qwen3.5 may still emit a thinking block in decoded text; strip before JSON parse.
_THINKING_BLOCK = re.compile(
    r"<(?:redacted_)?thinking>[\s\S]*?</(?:redacted_)?thinking>",
    re.IGNORECASE,
)


def _strip_thinking_and_fences(s: str) -> str:
    t = _THINKING_BLOCK.sub("", s).strip()
    return t


def world_xyz_median_from_depth(obs: Observations) -> np.ndarray:
    """
    Robust scene point in world frame: median of valid head-camera depth points.
    Falls back to camera optical center if depth or intrinsics missing.
    """
    if obs.camera_pose is None:
        raise ValueError("Observations.camera_pose is required for world xyz")

    if obs.depth is None or obs.camera_K is None:
        return np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)

    obs.compute_xyz(scaling=1.0)
    pts_world = obs.get_xyz_in_world_frame(scaling=1.0)
    if pts_world is None:
        return np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)

    flat = pts_world.reshape(-1, 3)
    d = obs.depth.reshape(-1)
    valid = (d > 0.05) & (d < 10.0) & np.isfinite(d)
    sel = flat[valid]
    if sel.shape[0] == 0:
        return np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)
    return np.median(sel, axis=0).astype(np.float64)


def _extract_json_object_blob(raw: str) -> str | None:
    """Return JSON object substring from model output (strip fences, find braces)."""
    s = _strip_thinking_and_fences(raw)
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = s[start : end + 1]
    return _json_object_blob_or_none(blob, s, start)


def _json_object_blob_or_none(blob: str, full: str, start: int) -> str | None:
    """Try json.loads(blob); on failure, try brace-balanced slice from ``start``."""
    if not blob:
        return None
    try:
        json.loads(blob)
        return blob
    except json.JSONDecodeError:
        pass
    balanced = _balanced_json_object_from(full, start)
    if balanced:
        try:
            json.loads(balanced)
            return balanced
        except json.JSONDecodeError:
            return None
    return None


def _balanced_json_object_from(s: str, start: int) -> str | None:
    """Extract outermost `{...}` from ``start`` using brace depth (handles `{` in strings poorly)."""
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
                quote = ""
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_graph_object_json(raw: str) -> dict | None:
    """Parse first JSON object from ``raw``; return dict or None."""
    blob = _extract_json_object_blob(raw)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def _normalize_extract_label(s: str, max_len: int = 48) -> str | None:
    t = " ".join(s.split()).strip()
    if not t or len(t) > max_len:
        return None
    if t.startswith("*"):
        return None
    if "**" in t:
        return None
    if _DIGITS_ONLY.match(t):
        return None
    if _LABEL_REJECT.search(t):
        return None
    return t


def labels_from_extract_response(raw: str, *, max_labels: int = 8, max_len: int = 48) -> list[str] | None:
    """
    If ``raw`` contains valid extract JSON, return normalized short object names.
    Returns None if JSON missing or invalid or no usable labels.
    """
    data = parse_graph_object_json(raw)
    if not isinstance(data, dict):
        return None
    names: list[str] = []
    if "objects" in data and isinstance(data["objects"], list):
        for o in data["objects"]:
            if not isinstance(o, dict):
                continue
            n = o.get("name") or o.get("label")
            if n is None:
                continue
            nn = _normalize_extract_label(str(n), max_len=max_len)
            if nn:
                names.append(nn)
    elif "labels" in data and isinstance(data["labels"], list):
        for x in data["labels"]:
            nn = _normalize_extract_label(str(x), max_len=max_len)
            if nn:
                names.append(nn)
    if not names:
        return None
    return names[:max_labels]


def parse_voxel_label_line(text: str, max_labels: int = 16, max_segment_len: int = 64) -> list[str]:
    """
    Split voxel / legacy comma-separated label lines.

    Does **not** replace periods (that splits CoT into fake labels). Uses comma,
    semicolon, and newline only.
    """
    cleaned = text.replace(";", ",").replace("\n", ",")
    out: list[str] = []
    for part in cleaned.split(","):
        s = part.strip().strip("-•").strip()
        if not s or len(s) > max_segment_len:
            continue
        out.append(s)
        if len(out) >= max_labels:
            break
    return out


def parse_comma_separated_labels(text: str, max_labels: int = 16) -> list[str]:
    """Parse comma / newline / semicolon-separated labels (no period splitting)."""
    return parse_voxel_label_line(text, max_labels=max_labels, max_segment_len=64)


def short_labels_from_voxel_descriptions(voxel_labels: list[str] | None, max_labels: int = 16) -> list[str]:
    """Turn possibly noisy voxel/VLM strings into short label tokens."""
    if not voxel_labels:
        return ["object"]
    out: list[str] = []
    for line in voxel_labels:
        out.extend(parse_voxel_label_line(str(line), max_labels=max_labels))
        if len(out) >= max_labels:
            break
    return out[:max_labels] if out else ["object"]


def _graph_extract_config(parameters: Parameters | dict | None) -> tuple[int, str, str]:
    """Returns (max_new_tokens, system_prompt, user_text_suffix)."""
    default_tokens = 128
    if parameters is None:
        return default_tokens, _EXTRACT_SYSTEM_DEFAULT, ""
    if isinstance(parameters, dict):
        block = parameters.get("graph_eqa_extract") or {}
        mt = int(block.get("max_tokens", default_tokens))
        sys_p = block.get("system_prompt") or _EXTRACT_SYSTEM_DEFAULT
        suf = str(block.get("user_suffix", "") or "").strip()
        return mt, str(sys_p), suf
    block = parameters.get("graph_eqa_extract") or {}
    if not isinstance(block, dict):
        block = {}
    mt = int(block.get("max_tokens", default_tokens))
    sys_p = block.get("system_prompt") or _EXTRACT_SYSTEM_DEFAULT
    suf = str(block.get("user_suffix", "") or "").strip()
    return mt, str(sys_p), suf


class SensorGraphBuilder:
    """
    Produces object labels (VLM) and a world-frame anchor xyz from Observations.

    If ``cpu_only`` or no GPU, skips loading Qwen3.5 multimodal and relies on
    ``voxel_labels`` / fallback ``["object"]``.
    """

    def __init__(
        self,
        *,
        perception_client: Callable[..., str] | None = None,
        use_voxel_fallback: bool = True,
        device: str = "cuda",
        cpu_only: bool = False,
        parameters: Parameters | dict | None = None,
    ):
        self._perception = perception_client
        self.use_voxel_fallback = use_voxel_fallback
        self._device = device
        self.cpu_only = cpu_only
        self._parameters = parameters
        self._lazy_vl_client: Callable[..., str] | None = None

    def _get_default_vl_client(self) -> Callable[..., str] | None:
        if self.cpu_only:
            return None
        try:
            from emet.llms.eqa_qwen import get_shared_qwen35_vl_client
        except ImportError as e:
            logger.warning(f"get_shared_qwen35_vl_client unavailable ({e}); using voxel/fallback labels")
            return None
        dev = self._device
        if dev not in ("cuda", "mps"):
            dev = "cuda"
        try:
            return get_shared_qwen35_vl_client(
                device=dev,
                quantization=None,
                parameters=self._parameters,
            )
        except Exception as e:
            logger.warning(f"Could not load Qwen3.5 multimodal ({e}); using voxel/fallback labels")
            return None

    def _client(self) -> Callable[..., str] | None:
        if self._perception is not None:
            return self._perception
        if self.cpu_only:
            return None
        if self._lazy_vl_client is None:
            self._lazy_vl_client = self._get_default_vl_client()
        return self._lazy_vl_client

    def labels_and_description_from_observation(
        self,
        obs: Observations,
        voxel_labels: list[str] | None = None,
    ) -> tuple[list[str], str | None]:
        """Short labels for graph nodes; optional long raw VLM text as ``description`` (not used as labels)."""
        client = self._client()
        if client is None:
            if voxel_labels:
                return short_labels_from_voxel_descriptions(voxel_labels), None
            return ["object"], None

        extract_max_tokens, system_extract, user_suffix = _graph_extract_config(self._parameters)
        user_prompt = _EXTRACT_USER_DEFAULT + (f" {user_suffix}" if user_suffix else "")

        try:
            try:
                out = client(
                    [user_prompt, obs.rgb],
                    system_prompt=system_extract,
                    max_new_tokens=extract_max_tokens,
                )
            except TypeError:
                # Injected clients / older callables: (cmd, image) only
                out = client([user_prompt, obs.rgb])
            if not isinstance(out, str):
                out = str(out)
            raw = out.strip()
            structured = labels_from_extract_response(raw)
            if structured:
                desc = raw if len(raw) > 200 else None
                return structured, desc
            logger.warning("Graph label extract: JSON parse failed or empty; using object fallback")
            return ["object"], raw if raw else None
        except Exception as e:
            logger.warning(f"Perception VLM failed ({e})")

        if voxel_labels and self.use_voxel_fallback:
            return short_labels_from_voxel_descriptions(voxel_labels), None
        return ["object"], None

    def labels_from_observation(
        self,
        obs: Observations,
        voxel_labels: list[str] | None = None,
    ) -> list[str]:
        labs, _ = self.labels_and_description_from_observation(obs, voxel_labels=voxel_labels)
        return labs

    def world_xyz_for_observation(self, obs: Observations) -> np.ndarray:
        return world_xyz_median_from_depth(obs)
