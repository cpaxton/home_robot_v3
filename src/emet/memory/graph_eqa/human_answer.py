# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Format GraphEQA / Dynagraph mLLM outputs for humans and agent tools (not "image 1").

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


_IMAGE_ID_RE = re.compile(
    r"\b(?:image|view|observation|frame)\s*#?\s*(\d+)\b",
    re.IGNORECASE,
)
_BARE_IMAGE_NUM_RE = re.compile(r"^\s*(?:image\s*)?(\d+)\s*\.?\s*$", re.IGNORECASE)


@dataclass
class HumanEQAResult:
    """User-facing EQA text derived from raw mLLM parse fields."""

    user_answer: str
    location_hint: str | None
    confidence_summary: str
    debug_reasoning: str


def answer_looks_like_image_index(answer: str, question: str = "") -> bool:
    """True when the model answer is an internal image id, not a human location."""
    a = (answer or "").strip()
    if not a:
        return False
    if _BARE_IMAGE_NUM_RE.match(a):
        return True
    if _IMAGE_ID_RE.search(a):
        return True
    q = (question or "").lower()
    if q.startswith(("where ", "where's", "wheres", "locate ")) and re.fullmatch(r"\d+", a):
        return True
    if len(a) <= 24 and re.search(r"\bimage\s*\d+\b", a, re.I):
        return True
    return False


def _primary_label(labels: list[str]) -> str:
    if not labels:
        return "object"
    return ", ".join(labels[:2])


def _format_xyz(xyz: np.ndarray) -> str:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return f"({x:.2f}, {y:.2f}, {z:.2f}) m"


def _extract_keywords(question: str) -> list[str]:
    stop = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "there",
        "where",
        "what",
        "which",
        "how",
        "many",
        "do",
        "does",
        "in",
        "on",
        "at",
        "of",
        "to",
        "and",
        "or",
        "room",
        "kitchen",
        "scene",
    }
    words = re.findall(r"[a-z0-9]+", question.lower())
    return [w for w in words if len(w) > 2 and w not in stop]


def _best_graph_node_for_question(
    nodes: list[Any],
    question: str,
    image_id: int | None = None,
) -> Any | None:
    if not nodes:
        return None
    if image_id is not None:
        for n in nodes:
            if int(n.obs_id) == int(image_id):
                return n
    keys = _extract_keywords(question)
    if keys:
        scored: list[tuple[int, Any]] = []
        for n in nodes:
            blob = " ".join(n.labels).lower()
            score = sum(1 for k in keys if k in blob)
            if score > 0:
                scored.append((score, n))
        if scored:
            scored.sort(key=lambda t: (-t[0], t[1].node_id))
            return scored[0][1]
    return nodes[0]


def _observation_for_image_index(
    observations: list[Any],
    nav_samples: list[Any],
    image_id: int,
) -> tuple[str, np.ndarray | None]:
    """Resolve 1-based image index to label text and anchor xyz."""
    if 1 <= image_id <= len(observations):
        obs = observations[image_id - 1]
        lbl = _primary_label(obs.labels)
        return lbl, np.asarray(obs.xyz, dtype=float).reshape(-1)[:3]
    if 1 <= image_id <= len(nav_samples):
        nv = nav_samples[image_id - 1]
        return "viewpoint", np.asarray(nv.xyz, dtype=float).reshape(-1)[:3]
    return "viewpoint", None


def _infer_image_id(answer: str, reasoning: str) -> int | None:
    for text in (answer, reasoning):
        m = _IMAGE_ID_RE.search(text or "")
        if m:
            return int(m.group(1))
        m2 = _BARE_IMAGE_NUM_RE.match((text or "").strip())
        if m2:
            return int(m2.group(1))
    m3 = re.search(r"\b(\d+)\b", (answer or "").strip())
    if m3 and answer_looks_like_image_index(answer):
        return int(m3.group(1))
    return None


def _spatial_phrase_for_node(node: Any, question: str) -> str:
    lbl = _primary_label(node.labels)
    xyz = np.asarray(node.xyz, dtype=float).reshape(-1)[:3]
    q = (question or "").lower()
    if q.startswith(("where ", "where's", "wheres", "locate ")):
        return f"The {lbl} is at approximately {_format_xyz(xyz)} in the scene."
    return f"The {lbl} is at approximately {_format_xyz(xyz)}."


def format_human_eqa_answer(
    question: str,
    answer: str,
    reasoning: str,
    graph_memory: Any,
    *,
    confidence: bool,
    confidence_reasoning: str = "",
    selected_obs_ids: list[int] | None = None,
) -> HumanEQAResult:
    """Turn raw mLLM answer/reasoning into a short human-readable reply."""
    nodes = graph_memory.get_nodes()
    observations = list(getattr(graph_memory, "_observations", []))
    nav_samples = list(getattr(graph_memory, "_nav_samples", []))

    user_answer = (answer or "").strip()
    debug_reasoning = (reasoning or "").strip()
    if confidence_reasoning and not confidence:
        debug_reasoning = f"{debug_reasoning}\n{confidence_reasoning}".strip()

    image_id = _infer_image_id(user_answer, debug_reasoning)
    location_hint: str | None = None

    if answer_looks_like_image_index(user_answer, question) or image_id is not None:
        node = _best_graph_node_for_question(nodes, question, image_id=image_id)
        if node is not None:
            user_answer = _spatial_phrase_for_node(node, question)
            location_hint = _format_xyz(np.asarray(node.xyz, dtype=float))
        elif image_id is not None:
            lbl, xyz = _observation_for_image_index(observations, nav_samples, image_id)
            if xyz is not None:
                user_answer = (
                    f"From the robot's view where it saw the {lbl}, "
                    f"the target is near {_format_xyz(xyz)}."
                )
                location_hint = _format_xyz(xyz)
            else:
                user_answer = (
                    f"I saw something related near view {image_id}, "
                    "but do not have a precise world position yet."
                )
        elif nodes:
            user_answer = _spatial_phrase_for_node(_best_graph_node_for_question(nodes, question), question)
            location_hint = _format_xyz(np.asarray(nodes[0].xyz, dtype=float))
        elif not user_answer or answer_looks_like_image_index(user_answer, question):
            user_answer = (
                "I do not have a confident map location yet; "
                "try exploring more or asking again after the scene graph updates."
            )

    elif nodes and _extract_keywords(question):
        node = _best_graph_node_for_question(nodes, question)
        if node is not None:
            keys = _extract_keywords(question)
            blob = " ".join(node.labels).lower()
            if any(k in blob for k in keys) and len(user_answer) < 12:
                user_answer = _spatial_phrase_for_node(node, question)
                location_hint = _format_xyz(np.asarray(node.xyz, dtype=float))

    if not user_answer:
        user_answer = "I could not determine an answer from the current scene graph."

    confidence_summary = "confident" if confidence else "not confident"
    return HumanEQAResult(
        user_answer=user_answer,
        location_hint=location_hint,
        confidence_summary=confidence_summary,
        debug_reasoning=debug_reasoning,
    )


def format_eqa_tool_response(human: HumanEQAResult) -> str:
    """Stable multi-line string for ``query_scene_graph`` / ``query_memory`` tools."""
    lines = [f"Answer: {human.user_answer}", f"Confidence: {human.confidence_summary}"]
    if human.location_hint:
        lines.insert(1, f"Location: {human.location_hint}")
    return "\n".join(lines)
