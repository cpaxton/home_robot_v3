# Copyright (c) Chris Paxton 2026
"""VLM-first target extract + multimodal answerability assess for agentic EQA.

Cheap SigLIP/OWL scores are **where-next** signals for navigation and graph
growth (drive to a promising place, then populate with higher-confidence LLM
views). They must **not** enter the assess inventory — ABSENT/PRESENT proposals
color Qwen answers. Assess looks at pixels + neutral map context only.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_TARGET_SYSTEM = (
    "You extract the visual target for a robot verifying a home question. Reply with ONLY a JSON object (no markdown)."
)

_ASSESS_SYSTEM = (
    "You are a robot looking at a camera image to decide if you can answer a question. "
    "Use the image and the inventory. Do not assume objects are absent from "
    "detector scores — judge presence from the image. "
    "Reply with ONLY a JSON object (no markdown)."
)


@dataclass
class TargetExtract:
    target_phrase: str
    question_type: str = "other"  # count | location | state | other
    notes: str = ""
    requires_close_look: bool = False  # time/state/count/detail questions need a close look

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ViewAssessment:
    target: str
    present: bool
    answerable: bool
    need_more_views: bool
    suggested_answer: str | None = None
    reason: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _call_eqa_client(
    client: Any,
    payload: str | list[Any],
    *,
    system_prompt: str,
    max_new_tokens: int = 192,
) -> str:
    try:
        return str(
            client(
                payload if isinstance(payload, list) else [payload],
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
            )
        )
    except TypeError:
        try:
            return str(client([f"{system_prompt}\n\n{payload}"]))
        except Exception:
            return ""
    except Exception:
        return ""


def extract_target_from_question(
    client: Any,
    question: str,
    *,
    fallback_phrase: str = "",
) -> TargetExtract:
    """Text-only VLM: pick the object/target phrase to seek/verify."""
    q = (question or "").strip()
    prompt = (
        "Question:\n"
        f"{q}\n\n"
        "Return JSON with keys:\n"
        "  target_phrase: short noun phrase for the object to find/verify "
        '(e.g. "utensils", "fruit bowl", "air conditioning")\n'
        "  question_type: one of count, location, state, other\n"
        "  requires_close_look: bool — true when answering needs a close look at the "
        "target (reading a clock/display/label, counting, checking on/off or open/closed "
        "state, reading fine detail); false for location questions where seeing the "
        "object at a distance suffices\n"
        "  notes: optional short note\n"
        "Do not include MCQ option text in target_phrase."
    )
    raw = _call_eqa_client(client, prompt, system_prompt=_TARGET_SYSTEM, max_new_tokens=128)
    data = _parse_json_object(raw)
    phrase = str(data.get("target_phrase") or data.get("target") or "").strip()
    qtype = str(data.get("question_type") or "other").strip().lower()
    if qtype not in {"count", "location", "state", "other"}:
        qtype = "other"
    close_look = bool(data.get("requires_close_look", False))
    if not phrase:
        phrase = (fallback_phrase or q).strip()
    return TargetExtract(
        target_phrase=phrase,
        question_type=qtype,
        notes=str(data.get("notes") or "").strip(),
        requires_close_look=close_look,
    )


def build_inventory_brief(
    *,
    n_observations: int = 0,
    graph_labels: list[str] | None = None,
    tried_obs_ids: list[int] | None = None,
    n_rounds: int = 0,
    n_nav: int = 0,
) -> str:
    """Neutral map/status context for VLM assess — no SigLIP/OWL verdicts.

    Detector scores guide *where* to navigate next (graph growth); they must not
    appear here or they bias ``present`` / MCQ letters (e.g. ABSENT → "None").
    """
    labels = [str(x) for x in (graph_labels or []) if str(x).strip()][:12]
    parts = [
        f"observations_seen={int(n_observations)}",
        f"rounds={int(n_rounds)}",
        f"nav_steps={int(n_nav)}",
    ]
    if labels:
        parts.append("graph_labels=" + ", ".join(labels))
    if tried_obs_ids:
        parts.append("tried_obs_ids=" + ",".join(str(i) for i in tried_obs_ids[-12:]))
    return "\n".join(parts)


def assess_view_with_vlm(
    client: Any,
    *,
    question: str,
    rgb: np.ndarray | None,
    inventory: str = "",
    target_phrase: str = "",
    is_mcq: bool = True,
) -> ViewAssessment:
    """Multimodal VLM: is this image enough to answer the question?"""
    q = (question or "").strip()
    target = (target_phrase or "").strip()
    if is_mcq:
        user = (
            f"Question:\n{q}\n\n"
            f"Target phrase (hint): {target or '(none)'}\n\n"
            f"Inventory:\n{inventory or '(none)'}\n\n"
            "Look at the image. Return JSON with keys:\n"
            "  target: string\n"
            "  present: bool — is the target / relevant evidence visible?\n"
            "  answerable: bool — can you confidently pick the MCQ answer from THIS view "
            "(and inventory)? For location/state questions, presence alone is not enough.\n"
            "  need_more_views: bool\n"
            "  suggested_answer: MCQ letter A-D or null if not answerable\n"
            "  reason: short explanation\n"
        )
    else:
        # Open-ended find / localize questions (OVMM find: "Where is the table?").
        # There is no MCQ letter set, so answerable means the target is actually in
        # view and localizable — not "can I pick A-D".
        user = (
            f"Question:\n{q}\n\n"
            f"Target phrase (hint): {target or '(none)'}\n\n"
            f"Inventory:\n{inventory or '(none)'}\n\n"
            "Look at the image. Return JSON with keys:\n"
            "  target: string\n"
            "  present: bool — is the target / relevant evidence visible in this view?\n"
            "  answerable: bool — is the target clearly visible so its location can be "
            "determined from THIS view (and inventory)? Presence alone is enough.\n"
            "  need_more_views: bool\n"
            "  suggested_answer: short answer text or null if not answerable\n"
            "  reason: short explanation\n"
        )
    if rgb is None:
        return ViewAssessment(
            target=target,
            present=False,
            answerable=False,
            need_more_views=True,
            reason="no rgb for VLM assess",
        )

    # Prefer generate_multimodal when the shared VL client is exposed.
    raw = ""
    vl = getattr(client, "_vl", None)
    if vl is not None and hasattr(vl, "generate_multimodal"):
        try:
            raw = str(
                vl.generate_multimodal(
                    user,
                    system_prompt=_ASSESS_SYSTEM,
                    max_new_tokens=192,
                    image=np.asarray(rgb),
                    reset_context=True,
                )
            )
        except Exception:
            raw = ""
    if not raw:
        # Gemini-style list command: text + PIL
        try:
            from PIL import Image

            pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
            raw = _call_eqa_client(
                client,
                [user, pil],
                system_prompt=_ASSESS_SYSTEM,
                max_new_tokens=192,
            )
        except Exception:
            raw = _call_eqa_client(
                client,
                user + "\n(Image unavailable — answerable must be false.)",
                system_prompt=_ASSESS_SYSTEM,
                max_new_tokens=128,
            )

    data = _parse_json_object(raw)
    suggested = data.get("suggested_answer")
    if suggested is not None:
        suggested = str(suggested).strip() or None
        if suggested and suggested.upper()[:1] in "ABCD":
            suggested = suggested.upper()[:1]
    return ViewAssessment(
        target=str(data.get("target") or target).strip(),
        present=bool(data.get("present", False)),
        answerable=bool(data.get("answerable", False)),
        need_more_views=bool(data.get("need_more_views", not bool(data.get("answerable", False)))),
        suggested_answer=suggested,
        reason=str(data.get("reason") or "").strip(),
        raw=raw[:1000],
    )
