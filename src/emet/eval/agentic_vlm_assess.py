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
from collections.abc import Iterable
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


def _json_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a JSON-ish boolean without treating non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "null", "none", ""}:
        return False
    return bool(default)


def _normalize_suggested_answer(value: Any) -> str | None:
    """Preserve semantic choice text while accepting legacy letter-only replies."""
    text = str(value or "").strip()
    if not text:
        return None
    match = re.fullmatch(
        r"(?:answer\s*[:=-]\s*)?([A-D])(?:\s*[\).}]|\s*)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return text


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
    close_look = _json_bool(data.get("requires_close_look", False))
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
    appear here or they bias ``present`` / MCQ answers (e.g. ABSENT → "None").
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


def unique_image_arrays(images: Iterable[Any], *, max_images: int | None = None) -> list[np.ndarray]:
    """Return non-empty image arrays once each, preserving input order."""

    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], str, bytes]] = set()
    for image in images:
        if image is None:
            continue
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.size == 0:
            continue
        arr = np.ascontiguousarray(arr)
        key = (tuple(int(dim) for dim in arr.shape), arr.dtype.str, arr.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(arr)
        if max_images is not None and len(out) >= int(max_images):
            break
    return out


def assess_view_with_vlm(
    client: Any,
    *,
    question: str,
    rgb: np.ndarray | None,
    inventory: str = "",
    target_phrase: str = "",
    is_mcq: bool = True,
    siglip_evidence: str = "",
    close_look_crop: np.ndarray | None = None,
    multi_close_look_crops: list[np.ndarray] | None = None,
) -> ViewAssessment:
    """Multimodal VLM: is this image enough to answer the question?

    ``siglip_evidence`` is an optional image-similarity hint (e.g. a SigLIP score for
    the target phrase on this view). It is **evidence, not ground truth**: small /
    visually ambiguous targets (a sugar cube vs a brick) can score high on both, so
    the VLM should weigh it against what it actually sees rather than treat it as a
    verdict.

    ``close_look_crop`` is an optional zoomed region around the target (for count /
    clock / fine-detail questions). When provided, it is shown to the VLM *in
    addition to* the wide frame so fine detail is readable.

    ``multi_close_look_crops`` (opt-in) adds several crops from different views of
    the same target so the VLM can aggregate temporal evidence before deciding
    (DeWorldSG-style) — useful when a single glance, even zoomed, is ambiguous.
    """
    q = (question or "").strip()
    target = (target_phrase or "").strip()
    evidence = (siglip_evidence or "").strip()
    evidence_line = f"\nVisual evidence (image-text similarity): {evidence}\n" if evidence else "\n"
    if is_mcq:
        user = (
            f"Question:\n{q}\n\n"
            f"Target phrase (hint): {target or '(none)'}\n\n"
            f"Inventory:\n{inventory or '(none)'}\n"
            f"{evidence_line}"
            "Look at the image. Return JSON with keys:\n"
            "  target: string\n"
            "  present: bool — is the target / relevant evidence visible?\n"
            "  answerable: bool — can you confidently pick the MCQ answer from THIS view "
            "(and inventory)? For location/state questions, presence alone is not enough.\n"
            "  need_more_views: bool\n"
            "  suggested_answer: exact semantic option text without its A/B/C/D label, "
            "or null if not answerable\n"
            "  reason: short explanation\n"
        )
    else:
        # Open-ended find / localize questions (OVMM find: "Where is the table?").
        # There is no MCQ option set, so answerable means the target is actually in
        # view and localizable.
        user = (
            f"Question:\n{q}\n\n"
            f"Target phrase (hint): {target or '(none)'}\n\n"
            f"Inventory:\n{inventory or '(none)'}\n"
            f"{evidence_line}"
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

    # When a zoomed crop is available, describe it so the VLM knows to read it for detail.
    crop_line = ""
    current_crop = (
        np.asarray(close_look_crop)
        if close_look_crop is not None and getattr(close_look_crop, "ndim", 0) == 3
        else None
    )
    multi_candidates = [
        np.asarray(c) for c in (multi_close_look_crops or []) if c is not None and getattr(c, "ndim", 0) == 3
    ]
    crops = unique_image_arrays(
        ([current_crop] if current_crop is not None else []) + multi_candidates,
        max_images=3,
    )
    has_current_crop = current_crop is not None and bool(crops) and np.array_equal(crops[0], current_crop)
    if has_current_crop:
        current_crop = crops[0]
        multi = crops[1:]
        crop_line = (
            "\nA zoomed crop of the target region is attached as a second image. "
            "Use it to read fine detail (count objects, read a clock/label) — do not "
            "rely on the wide frame alone for detail.\n"
        )
    else:
        current_crop = None
        multi = crops
    n_extra = len(crops)
    if multi:
        crop_line += (
            f"\n{n_extra} zoomed crops of the target from different views are attached "
            "(the current view plus earlier close looks). Aggregate across all of them "
            "before answering — a single view may be ambiguous.\n"
        )
    user = f"{user}{crop_line}"

    # Prefer generate_multimodal when the shared VL client is exposed.
    raw = ""
    images = [np.asarray(rgb)]
    if current_crop is not None:
        images.append(current_crop)
    images.extend(multi)
    vl = getattr(client, "_vl", None)
    if vl is not None and hasattr(vl, "generate_multimodal"):
        try:
            raw = str(
                vl.generate_multimodal(
                    user,
                    system_prompt=_ASSESS_SYSTEM,
                    max_new_tokens=192,
                    image=images[0] if len(images) == 1 else images,
                    reset_context=True,
                )
            )
        except Exception:
            raw = ""
    if not raw:
        # Gemini-style list command: text + PIL
        try:
            from PIL import Image

            pil = [Image.fromarray(np.asarray(im, dtype=np.uint8), mode="RGB") for im in images]
            raw = _call_eqa_client(
                client,
                [user, *pil],
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
    suggested = _normalize_suggested_answer(data.get("suggested_answer"))
    answerable = _json_bool(data.get("answerable", False))
    return ViewAssessment(
        target=str(data.get("target") or target).strip(),
        present=_json_bool(data.get("present", False)),
        answerable=answerable,
        need_more_views=_json_bool(data.get("need_more_views", not answerable), default=not answerable),
        suggested_answer=suggested,
        reason=str(data.get("reason") or "").strip(),
        raw=raw[:1000],
    )
