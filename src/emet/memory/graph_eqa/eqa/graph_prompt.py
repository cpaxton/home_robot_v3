# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""EQA answer parse, prompt assembly, and MCQ salvage/vote."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

from emet.habitat.metrics import (
    extract_mcq_letter,
)
from emet.memory.graph_eqa.eqa.mcq_debias import (
    LETTERS,
    extract_single_letter,
    letter_to_original_index,
    match_freeform_to_choice,
    rotated_choice_order,
    tally_choice_votes,
)
from emet.memory.graph_eqa.graph_types import (
    _LANDMARK_GENERIC_TOKENS,
    SIGLIP_CONFIRM_THRESHOLD,
    SIGLIP_PRESENT_THRESHOLD,
    distinctive_choice_tokens,
    label_matches_relevant_object,
    parse_eqa_action,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)



def _get_image_descriptions_str(
    self,
    obs_ids: list[int],
    *,
    omit_labels_for_obs: set[int] | None = None,
) -> str:
    """Build IMAGE_DESCRIPTIONS for attached EQA images only (Image 1..N).

    When ``omit_labels_for_obs`` contains an obs id (already tagged on SCENE_GRAPH
    Image-N lines), emit coords/nav suffix only so labels are not restated.
    """
    if not obs_ids:
        return "IMAGE_DESCRIPTIONS: (none)"
    skip_labels = {int(x) for x in (omit_labels_for_obs or set())}
    id_to_obs = {int(o.obs_id): o for o in self._observations}
    options: list[str] = []
    for img_idx, oid in enumerate(obs_ids, start=1):
        obs = id_to_obs.get(int(oid))
        if obs is None:
            continue
        if int(obs.obs_id) in skip_labels and self._obs_is_object_place(int(obs.obs_id)):
            line = f"Image {img_idx}. at ({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f});"
        else:
            lbl = ", ".join(obs.labels) if obs.labels else "object"
            line = f"Image {img_idx}. {lbl} at ({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f});"
        node = next((n for n in self._nodes if int(n.obs_id) == int(obs.obs_id)), None)
        if node is not None:
            line += self._node_nav_status_suffix(node)
        if self._obs_is_frontier(obs.obs_id):
            line += " unexplored frontier;"
        elif obs.description and "unexplored" in obs.description.lower():
            line += f" {obs.description.strip()};"
        options.append(line)
    return "IMAGE_DESCRIPTIONS: " + "\n".join(options) if options else "IMAGE_DESCRIPTIONS: (none)"

@staticmethod
def _coerce_eqa_confidence(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    s = str(raw).strip().lower().replace(" ", "")
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    return "true" in s

@staticmethod
def _normalize_eqa_answer_field(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        s = str(int(raw)) if float(raw) == int(raw) else str(raw)
    else:
        s = str(raw).strip()
    # JSON / labeled MCQ often wraps the letter in quotes or trailing punctuation.
    m = re.search(r"\b([A-Da-d])\b", s)
    if m and len(s) <= 4:
        return m.group(1).upper()
    return s.replace("\n", " ").replace("\t", " ").strip()

@classmethod
def _parse_answer_from_json_dict(cls, data: dict[str, Any]) -> tuple[str, str, bool, str, str] | None:
    """Map a JSON answer object onto the labeled-field tuple, or None if unusable."""
    if not isinstance(data, dict):
        return None
    # Accept common key aliases from chatty VLMs.
    key_map = {str(k).strip().lower(): k for k in data}

    def _get(*names: str) -> Any:
        for n in names:
            k = key_map.get(n)
            if k is not None:
                return data[k]
        return None

    if _get("answer", "ans") is None and _get("reasoning", "reason") is None:
        return None
    reasoning = str(_get("reasoning", "reason") or "").strip().replace("\n", " ").replace("\t", " ")
    answer = cls._normalize_eqa_answer_field(_get("answer", "ans"))
    confidence = cls._coerce_eqa_confidence(_get("confidence", "confident"))
    action_raw = _get("action", "next_action")
    if action_raw is None:
        action = ""
    else:
        action = str(action_raw).strip().replace("\n", " ").replace("\t", " ")
    confidence_reasoning = (
        str(_get("confidence_reasoning", "confidence_reason", "conf_reasoning") or "")
        .strip()
        .replace("\n", " ")
        .replace("\t", " ")
    )
    return reasoning, answer, confidence, action, confidence_reasoning

def parse_answer(
    self,
    answer_outputs: str,
    *,
    prefer_json: bool = True,
    json_prefill: str | None = None,
) -> tuple[str, str, bool, str, str]:
    """Parse mLLM output into reasoning, answer, confidence, action, confidence_reasoning.

    Tries a JSON object first (HM-EQA / chat-style contract), then the legacy labeled
    ``Reasoning:/Answer:/…`` scrape so old HISTORY / DualMem traces still work.
    """
    text = answer_outputs or ""
    if prefer_json:
        from emet.utils.json_parse import first_json_dict_lenient

        data = first_json_dict_lenient(text, prefill=json_prefill)
        if data is not None:
            parsed = self._parse_answer_from_json_dict(data)
            if parsed is not None:
                return parsed

    # Labeled scrape is case-insensitive; strip light markdown noise.
    lowered = text.replace("*", "").replace("#", "").lower()

    def extract_between(src: str, start: str, end: str) -> str:
        pattern = re.compile(
            rf"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}",
            flags=re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(src)
        if not m:
            return ""
        return m.group(1).strip().replace("\n", " ").replace("\t", " ")

    def extract_after(src: str, start: str) -> str:
        pattern = re.compile(rf"{re.escape(start)}\s*(.*)", flags=re.IGNORECASE | re.DOTALL)
        m = pattern.search(src)
        if not m:
            return ""
        return m.group(1).strip().replace("\n", " ").replace("\t", " ")

    reasoning = extract_between(lowered, "reasoning:", "answer:")
    answer = extract_between(lowered, "answer:", "confidence:")
    confidence_text = extract_between(lowered, "confidence:", "action:")
    confidence = "true" in confidence_text.replace(" ", "").lower()
    action = extract_between(lowered, "action:", "confidence_reasoning:")
    confidence_reasoning = extract_after(lowered, "confidence_reasoning:")
    if not answer.strip():
        m = re.search(r"answer\s*:\s*([a-d])\b", lowered)
        if m:
            answer = m.group(1).upper()
    if not answer.strip():
        m = re.search(r"(?:^|\n)\s*([a-d])\s*(?:\n|$)", lowered)
        if m:
            answer = m.group(1).upper()
    # Terse letter-only replies (``A}``, ``A) <choice text>``, ``A.``) that skip
    # both the JSON contract and the labeled scrape — common under a trailing
    # ``Answer:`` cue with remote/weaker VLMs.
    if not answer.strip():
        terse = extract_mcq_letter(text)
        if terse:
            answer = terse
    answer = self._normalize_eqa_answer_field(answer) if answer.strip() else answer
    return reasoning, answer, confidence, action, confidence_reasoning

@staticmethod
def format_eqa_history_outcome(
    *,
    answer: str,
    confidence: bool,
    action: str,
    reasoning: str,
    salvage: bool = False,
    obs_ids: list[int] | None = None,
) -> str:
    """One-line HISTORY entry — semantic answer/outcome, not a raw model replay."""
    ans = (answer or "").strip().replace("\n", " ")[:40] or "?"
    act = (action or "").strip().replace("\n", " ")
    act_bit = ""
    if act:
        kind, display_index = parse_eqa_action(act)
        if display_index is not None:
            act_bit = f"read{display_index}" if kind == "read" else str(display_index)
    reason = (reasoning or "").replace("\n", " ").strip()[:80]
    obs_bit = ""
    if obs_ids:
        obs_bit = f" obs={','.join(str(int(x)) for x in obs_ids)}"
    return (
        f"Iter: answer={ans} conf={str(bool(confidence)).lower()} "
        f"action={act_bit or '-'} salvage={1 if salvage else 0}{obs_bit} | {reason}"
    )

@staticmethod
def estimate_eqa_prompt_tokens(text: str) -> int:
    from emet.llms.eqa_vl_settings import estimate_eqa_prompt_tokens

    return estimate_eqa_prompt_tokens(text)

@classmethod
def _truncate_scene_graph_text(cls, graph_str: str, *, drop_edges: bool, max_node_lines: int | None) -> str:
    """Trim SCENE_GRAPH edges and/or lowest-ranked (trailing) node lines for budget."""
    if not graph_str.startswith("SCENE_GRAPH"):
        return graph_str
    prefix, _, body = graph_str.partition("\n")
    if not body.strip():
        return graph_str
    lines = body.split("\n")
    node_lines: list[str] = []
    edge_lines: list[str] = []
    tail_lines: list[str] = []
    in_tail = False
    for line in lines:
        if in_tail or line.startswith(("CONFIRMED_MEMORY", "Rooms:", "GRAPH_COUNT")):
            in_tail = True
            tail_lines.append(line)
            continue
        if line.startswith("  ") and "(" in line:
            edge_lines.append(line)
        else:
            node_lines.append(line)
    if max_node_lines is not None and max_node_lines >= 0:
        # Keep highest-ranked nodes (to_string emits best-first).
        node_lines = node_lines[:max_node_lines]
    if drop_edges:
        edge_lines = []
    kept = node_lines + edge_lines + tail_lines
    return prefix + ("\n" + "\n".join(kept) if kept else "")

@classmethod
def _trim_confirmed_memory_block(cls, block: str, *, max_lines: int) -> str:
    if max_lines < 0 or not block:
        return block
    lines = block.split("\n")
    if len(lines) <= 1:
        return block
    head, rest = lines[0], lines[1:]
    return "\n".join([head] + rest[:max_lines])

@classmethod
def build_eqa_prompt_text(
    cls,
    *,
    question_line: str,
    extra_hints: list[str] | None = None,
    memory_summary: str | None = None,
    history_entries: list[str] | None = None,
    history_start_index: int = 0,
    graph_str: str,
    view_status_str: str | None = None,
    img_desc_str: str,
    max_tokens: int = 2500,
) -> list[str]:
    """Assemble EQA text blocks under an approximate token budget.

    Truncation order: oldest HISTORY → CONFIRMED_MEMORY / merged tail lines →
    SCENE_GRAPH edges → lowest-ranked SCENE_GRAPH node labels.
    """
    hints = list(extra_hints or [])
    history = list(history_entries or [])
    mem = memory_summary or ""
    graph = graph_str
    view_status = (view_status_str or "").strip()
    img = img_desc_str
    max_tok = int(max_tokens)

    def _parts(hist: list[str], mem_block: str, graph_block: str) -> list[str]:
        out: list[str] = [question_line]
        out.extend(hints)
        if mem_block:
            out.append(mem_block)
        out.append("HISTORY: ")
        for i, h in enumerate(hist):
            out.append("Iteration_" + str(history_start_index + i) + ":" + h)
        out.append(graph_block)
        if view_status:
            out.append(view_status)
        out.append(img)
        return out

    def _tok(parts: list[str]) -> int:
        return cls.estimate_eqa_prompt_tokens("\n".join(parts))

    parts = _parts(history, mem, graph)
    if max_tok <= 0 or _tok(parts) <= max_tok:
        return parts

    # 1) Drop oldest HISTORY entries.
    while history and _tok(_parts(history, mem, graph)) > max_tok:
        history = history[1:]
        history_start_index += 1
    parts = _parts(history, mem, graph)
    if _tok(parts) <= max_tok:
        return parts

    # 2) Trim CONFIRMED_MEMORY / merged tail lines.
    if mem:
        for n in (8, 4, 2, 1, 0):
            mem = cls._trim_confirmed_memory_block(mem, max_lines=n) if n else ""
            if _tok(_parts(history, mem, graph)) <= max_tok:
                return _parts(history, mem, graph)
    # Also trim merged-memory tail inside SCENE_GRAPH.
    if "CONFIRMED_MEMORY" in graph:
        g_lines = graph.split("\n")
        try:
            idx = next(i for i, ln in enumerate(g_lines) if ln.startswith("CONFIRMED_MEMORY"))
        except StopIteration:
            idx = -1
        if idx >= 0:
            for n_tail in (4, 2, 0):
                head = g_lines[: idx + (0 if n_tail == 0 else 1)]
                tail = [] if n_tail == 0 else g_lines[idx + 1 : idx + 1 + n_tail]
                # Keep Rooms: and GRAPH_COUNT lines even when the preceding
                # merged-memory tail is trimmed.
                protected = [ln for ln in g_lines[idx + 1 :] if ln.startswith(("Rooms:", "GRAPH_COUNT"))]
                tail = [ln for ln in tail if not ln.startswith(("Rooms:", "GRAPH_COUNT"))]
                graph_try = "\n".join(head + tail + protected)
                if _tok(_parts(history, mem, graph_try)) <= max_tok:
                    graph = graph_try
                    return _parts(history, mem, graph)

    # 3) Drop SCENE_GRAPH edges.
    graph = cls._truncate_scene_graph_text(graph, drop_edges=True, max_node_lines=None)
    parts = _parts(history, mem, graph)
    if _tok(parts) <= max_tok:
        return parts

    # 4) Drop lowest-ranked node labels (trailing lines after rank order).
    body_lines = graph.split("\n")[1:] if "\n" in graph else []
    n_nodes = sum(
        1
        for ln in body_lines
        if (ln and not ln.startswith("  ") and not ln.startswith(("CONFIRMED_MEMORY", "Rooms:", "GRAPH_COUNT")))
    )
    for keep in list(range(max(0, n_nodes - 1), -1, -1)):
        graph_try = cls._truncate_scene_graph_text(graph, drop_edges=True, max_node_lines=keep)
        if _tok(_parts(history, mem, graph_try)) <= max_tok:
            return _parts(history, mem, graph_try)
    return _parts(history, mem, cls._truncate_scene_graph_text(graph, drop_edges=True, max_node_lines=0))

def _any_confirmed_phrase_present(self) -> bool:
    for phrase in self._confirmed_memory_phrases():
        if self._object_present_in_graph_or_siglip(phrase):
            return True
    return False

def _visibility_location_mcq_hint(self, choices: list[str]) -> str:
    lines = "\n".join(f"  {chr(65 + i)}) {choice}" for i, choice in enumerate(choices[:4]))
    return (
        "LOCATION_MCQ: The options are places, not yes/no. When the question asks "
        "'did you see … anywhere?', answer with the matching location option text for "
        "WHERE the object was observed. Do not output its A/B/C/D label. "
        "Prefer landmarks visible in the attached images of the object; "
        "treat WORKING_MEMORY / CONFIRMED_MEMORY / SCENE_GRAPH as views to look at, "
        "not as the WHERE answer. Do not cite Node N or xyz as the location. "
        "Never answer yes/no on answer:.\n"
        f"{lines}"
    )

def _salvage_answer_letter(self, question: str, commands: list[Any]) -> str:
    """Terse follow-up when the main EQA output never produced an ``answer:`` field.

    The VLM names the semantic answer; this compatibility helper maps that text to
    the benchmark letter expected by older callers.
    """
    if self.eqa_client is None:
        return ""
    from emet.habitat.metrics import parse_mcq_choices_from_question

    choices = parse_mcq_choices_from_question(question)
    if not choices:
        return ""
    images = [c for c in commands if isinstance(c, Image.Image)]
    directive = (
        "Answer the multiple-choice question with the exact text of the best option. "
        "Do not output its A/B/C/D label. Do not caption images or explain.\n"
        f"Question: {question}"
    )
    try:
        salvage_raw = self.eqa_client([directive, *images])
    except Exception as e:
        _logger.warning(f"EQA answer salvage failed ({e})")
        return ""
    text = (salvage_raw or "").strip()
    idx = match_freeform_to_choice(text, choices)
    if idx is not None and 0 <= idx < min(len(choices), len(LETTERS)):
        return LETTERS[idx]
    # Read-only compatibility for old clients/traces that still return a letter.
    return extract_mcq_letter(text, choices)

def _neighbor_label_blob_for_present_objects(self) -> str:
    """Concatenate nearest-furniture labels around PRESENT question objects."""
    object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
    labels: list[str] = []
    for obj in self._confirmed_memory_phrases():
        matches = [n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)]
        sig = self._siglip_match_for_phrase(obj)
        sig_present = sig is not None and float(sig[0]) >= SIGLIP_PRESENT_THRESHOLD
        if not matches and not sig_present:
            continue
        if matches:
            anchor_xyz = np.asarray(matches[0].xyz, dtype=np.float64)
            exclude_ids = {int(n.node_id) for n in matches}
        else:
            assert sig is not None
            anchor_xyz = np.asarray(sig[1], dtype=np.float64)
            exclude_ids = set()
        for n, _dist in self._nearest_object_neighbors(
            anchor_xyz, exclude_node_ids=exclude_ids, max_neighbors=2, max_dist_m=3.0
        ):
            labels.extend(str(lab) for lab in (n.labels or []) if lab)
    return " ".join(labels).lower()

def _score_choices_against_label_blob(
    self,
    choices: list[str],
    blob: str,
    *,
    ignore_generic: bool = False,
) -> list[int]:
    """Per-option token overlap scores against a lowercase label blob."""
    blob_l = (blob or "").lower()
    scores: list[int] = []
    for ch in choices[:4]:
        tokens = distinctive_choice_tokens(ch)
        score = 0
        for t in tokens:
            if ignore_generic and t in _LANDMARK_GENERIC_TOKENS:
                continue
            if t in blob_l:
                score += 2
            elif any(lab.startswith(t) or t.startswith(lab) for lab in blob_l.split()):
                score += 1
        scores.append(score)
    return scores

def _unique_best_choice_letter(self, scores: list[int]) -> str:
    if not scores or max(scores) < 1:
        return ""
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    if len(winners) != 1:
        return ""
    return chr(65 + winners[0])

def _any_graph_label_match_for_confirmed(self) -> bool:
    """True when at least one confirmed phrase matches a non-frontier graph/obs label."""
    object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
    for obj in self._confirmed_memory_phrases():
        if any(label_matches_relevant_object(obj, lab) for n in object_nodes for lab in (n.labels or [])):
            return True
        if any(label_matches_relevant_object(obj, lab) for o in self._observations for lab in (o.labels or [])):
            return True
    return False

def _location_letter_from_option_label_hits(self, choices: list[str]) -> str:
    """Map MCQ options onto graph/obs labels (e.g. refrigerator in graph → that letter)."""
    from emet.habitat.metrics import choices_are_location_mcq

    if not choices_are_location_mcq(choices):
        return ""
    parts: list[str] = []
    for n in self._nodes:
        if n.is_frontier or n.is_viewpoint:
            continue
        parts.extend(str(lab) for lab in (n.labels or []) if lab)
    for o in self._observations:
        parts.extend(str(lab) for lab in (o.labels or []) if lab)
    blob = " ".join(parts).lower()
    return self._unique_best_choice_letter(self._score_choices_against_label_blob(choices, blob))

def _location_letter_from_attached_images(self, choices: list[str], obs_ids: list[int]) -> str:
    """Map MCQ options onto labels of the attached Image 1..N observations.

    Prefer Image 1 landmarks; only fall back to the full attached set when Image 1
    does not uniquely map to a choice (avoids bowl-on-table / kitchen-cabinet noise).
    """
    from emet.habitat.metrics import choices_are_location_mcq

    if not choices_are_location_mcq(choices) or not obs_ids:
        return ""
    by_id = {int(o.obs_id): o for o in self._observations}

    def _blob_for(oids: list[int]) -> str:
        parts: list[str] = []
        for oid in oids:
            o = by_id.get(int(oid))
            if o is None:
                continue
            parts.extend(str(lab) for lab in (o.labels or []) if lab)
        return " ".join(parts).lower()

    primary = self._unique_best_choice_letter(
        self._score_choices_against_label_blob(choices, _blob_for(obs_ids[:1]), ignore_generic=True)
    )
    if primary:
        return primary
    return self._unique_best_choice_letter(
        self._score_choices_against_label_blob(choices, _blob_for(obs_ids), ignore_generic=True)
    )

def _equipment_letter_from_target_distances(self, choices: list[str]) -> str:
    """For under-equipment MCQs, pick the option whose equipment label is closest to the target."""
    from emet.habitat.metrics import choices_are_location_mcq

    if not choices_are_location_mcq(choices):
        return ""
    # Need a target object with known xyz (graph or strong SigLIP).
    object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
    anchors: list[np.ndarray] = []
    for obj in self._confirmed_memory_phrases():
        matches = [n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)]
        for n in matches[:3]:
            anchors.append(np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2])
        sig = self._siglip_match_for_phrase(obj)
        if sig is not None and float(sig[0]) >= SIGLIP_CONFIRM_THRESHOLD:
            anchors.append(np.asarray(sig[1], dtype=np.float64).reshape(-1)[:2])
    if not anchors:
        return ""
    anchor = anchors[0]

    # Only apply when ≥2 options look like "under <equipment>".
    underish = sum(1 for ch in choices[:4] if "under" in (ch or "").lower())
    if underish < 2:
        return ""

    best_letter = ""
    best_dist = float("inf")
    ties = 0
    matched_options = 0
    for i, ch in enumerate(choices[:4]):
        tokens = distinctive_choice_tokens(ch)
        if not tokens:
            continue
        # Find nearest graph node matching this option's equipment tokens.
        option_hit = False
        for n in object_nodes:
            labs = [str(lab).lower() for lab in (n.labels or []) if lab]
            if not labs:
                continue
            if not any(any(t in lab or lab.startswith(t) or t.startswith(lab) for lab in labs) for t in tokens):
                continue
            option_hit = True
            xy = np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2]
            dist = float(np.linalg.norm(anchor - xy))
            if dist < best_dist - 1e-6:
                best_dist = dist
                best_letter = chr(65 + i)
                ties = 1
            elif abs(dist - best_dist) <= 1e-6 and chr(65 + i) != best_letter:
                ties += 1
        if option_hit:
            matched_options += 1
    # Need ≥2 equipment options grounded in the graph (bike alone must not win).
    if matched_options < 2 or ties != 1 or best_dist == float("inf"):
        return ""
    return best_letter

def _location_letter_from_nearest_memory(self, choices: list[str]) -> str:
    """Map PRESENT nearest-furniture labels onto a location MCQ letter (no VLM).

    Used when the model answers yes/no or picks a room that conflicts with
    CONFIRMED_MEMORY neighbors (e.g. woven basket nearest armchair → D).
    Prefer graph-label matches for the question object; SigLIP-only PRESENT is
    weaker and should not override a letter that attached images support.
    """
    from emet.habitat.metrics import choices_are_location_mcq

    if not choices_are_location_mcq(choices) or not self._any_confirmed_phrase_present():
        return ""
    # Prefer direct option↔graph label hits when unique (fridge vs dining table).
    direct = self._location_letter_from_option_label_hits(choices)
    equip = self._equipment_letter_from_target_distances(choices)
    if equip:
        return equip
    blob = self._neighbor_label_blob_for_present_objects()
    nearest = self._unique_best_choice_letter(self._score_choices_against_label_blob(choices, blob))
    if direct and nearest and direct != nearest:
        # Conflict: trust option landmarks in the graph over nearest-furniture of a
        # possibly wrong SigLIP anchor when we lack a graph label on the target.
        if self._any_graph_label_match_for_confirmed():
            return nearest
        return direct
    return nearest or direct

def _salvage_location_mcq_letter(
    self,
    question: str,
    choices: list[str],
    commands: list[Any],
) -> str:
    """Re-ask for semantic location text, then map it for legacy callers."""
    if self.eqa_client is None or len(choices) < 2:
        return ""
    images = [c for c in commands if isinstance(c, Image.Image)]
    memory = self._relevant_memory_summary()
    choice_lines = "\n".join(f"  {chr(65 + i)}) {choice}" for i, choice in enumerate(choices[:4]))
    stem = question.split("Answer:")[0].strip()
    directive = (
        "CONFIRMED_MEMORY lists candidate views to inspect, not the answer. "
        "This is a WHERE-did-you-see-it multiple choice question. "
        "Answer from the attached images with the exact text of the single best "
        "location option, without its A/B/C/D label. Do NOT answer yes/no or explain.\n"
    )
    if memory:
        directive += memory + "\n"
    directive += f"Question: {stem}\nOptions:\n{choice_lines}"
    try:
        salvage_raw = self.eqa_client([directive, *images])
    except Exception as e:
        _logger.warning(f"Location-MCQ salvage failed ({e})")
        return ""
    text = (salvage_raw or "").strip()
    idx = match_freeform_to_choice(text, choices)
    if idx is not None and 0 <= idx < min(len(choices), len(LETTERS)):
        return LETTERS[idx]
    return extract_mcq_letter(text, choices)

def vote_mcq_letter(
    self,
    question: str,
    choices: list[str],
    *,
    max_votes: int = -1,
) -> str:
    """Debiased final MCQ letter (see mcq_debias.py).

    Two stages, both letter-token-free at the selection step:
      1. Free-form ask ("answer in a few words", no choices shown) matched to the
         closest choice by token overlap — immune to MCQ position bias.
      2. Fallback: choice-rotation voting — re-ask with cyclically rotated choice
         orders, map each reply back to the original choice index, majority-vote.

    ``max_votes`` caps stage 2 when the caller is latency-sensitive (e.g. the
    agentic forced-answer ladder at budget exhaustion); ``-1`` = unlimited.
    Returns the winning original letter, or ``""`` when neither stage finds a
    clear signal (caller keeps its main answer). Details in ``self.last_mcq_debias``.
    """
    self.last_mcq_debias = {}
    if self.eqa_client is None or len(choices) < 2:
        return ""
    n = min(4, len(choices))
    images = [
        Image.fromarray(o.rgb.astype(np.uint8), mode="RGB")
        for o in self._observations
        if o.obs_id in set(self.last_eqa_obs_ids)
    ]

    from emet.habitat.metrics import choices_are_location_mcq

    memory = ""
    if self.memory_summary_enabled and choices_are_location_mcq(choices):
        memory = self._relevant_memory_summary()
    freeform_directive = (
        "Look at the images and answer the question in a few words. "
        "Do not use option letters. Do not caption images. Do not explain.\n"
        f"Question: {question}"
    )
    if memory:
        freeform_directive = memory + "\n" + freeform_directive
    try:
        freeform = (self.eqa_client([freeform_directive, *images]) or "").strip()
    except Exception as e:
        _logger.warning(f"MCQ freeform vote failed ({e})")
        freeform = ""
    ff_idx = match_freeform_to_choice(freeform, choices[:n])
    if ff_idx is not None:
        letter = LETTERS[ff_idx]
        self.last_mcq_debias = {
            "letter": letter,
            "freeform": freeform[:300],
            "freeform_match": letter,
            "votes": [],
            "prior": None,
            "replies": [],
        }
        return letter

    prior_text = str(self.last_eqa_parsed[1] or "")
    prior_index = match_freeform_to_choice(prior_text, choices[:n])
    if prior_index is None:
        prior_index = letter_to_original_index(extract_single_letter(prior_text, n), rotation=0, n_choices=n)
    votes: list[int | None] = []
    replies: list[str] = []
    n_votes = int(max_votes) if int(max_votes) >= 0 else n
    question_stem = re.split(r"\s+A\s*[\).]\s*", question, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    for r in range(min(n, n_votes)):
        order = rotated_choice_order(n, r)
        option_lines = "\n".join(f"- {choices[idx]}" for idx in order)
        directive = (
            "Answer the multiple-choice question with the exact text of the best "
            "option. Do not output an option letter, caption images, or explain.\n"
            f"Question: {question_stem}\nOptions:\n{option_lines}"
        )
        try:
            reply = self.eqa_client([directive, *images])
        except Exception as e:
            _logger.warning(f"MCQ letter vote failed ({e})")
            reply = ""
        replies.append((reply or "").strip()[:200])
        votes.append(match_freeform_to_choice(reply, choices[:n]))
    win = tally_choice_votes(votes, choices[:n], prior_index=prior_index)
    letter = LETTERS[win] if win is not None else ""
    self.last_mcq_debias = {
        "letter": letter,
        "freeform": freeform[:300],
        "freeform_match": None,
        "votes": [None if v is None else LETTERS[v] for v in votes],
        "prior": None if prior_index is None else LETTERS[prior_index],
        "replies": replies,
    }
    return letter
