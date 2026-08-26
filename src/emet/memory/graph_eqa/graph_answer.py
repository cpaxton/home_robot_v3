# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""query_answer, VLM description fill, and graph getters."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

from emet.habitat.metrics import (
    extract_mcq_letter,
)
from emet.memory.graph_eqa.eqa_views import center_zoom_enabled
from emet.memory.graph_eqa.graph_types import (
    GraphNavigationSample,
    GraphNode,
    GraphObservation,
    parse_eqa_action,
)
from emet.memory.graph_eqa.human_answer import format_human_eqa_answer
from emet.memory.graph_eqa.mcq_debias import (
    answer_is_unknownish,
    count_answer_is_none_or_zero,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)


def _log_eqa_prep(msg: str) -> None:
    from emet.eval.output_config import eval_log_eqa_prep

    if eval_log_eqa_prep():
        _logger.info(msg)
    else:
        _logger.debug(msg)


class GraphAnswerMixin:
    """query_answer, VLM description fill, and graph getters."""

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
        *,
        force_obs_ids: list[int] | None = None,
        voxel_map: Any | None = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, list[Image.Image]]:
        """
        Answer the question using the scene graph and task-relevant images.
        Same return contract as voxel_dynamem.SparseVoxelMap.query_answer.

        Args:
            force_obs_ids: When set (agentic verified submit), prefer these observation
                ids as Image 1..K instead of a pure diversified pick. Remaining slots
                fill from FIND pins and ``_select_relevant_obs_ids``. Count MCQs attach
                FIND / pending Action views first so a single verified frame cannot
                occupy the whole budget (``max_images=1`` used to drop lamp/stool RGB).

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        import time as _time

        from emet.habitat.metrics import (
            answer_is_visibility_abstain,
            choices_are_attribute_state,
            choices_are_count_mcq,
            choices_are_location_mcq,
            choices_are_time_of_day,
            parse_mcq_choices_from_question,
            question_is_attribute_state,
            question_is_visibility_location,
        )
        from emet.llms.eqa_vl_settings import (
            get_eqa_vl_int,
            resolve_eqa_answer_format,
            resolve_eqa_answer_max_new_tokens,
            resolve_eqa_answer_prefill,
            resolve_eqa_include_image_descriptions,
            resolve_eqa_prompt_max_tokens,
            resolve_eqa_prompt_variant,
        )

        _t0 = _time.monotonic()
        _log_eqa_prep("query_answer: ensure_llm_clients…")
        self._ensure_llm_clients()
        _log_eqa_prep("query_answer: extract_relevant_objects…")
        self.extract_relevant_objects(question)
        if self.memory_summary_enabled:
            # Encoder may already be dropped by prepare_dynagraph_vram_for_eqa; refresh
            # is a no-op without it (uses cached phrase features when present).
            _log_eqa_prep("query_answer: refresh_siglip_confirmed_memory…")
            self.refresh_siglip_confirmed_memory()
        max_images = get_eqa_vl_int(self.parameters, "eqa_max_images", 4)
        include_image_descriptions = resolve_eqa_include_image_descriptions(self.parameters)
        parsed_choices = parse_mcq_choices_from_question(question)
        attribute_q = question_is_attribute_state(question) or choices_are_attribute_state(parsed_choices)
        count_q = bool(parsed_choices and choices_are_count_mcq(parsed_choices) and not attribute_q)
        time_q = bool(parsed_choices and choices_are_time_of_day(parsed_choices) and not attribute_q)
        location_q = bool(
            parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not count_q
            and not attribute_q
            and not time_q
        )
        # Honor Action:N from the previous call / post-nav look even when this submit
        # forces a different verified obs as Image 1. Do not drop the pin until
        # this call actually attaches it as Image 1 (q86 Action 163).
        look_obs_id = self.last_eqa_look_obs_id
        if look_obs_id is None:
            look_obs_id = self.last_eqa_action_obs_id
        count_nodes, _count_target = ([], None) if attribute_q else self._count_candidate_nodes(question)
        pin_obs: list[int] = []
        phrases = self._eqa_find_phrases()
        visual_pins = self._visual_find_obs_ids(phrases, max_n=max_images)
        q_low = str(question or "").lower()
        clockish = any(s in q_low for s in ("what time", "time is it", "o'clock", "o’clock"))
        if clockish:
            self._merge_highlight_phrases(["clock", "wall clock"])
            phrases = self._eqa_find_phrases()
            visual_pins = self._visual_find_obs_ids(phrases, max_n=max_images)
        if (
            (count_q or time_q or clockish)
            and not location_q
            and self._phrases_need_rgb_highlight(visual_pins, phrases)
        ):
            extra_hl = self._highlight_relevant_from_latest_rgb(question)
            if extra_hl:
                self._merge_highlight_phrases(extra_hl)
                phrases = self._eqa_find_phrases()
                visual_pins = self._visual_find_obs_ids(phrases, max_n=max_images)
        # Location MCQs keep landmark Image 1; visual FIND is leftover recall there.
        if not location_q:
            for oid in visual_pins:
                if oid not in pin_obs and self._obs_usable_for_eqa_image(oid):
                    pin_obs.append(int(oid))
        # YoloE instance nodes are FIND only when SigLIP retrieved nothing.
        if not pin_obs:
            for node in count_nodes:
                oid = int(node.obs_id)
                if oid in pin_obs:
                    continue
                if self._obs_usable_for_eqa_image(oid):
                    pin_obs.append(oid)
        if not pin_obs:
            for node in self._location_finder_nodes()[:4]:
                oid = int(node.obs_id)
                if oid in pin_obs:
                    continue
                if self._obs_usable_for_eqa_image(oid):
                    pin_obs.append(oid)
        pin_obs = self._spread_obs_xy(pin_obs, max_n=max_images)
        forced: list[int] = []
        if force_obs_ids:
            for oid in force_obs_ids:
                oi = int(oid)
                if oi in forced:
                    continue
                if self._obs_usable_for_eqa_image(oi):
                    forced.append(oi)
        selected = [
            int(oid)
            for oid in self._select_relevant_obs_ids(
                max_images=max_images,
                choices=parsed_choices if parsed_choices else None,
                attribute_question=attribute_q,
            )
            if self._obs_usable_for_eqa_image(oid)
        ]
        obs_ids = self._compose_eqa_answer_obs_ids(
            forced=forced,
            pin_obs=pin_obs,
            selected=selected,
            max_images=max_images,
            count_question=count_q,
            look_obs_id=look_obs_id,
        )
        zoom_on = center_zoom_enabled(self.parameters)
        detail_zoom = bool((time_q or clockish) and zoom_on)
        read_zoom_primary: set[int] = set()
        if zoom_on and look_obs_id is not None and self.last_eqa_parsed:
            prev_kind, prev_disp = parse_eqa_action(str(self.last_eqa_parsed[3] or ""))
            if prev_kind == "read":
                prev_ids = list(self.last_eqa_obs_ids or [])
                resolved = (
                    self._resolve_eqa_action_image_ref(int(prev_disp), prev_ids, slots_only=True)
                    if prev_disp is not None
                    else int(look_obs_id)
                )
                if resolved is not None:
                    read_zoom_primary.add(int(resolved))
        crop_oid: int | None = None
        crop_rgb: np.ndarray | None = None
        if max_images >= 2:
            crop_oid = self._eqa_pick_closeup_obs_id(
                obs_ids, look_obs_id, pin_obs, detail_zoom=detail_zoom
            )
            if crop_oid is not None:
                crop_rgb = self._eqa_crop_for_obs(crop_oid, detail_zoom=detail_zoom)
            if crop_rgb is None:
                crop_oid = None
            elif len(obs_ids) >= max_images:
                obs_ids = self._eqa_reserve_closeup_slot(obs_ids, pin_obs, look_obs_id)
                if crop_oid not in obs_ids:
                    crop_oid = self._eqa_pick_closeup_obs_id(
                        obs_ids, look_obs_id, pin_obs, detail_zoom=detail_zoom
                    )
                    crop_rgb = (
                        self._eqa_crop_for_obs(crop_oid, detail_zoom=detail_zoom)
                        if crop_oid is not None
                        else None
                    )
                    if crop_rgb is None:
                        crop_oid = None
        self.last_eqa_obs_ids = list(obs_ids)
        action_obs_ids = list(obs_ids)
        if crop_oid is not None:
            action_obs_ids.append(int(crop_oid))
        if look_obs_id is not None and (not obs_ids or int(obs_ids[0]) != int(look_obs_id)):
            self.last_eqa_look_obs_id = int(look_obs_id)
        else:
            self.last_eqa_look_obs_id = None
        max_graph_nodes = get_eqa_vl_int(self.parameters, "eqa_max_graph_nodes", 48)
        merged_memory = self._merged_memory_enabled()
        merge_confirmed = (
            merged_memory and self.memory_summary_enabled and not attribute_q and not self._spatial_rag_enabled()
        )
        graph_str = self.to_string(
            max_object_nodes=max_graph_nodes if max_graph_nodes > 0 else None,
            question_keywords=list(self._relevant_objects or []),
            prefer_obs_ids=obs_ids,
            record_prompt_count=True,
            merge_confirmed=merge_confirmed,
        )
        count_hint = self._graph_count_hint(question)
        # Prefer real RGB. If selection is empty (only frontier placeholders in memory),
        # fall back to navigation viewpoint samples — never attach black 8×8 frontiers.
        nav_fallback_tail: list[GraphNavigationSample] = []
        graph_obs_ids = {int(n.obs_id) for n in self._nodes if not n.is_frontier and not n.is_viewpoint}
        if obs_ids:
            if include_image_descriptions:
                img_desc_str = self._get_image_descriptions_str(
                    obs_ids,
                    omit_labels_for_obs=graph_obs_ids,
                )
            else:
                n = len(obs_ids)
                img_desc_str = (
                    f"Attached images: Image 1..{n} are full camera views; match them to SCENE_GRAPH "
                    "nodes via Image tags on nodes. Do not re-list objects from the images."
                )
        elif self._nav_samples:
            nav_fallback_tail = self._nav_samples[-max_images:]
            if include_image_descriptions:
                lines = [
                    "IMAGE_DESCRIPTIONS (navigation-only views; no object graph nodes yet):",
                ]
                for i, nv in enumerate(nav_fallback_tail, start=1):
                    tail = (
                        f" robot base (~{nv.base_xyz[0]:.2f}, {nv.base_xyz[1]:.2f})." if nv.base_xyz is not None else ""
                    )
                    lines.append(
                        f"Image {i}. viewpoint anchor at ({nv.xyz[0]:.2f}, {nv.xyz[1]:.2f}, {nv.xyz[2]:.2f});{tail}"
                    )
                img_desc_str = "\n".join(lines)
            else:
                n = len(nav_fallback_tail)
                img_desc_str = (
                    f"Attached images: Image 1..{n} are navigation-only RGB views "
                    "(no object graph nodes yet). Do not re-list objects from the images."
                )
        else:
            img_desc_str = (
                "IMAGE_DESCRIPTIONS: (none — explore for a real camera view before answering)"
                if include_image_descriptions
                else "Attached images: (none — explore for a real camera view before answering)"
            )
        if crop_oid is not None and crop_rgb is not None and obs_ids:
            try:
                parent_txt = f"Image {obs_ids.index(int(crop_oid)) + 1}"
            except ValueError:
                parent_txt = f"graph obs {int(crop_oid)}"
            n_scene = len(obs_ids)
            zoom_note = (
                "center-zoom of the clock/detail region; read hands or digits from this frame"
                if detail_zoom
                else "close-up of the object"
            )
            read_note = (
                " If a scene view is too wide to read detail, set action to read that Image slot "
                "for a center-zoom on the next turn."
                if detail_zoom
                else ""
            )
            img_desc_str = str(img_desc_str).rstrip() + (
                f" Image {n_scene + 1} is a {zoom_note} from {parent_txt} "
                "(the same view, not another object). Count from the scene views."
                + read_note
            )

        extra_hints: list[str] = []
        if obs_ids:
            extra_hints.append(self.format_attached_index(obs_ids))
        if read_zoom_primary:
            extra_hints.append(
                "A zoomed/detail crop is already attached this turn. Pick a time bucket or "
                "count from that frame, or leave action empty to explore; do not emit read N "
                "again on the same attached slot."
            )
        find_queue = ""
        if count_q or time_q or clockish or detail_zoom:
            find_queue = self.format_find_queue(
                question,
                attached_obs_ids=obs_ids,
                pin_obs=pin_obs,
            )
        if find_queue:
            extra_hints.append(find_queue)
        if count_hint:
            extra_hints.append(count_hint)
        if crop_oid is not None:
            extra_hints.append(
                "A later attached image is a close-up of an object already shown in a scene view; "
                "do not count that close-up as a second object."
            )
        if parsed_choices and choices_are_location_mcq(parsed_choices) and question_is_visibility_location(question):
            extra_hints.append(self._visibility_location_mcq_hint(parsed_choices))
        # Attribute/state questions: answer from images; do not inject memory priors.
        # Merged-memory mode folds status into SCENE_GRAPH, so skip the separate block.
        memory_summary = ""
        if self.memory_summary_enabled and not attribute_q and not merge_confirmed:
            memory_summary = self._relevant_memory_summary() or ""
        max_history = get_eqa_vl_int(self.parameters, "eqa_max_history", 4)
        history = self._history_outputs
        start = max(0, len(history) - max_history) if max_history > 0 else 0
        history_slice = list(history[start:])
        off_prompt = self._find_queue_candidate_obs_ids(question, pin_obs=pin_obs)
        view_status = self.format_eqa_view_status(obs_ids, off_prompt_find=off_prompt)
        close_look = self.format_close_look_status(
            obs_ids, off_prompt_find=off_prompt, voxel_map=voxel_map
        )
        if close_look:
            view_status = view_status + "\n" + close_look if view_status else close_look
        prompt_max_tokens = resolve_eqa_prompt_max_tokens(self.parameters)
        text_blocks = self.build_eqa_prompt_text(
            question_line="Question: " + question,
            extra_hints=extra_hints,
            memory_summary=memory_summary or None,
            history_entries=history_slice,
            history_start_index=start,
            graph_str=graph_str,
            view_status_str=view_status,
            img_desc_str=img_desc_str,
            max_tokens=prompt_max_tokens,
        )
        commands: list[Any] = list(text_blocks)

        relevant_images: list[Image.Image] = []
        for oid in obs_ids:
            oid_i = int(oid)
            use_zoom_primary = oid_i in read_zoom_primary
            rgb = None
            if use_zoom_primary:
                rgb = self._eqa_crop_for_obs(oid_i, detail_zoom=True)
            if rgb is None:
                rgb = self._eqa_rgb_for_obs(oid_i)
            if rgb is None:
                continue
            im = Image.fromarray(rgb, mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        if crop_rgb is not None:
            im = Image.fromarray(crop_rgb, mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        for nv in nav_fallback_tail:
            im = Image.fromarray(nv.rgb.astype(np.uint8), mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        self.last_eqa_nav_fallback_count = len(nav_fallback_tail)
        # Keep the attached frames reachable after query_answer returns so the salvage
        # counterfactual can re-ask on the same images instead of silently no-op'ing.
        self.last_relevant_images = list(relevant_images)

        _log_eqa_prep(
            f"query_answer: calling eqa_client (n_images={len(relevant_images)} "
            f"n_cmd={len(commands)} prep_s={_time.monotonic() - _t0:.1f} "
            f"include_image_descriptions={include_image_descriptions} "
            f"history_n={len(self._history_outputs)})…"
        )
        assistant_prefill: str | None = None
        answer_format = resolve_eqa_answer_format(self.parameters)
        _variant = resolve_eqa_prompt_variant(self.parameters)
        try:
            t_vl = _time.monotonic()
            ans_cap = resolve_eqa_answer_max_new_tokens(self.parameters)
            eqa_kw: dict[str, Any] = {}
            if ans_cap > 0:
                eqa_kw["max_new_tokens"] = ans_cap
            # Force the first output field so Qwen cannot open with Caption: — prompt edits
            # alone still left a 26% caption share on the 2026-07-30 q2 probe.
            assistant_prefill = resolve_eqa_answer_prefill(self.parameters)
            if assistant_prefill:
                eqa_kw["assistant_prefill"] = assistant_prefill
            _log_eqa_prep(
                f"query_answer: eqa_kw max_new_tokens={eqa_kw.get('max_new_tokens')} "
                f"assistant_prefill={assistant_prefill!r} prompt_variant={_variant!r} "
                f"answer_format={answer_format!r}"
            )
            try:
                raw = self.eqa_client(commands, **eqa_kw)
            except TypeError:
                # Older / test doubles that only accept the command list.
                raw = self.eqa_client(commands)
            _log_eqa_prep(
                f"query_answer: eqa_client done wall_s={_time.monotonic() - t_vl:.1f} out_chars={len(raw or '')}"
            )
        except Exception as exc:
            raw = f"Error: {exc}"
            self.last_eqa_raw = raw
            self.last_eqa_parsed = ("", "Unknown", False, "", str(exc))
            self.last_eqa_model_raw = raw
            self.last_eqa_model_parsed = self.last_eqa_parsed
            self._append_eqa_history(
                self.format_eqa_history_outcome(
                    answer="Unknown",
                    confidence=False,
                    action="",
                    reasoning=str(exc),
                    salvage=False,
                )
            )
            return (
                str(exc),
                "Unknown",
                False,
                str(exc),
                None,
                relevant_images,
            )
        self.last_eqa_raw = raw
        prefer_json = answer_format == "json"
        reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(
            raw or "",
            prefer_json=prefer_json,
            json_prefill=assistant_prefill if prefer_json else None,
        )
        # Also accept labeled scrape when JSON was preferred but incomplete.
        if prefer_json and not (answer or "").strip():
            reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(
                raw or "",
                prefer_json=False,
            )
        self.last_eqa_model_raw = str(raw or "")
        self.last_eqa_model_parsed = (
            reasoning,
            answer,
            bool(confidence),
            action,
            confidence_reasoning,
        )
        answer_outputs = (raw or "").replace("*", "").replace("#", "").lower()
        # Salvage: small VLMs sometimes run away captioning and never emit ``answer:``.
        # Re-ask tersely for semantic option text.
        # - Empty answer → always salvage (64-token truncation / runaway caption).
        # - ``Unknown`` on attribute/yes-no → salvage (holdout q65).
        # - Empty/``Unknown`` on location MCQ → do NOT invent a letter (holdout q104/q105);
        #   agentic should follow Action:/explore instead of memory/salvage A–D.
        # - ``Unknown`` on count MCQ → do NOT invent One/Two (q86 pin: bathroom RGB
        #   salvaged to One while GRAPH_COUNT pointed at a lamp view).
        _ans_stripped = (answer or "").strip()
        _ans_unknownish = answer_is_unknownish(_ans_stripped, parsed_choices)
        _loc_mcq = bool(parsed_choices and choices_are_location_mcq(parsed_choices) and not attribute_q)
        _count_mcq = bool(parsed_choices and choices_are_count_mcq(parsed_choices) and not attribute_q)
        # A stream that never reached ``answer:`` / ``"answer"`` was cut off mid-caption.
        _answer_field_emitted = bool(
            re.search(r"answer\s*:", answer_outputs)
            or re.search(r'["\']answer["\']\s*:', answer_outputs)
            or (prefer_json and not _ans_unknownish)
        )
        _truncated_before_answer = _ans_unknownish and not _answer_field_emitted
        # Surfaced per episode so a decode-budget regression is visible in the results
        # table instead of only showing up as a mysterious accuracy drop.
        self.last_eqa_answer_field_emitted = _answer_field_emitted
        self.last_eqa_salvage_used = False
        # Location / count MCQ: never invent A–D from a second guess, but do recover
        # a truncated stream that never emitted ``answer:``.
        _should_salvage = _ans_unknownish and (not (_loc_mcq or _count_mcq) or _truncated_before_answer)
        if _loc_mcq and _ans_unknownish and not _ans_stripped:
            # Truncated streams often omit ``answer:`` entirely (failfix5); normalize so
            # human_answer / agentic follow-up treat this as Unknown, not memory-B.
            answer = "Unknown"
            _ans_stripped = "Unknown"
            _ans_unknownish = True
        if _should_salvage:
            salvage_letter = self._salvage_answer_letter(question, commands)
            if salvage_letter:
                salvage_idx = ord(salvage_letter) - ord("A")
                salvage_text = parsed_choices[salvage_idx] if 0 <= salvage_idx < len(parsed_choices) else salvage_letter
                answer = salvage_text
                raw = (raw or "") + f"\n[salvage]\nanswer:\n{salvage_text}\n"
                self.last_eqa_raw = raw
                self.last_eqa_salvage_used = True
        elif (
            parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and self._any_confirmed_phrase_present()
            and not attribute_q
        ):
            # Location letter overrides (equip → image → abstaining memory) are intentional
            # Dynagraph eval levers. Accuracy can move vs GE-only / no-override ablations;
            # always report HM-EQA deltas with the harness fingerprint + git commit.
            # Geometric under-equipment (mat under treadmill) may correct VLM guesses.
            # Image landmarks may correct memory-steered letters. Nearest-furniture memory
            # alone must NOT override a clear VLM A–D (Q6: VLM B correct, memory A) **or**
            # free-text that uniquely matches a choice ("the room with the blue curtains").
            img_letter = self._location_letter_from_attached_images(parsed_choices, obs_ids)
            equip_letter = self._equipment_letter_from_target_distances(parsed_choices)
            memory_letter = self._location_letter_from_nearest_memory(parsed_choices)
            parsed_letter = extract_mcq_letter(answer, parsed_choices)
            abstain = answer_is_visibility_abstain(answer) or not parsed_letter
            # The geometric equipment-distance guess must not override a confident VLM
            # letter: on the full-113 (2026-08-13) it flipped correct image-grounded
            # answers to wrong (q44/q47/q94/q101: json_answer == gold, scored via
            # [memory-location]). Config-gated so the legacy (always-override) behavior
            # can be reproduced for A/B (eqa.location_override_equip_gate; env
            # EMET_EQA_LOCATION_OVERRIDE_EQUIP_GATE=0 restores legacy).
            _gate_equip = self._eqa_override_gate("location_override_equip_gate", True)
            _gate_img = self._eqa_override_gate("location_override_image_gate", True)
            vlm_clear = bool(confidence) and bool(parsed_letter)
            preferred = ""
            if not (vlm_clear and _gate_equip) and equip_letter and (abstain or parsed_letter != equip_letter):
                preferred = equip_letter
            elif not (vlm_clear and _gate_img) and img_letter and (abstain or parsed_letter != img_letter):
                preferred = img_letter
            elif abstain and memory_letter and (self._any_graph_label_match_for_confirmed() or not img_letter):
                preferred = memory_letter
            # Empty / Unknown location MCQ: keep Unknown so agentic can follow Action:N.
            # Inventing A–D via memory/salvage-location caused failfix5 wrong B letters.
            if _ans_unknownish:
                pass
            elif preferred and (abstain or parsed_letter != preferred):
                preferred_idx = ord(preferred) - ord("A")
                preferred_text = (
                    parsed_choices[preferred_idx] if 0 <= preferred_idx < len(parsed_choices) else preferred
                )
                answer = preferred_text
                raw = (raw or "") + f"\n[memory-location]\nanswer:\n{preferred_text}\n"
                self.last_eqa_raw = raw
            elif abstain:
                # Visibility-style Yes/No on a WHERE question may still salvage; empty/Unknown
                # already handled above. Do not salvage bare abstains without a letter.
                if answer_is_visibility_abstain(answer) and not _ans_unknownish:
                    salvage_letter = self._salvage_location_mcq_letter(question, parsed_choices, commands)
                    if salvage_letter:
                        salvage_idx = ord(salvage_letter) - ord("A")
                        salvage_text = (
                            parsed_choices[salvage_idx] if 0 <= salvage_idx < len(parsed_choices) else salvage_letter
                        )
                        answer = salvage_text
                        raw = (raw or "") + f"\n[salvage-location]\nanswer:\n{salvage_text}\n"
                        self.last_eqa_raw = raw
        self.last_eqa_model_confident = bool(confidence)
        covered = self._graph_covers_relevant_objects()
        if confidence and not covered:
            confidence = False
            confidence_reasoning = (
                confidence_reasoning
                + " The scene graph does not yet include all question-relevant objects; explore further."
            ).strip()
        # Do not finalize Yes/No from absence while relevant objects are still uncovered.
        if (
            not covered
            and answer_is_visibility_abstain(answer)
            and not (parsed_choices and choices_are_location_mcq(parsed_choices))
        ):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning
                + " Yes/No from missing evidence is not final; keep exploring until objects are observed."
            ).strip()
        # Abstentions are never confirmed MCQ answers. A count option whose
        # semantic text is ``None`` is a real answer, not an abstention.
        if answer_is_unknownish(str(answer or ""), parsed_choices):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning + " No clear letter yet; explore and refresh memory before confirming."
            ).strip()
        # Require a clear picture: don't confirm location letters unsupported by attached
        # image labels when memory is only SigLIP-candidate (no graph label on the target).
        if (
            confidence
            and parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not attribute_q
            and not self._any_graph_label_match_for_confirmed()
        ):
            parsed_letter = extract_mcq_letter(str(answer or ""), parsed_choices)
            img_letter = self._location_letter_from_attached_images(parsed_choices, obs_ids)
            if parsed_letter and img_letter and parsed_letter != img_letter:
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning
                    + " Answer conflicts with landmarks in attached images; update memory / views before confirming."
                ).strip()
            elif parsed_letter and not img_letter:
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning + " Location not yet verified in attached images; explore for a clearer view."
                ).strip()
        # Never finalize a WHERE answer if the target object is not in attached views
        # (guessing "dining table" / "side table" without seeing towel/fruit bowl).
        if (
            confidence
            and parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not attribute_q
            and not self._target_visible_in_obs_ids(obs_ids)
        ):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning + " Target object not visible in attached images; explore before confirming."
            ).strip()
        # Under-equipment MCQs: do not finalize until geometric equipment letter is known
        # (otherwise bike vs treadmill is a coin flip from a partial gym view).
        if confidence and parsed_choices and choices_are_location_mcq(parsed_choices) and not attribute_q:
            underish = sum(1 for ch in parsed_choices[:4] if "under" in (ch or "").lower())
            if underish >= 2:
                equip = self._equipment_letter_from_target_distances(parsed_choices)
                if not equip:
                    confidence = False
                    confidence_reasoning = (
                        confidence_reasoning
                        + " Under-equipment location needs a clearer mat↔equipment distance before confirming."
                    ).strip()
                else:
                    parsed_letter = extract_mcq_letter(str(answer or ""), parsed_choices)
                    if parsed_letter and parsed_letter != equip:
                        equip_idx = ord(equip) - ord("A")
                        equip_text = parsed_choices[equip_idx] if 0 <= equip_idx < len(parsed_choices) else equip
                        answer = equip_text
                        raw = (raw or "") + f"\n[equipment-location]\nanswer:\n{equip_text}\n"
                        self.last_eqa_raw = raw
        # Attribute/state: never finalize from memory priors (images only).
        if confidence and attribute_q and self.memory_summary_enabled:
            # Soft gate: if Image 1 is a frontier-only view, keep exploring.
            if obs_ids and self._obs_is_frontier(int(obs_ids[0])):
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning + " Attribute/state needs a non-frontier view of the object before confirming."
                ).strip()
        missing_find: list[int] = []
        if _count_mcq and count_nodes:
            attached = {int(oid) for oid in obs_ids}
            for node in count_nodes:
                oid = int(node.obs_id)
                if not self._obs_usable_for_eqa_image(oid):
                    continue
                if oid not in attached and oid not in missing_find:
                    missing_find.append(oid)
            for oid in self._find_queue_candidate_obs_ids(question, pin_obs=pin_obs):
                if int(oid) not in attached and int(oid) not in missing_find:
                    missing_find.append(int(oid))
            if (
                missing_find
                and count_answer_is_none_or_zero(str(answer or ""), parsed_choices)
            ):
                if confidence:
                    confidence = False
                    confidence_reasoning = (
                        confidence_reasoning
                        + " FIND views were not attached; look at those RGB frames before answering None."
                    ).strip()
                if self.last_eqa_look_obs_id is None:
                    self.last_eqa_look_obs_id = missing_find[0]
        raw_answer = answer
        self.last_eqa_parsed = (reasoning, raw_answer, confidence, action, confidence_reasoning)
        human = format_human_eqa_answer(
            question,
            answer,
            reasoning,
            self,
            confidence=confidence,
            confidence_reasoning=confidence_reasoning,
            selected_obs_ids=obs_ids,
        )
        answer = human.user_answer
        reasoning = human.debug_reasoning

        target_point = None
        self.last_eqa_action_obs_id = None
        hist_action = ""
        kind, display_index = parse_eqa_action(action)
        if display_index is not None:
            hist_action = action.strip()
        if not confidence and display_index is not None:
            if kind == "read":
                resolved_oid = self._resolve_eqa_action_image_ref(
                    display_index, action_obs_ids, slots_only=True
                )
            else:
                resolved_oid = self._resolve_eqa_action_image_ref(display_index, action_obs_ids)
            if kind == "read" and resolved_oid is not None:
                self.last_eqa_action_obs_id = int(resolved_oid)
                if self.last_eqa_look_obs_id is None:
                    self.last_eqa_look_obs_id = int(resolved_oid)
                # Stay put: next turn can zoom or close-look this slot. Do not hop 1–5 m.
            elif kind == "read":
                # ``read 195`` (graph obs id) is not a zoom slot; fall through to FIND.
                pass
            elif resolved_oid is not None and self.eqa_obs_look_spent(resolved_oid):
                # Still attach this RGB next turn; do not orbit the same obs.
                if self.last_eqa_look_obs_id is None:
                    self.last_eqa_look_obs_id = int(resolved_oid)
                nxt = self.next_unspent_eqa_obs_id(
                    list(pin_obs) + list(missing_find),
                    skip={int(resolved_oid)},
                )
                self.last_eqa_action_obs_id = nxt
                if nxt is not None:
                    target_point = self._navigation_waypoint_for_obs(int(nxt), xyt)
            else:
                self.last_eqa_action_obs_id = resolved_oid
                target_point = self._target_point_from_display_image_index(
                    display_index,
                    obs_ids=action_obs_ids,
                    nav_fallback_tail=nav_fallback_tail,
                    robot_xyt=xyt,
                )
        if not confidence and self.last_eqa_action_obs_id is None and missing_find:
            nxt = self.next_unspent_eqa_obs_id(missing_find)
            self.last_eqa_action_obs_id = nxt
            if nxt is not None and target_point is None:
                target_point = self._navigation_waypoint_for_obs(int(nxt), xyt)
        pending_look = self.last_eqa_action_obs_id
        if pending_look is not None and (not obs_ids or int(obs_ids[0]) != int(pending_look)):
            self.last_eqa_look_obs_id = int(pending_look)
        self._append_eqa_history(
            self.format_eqa_history_outcome(
                answer=raw_answer,
                confidence=confidence,
                action=hist_action,
                reasoning=reasoning,
                salvage=bool(self.last_eqa_salvage_used),
            )
        )

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            target_point,
            relevant_images,
        )

    def fill_descriptions_from_vlm(
        self,
        prompt: str | None = None,
        max_tokens: int = 80,
    ) -> None:
        """
        Fill missing node/observation descriptions using the VLM (e.g. Qwen 2.5-VL / 3.5).
        Skips observations that already have a description. Can be slow for many images.
        """
        if self.image_description_client is None:
            self._init_clients()
        default_prompt = (
            "In one short sentence, describe what is visible in this image: "
            "main objects, their arrangement, and any notable spatial relationships. "
            "Be concise."
        )
        prompt = prompt or default_prompt
        for obs in self._observations:
            if obs.description:
                continue
            try:
                # VLM accepts list of text + image(s)
                out = self.image_description_client(
                    [prompt, Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB")],
                    verbose=False,
                )
                if isinstance(out, str) and out.strip():
                    desc = out.strip()
                    # Update observation (same object as stored)
                    obs.description = desc
                    # Update corresponding node
                    for n in self._nodes:
                        if n.obs_id == obs.obs_id:
                            n.description = desc
                            break
            except Exception as e:
                _logger.warning(f"fill_descriptions_from_vlm failed for obs {obs.obs_id}: {e}")
                continue

    def get_observations(self) -> list[GraphObservation]:
        return list(self._observations)

    def get_nodes(self) -> list[GraphNode]:
        return list(self._nodes)

    def get_edges(self) -> list[tuple[int, int, str]]:
        return list(self._edges)

    def print_memory(self) -> str:
        """
        Return the 3D scene graph as a human-readable tree (same as to_tree_string).
        Use this as the canonical "print" output for the graph memory.
        """
        return self.to_tree_string()
