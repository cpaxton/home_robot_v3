# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Per-iteration EQA VLM decision export (prompt text + attached images)."""

from __future__ import annotations

from typing import Any


class GraphAnswerTraceMixin:
    """Write ``eqa_decisions/`` when ``_eqa_decision_trace_dir`` is set on the memory."""

    def _record_eqa_decision_trace(
        self,
        *,
        iteration: int,
        question: str,
        text_blocks: list[Any],
        obs_ids: list[int],
        crop_obs_id: int | None,
        nav_fallback_count: int,
        relevant_images: list[Any],
        view_status: str,
        close_look_status: str,
        vlm_raw: str,
        parsed: dict[str, Any],
    ) -> None:
        trace_root = getattr(self, "_eqa_decision_trace_dir", None)
        if not trace_root:
            return
        from emet.eval.eqa_decision_trace import record_eqa_decision_iteration

        record_eqa_decision_iteration(
            trace_root,
            iteration,
            question=question,
            text_blocks=text_blocks,
            obs_ids=obs_ids,
            crop_obs_id=crop_obs_id,
            nav_fallback_count=nav_fallback_count,
            relevant_images=relevant_images,
            view_status=view_status,
            close_look_status=close_look_status,
            vlm_raw=vlm_raw,
            parsed=parsed,
        )
