# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SigLIP and detector verify_siglip for the agentic GraphEQA executor."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from emet.mapping.voxel_localize import (
    is_current_view_sentinel,
    is_proposal_handle,
)
from emet.memory.graph_eqa.agentic.config import (
    SIGLIP_IMAGE_ABSENT_THRESHOLD,
    SIGLIP_IMAGE_PRESENT_THRESHOLD,
    _feat_list,
)
from emet.memory.graph_eqa.agentic.policy import AgenticState, EvidenceRecord
from emet.memory.graph_eqa.graph_memory import (
    SIGLIP_PRESENT_THRESHOLD,
    VerifyResult,
    label_matches_relevant_object,
    question_stem_for_keywords,
)
from emet.memory.graph_eqa.labels import _QUESTION_VERB_FILLERS
from emet.utils.logger import Logger

_logger = Logger(__name__)


def _dense_max_sim_for_rgb(self, phrase: str, rgb: np.ndarray | None) -> float | None:
    """Max patch-token SigLIP cosine for *phrase* vs *rgb* (MaskSigLIP space).

    Full-frame pool cosines rarely exceed ~0.12; dense max is closer to DynaMem's
    per-point features and can reach the 0.21 PRESENT bar when the object is in view.
    """
    text = (phrase or "").strip()
    if not text or rgb is None:
        return None
    enc = None
    gm = self.graph_memory
    if gm is not None:
        enc = getattr(gm, "_confirmed_memory_siglip_encoder", None)
    if enc is None:
        try:
            from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

            enc = get_shared_mask_siglip_encoder()
        except Exception:
            return None
    try:
        import torch
        import torch.nn.functional as F

        text_t = enc.encode_text(text).detach().float().reshape(-1)
        text_t = text_t / (text_t.norm() + 1e-12)
        inputs = enc._to_model_inputs(enc.processor(images=np.asarray(rgb, dtype=np.uint8), return_tensors="pt"))
        with torch.no_grad():
            out = enc.model.vision_model(inputs["pixel_values"], output_hidden_states=True)
            feat = F.normalize(out.last_hidden_state.float(), dim=-1)
            sims = feat @ text_t.to(device=feat.device, dtype=feat.dtype).reshape(-1, 1)
            return float(sims.max().item())
    except Exception as e:
        _logger.warning(f"dense_max_sim_for_rgb failed: {e}")
        return None


def _voxel_max_sim_for_obs(self, phrase: str, obs_id: int) -> tuple[float, str] | None:
    """Max per-point SigLIP cosine for *phrase* on voxel features in *obs_id*.

    Full-frame ``encode_image`` cosines top out ~0.10–0.16 in Habitat, while DynaMem's
    0.21 / 0.28 thresholds were calibrated on dense voxel point features. Prefer those
    when the voxel map is available so PRESENT can actually fire.
    """
    text = (phrase or "").strip()
    if not text:
        return None
    vm = getattr(self.agent, "voxel_map", None)
    if vm is None or not hasattr(vm, "find_alignment_over_model"):
        return None
    try:
        alignments = vm.find_alignment_over_model(text)
    except Exception:
        return None
    if alignments is None:
        return None
    try:
        import torch

        if not isinstance(alignments, torch.Tensor):
            return None
    except ImportError:
        return None
    a = alignments.detach().cpu().float().reshape(-1)
    if a.numel() == 0:
        return None
    sm = getattr(vm, "semantic_memory", None)
    counts = getattr(sm, "_obs_counts", None) if sm is not None else None
    channel = "voxel_global"
    if counts is not None:
        try:
            import torch

            c = counts.detach().cpu().long().reshape(-1)
            if c.numel() == a.numel():
                mask = c == int(obs_id)
                if bool(mask.any()):
                    a = a[mask]
                    channel = "voxel_obs"
        except Exception:
            pass
    return float(a.max().item()), channel


def _detector_for_verify(self) -> Any | None:
    if self._presence_detector_initialized:
        return self._presence_detector
    self._presence_detector_initialized = True
    backend = os.environ.get("EMET_EQA_AGENTIC_VERIFIER", "").strip().lower()
    if backend in ("", "none", "siglip"):
        return None
    try:
        from emet.eval.presence_verifiers import (
            OwlV2PresenceDetector,
            YoloEPresenceDetector,
        )

        if backend == "owlv2":
            self._presence_detector = OwlV2PresenceDetector()
        elif backend == "yoloe":
            self._presence_detector = YoloEPresenceDetector()
        else:
            raise ValueError(f"unsupported EMET_EQA_AGENTIC_VERIFIER={backend!r}")
    except Exception as exc:
        _logger.warning(f"presence detector unavailable: {exc}")
        self._presence_detector = None
    return self._presence_detector


def _graph_label_matches(self, phrase: str, obs_id: int) -> bool:
    gm = self.graph_memory
    stored = gm._observation_by_id(int(obs_id)) if gm is not None and hasattr(gm, "_observation_by_id") else None
    labels = getattr(stored, "labels", None) or []
    return any(label_matches_relevant_object(phrase, str(label)) for label in labels)


def _relation_sufficient_for_obs(self, obs_id: int) -> bool:
    from emet.habitat.metrics import (
        choices_are_count_mcq,
        choices_are_location_mcq,
        parse_mcq_choices_from_question,
    )

    choices = parse_mcq_choices_from_question(self.question)
    if not choices:
        choices = list(getattr(self, "_mcq_choices", None) or [])
    if not choices:
        return True
    # Cardinality MCQs: fused target presence is enough; the answer VLM counts.
    if choices_are_count_mcq(choices):
        return True
    if not choices_are_location_mcq(choices):
        return True
    gm = self.graph_memory
    stored = gm._observation_by_id(int(obs_id)) if gm is not None and hasattr(gm, "_observation_by_id") else None
    labels = [str(label) for label in (getattr(stored, "labels", None) or [])]
    if any(label_matches_relevant_object(choice, label) for choice in choices for label in labels):
        return True
    # Landmark overlap with nearby graph nodes (room/fixture context).
    if gm is not None and hasattr(gm, "labels_near_obs"):
        try:
            near_labels = [str(x) for x in (gm.labels_near_obs(int(obs_id)) or [])]
        except Exception:
            near_labels = []
        if any(label_matches_relevant_object(choice, label) for choice in choices for label in near_labels):
            return True
    return False


def _inventory_labels(self, *, limit: int = 12) -> list[str]:
    gm = self.graph_memory
    labels: list[str] = []
    seen: set[str] = set()
    for obs in list(getattr(gm, "_observations", None) or []):
        for lab in list(getattr(obs, "labels", None) or []):
            s = str(lab).strip()
            if not s or s.lower() in seen:
                continue
            seen.add(s.lower())
            labels.append(s)
            if len(labels) >= limit:
                return labels
    return labels


def _tool_verify_siglip(self, phrase: str, obs_id: int | None) -> dict[str, Any]:
    gm = self.graph_memory
    if gm is None:
        return {"ok": False, "error": "no graph_memory"}
    text = (phrase or "").strip()
    if not text:
        if self._target_phrase:
            text = self._target_phrase
        else:
            phrases = list(getattr(gm, "_relevant_phrases", None) or []) + list(
                getattr(gm, "_relevant_objects", None) or []
            )
            # Prefer phrases from the question stem over MCQ-option nouns
            # (``fruit bowl`` > ``kitchen island``), then noun compounds over
            # leading verb fillers (``fruit bowl`` > ``looking``).
            stem = question_stem_for_keywords(self.question).lower()
            ranked = sorted(
                phrases,
                key=lambda p: (
                    1 if (p or "").strip().lower() in stem else 0,
                    0 if (p or "").split()[:1] and (p or "").split()[0].lower() in _QUESTION_VERB_FILLERS else 1,
                    len((p or "").split()),
                    len(p or ""),
                ),
                reverse=True,
            )
            text = ranked[0] if ranked else self.question
    oid = obs_id
    if oid is None or is_current_view_sentinel(oid) or is_proposal_handle(oid):
        # -1 = live camera; voxel handles are not frames. Score the current view, or fail.
        oid = self._latest_obs_id()
    if oid is None:
        return {
            "ok": False,
            "status": "NOT_A_VIEW",
            "obs_id": obs_id,
            "phrase": text,
            "error": (
                "obs_id is a detection handle or missing, and there is no captured "
                "camera frame; investigate() to capture, then verify the new obs_id"
            ),
            "verified": self._verified,
        }
    oid = int(oid)
    if is_proposal_handle(oid):
        return {
            "ok": False,
            "status": "NOT_A_VIEW",
            "obs_id": oid,
            "phrase": text,
            "error": "cannot verify a detection handle; capture a real observation first",
            "verified": self._verified,
        }
    verify_target = self._action_target_for_obs(oid)
    if self._router_enabled and self._evidence_policy.state != AgenticState.VERIFY and oid not in self._fresh_obs_ids:
        return {
            "ok": False,
            "error": (f"obs_id {oid} is stale; SEARCH must APPROACH/capture a fresh view before VERIFY"),
            "status": "REQUIRES_FRESH_VIEW",
            "obs_id": oid,
            "phrase": text,
            "target_kind": verify_target.kind,
            "target_id": verify_target.stable_id,
            "view_id": verify_target.view_id,
            "room": verify_target.room,
            "verified": self._verified,
        }
    # Interactive rule: one verify per view. Re-checking the same obs burns rounds
    # without new evidence — move first (nav / explore), then verify the fresh frame.
    if self._obs_already_verified(oid):
        self._append_trace(
            {
                "tool": "verify_siglip",
                "phrase": text,
                "obs_id": oid,
                "decision": "SKIPPED_SAME_VIEW",
                "sim": 0.0,
                "prior": self._tried.get(oid),
            }
        )
        return {
            "ok": False,
            "error": f"obs_id {oid} already verified ({self._tried.get(oid)}); navigate or explore for a new view",
            "status": "SKIPPED_SAME_VIEW",
            "obs_id": oid,
            "phrase": text,
            "target_kind": verify_target.kind,
            "target_id": verify_target.stable_id,
            "view_id": verify_target.view_id,
            "room": verify_target.room,
            "verified": self._verified,
        }
    rgb = None
    live_obs = None
    robot = getattr(self.agent, "robot", None)
    if robot is not None and hasattr(robot, "get_observation"):
        try:
            live_obs = robot.get_observation()
            if live_obs is not None and getattr(live_obs, "rgb", None) is not None:
                rgb = np.asarray(live_obs.rgb)
        except Exception:
            pass
    if rgb is None:
        stored = gm._observation_by_id(int(oid)) if hasattr(gm, "_observation_by_id") else None
        stored_rgb = getattr(stored, "rgb", None) if stored is not None else None
        if isinstance(stored_rgb, np.ndarray) and stored_rgb.ndim == 3:
            rgb = np.asarray(stored_rgb)
    result = gm.verify_phrase_at_obs(text, int(oid), rgb=rgb, min_sim=self.verify_min_sim)
    full_frame_sim = float(result.sim)
    voxel = self._voxel_max_sim_for_obs(text, int(oid))
    voxel_sim = float(voxel[0]) if voxel is not None else None
    voxel_ch = voxel[1] if voxel is not None else None
    dense_sim = self._dense_max_sim_for_rgb(text, rgb)
    detector_evidence = None
    detector = self._detector_for_verify()
    if detector is not None and rgb is not None:
        try:
            from emet.eval.presence_verifiers import detector_crop_evidence

            enc = getattr(gm, "_confirmed_memory_siglip_encoder", None)
            detector_evidence = (
                detector_crop_evidence(detector, enc, rgb, text) if enc is not None else detector.score(rgb, text)
            )
        except Exception as exc:
            _logger.warning(f"hybrid presence verify failed: {exc}")
    # Best image-space score (full-frame pool vs dense patch).
    best_img = full_frame_sim
    verify_channel = "full_frame"
    if dense_sim is not None and float(dense_sim) > best_img:
        best_img = float(dense_sim)
        verify_channel = "dense_patch"
    # Voxel wins when it clears the DynaMem bar; else use image three-band.
    if voxel_sim is not None and voxel_sim >= SIGLIP_PRESENT_THRESHOLD:
        status, ok, sim_out, verify_channel = "PRESENT", True, float(voxel_sim), str(voxel_ch)
    elif best_img >= SIGLIP_IMAGE_PRESENT_THRESHOLD:
        status, ok, sim_out = "PRESENT", True, float(best_img)
    elif best_img >= SIGLIP_IMAGE_ABSENT_THRESHOLD:
        status, ok, sim_out = "CANDIDATE", False, float(best_img)
    else:
        status, ok, sim_out = "ABSENT", False, float(best_img)
    result = VerifyResult(
        status=status,
        sim=float(sim_out),
        obs_id=int(oid),
        phrase=text,
        ok=ok,
        text_feat=result.text_feat,
        img_feat=result.img_feat,
    )
    self._last_verify = result
    self._tried[int(result.obs_id)] = f"verify {result.status} sim={float(result.sim):.2f}"
    self._fresh_obs_ids.discard(int(result.obs_id))
    # After motion, capture may not mint a new obs_id (unit mocks / no map growth).
    # Complete APPROACH → VERIFY on the view we are scoring; otherwise open a view hyp.
    if self._evidence_policy.state == AgenticState.APPROACH:
        active = self._evidence_policy.active_hypothesis_id
        if active is not None:
            self._policy_approached(active, int(result.obs_id))
    if self._evidence_policy.state != AgenticState.VERIFY:
        hypothesis_id = self._begin_policy_approach("view", int(result.obs_id), text)
        self._policy_approached(hypothesis_id, int(result.obs_id))
    hypothesis_id = self._evidence_policy.active_hypothesis_id or f"view:{int(result.obs_id)}"
    graph_label_match = self._graph_label_matches(text, int(result.obs_id))
    assessment = None
    try:
        evidence = EvidenceRecord(
            hypothesis_id=hypothesis_id,
            obs_id=int(result.obs_id),
            phrase=text,
            full_frame_sim=full_frame_sim,
            dense_sim=dense_sim,
            voxel_sim=voxel_sim,
            detector_score=(float(detector_evidence.score) if detector_evidence is not None else None),
            crop_siglip_sim=(detector_evidence.crop_siglip_sim if detector_evidence is not None else None),
            graph_label_match=graph_label_match,
            detector_backend=(str(detector_evidence.backend) if detector_evidence is not None else None),
            bbox_xyxy=(detector_evidence.bbox_xyxy if detector_evidence is not None else None),
            provenance=tuple(
                channel
                for channel, value in (
                    ("full_frame", full_frame_sim),
                    ("dense_patch", dense_sim),
                    (str(voxel_ch or "voxel"), voxel_sim),
                    (
                        str(detector_evidence.backend) if detector_evidence is not None else "detector",
                        detector_evidence.score if detector_evidence is not None else None,
                    ),
                )
                if value is not None
            ),
        )
        self._evidence_policy.add_evidence(evidence)
        assessment = self._evidence_policy.assess(
            relation_sufficient=self._relation_sufficient_for_obs(int(result.obs_id))
        )
    except (RuntimeError, ValueError) as exc:
        _logger.warning(f"evidence-policy verify rejected: {exc}")
    # Cheap channels never unlock submit — multimodal VLM assess is the gate.
    proposal = {
        "phrase": text,
        "detector_score": (float(detector_evidence.score) if detector_evidence is not None else None),
        "dense_sim": dense_sim,
        "full_frame_sim": full_frame_sim,
        "decision": result.status,
        "obs_id": int(result.obs_id),
    }
    proposal_threshold = (
        SIGLIP_PRESENT_THRESHOLD if verify_channel.startswith("voxel") else SIGLIP_IMAGE_PRESENT_THRESHOLD
    )
    proposal_event_id = self._persist_agentic_evidence(
        stage="siglip_proposal",
        outcome=result.status.lower(),
        obs_id=int(result.obs_id),
        phrase=text,
        confidence=min(1.0, max(0.0, float(result.sim))),
        source=verify_channel,
        score=float(result.sim),
        threshold=float(proposal_threshold),
        payload={
            "full_frame_sim": full_frame_sim,
            "dense_sim": dense_sim,
            "voxel_sim": voxel_sim,
            "graph_label_match": bool(graph_label_match),
        },
    )
    proposal["evidence_event_id"] = proposal_event_id or None
    vlm_out = self._run_vlm_view_assess(
        rgb=rgb,
        phrase=text,
        obs_id=int(result.obs_id),
        proposal=proposal,
    )
    row = {
        "tool": "verify_siglip",
        "phrase": text,
        "obs_id": int(result.obs_id),
        "target_kind": verify_target.kind,
        "target_id": verify_target.stable_id,
        "view_id": verify_target.view_id,
        "room": verify_target.room,
        "sim": float(result.sim),
        "decision": result.status,
        "verify_channel": verify_channel,
        "full_frame_sim": full_frame_sim,
        "voxel_sim": voxel_sim,
        "dense_sim": dense_sim,
        "detector_backend": (detector_evidence.backend if detector_evidence is not None else None),
        "detector_score": (float(detector_evidence.score) if detector_evidence is not None else None),
        "detector_bbox_xyxy": (
            list(detector_evidence.bbox_xyxy)
            if detector_evidence is not None and detector_evidence.bbox_xyxy is not None
            else None
        ),
        "crop_siglip_sim": (detector_evidence.crop_siglip_sim if detector_evidence is not None else None),
        "graph_label_match": graph_label_match,
        "policy_state": self._evidence_policy.state,
        "presence_probability": (assessment.presence_probability if assessment is not None else None),
        "answerability_probability": (
            self._evidence_policy.beliefs[self._evidence_policy.active_hypothesis_id].answerability_probability
            if self._evidence_policy.active_hypothesis_id
            and self._evidence_policy.active_hypothesis_id in self._evidence_policy.beliefs
            else (assessment.answerability_probability if assessment is not None else None)
        ),
        "positive_channels": (list(assessment.positive_channels) if assessment is not None else []),
        "contradiction_channels": (list(assessment.contradiction_channels) if assessment is not None else []),
        # Submit unlock = VLM answerable (not cheap SigLIP/OWL alone).
        "fused_verified": bool(self._verified),
        "vlm_answerable": bool(self._verified),
        "answerable": bool(self._evidence_policy.state == AgenticState.ANSWER),
        "vlm_assess": vlm_out,
        "present_bar": (
            SIGLIP_PRESENT_THRESHOLD if verify_channel.startswith("voxel") else SIGLIP_IMAGE_PRESENT_THRESHOLD
        ),
        "absent_bar": None if verify_channel.startswith("voxel") else SIGLIP_IMAGE_ABSENT_THRESHOLD,
        "text_feat": _feat_list(result.text_feat),
        "img_feat": _feat_list(result.img_feat),
        "siglip_evidence_event_id": proposal_event_id or None,
    }
    labeler = getattr(robot, "hm3d_semantic_labeler", None) if robot is not None else None
    semantic = getattr(live_obs, "semantic", None) if live_obs is not None else None
    if labeler is not None and semantic is not None and hasattr(labeler, "visibility_for_phrase"):
        row.update(
            labeler.visibility_for_phrase(
                semantic,
                text,
                getattr(live_obs, "depth", None),
            )
        )
    xyt = self._robot_xyt_world()
    if xyt is not None:
        row["xyt"] = [float(x) for x in xyt.reshape(-1)[:3]]
    hyp = next((h for h in self._hypotheses if int(h.obs_id) == int(result.obs_id)), None)
    if hyp is not None:
        row["target_xyz"] = [float(x) for x in np.asarray(hyp.xyz).reshape(-1)[:3]]
        row["source"] = hyp.source
        self._attach_gt(row, hyp.xyz)
    else:
        row["source"] = "current_view"
    self._append_trace(row)
    return {
        "ok": True,
        "status": result.status,
        "sim": float(result.sim),
        "verified": self._verified,
        "vlm_answerable": bool(self._verified),
        "obs_id": int(result.obs_id),
        "phrase": text,
        "target_kind": verify_target.kind,
        "target_id": verify_target.stable_id,
        "view_id": verify_target.view_id,
        "room": verify_target.room,
        "verify_channel": verify_channel,
        "fused_verified": bool(self._verified),
        "answerable": bool(self._evidence_policy.state == AgenticState.ANSWER),
        "presence_probability": (assessment.presence_probability if assessment is not None else None),
        "answerability_probability": (
            self._evidence_policy.beliefs[self._evidence_policy.active_hypothesis_id].answerability_probability
            if self._evidence_policy.active_hypothesis_id
            and self._evidence_policy.active_hypothesis_id in self._evidence_policy.beliefs
            else (assessment.answerability_probability if assessment is not None else None)
        ),
        "siglip_evidence_event_id": proposal_event_id or None,
        "verified_evidence_event_ids": list(self._verified_evidence_event_ids),
        "vlm_assess": vlm_out,
    }
