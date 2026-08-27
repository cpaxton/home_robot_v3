# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Nav-target recall, verify/retract, and relevant-observation selection."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from emet.habitat.metrics import choices_are_location_mcq
from emet.memory.graph_eqa.graph_types import (
    _GRAPH_CANDIDATE_COUNT_DISCLAIMER,
    _LANDMARK_GENERIC_TOKENS,
    _QUESTION_LANDMARK_BOOST,
    SIGLIP_CONFIRM_THRESHOLD,
    SIGLIP_PRESENT_THRESHOLD,
    GraphObservation,
    NavHypothesis,
    VerifyResult,
    _object_match_tokens,
    countable_primary_label_matches,
    distinctive_choice_tokens,
    format_graph_node_candidates,
    heuristic_relevant_objects,
    heuristic_relevant_phrases,
    label_matches_relevant_object,
    location_mcq_landmark_phrases,
    node_display_name,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)



def hypothesize_nav_targets(
    self,
    question: str,
    max_k: int = 6,
    robot_xyt: np.ndarray | None = None,
    boost_phrases: list[str] | None = None,
) -> list[NavHypothesis]:
    """Retrieve a small diversified set of nav evidence cards for the router/fallback.

    Ranking is **recall only** (source tier + keyword/landmark hit + distance
    tiebreak). The VLM router decides where to go among the returned cards.
    """
    if not self._observations and not any(getattr(n, "is_frontier", False) for n in self._nodes):
        return []
    phrases: list[str] = []
    for raw in list(boost_phrases or []):
        p = str(raw or "").strip()
        if p and p not in phrases:
            phrases.append(p)
    for p in list(self._confirmed_memory_phrases()) + list(self._relevant_objects or []):
        if p and p not in phrases:
            phrases.append(p)
    if not phrases and question:
        self.extract_relevant_objects(question)
        for p in list(self._confirmed_memory_phrases()) + list(self._relevant_objects or []):
            if p and p not in phrases:
                phrases.append(p)
    # Always merge location-MCQ landmarks (even if extract already ran thin).
    for landmark in location_mcq_landmark_phrases(question):
        if landmark not in phrases:
            phrases.append(landmark)
    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
        cand_labels = [
            (
                int(n.obs_id),
                str(n.labels)[:40],
                bool(getattr(n, "is_viewpoint", False)),
                bool(getattr(n, "is_frontier", False)),
            )
            for n in self._nodes
        ]
        _logger.info(
            "[recall] q=%r phrases=%r n_obs=%d n_nodes=%d all_nodes=%s",
            question[:40],
            phrases[:6],
            len(self._observations),
            len(self._nodes),
            cand_labels,
        )
    scored: list[NavHypothesis] = []
    seen: set[int] = set()
    retracted = getattr(self, "_retracted_nav_claims", None) or set()

    def _claim_blocked(oid: int, phrase: str) -> bool:
        key = (int(oid), str(phrase or "").strip().lower())
        return key in retracted

    # Object / SigLIP cards need phrases; frontiers are still valid cold-start evidence.
    for phrase in phrases:
        for o in self._observations:
            oid = int(o.obs_id)
            if oid in seen or _claim_blocked(oid, phrase):
                continue
            if self._obs_is_frontier(oid):
                continue
            # Viewpoint-only / camera-station obs are not place cards.
            if not self._obs_is_object_place(oid):
                continue
            if any(label_matches_relevant_object(phrase, lab) for lab in (o.labels or [])):
                seen.add(oid)
                scored.append(
                    NavHypothesis(
                        phrase=phrase,
                        obs_id=oid,
                        xyz=np.asarray(o.xyz, dtype=float).reshape(-1)[:3].copy(),
                        score=0.0,
                        source="graph",
                    )
                )
        # Also match graph nodes (centroid) when observations lack the label string.
        for node in self._nodes:
            if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                continue
            oid = int(node.obs_id)
            if oid in seen or _claim_blocked(oid, phrase):
                continue
            if any(label_matches_relevant_object(phrase, lab) for lab in (node.labels or [])):
                seen.add(oid)
                scored.append(
                    NavHypothesis(
                        phrase=phrase,
                        obs_id=oid,
                        xyz=np.asarray(node.xyz, dtype=float).reshape(-1)[:3].copy(),
                        score=0.0,
                        source="graph",
                    )
                )
    for phrase in phrases:
        sig = self._siglip_match_for_phrase(phrase)
        if sig is None:
            continue
        sim, xyz, oid = float(sig[0]), np.asarray(sig[1], dtype=float), sig[2]
        if oid is None:
            continue
        oid = int(oid)
        if oid in seen or self._obs_is_frontier(oid) or _claim_blocked(oid, phrase):
            continue
        if not self._obs_is_object_place(oid):
            continue
        if sim >= SIGLIP_CONFIRM_THRESHOLD:
            source = "confirmed"
        elif sim >= SIGLIP_PRESENT_THRESHOLD:
            source = "siglip"
        else:
            continue
        seen.add(oid)
        scored.append(
            NavHypothesis(
                phrase=phrase,
                obs_id=oid,
                xyz=xyz.reshape(-1)[:3].copy(),
                score=0.0,
                source=source,
                siglip_sim=float(sim),
            )
        )
    for node in self._nodes:
        if not node.is_frontier or int(node.obs_id) in seen:
            continue
        # Do NOT attach the question object as the frontier phrase — that made
        # every frontier look like a "fruit bowl" hit and drowned graph places.
        scored.append(
            NavHypothesis(
                phrase="unexplored frontier",
                obs_id=int(node.obs_id),
                xyz=np.asarray(node.xyz, dtype=float).copy(),
                score=0.0,
                source="frontier",
            )
        )
        seen.add(int(node.obs_id))
    if not scored:
        return []
    scored = [self._recall_rank_score(hypothesis, question, robot_xyt) for hypothesis in scored]
    scored.sort(key=lambda h: (-h.score, h.path_cost, -h.obs_id))
    return self._pack_diversified_hypotheses(scored, max_k)

def retire_frontier_obs(self, obs_id: int) -> bool:
    """Drop a frontier node after visit — visited space is not a frontier."""
    oid = int(obs_id)
    drop_nodes: set[int] = set()
    frontier_ids: set[str] = set()
    from emet.memory.graph_eqa.spatial.frontier_nodes import FRONTIER_DESC_PREFIX

    for n in self._nodes:
        if n.is_frontier and int(n.obs_id) == oid:
            drop_nodes.add(int(n.node_id))
            desc = str(n.description or "")
            if desc.startswith(FRONTIER_DESC_PREFIX):
                frontier_ids.add(desc[len(FRONTIER_DESC_PREFIX) :])
    if not drop_nodes:
        return False
    for frontier_id in frontier_ids:
        self.world_evidence.set_frontier_status(frontier_id, "visited")
    self._nodes = [n for n in self._nodes if int(n.node_id) not in drop_nodes]
    self._observations = [o for o in self._observations if int(o.obs_id) != oid]
    for i, n in enumerate(self._nodes, start=1):
        self._nodes[i - 1] = replace(n, node_id=i)
    self._reindex_world_entities()
    self._rebuild_viewpoint_index()
    self._update_edges()
    return True

def frontier_id_near_xy(self, xyz: Any, *, max_dist_m: float = 1.5) -> str:
    track = self.world_evidence.frontier_near_xyz(
        xyz,
        max_dist_m=max_dist_m,
    )
    return track.frontier_id if track is not None else ""

def retract_phrase_claim_at_obs(
    self,
    obs_id: int,
    phrase: str,
    *,
    strip_matching_labels: bool = True,
    strip_across_obs: bool = False,
    apply_blacklist: bool = True,
    evidence_obs_id: int | None = None,
    evidence_source: str = "vlm",
    room: str | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Stop offering a disproved stem-object claim without deleting the place.

    After a close look verifies ABSENT for ``phrase`` at ``obs_id``, blacklist
    that (obs, phrase) for hyp recall and optionally strip matching labels from
    the observation / node. Location-MCQ *place* landmarks should not call this
    for the place name itself (the island is real; only the object was missing).

    ``strip_across_obs`` (default False): cross-view stripping is opt-in and
    corroborated-only — a closer look at one view disproving the object should NOT
    strip the label from nodes at OTHER views unless ABSENT is corroborated at 2+
    distinct views (callers that prove corroboration pass True). One weak glance
    must not delete a node seen elsewhere (exp1 regression fix).

    ``evidence_obs_id`` identifies the fresh view that produced ABSENT. It may
    differ from ``obs_id``, which remains the navigation/place claim being
    retracted.
    """
    oid = int(obs_id)
    key_phrase = str(phrase or "").strip().lower()
    if not key_phrase:
        return {"ok": False, "error": "empty phrase", "obs_id": oid}
    if not hasattr(self, "_retracted_nav_claims"):
        self._retracted_nav_claims = set()
    if apply_blacklist:
        self._retracted_nav_claims.add((oid, key_phrase))
    if not hasattr(self, "_retraction_evidence_views"):
        self._retraction_evidence_views = set()
    evidence_oid = int(evidence_obs_id) if evidence_obs_id is not None else oid
    self._retraction_evidence_views.add((evidence_oid, key_phrase))
    source = str(evidence_source or "unknown").strip().lower()[:40]
    stripped_obs = 0
    stripped_nodes = 0
    if strip_matching_labels:
        for o in self._observations:
            if not strip_across_obs and int(o.obs_id) != oid:
                continue
            before = list(o.labels or [])
            kept = [lab for lab in before if not label_matches_relevant_object(key_phrase, lab)]
            if len(kept) != len(before):
                o.labels = kept if kept else ["object"]
                stripped_obs += 1
        for i, n in enumerate(self._nodes):
            if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
                continue
            if not strip_across_obs and int(n.obs_id) != oid:
                continue
            before = list(n.labels or [])
            kept = [lab for lab in before if not label_matches_relevant_object(key_phrase, lab)]
            if len(kept) != len(before):
                self._nodes[i] = replace(
                    n,
                    labels=kept if kept else ["object"],
                )
                stripped_nodes += 1
    room_ev = self.record_room_event(
        room=room,
        kind="verify_absent",
        step=step,
        phrase=key_phrase,
        obs_id=evidence_oid,
        note=f"{source} absent; claim_obs={oid}",
    )
    room_label = str((room_ev or {}).get("room") or "")
    # Persist ABSENT as a verify attempt when the ledger is on (does not change
    # per-view semantics — ABSENT is not scene-wide proof of absence).
    self.record_attempt(
        action_kind="verify",
        outcome="absent",
        status_code=f"{source}_absent",
        note=f"negative evidence {key_phrase!r}: claim_obs={oid} evidence_obs={evidence_oid}",
        obs_id=evidence_oid,
        phrase=key_phrase,
        source="eqa",
        room=room_label,
        step=step,
    )
    return {
        "ok": True,
        "obs_id": oid,
        "claim_obs_id": oid,
        "evidence_obs_id": evidence_oid,
        "evidence_source": source,
        "phrase": key_phrase,
        "stripped_obs": stripped_obs,
        "stripped_nodes": stripped_nodes,
        "n_retracted": len(self._retracted_nav_claims),
        "recorded_only": not apply_blacklist and not strip_matching_labels,
        "room": room_label or None,
    }

def has_absent_retraction_at_other_view(self, phrase: str, evidence_obs_id: int) -> bool:
    """Return whether the phrase was already ABSENT at a different evidence view."""
    key_phrase = str(phrase or "").strip().lower()
    evidence_oid = int(evidence_obs_id)
    views = getattr(self, "_retraction_evidence_views", None) or set()
    return any(
        int(view_id) != evidence_oid and str(view_phrase).strip().lower() == key_phrase
        for view_id, view_phrase in views
    )

def clear_retracted_nav_claims(self) -> None:
    """Drop claim blacklist (e.g. new question).

    When ``persist_absent_claims`` is on (``eqa.attempt_ledger.persist_absent_claims``
    / ``EMET_ATTEMPT_LEDGER_PERSIST_ABSENT``), keep the blacklist across questions.
    Ledger rows always persist for the graph lifetime regardless.
    """
    if self.persist_absent_claims:
        return
    self._retracted_nav_claims = set()
    self._retraction_evidence_views = set()

def retire_frontier_near_xy(
    self,
    xy: Any,
    *,
    radius_m: float = 1.25,
) -> int:
    """Retire frontier nodes within ``radius_m`` of a visited explore goal."""
    try:
        pt = np.asarray(xy, dtype=float).reshape(-1)[:2]
    except Exception:
        return 0
    if pt.size < 2:
        return 0
    r2 = float(radius_m) ** 2
    drop_obs: list[int] = []
    for n in self._nodes:
        if not n.is_frontier:
            continue
        nxy = np.asarray(n.xyz, dtype=float).reshape(-1)[:2]
        if nxy.size < 2:
            continue
        if float(np.sum((nxy - pt) ** 2)) <= r2:
            drop_obs.append(int(n.obs_id))
    n_dropped = 0
    for oid in drop_obs:
        if self.retire_frontier_obs(oid):
            n_dropped += 1
    return n_dropped

def verify_phrase_at_obs(
    self,
    phrase: str,
    obs_id: int,
    rgb: np.ndarray | None = None,
    *,
    min_sim: float | None = None,
) -> VerifyResult:
    """SigLIP-verify *phrase* against observation *obs_id* (optional live *rgb*)."""
    thresh = float(min_sim if min_sim is not None else SIGLIP_CONFIRM_THRESHOLD)
    oid = int(obs_id)
    text = (phrase or "").strip()
    obs = self._observation_by_id(oid)
    if obs is None and rgb is None:
        return VerifyResult(status="ABSENT", sim=0.0, obs_id=oid, phrase=text, ok=False)

    label_hit = False
    if obs is not None and text:
        label_hit = any(label_matches_relevant_object(text, lab) for lab in (obs.labels or []))

    enc = self._confirmed_memory_siglip_encoder
    text_feat: np.ndarray | None = None
    img_feat: np.ndarray | None = None
    sim = 0.0
    if enc is not None and text:
        try:
            from emet.memory.graph_eqa.eqa.graph_eqa_siglip import (
                _feature_vector,
                encode_observation_rgb,
            )

            text_feat = _feature_vector(enc.encode_text(text))
            if rgb is not None:
                img_feat = encode_observation_rgb(enc, np.asarray(rgb, dtype=np.uint8))
            elif oid in self._obs_siglip_features:
                img_feat = np.asarray(self._obs_siglip_features[oid], dtype=np.float32)
            elif obs is not None:
                img_feat = encode_observation_rgb(enc, obs.rgb)
                if img_feat is not None:
                    self._obs_siglip_features[oid] = img_feat
            if text_feat is not None and img_feat is not None:
                sim = float(np.dot(text_feat, img_feat))
        except Exception as e:
            _logger.warning(f"verify_phrase_at_obs SigLIP failed: {e}")

    if sim >= thresh:
        status, ok = "PRESENT", True
    elif sim >= SIGLIP_PRESENT_THRESHOLD or label_hit:
        status, ok = "CANDIDATE", False
    elif text and img_feat is None and text_feat is None:
        # SigLIP is released before submit_answer to free VRAM for the VLM, so any
        # verify after the first submit computes no features. Reporting that as
        # ABSENT looks like real negative evidence in traces and to the loop.
        status, ok = "UNAVAILABLE", False
    else:
        status, ok = "ABSENT", False
    return VerifyResult(
        status=status,
        sim=float(sim),
        obs_id=oid,
        phrase=text,
        ok=ok,
        text_feat=text_feat,
        img_feat=img_feat,
    )

def select_obs_ids_for_verified_answer(
    self,
    verified_obs_id: int,
    max_images: int = 1,
) -> list[int]:
    """Prefer the verified observation; cap at *max_images*."""
    if max_images <= 0:
        return []
    oid = int(verified_obs_id)
    if self._observation_by_id(oid) is None:
        return []
    return [oid][:max_images]

def _confirmed_memory_phrases(self) -> list[str]:
    if self._relevant_phrases:
        return list(self._relevant_phrases)
    return list(self._relevant_objects or [])

def _siglip_match_for_phrase(self, phrase: str) -> tuple[float, np.ndarray, int | None] | None:
    key = (phrase or "").strip().lower()
    if not key:
        return None
    cached = self._siglip_phrase_cache.get(key)
    if cached is not None:
        return cached
    grounder = self._text_grounder
    if grounder is None:
        return None
    try:
        sig = grounder(phrase)
    except Exception as e:
        _logger.warning(f"SigLIP grounder failed for {phrase!r}: {e}")
        return None
    if sig is None:
        return None
    sim, xyz = float(sig[0]), np.asarray(sig[1], dtype=float)
    return sim, xyz, None

def _object_present_in_graph_or_siglip(self, obj: str) -> bool:
    if any(label_matches_relevant_object(obj, lab) for o in self._observations for lab in o.labels):
        return True
    sig = self._siglip_match_for_phrase(obj)
    return sig is not None and float(sig[0]) >= SIGLIP_PRESENT_THRESHOLD

def _obs_usable_for_eqa_image(self, obs_id: int) -> bool:
    """True when ``obs_id`` may be attached as a VLM answer image.

    Frontier sync stores black 8×8 placeholders — never answer off those.
    Frontiers remain in the SCENE_GRAPH text for Action navigation targets.
    """
    if self._obs_is_frontier(int(obs_id)):
        return False
    obs = self._observation_by_id(int(obs_id))
    if obs is None:
        return False
    rgb = np.asarray(obs.rgb)
    return bool(rgb.ndim == 3 and rgb.shape[0] >= 2 and rgb.shape[1] >= 2)

def _select_relevant_obs_ids(
    self,
    max_images: int = 6,
    choices: list[str] | None = None,
    attribute_question: bool = False,
) -> list[int]:
    """Select a diverse set of observation IDs for the EQA prompt (1-based).

    P2 diversification: instead of "all keyword matches then fill", build a
    prioritized pool so the VLM sees question-relevant views *and* a recent
    view *and* spatially spread context, capped at ``max_images``. Falls back
    to the most recent non-frontier observations when there are no keyword
    objects. Frontier placeholder RGB is never selected for answering.

    When ``choices`` are location MCQ options, prefer views whose labels match
    option landmarks (refrigerator, treadmill, …) *before* SigLIP nearest —
    false CONFIRMED_MEMORY coords must not steal Image 1. Count / clock / other
    questions rank stored RGB by visual FIND (DynaMem ``find_all_images`` /
    SigLIP top-k) first; YoloE class strings are leftover recall only, spread
    in XY so one cluster cannot fill the budget.

    For attribute/state questions, prefer views with lamp/light/curtain labels
    over frontiers before answering on/off or up/down.
    """
    if not self._observations:
        return []
    if max_images <= 0:
        return []
    if not self._relevant_objects:
        recent = [int(o.obs_id) for o in self._observations if self._obs_usable_for_eqa_image(o.obs_id)]
        return recent[-max_images:]

    by_id = {int(o.obs_id): o for o in self._observations}
    selected: list[int] = []

    def take(oid: int) -> bool:
        oid = int(oid)
        if oid in selected or oid not in by_id:
            return False
        if not self._obs_usable_for_eqa_image(oid):
            return False
        selected.append(oid)
        return len(selected) >= max_images

    reserved = 0
    if max_images >= 3:
        reserved = min(2, max_images - 1)
    keyword_budget = max(1, max_images - reserved)

    boost: set[str] = set()
    if choices and not attribute_question:
        for phrase in list(self._confirmed_memory_phrases()) + list(self._relevant_objects or []):
            for tok in _object_match_tokens(phrase):
                boost |= set(_QUESTION_LANDMARK_BOOST.get(tok, frozenset()))

    def _obs_blob(o: GraphObservation) -> str:
        return " ".join(str(lab) for lab in (o.labels or [])).lower()

    def _direct_target_match(o: GraphObservation) -> bool:
        """True when a label matches the question object without relying only on aliases
        that are absent from the MCQ options (recycle bin vs refrigerator)."""
        labels = [str(lab) for lab in (o.labels or []) if lab]
        if not labels:
            return False
        phrases = list(self._relevant_objects or []) + list(self._confirmed_memory_phrases())
        choice_blob = " ".join(choices or []).lower()
        for lab in labels:
            for phrase in phrases:
                if not label_matches_relevant_object(phrase, lab):
                    continue
                # Direct token overlap with the question phrase/object.
                if _object_match_tokens(phrase) & _object_match_tokens(lab):
                    return True
                # Alias match (trash↔recycle): keep only if the label appears in options.
                lab_toks = _object_match_tokens(lab)
                if any(t in choice_blob for t in lab_toks):
                    return True
        return False

    location_q = bool(choices) and not attribute_question and choices_are_location_mcq(list(choices))

    # Unified Image-1 ranking for location MCQs:
    # boosted choice landmarks (fridge) > direct target (ladder) > weak aliases / generics.
    if location_q:
        scored: list[tuple[float, int]] = []
        for o in self._observations:
            oid = int(o.obs_id)
            blob = _obs_blob(o)
            if not blob.strip():
                continue
            score = 0.0
            if _direct_target_match(o):
                score += 10.0
            for ch in choices[:4]:
                for tok in distinctive_choice_tokens(ch):
                    hit = tok in blob or any(lab.startswith(tok) or tok.startswith(lab) for lab in blob.split())
                    if not hit:
                        continue
                    if tok in _LANDMARK_GENERIC_TOKENS:
                        score += 0.25
                    elif tok in boost:
                        score += 12.0  # fridge for trash beats recycle-alias (+10)
                    else:
                        score += 1.0
            for tok in boost:
                if tok in blob:
                    score += 0.5  # recycle/bin mild, not enough to beat fridge
            if score > 0:
                scored.append((score, oid))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        for _score, oid in scored[:keyword_budget]:
            if take(oid):
                return selected

    # Visual FIND (DynaMem SigLIP top-k) before YoloE/caption labels.
    phrases = self._eqa_find_phrases()
    for oid in self._visual_find_obs_ids(phrases, max_n=keyword_budget):
        if take(oid):
            return selected

    # Target keyword / confirmed-memory label matches (extra recall).
    # Round-robin across phrases; within a noun, pick views far apart in XY.
    seen_kw: set[int] = set()
    buckets: list[list[int]] = []
    for obj in phrases:
        bucket: list[int] = []
        for o in reversed(self._observations):
            oid = int(o.obs_id)
            if oid in seen_kw:
                continue
            if any(label_matches_relevant_object(obj, lab) for lab in o.labels):
                bucket.append(oid)
                seen_kw.add(oid)
        if bucket:
            buckets.append(bucket)
    keyword_hits: list[int] = []
    while buckets and len(keyword_hits) < keyword_budget:
        bucket = buckets.pop(0)
        pick = self._spread_obs_xy(keyword_hits + bucket, max_n=len(keyword_hits) + 1)
        chosen = pick[-1] if pick else bucket[0]
        if chosen not in bucket:
            chosen = bucket[0]
        keyword_hits.append(chosen)
        rest = [oid for oid in bucket if oid != chosen]
        if rest:
            buckets.append(rest)
    for oid in keyword_hits[:keyword_budget]:
        if take(oid):
            return selected

    # Attribute/state: prefer lamp/light/curtain views over frontiers for Image 1.
    if attribute_question:
        attr_tokens = (
            "lamp",
            "light",
            "lights",
            "ceiling",
            "curtain",
            "curtains",
            "window",
            "fixture",
        )
        attr_hits: list[int] = []
        for o in reversed(self._observations):
            oid = int(o.obs_id)
            if oid in selected or self._obs_is_frontier(oid):
                continue
            blob = _obs_blob(o)
            if any(t in blob for t in attr_tokens):
                attr_hits.append(oid)
        for oid in attr_hits:
            if take(oid):
                return selected

    # SigLIP phrase cache (caption-independent) for targets not already selected.
    for phrase in self._confirmed_memory_phrases():
        cached = self._siglip_phrase_cache.get(phrase.strip().lower())
        if cached is None or cached[2] is None:
            continue
        if float(cached[0]) >= SIGLIP_PRESENT_THRESHOLD and take(int(cached[2])):
            return selected

    # SigLIP obs grounder per relevant object.
    obs_grounder = getattr(self, "_obs_id_grounder", None)
    if obs_grounder is not None:
        for obj in self._relevant_objects:
            try:
                oid = obs_grounder(obj)
            except Exception:
                oid = None
            if oid is not None and take(int(oid)):
                return selected

    # Most recent non-frontier observation (fresh context).
    for o in reversed(self._observations):
        if take(int(o.obs_id)):
            return selected
        break

    # Spatial spread: greedily add observations farthest from those chosen.
    remaining = [
        int(o.obs_id)
        for o in self._observations
        if int(o.obs_id) not in selected and self._obs_usable_for_eqa_image(o.obs_id)
    ]
    while remaining and len(selected) < max_images:
        best_oid = None
        best_dist = -1.0
        for oid in remaining:
            cand = by_id[oid].xyz[:2]
            if selected:
                d = min(float(np.linalg.norm(cand - by_id[s].xyz[:2])) for s in selected if s in by_id)
            else:
                d = 0.0
            if d > best_dist:
                best_dist = d
                best_oid = oid
        if best_oid is None:
            break
        remaining.remove(best_oid)
        if take(best_oid):
            return selected

    return selected

def set_text_grounder(self, grounder: Callable[[str], tuple[float, np.ndarray] | None] | None) -> None:
    """Register an open-vocab visual grounder: ``text -> (similarity, xyz) | None``.

    Backed by the voxel map's SigLIP features so existence/location can be grounded in
    pixels rather than the VLM's caption-derived node labels.
    """
    self._text_grounder = grounder

def set_obs_id_grounder(self, grounder: Callable[[str], int | None] | None) -> None:
    """Register an open-vocab ``text -> obs_id`` selector (SigLIP-backed).

    Used by image selection to force the best-aligned observation of each relevant object
    into the VLM prompt regardless of its caption label.
    """
    self._obs_id_grounder = grounder

def set_visual_find_fn(self, fn: Callable[..., list[Any]] | None) -> None:
    """Register DynaMem retrieve: ``phrase, max_n -> [(similarity, graph_obs_id), ...]``.

    Backed by voxel ``find_all_images`` (top-k, not argmax) mapped onto graph
    observation ids. SigLIP only proposes RGB; the VLM still reads the frames.
    """
    self._visual_find_fn = fn

def snapshot_visual_find_ranks(self, *, question: str = "") -> None:
    """Cache top-k FIND ranks while SigLIP can still ``encode_text``.

    Habitat dynagraph calls ``prepare_dynagraph_vram_for_eqa`` after
    ``extract_relevant_objects`` and then drops GPU SigLIP so Qwen can load.
    ``query_answer`` reads this cache instead of calling ``find_all_images``.
    """
    q = str(question or self._question or "").strip()
    phrases: list[str] = []
    for raw in self._eqa_find_phrases():
        text = str(raw or "").strip()
        if text and text not in phrases:
            phrases.append(text)
    if q:
        for raw in heuristic_relevant_phrases(q) + heuristic_relevant_objects(q):
            text = str(raw or "").strip()
            if text and text not in phrases:
                phrases.append(text)
        q_low = q.lower()
        if any(s in q_low for s in ("what time", "time is it", "o'clock", "o’clock")):
            for extra in ("clock", "wall clock"):
                if extra not in phrases:
                    phrases.append(extra)
    if not phrases:
        return
    cache = dict(self._visual_find_rank_cache)
    visual_fn = self._visual_find_fn
    enc = self._confirmed_memory_siglip_encoder
    feats = self._obs_siglip_features or {}

    def _merge(rows: list[tuple[float, int]], sim: float, oid: int | None) -> None:
        if oid is None:
            return
        oi = int(oid)
        for i, (old_sim, old_oid) in enumerate(rows):
            if old_oid == oi:
                if float(sim) > old_sim:
                    rows[i] = (float(sim), oi)
                return
        rows.append((float(sim), oi))

    for phrase in phrases:
        key = phrase.strip().lower()
        if not key:
            continue
        rows: list[tuple[float, int]] = list(cache.get(key, []))
        if visual_fn is not None:
            hits: Any = []
            try:
                hits = visual_fn(str(phrase), 12)
            except TypeError:
                try:
                    hits = visual_fn(str(phrase))
                except Exception:
                    hits = []
            except Exception:
                hits = []
            for item in hits or []:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    _merge(rows, float(item[0]), int(item[1]))
                else:
                    try:
                        _merge(rows, max(SIGLIP_PRESENT_THRESHOLD, 0.25), int(item))
                    except (TypeError, ValueError):
                        continue
        if enc is not None and feats:
            from emet.memory.graph_eqa.eqa.graph_eqa_siglip import rank_observations_for_phrase

            for sim, oid in rank_observations_for_phrase(str(phrase), enc, feats):
                if float(sim) < SIGLIP_PRESENT_THRESHOLD:
                    break
                _merge(rows, float(sim), int(oid))
        if rows:
            rows.sort(key=lambda t: -t[0])
            cache[key] = rows
    self._visual_find_rank_cache = cache

def _nearest_object_neighbors(
    self,
    xyz: np.ndarray,
    *,
    exclude_node_ids: set[int] | None = None,
    max_neighbors: int = 2,
    max_dist_m: float = 3.0,
) -> list[tuple[Any, float]]:
    """Nearest non-frontier/viewpoint object nodes to ``xyz`` (planar XY)."""
    exclude = exclude_node_ids or set()
    anchor = np.asarray(xyz, dtype=np.float64).reshape(-1)[:2]
    scored: list[tuple[Any, float]] = []
    for n in self._nodes:
        if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
            continue
        if int(n.node_id) in exclude:
            continue
        other = np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2]
        dist = float(np.linalg.norm(anchor - other))
        if dist <= max_dist_m:
            scored.append((n, dist))
    scored.sort(key=lambda t: t[1])
    return scored[:max_neighbors]

def _confirmed_phrase_statuses(
    self,
) -> dict[str, tuple[str, list[int], float | None, np.ndarray | None, int | None]]:
    """Map each confirmed-memory phrase -> (status, node_ids, sig_sim, sig_xyz, sig_obs_id).

    status is one of:
      * present       — graph label match (grounded in SCENE_GRAPH nodes)
      * candidate     — SigLIP >= PRESENT threshold, no graph match (sighted only)
      * weak_siglip   — SigLIP below PRESENT (do not treat as absence)
      * not_observed  — no graph match and no SigLIP signal
    Used by merged ``to_string(merge_confirmed=True)``. Stricter than the legacy
    summary: only graph matches are ``present``; SigLIP never asserts presence/absence.
    """
    phrases = self._confirmed_memory_phrases()
    if not phrases:
        return {}
    object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
    out: dict[str, tuple[str, list[int], float | None, np.ndarray | None, int | None]] = {}
    for obj in phrases:
        matches = [
            int(n.node_id) for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)
        ]
        sig = self._siglip_match_for_phrase(obj)
        sim: float | None = None
        xyz: np.ndarray | None = None
        obs_id: int | None = None
        if sig is not None:
            sim = float(sig[0])
            xyz = np.asarray(sig[1], dtype=float)
            if sig[2] is not None:
                obs_id = int(sig[2])
        # Graph label match is the only path to "present" (grounded in SCENE_GRAPH).
        # SigLIP-only stays candidate even above CONFIRM — never assert presence/absence
        # from detector scores in the answer prompt (ABSENT coloring / false presents).
        if matches:
            status = "present"
        elif sim is not None and sim >= SIGLIP_PRESENT_THRESHOLD:
            status = "candidate"
        elif sim is not None:
            status = "weak_siglip"
        else:
            status = "not_observed"
        out[obj] = (status, matches, sim, xyz, obs_id)
    return out

def _node_room_by_id(self) -> dict[int, str]:
    """Map node_id -> stamped room-cluster name for object nodes (unknown rooms skipped)."""
    if not self._room_clusters:
        self.refresh_room_clusters()
    out: dict[int, str] = {}
    for c in self._room_clusters:
        name = str(getattr(c, "room_name", "") or "")
        if not name or name == "unknown":
            continue
        for nid in getattr(c, "node_ids", ()) or ():
            out[int(nid)] = name
    return out

def _relevant_memory_summary(self) -> str:
    """Surface question-relevant objects as 'confirmed memory' for the VLM.

    Graph label matches are LOOK (candidate views to inspect). SigLIP matches over
    observed points are CANDIDATE / weak-SigLIP navigation hints — they catch
    mislabeled sightings but must not assert presence, absence, or a count.
    """
    if not self._confirmed_memory_phrases():
        return ""
    object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
    present_thresh = SIGLIP_PRESENT_THRESHOLD
    lines: list[str] = []
    for obj in self._confirmed_memory_phrases():
        matches = [n for n in object_nodes if countable_primary_label_matches(obj, n)]
        if not matches:
            matches = [
                n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in (n.labels or []))
            ]
        sig = self._siglip_match_for_phrase(obj)
        parts: list[str] = []
        if matches:
            parts.append("candidate views: " + format_graph_node_candidates(matches, max_nodes=6))
            parts.append(_GRAPH_CANDIDATE_COUNT_DISCLAIMER)
        sig_present = sig is not None and float(sig[0]) >= present_thresh
        if sig is not None:
            sim, xyz = float(sig[0]), sig[1]
            obs_note = f", obs_id={int(sig[2])}" if sig[2] is not None else ""
            if sig_present:
                parts.append(f"SigLIP phrase match sim={sim:.2f} near ({xyz[0]:.1f}, {xyz[1]:.1f}){obs_note}")
            else:
                parts.append(f"no strong SigLIP match (sim={sim:.2f})")
        # Graph label match is a FIND pointer (LOOK), not a presence/count answer.
        if matches:
            anchor_xyz = np.asarray(matches[0].xyz, dtype=np.float64)
            exclude_ids = {int(n.node_id) for n in matches}
            status = "LOOK"
        elif sig_present:
            anchor_xyz = np.asarray(sig[1], dtype=np.float64) if sig is not None else None
            exclude_ids = set()
            status = (
                "CANDIDATE (SigLIP-only — verify in attached images before finalizing; "
                "do not treat as confirmed present or absent)"
            )
        elif sig is not None:
            lines.append(
                f"- {obj}: weak SigLIP only — "
                + "; ".join(parts)
                + " — not evidence of absence; trust attached images"
            )
            continue
        else:
            lines.append(f"- {obj}: not observed during exploration")
            continue
        if anchor_xyz is not None:
            neighbors = self._nearest_object_neighbors(
                anchor_xyz, exclude_node_ids=exclude_ids, max_neighbors=2, max_dist_m=3.0
            )
            if neighbors:
                near_bits = []
                for n, dist in neighbors:
                    lab = node_display_name(n)
                    near_bits.append(f"{lab} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f}) {dist:.1f}m")
                parts.append("nearest: " + "; ".join(near_bits))
        # Compact attempt-ledger tags for matched obs ids (opt-in; empty when off).
        attempt_bits: list[str] = []
        for n in matches[:3]:
            bit = self.attempt_summary_for_obs(int(n.obs_id), max_bits=2)
            if bit:
                attempt_bits.append(bit)
        if sig is not None and sig[2] is not None:
            bit = self.attempt_summary_for_obs(int(sig[2]), max_bits=2)
            if bit and bit not in attempt_bits:
                attempt_bits.append(bit)
        if attempt_bits:
            parts.append("attempts: " + " | ".join(attempt_bits[:2]))
        lines.append(f"- {obj}: {status} — " + "; ".join(parts))
    if not lines:
        return ""
    header = (
        "CONFIRMED_MEMORY (index of views to look at, not the answer. LOOK = "
        "candidate Image N to look at; CANDIDATE/weak SigLIP are navigation hints. "
        "Detector class names are proposals for WHERE to look. Identify and count "
        "from attached RGB; if images contradict memory, trust the images and keep "
        "exploring; for location MCQs, prefer option landmarks visible in Image 1 "
        "over nearest-furniture guesses):"
    )
    return header + "\n" + "\n".join(lines)

def _graph_covers_relevant_objects(self) -> bool:
    """True when every keyword object appears in at least one graph node label."""
    eqa_cfg = self.parameters.get("eqa", {}) if hasattr(self.parameters, "get") else {}
    if isinstance(eqa_cfg, dict) and eqa_cfg.get("sqa3d_allow_partial_graph"):
        return True
    if not self._confirmed_memory_phrases() or not self._observations:
        return True
    for obj in self._confirmed_memory_phrases():
        if not self._object_present_in_graph_or_siglip(obj):
            return False
    return True

def _target_visible_in_obs_ids(self, obs_ids: list[int]) -> bool:
    """True when a question target label appears on an attached Image 1..N view."""
    if not obs_ids:
        return False
    by_id = {int(o.obs_id): o for o in self._observations}
    phrases = list(self._confirmed_memory_phrases()) + list(self._relevant_objects or [])
    for oid in obs_ids:
        o = by_id.get(int(oid))
        if o is None:
            continue
        for phrase in phrases:
            if any(label_matches_relevant_object(phrase, lab) for lab in (o.labels or [])):
                return True
    return False

def _obs_is_frontier(self, obs_id: int) -> bool:
    for n in self._nodes:
        if int(n.obs_id) == int(obs_id) and n.is_frontier:
            return True
    return False

def _obs_is_object_place(self, obs_id: int) -> bool:
    """True when ``obs_id`` anchors a real object node (not frontier/viewpoint-only)."""
    for n in self._nodes:
        if int(n.obs_id) != int(obs_id):
            continue
        if n.is_frontier or n.is_viewpoint:
            continue
        return True
    return False
