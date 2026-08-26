# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Look-around, capture, and hypothesis refresh for the agentic GraphEQA executor."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from emet.memory.graph_eqa.agentic_config import (
    INVESTIGATE_SOURCES,
    NAV_SAME_OBS_LOOP_LIMIT,
    PLACE_APPROACH_SAMPLES,
)
from emet.memory.graph_eqa.agentic_policy import AgenticState
from emet.memory.graph_eqa.graph_memory import NavHypothesis, question_stem_for_keywords
from emet.memory.graph_eqa.query_images import dump_query_rgb
from emet.utils.logger import Logger

_logger = Logger(__name__)


class AgenticCaptureMixin:
    """Look-around, capture_and_update, and hypothesis refresh."""

    def _pin_eqa_look_obs(self, obs_id: int | None) -> None:
        """Next ``query_answer`` attaches this graph RGB as Image 1."""
        gm = self.graph_memory
        if gm is None or obs_id is None:
            return
        try:
            oid = int(obs_id)
        except (TypeError, ValueError):
            return
        if oid <= 0:
            return
        gm.last_eqa_look_obs_id = oid

    def _tool_look_around(self, *, verify: bool = True) -> dict[str, Any]:
        agent = self.agent
        hypothesis_id = None
        if verify:
            hypothesis_id = self._begin_policy_approach(
                "look",
                -2_000_000 - self._n_nav - self._n_explore,
                self.query_text,
            )
        ok = False
        if hasattr(agent, "look_around"):
            try:
                agent.look_around()
                ok = True
            except Exception as e:
                _logger.warning(f"look_around failed: {e}")
        if ok:
            self._refresh_room_after_motion()
        cap = self._tool_capture_and_update()
        verify_out = None
        if hypothesis_id is not None and cap.get("ok") and cap.get("obs_id") is not None:
            self._policy_approached(hypothesis_id, int(cap["obs_id"]))
            if verify and self.mode == "answer":
                verify_out = self._verify_after_motion(phrase=self.query_text)
        self._append_trace({"tool": "look_around", "ok": ok})
        return {"ok": ok, "capture": cap, "verify": verify_out}

    def _siglip_phrase(self, phrase: str = "") -> str:
        """Short object phrase for SigLIP — never feed the full MCQ question text."""
        text = (phrase or "").strip()
        q = (self.question or "").strip()
        # Callers sometimes pass query_text (== full question); prefer extracted target.
        if (not text) or (q and text == q) or ("?" in text and len(text.split()) > 6):
            text = (self._target_phrase or "").strip()
        if not text:
            # Last resort: stem without choices / trailing "Answer:"
            stem = question_stem_for_keywords(self.question or "")
            text = (stem or self.query_text or "").strip()
        return text

    def _verify_after_motion(self, *, phrase: str = "") -> dict[str, Any]:
        """Run verify on the newest captured view (router and fallback both need this)."""
        return self.handle_tool(
            "verify_siglip",
            {"phrase": self._siglip_phrase(phrase)},
        )

    def _verify_stalled_nav_view(self, obs_id: int, *, phrase: str = "") -> dict[str, Any]:
        """When capture does not advance, still score the current view once for the planner."""
        oid = int(obs_id)
        # Allow verify despite REQUIRES_FRESH_VIEW — we intentionally revisit this station.
        self._fresh_obs_ids.add(oid)
        # Clear prior same-view skip so this stall path can record ABSENT/CANDIDATE.
        self._tried.pop(oid, None)
        scored = getattr(self._evidence_policy, "_globally_scored_obs_ids", None)
        if isinstance(scored, set):
            scored.discard(oid)
        return self.handle_tool(
            "verify_siglip",
            {
                "phrase": self._siglip_phrase(phrase),
                "obs_id": oid,
            },
        )

    def _obs_revision_snapshot(self, gm: Any) -> dict[int, int]:
        """Safe obs_id→revision map (ignores MagicMock / non-int backends)."""
        out: dict[int, int] = {}
        if gm is None:
            return out
        fn = getattr(gm, "obs_revision", None)
        if not callable(fn):
            return out
        tracked: set[int] = set()
        before = self._latest_obs_id()
        if before is not None:
            tracked.add(int(before))
        for h in self._hypotheses[:5]:
            tracked.add(int(h.obs_id))
        for oid in tracked:
            try:
                out[int(oid)] = int(fn(int(oid)))
            except (TypeError, ValueError):
                continue
        return out

    def _obs_revisions_advanced(self, gm: Any, before_revs: dict[int, int]) -> list[int]:
        if gm is None or not before_revs:
            return []
        fn = getattr(gm, "obs_revision", None)
        if not callable(fn):
            return []
        advanced: list[int] = []
        check_ids = set(before_revs)
        last_u = getattr(gm, "_last_obs_content_update_id", None)
        if isinstance(last_u, int):
            check_ids.add(int(last_u))
        for oid in check_ids:
            try:
                cur = int(fn(int(oid)))
            except (TypeError, ValueError):
                continue
            if cur > int(before_revs.get(int(oid), 0)):
                advanced.append(int(oid))
        return advanced

    def _tool_capture_and_update(self) -> dict[str, Any]:
        before = self._latest_obs_id()
        gm = self.graph_memory
        before_revs = self._obs_revision_snapshot(gm)
        agent = self.agent
        if hasattr(agent, "update"):
            try:
                agent.update()
            except Exception as e:
                _logger.warning(f"capture_and_update agent.update failed: {e}")
        # Always refresh graph-side confirmed memory after a voxel update when enabled.
        if gm is not None and getattr(gm, "memory_summary_enabled", False):
            if hasattr(gm, "refresh_siglip_confirmed_memory"):
                gm.refresh_siglip_confirmed_memory()
        fresh = self._latest_obs_id()
        refreshed_ids = self._obs_revisions_advanced(gm, before_revs)

        # New observation id — full advance.
        if fresh is not None and (before is None or int(fresh) != int(before)):
            self._fresh_obs_ids.add(int(fresh))
            if self.mode == "answer" and before is not None:
                try:
                    self._refresh_hypotheses_from_graph()
                except Exception as exc:
                    _logger.warning(f"hypothesis refresh after capture failed: {exc}")
            self._last_capture_status = "OK"
            cap_paths = dump_query_rgb(self, int(fresh), kind="capture")
            self._append_trace({"tool": "capture_and_update", "ok": True, "obs_id": fresh, **cap_paths})
            return {"ok": True, "obs_id": fresh, "status": "NEW_OBS"}

        # Same obs_id but candidate RGB/evidence refreshed via spatial merge.
        if refreshed_ids:
            use_id = int(refreshed_ids[0])
            if fresh is not None and int(fresh) in refreshed_ids:
                use_id = int(fresh)
            refreshed_set = {int(item) for item in refreshed_ids}
            self._fresh_obs_ids.update(refreshed_set)
            # Allow re-verify: old evidence on these stable ids is stale once RGB changed.
            for refreshed_id in refreshed_set:
                self._tried.pop(refreshed_id, None)
                self._vlm_assessed_obs_ids.discard(refreshed_id)
                self._assess_history.pop(refreshed_id, None)
            self._answer_evidence = [item for item in self._answer_evidence if item.obs_id not in refreshed_set]
            if self._confirmed_answer_evidence is not None and self._confirmed_answer_evidence.obs_id in refreshed_set:
                self._confirmed_answer_evidence = None
            if self._pending_answerable is not None and self._pending_answerable.get("obs_id") in refreshed_set:
                self._pending_answerable = None
            if self._verified_obs_id in refreshed_set:
                self._verified = False
                self._verified_obs_id = None
                self._verified_evidence_event_ids = ()
                self._final_answer_decision = None
            decision_evidence = (
                self._final_answer_decision.evidence if self._final_answer_decision is not None else None
            )
            if decision_evidence is not None and decision_evidence.obs_id in refreshed_set:
                self._final_answer_decision = None
            if self._last_positive_obs_id in refreshed_set:
                self._last_positive_obs_id = None
                self._last_positive_letter = ""
            scored = getattr(self._evidence_policy, "_globally_scored_obs_ids", None)
            if isinstance(scored, set):
                scored.difference_update(refreshed_set)
            policy_invalidated = False
            for belief in self._evidence_policy.beliefs.values():
                prior_evidence = list(belief.evidence)
                belief.evidence = [item for item in prior_evidence if int(item.obs_id) not in refreshed_set]
                belief.attempted_obs_ids.difference_update(refreshed_set)
                policy_invalidated = policy_invalidated or len(belief.evidence) != len(prior_evidence)
            if policy_invalidated:
                self._evidence_policy.reset_for_new_approach()
            if self.mode == "answer":
                try:
                    self._refresh_hypotheses_from_graph()
                except Exception as exc:
                    _logger.warning(f"hypothesis refresh after content refresh failed: {exc}")
            self._last_capture_status = "CONTENT_REFRESHED"
            cap_paths = dump_query_rgb(self, int(use_id), kind="capture")
            self._append_trace(
                {
                    "tool": "capture_and_update",
                    "ok": True,
                    "obs_id": use_id,
                    "status": "CONTENT_REFRESHED",
                    "refreshed_obs_ids": refreshed_ids,
                    **cap_paths,
                }
            )
            return {
                "ok": True,
                "obs_id": use_id,
                "status": "CONTENT_REFRESHED",
                "refreshed_obs_ids": refreshed_ids,
            }

        # Reject non-advancing captures (same obs_id, no candidate refresh).
        if fresh is not None and before is not None and int(fresh) == int(before):
            self._last_capture_status = "NO_NEW_OBS"
            self._append_trace(
                {
                    "tool": "capture_and_update",
                    "ok": False,
                    "obs_id": fresh,
                    "prior_obs_id": before,
                    "status": "NO_NEW_OBS",
                }
            )
            return {
                "ok": False,
                "error": "capture did not advance observation — move before re-capturing",
                "obs_id": fresh,
                "status": "NO_NEW_OBS",
            }
        self._last_capture_status = "NO_OBS"
        self._append_trace({"tool": "capture_and_update", "ok": True, "obs_id": fresh})
        return {"ok": True, "obs_id": fresh}

    def _refresh_hypotheses_from_graph(self) -> None:
        """Re-retrieve nav evidence cards after voxel/graph grew — no VLM extract."""
        gm = self.graph_memory
        if gm is None or not hasattr(gm, "hypothesize_nav_targets"):
            return
        hypotheses = self._recall_nav_hypotheses()
        self._set_hypotheses(hypotheses)

    def _receptacle_adjacent_hypotheses(self, gm: Any) -> list[NavHypothesis]:
        """Container/fixture nodes to look at when a receptacle phrase has no direct
        place card (microwave/table/cab often sit on/under these).

        Two soft-search sources, both label-free:
          (1) SigLIP text grounding of the target phrase against the voxel semantic
              memory — the top-similarity world point is where the object is likely
              to be even if YoloE never made a labeled node for it.
          (2) container/fixture node labels (cabinet/counter/shelf/table/...) as a
              geometric fallback when SigLIP has nothing above threshold.
        """
        if gm is None:
            return []
        out: list[NavHypothesis] = []
        seen: set[int] = set()

        # (1) SigLIP soft ground: top-similarity voxel point for the target phrase.
        # A soft *explore* seed only needs to point at the most likely spot, not a
        # PRESENT-level confirmation, so use a low bar — if the semantic memory has
        # any microwave-like features we want to go look there.
        voxel_map, _ = self._voxel_planner()
        target = self._target_phrase or self._siglip_phrase()
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            _logger.info(
                "[siglip-seed] target=%r voxel=%s target_phrase=%r",
                target,
                bool(voxel_map is not None),
                self._target_phrase,
            )
        if voxel_map is not None and target:
            try:
                sim = voxel_map.find_alignment_over_model(target)
                points, _, _, _ = voxel_map.semantic_memory.get_pointcloud()
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    _logger.info(
                        "[siglip-seed] sim=%s n_points=%s",
                        "None" if sim is None else f"{sim.numel()}",
                        "None" if points is None else str(tuple(points.shape)),
                    )
                if sim is not None and points is not None and sim.numel() > 0:
                    best = int(sim.cpu().argmax(dim=-1))
                    best_sim = float(sim.cpu().max(dim=-1)[0].item())
                    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                        _logger.info(
                            "[siglip-seed] target=%r top_sim=%.3f n_points=%d",
                            target,
                            best_sim,
                            int(points.shape[0]),
                        )
                    if best_sim > 0.12:
                        xyz = np.asarray(points[best].detach().cpu().numpy(), dtype=float).reshape(-1)[:3]
                        if xyz.size >= 3 and np.isfinite(xyz).all():
                            out.append(
                                NavHypothesis(
                                    phrase=f"siglip {self._target_phrase or target}",
                                    obs_id=-2_000_000 - len(out),
                                    xyz=xyz,
                                    score=0.0,
                                    source="siglip",
                                )
                            )
            except Exception as e:
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    _logger.warning(f"siglip receptacle seed failed for {target!r}: {e}")

        # (2) container/fixture node labels as a geometric fallback.
        if not out and hasattr(gm, "get_nodes"):
            for node in gm.get_nodes():
                if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                    continue
                oid = int(getattr(node, "obs_id", -1))
                if oid < 0 or oid in seen:
                    continue
                labels = [str(lab).lower() for lab in (getattr(node, "labels", None) or [])]
                if not any(tok in lab for lab in labels for tok in self._FIXTURE_LABEL_TOKENS):
                    continue
                seen.add(oid)
                xyz = np.asarray(node.xyz, dtype=float).reshape(-1)
                out.append(
                    NavHypothesis(
                        phrase="nearby fixture",
                        obs_id=oid,
                        xyz=xyz[:3],
                        score=0.0,
                        source="graph",
                    )
                )
                if len(out) >= 4:
                    break
        return out

    def _set_hypotheses(self, hypotheses: list[NavHypothesis]) -> None:
        """Install recalled hyps: drop visited frontiers; prefer untried in order."""
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            inv = [h for h in hypotheses if str(h.source) in INVESTIGATE_SOURCES]
            exp = [h for h in hypotheses if str(h.source) not in INVESTIGATE_SOURCES]
            _logger.info(
                "[hyps] q=%r investigate=%d (%s) explore=%d (%s)",
                self.query_text[:50],
                len(inv),
                [f"{int(h.obs_id)}:{h.phrase}" for h in inv][:8],
                len(exp),
                [f"{int(h.obs_id)}:{h.phrase}" for h in exp][:8],
            )
        filtered: list[NavHypothesis] = []
        for h in hypotheses:
            oid = int(h.obs_id)
            if str(h.source) == "frontier" and (
                int(self._nav_to_obs_counts.get(oid, 0)) >= 1 or self._hypothesis_nav_blocked(oid)
            ):
                continue
            # Capture stations are verify views, not places to orbit next.
            if str(h.source) in INVESTIGATE_SOURCES and oid in self._station_obs_ids:
                continue
            filtered.append(h)
        # Anti-echo: untried / low visits first, then tried graph/siglip for context.
        untried: list[NavHypothesis] = []
        tried: list[NavHypothesis] = []
        for h in filtered:
            oid = int(h.obs_id)
            if self._hypothesis_nav_blocked(oid) or int(self._nav_to_obs_counts.get(oid, 0)) >= 1:
                if str(h.source) != "frontier":
                    tried.append(h)
            else:
                untried.append(h)
        packed = untried + tried
        self._hypotheses = packed
        self._hyp_i = 0
        _SOURCE_PRIOR = {
            "graph": 0.55,
            "confirmed": 0.5,
            "siglip": 0.4,
            "frontier": 0.2,
        }
        for h in self._hypotheses:
            self._evidence_policy.register_hypothesis(
                f"{h.source}:{int(h.obs_id)}",
                h.phrase,
                prior_probability=_SOURCE_PRIOR.get(str(h.source), 0.3),
            )

    def _latest_obs_id(self) -> int | None:
        """Newest non-frontier observation id (the frame just captured), if any."""
        gm = self.graph_memory
        observations = list(getattr(gm, "_observations", None) or [])
        for obs in reversed(observations):
            oid = int(obs.obs_id)
            usable = getattr(gm, "_obs_usable_for_eqa_image", None)
            if usable is not None and not usable(oid):
                continue
            return oid
        return None

    def _obs_already_verified(self, obs_id: int) -> bool:
        """True when this obs was already SigLIP-scored — do not score it again.

        Bare ``\"nav failed\"`` in ``_tried`` is a transient planner miss, not a
        verify score; callers may still navigate/verify that obs_id.
        """
        oid = int(obs_id)
        if oid in self._evidence_policy.scored_obs_ids:
            return True
        tried = str(self._tried.get(oid) or "")
        if not tried or tried == "nav failed":
            return False
        if tried.startswith("STALLED_NAV_LOOP") or tried.startswith("verify "):
            return True
        # Legacy / unknown tried markers — preserve no-reverify.
        return True

    def _begin_policy_approach(self, source: str, obs_id: int, phrase: str) -> str:
        # A prior verify may have left the policy in ANSWER (a different hypothesis
        # was confirmed). Starting a new investigate must reset to a fresh
        # SEARCH→APPROACH so the next capture+assess can confirm again — otherwise
        # apply_vlm_assessment raises 'invalid in state ANSWER' and _verified never
        # updates even when the VLM keeps reporting present=True. Only reset when
        # switching to a new hypothesis (not re-verifying the same confirmed view).
        if (
            self._evidence_policy.state in (AgenticState.REPLAN, AgenticState.ANSWER)
            and self._evidence_policy.active_hypothesis_id
            and self._evidence_policy.active_hypothesis_id != f"{source}:{int(obs_id)}"
        ):
            self._evidence_policy.reset_for_new_approach()
            self._verified = False
            self._verified_obs_id = None
        elif self._evidence_policy.state == AgenticState.REPLAN:
            self._evidence_policy.replan()
            self._verified = False
            self._verified_obs_id = None
        hypothesis_id = f"{source}:{int(obs_id)}"
        self._evidence_policy.register_hypothesis(hypothesis_id, phrase)
        if self._evidence_policy.state == AgenticState.SEARCH:
            self._evidence_policy.choose(hypothesis_id)
        return hypothesis_id

    def _policy_approached(self, hypothesis_id: str, fresh_obs_id: int) -> None:
        if self._evidence_policy.active_hypothesis_id != hypothesis_id:
            return
        try:
            self._evidence_policy.approached(int(fresh_obs_id))
        except (RuntimeError, ValueError) as exc:
            _logger.warning(f"evidence-policy approach rejected: {exc}")

    def _next_untried_hypothesis(self) -> NavHypothesis | None:
        """Prefer Investigate cards with unused approach samples left."""
        for h in self._investigate_hypotheses():
            oid = int(h.obs_id)
            if self._obs_already_verified(oid):
                continue
            if self._place_approaches_exhausted(oid) or self._hypothesis_nav_blocked(oid):
                continue
            rec = self._place_inspect.get(oid)
            if rec is None or not rec.approached_close or rec.approaches_left > 0:
                return h
        for h in self._investigate_hypotheses():
            oid = int(h.obs_id)
            if self._obs_already_verified(oid):
                continue
            if self._place_approaches_exhausted(oid) or self._hypothesis_nav_blocked(oid):
                continue
            return h
        return None

    def _hypothesis_nav_blocked(self, obs_id: int) -> bool:
        """True if investigate must refuse this id (approaches/coverage exhausted / stall)."""
        oid = int(obs_id)
        if self.action_progress_mode == "enforce":
            # Semantic dispatch already chose an eligible concrete approach.
            return False
        if self._place_approaches_exhausted(oid):
            return True
        if self._next_approach_index(oid) is None:
            return True
        tried = str(self._tried.get(oid) or "")
        if tried.startswith("STALLED_NAV_LOOP"):
            return True
        # Consecutive planner misses marked this candidate unreachable.
        if oid in self._unreachable_obs_ids:
            return True
        # Hard cap so planner thrashing cannot consume the whole nav budget.
        max_attempts = PLACE_APPROACH_SAMPLES + NAV_SAME_OBS_LOOP_LIMIT
        if int(self._nav_to_obs_counts.get(oid, 0)) >= max_attempts:
            return True
        return False
