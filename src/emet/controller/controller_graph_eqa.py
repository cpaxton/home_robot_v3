# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# GraphEQA agent: uses graph-based memory for EQA while reusing DynaMem-style
# voxel map for navigation and exploration. Re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480).


import os
import re
import time
from typing import Any

import numpy as np
from PIL import Image

from emet.controller.controller_dynamem import DynamemController
from emet.controller.habitat_nav import (
    apply_habitat_nav_resolution,
    explore_grid_resolution_m,
    goal_key_xy,
    habitat_body_scan,
    habitat_explore_frontiers_enabled,
    habitat_nav_would_be_noop,
    habitat_perfect_nav_enabled,
    habitat_random_walk_step,
    is_habitat_robot_client,
    pick_habitat_exploration_target,
    pick_uncovered_explore_target,
    robot_planar_xy,
)
from emet.core.parameters import Parameters
from emet.core.robot import AbstractRobotClient
from emet.memory.graph_eqa import GraphEQAMemory, SensorGraphBuilder
from emet.memory.graph_eqa.ingest.instance_observations import (
    DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M,
)
from emet.utils.logger import Logger
from emet.visualization.dynagraph_context import (
    log_vlm_context_to_visualizer,
    send_graph_memory_rerun_blueprint,
)
from emet.visualization.null_visualizer import visualizer_is_enabled

logger = Logger(__name__)


def _parse_image_pick(reply: str, n_candidates: int) -> int | None:
    """Parse a 1-based 'Image N' pick from a terse VLM reply; None when out of range."""
    m = re.search(r"\d+", reply or "")
    if not m:
        return None
    idx = int(m.group())
    if 1 <= idx <= n_candidates:
        return idx - 1
    return None


class GraphEQAController(DynamemController):
    """
    Robot controller that uses GraphEQA (graph-based) memory for EQA instead of
    the voxel map. Keeps the same voxel map for navigation and exploration;
    feeds the graph memory on each update and uses it in run_eqa.
    """

    ground_truth_mode = False

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters | dict,
        semantic_sensor=None,
        save_rerun: bool = False,
        enable_live_rerun: bool = False,
        use_instance_graph: bool = True,
        realtime_updates: bool = False,
        re: int = 3,
        manip_port: int = 5557,
        log: str | None = None,
        server_ip: str | None = "127.0.0.1",
        mllm: bool = False,
        manipulation_only: bool = False,
        cpu_only: bool = False,
        graph_memory_input_path: str | None = None,
        use_sensor_perception: bool = True,
        semantic_ingest_mode: str = "streaming_objects",
        perception_client=None,
        graph_instance_dedup_xy_m: float | None = None,
        eqa: bool | None = None,
        defer_eqa_vllm: bool = True,
    ):
        # Instance graph: YoloE + SparseVoxelMap Frame masks; voxel ``run_eqa`` off (no per-frame VLM list_objects).
        # Legacy ``--no-instance-graph``: voxel list_objects + VLM / voxel labels for graph nodes.
        # Optional ``eqa=True`` keeps instance graph but still loads caption/EQA clients (agent --eqa).
        voxel_eqa = (not use_instance_graph) if eqa is None else bool(eqa)
        super().__init__(
            robot=robot,
            parameters=parameters,
            semantic_sensor=semantic_sensor,
            save_rerun=save_rerun,
            enable_live_rerun=enable_live_rerun,
            use_instance_memory=use_instance_graph,
            realtime_updates=realtime_updates,
            re=re,
            manip_port=manip_port,
            log=log,
            server_ip=server_ip,
            mllm=mllm,
            manipulation_only=manipulation_only,
            cpu_only=cpu_only,
            eqa=voxel_eqa,
            defer_eqa_vllm=bool(defer_eqa_vllm) if voxel_eqa else True,
        )
        logger.info("Agent init: building GraphEQA memory")
        self.graph_memory = GraphEQAMemory(
            parameters=parameters,
            log_dir="graph_eqa_log",
            defer_llm_clients=True,
        )
        logger.info("Agent init: GraphEQA memory ready")
        if graph_memory_input_path:
            from pathlib import Path

            from emet.memory.backend import get_memory_backend
            from emet.memory.format import VOXEL_PICKLE_FILENAME

            backend = get_memory_backend(
                "graph_eqa",
                graph_memory=self.graph_memory,
                voxel_map=getattr(self, "voxel_map", None),
            )
            backend.load(graph_memory_input_path)
            voxel_pickle = Path(graph_memory_input_path) / VOXEL_PICKLE_FILENAME
            vm = getattr(self, "voxel_map", None)
            if (
                bool(getattr(backend, "loaded_has_voxel_pickle", False))
                and voxel_pickle.exists()
                and vm is not None
                and hasattr(vm, "read_from_pickle")
            ):
                vm.read_from_pickle(voxel_pickle)
            # Resume the staleness clock from the checkpoint so maintain() does not
            # immediately prune reloaded nodes (lifelong checkpoint resume).
            final_step = getattr(backend, "loaded_final_step", None)
            if final_step is not None and int(final_step) > 0:
                self.obs_count = max(int(self.obs_count), int(final_step))
                self.graph_memory.set_graph_timestep(self.obs_count)

        self.use_instance_graph = use_instance_graph
        self.use_sensor_perception = use_sensor_perception
        self._graph_eqa_use_instance_graph = self.use_instance_graph
        self._graph_eqa_use_sensor_perception = self.use_sensor_perception
        self._graph_eqa_semantic_ingest_mode = str(semantic_ingest_mode or "streaming_objects")
        if graph_instance_dedup_xy_m is not None:
            self._graph_dedup_xy_m = float(graph_instance_dedup_xy_m)
        elif isinstance(parameters, dict):
            self._graph_dedup_xy_m = float(
                parameters.get("graph_instance_dedup_xy_m", DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M)
            )
        else:
            self._graph_dedup_xy_m = float(
                parameters.get("graph_instance_dedup_xy_m", DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M)
            )

        from emet.memory.graph_eqa.graph_object_fusion.attach import attach_graph_object_fusion

        self._graph_object_fusion = attach_graph_object_fusion(
            self.graph_memory,
            parameters,
        )
        self._calibration_writer = None

        dev = self.device if self.device in ("cuda", "mps") else "cuda"
        self.sensor_builder = None
        if use_sensor_perception:
            self.sensor_builder = SensorGraphBuilder(
                perception_client=perception_client,
                use_voxel_fallback=True,
                device=dev,
                cpu_only=self.cpu_only,
                parameters=parameters,
            )

        # Baseline GraphEQA: improvements (SigLIP-grounded CONFIRMED_MEMORY + frontier
        # coverage override) stay OFF here so this controller is a clean baseline. The
        # DynagraphController turns them on.
        self._eqa_explore_when_uncovered = False
        self._eqa_explore_uncovered_habitat_frontier = False
        # Experiment flag: classic coverage path only. Agentic explore always calls
        # ``_vlm_frontier_choice`` when present. Enable classic with EMET_VLM_FRONTIER_SCORING=1.
        self._vlm_frontier_scoring = os.environ.get("EMET_VLM_FRONTIER_SCORING", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._habitat_blocked_goals: set[tuple[float, float]] = set()
        self._habitat_recent_goals: list[tuple[float, float]] = []

    def setup_custom_blueprint(self) -> None:
        """Context (VLM) + EQA mosaic — same column as live Dynagraph, including ``emet run graph-eqa``."""
        send_graph_memory_rerun_blueprint(self.rerun_visualizer)

    def _log_graph_eqa_rerun(self) -> None:
        if self.graph_memory is None or not visualizer_is_enabled(self.rerun_visualizer):
            return
        self.rerun_visualizer.log_dynagraph_state(
            self.graph_memory,
            ground_truth_mode=self.ground_truth_mode,
        )
        log_vlm_context_to_visualizer(self.rerun_visualizer, self.graph_memory)

    def update(self, *, full_perception: bool | None = None) -> None:
        super().update(full_perception=full_perception)
        self._log_graph_eqa_rerun()

    def look_around(self):
        """Habitat has no head actuators — rotate the base to build coverage."""
        if is_habitat_robot_client(self.robot):
            habitat_body_scan(self.robot, on_step=self.update)
            return
        super().look_around()

    def _best_frontier_point_from_graph(self, text: str | None) -> np.ndarray | None:
        """Use graph information gain/risk scoring before the nearest-frontier fallback."""
        gm = getattr(self, "graph_memory", None)
        if gm is not None and hasattr(gm, "hypothesize_nav_targets"):
            pose = self._planning_base_xyt(self.robot.get_base_pose())
            try:
                hypotheses = gm.hypothesize_nav_targets(
                    text or "",
                    max_k=12,
                    robot_xyt=pose,
                )
            except TypeError:
                hypotheses = gm.hypothesize_nav_targets(text or "", max_k=12)
            frontier = next(
                (hypothesis for hypothesis in hypotheses if hypothesis.source == "frontier"),
                None,
            )
            if frontier is not None:
                return np.array(
                    [float(frontier.xyz[0]), float(frontier.xyz[1]), 1.0],
                    dtype=float,
                )
        return super()._best_frontier_point_from_graph(text)

    def _habitat_should_prefer_frontier(
        self,
        *,
        confidence: bool,
        target_point: np.ndarray | None,
    ) -> bool:
        if confidence or not is_habitat_robot_client(self.robot):
            return False
        if not habitat_perfect_nav_enabled(self.parameters):
            return False
        if not habitat_explore_frontiers_enabled(self.parameters):
            return False
        if (
            self.graph_memory is not None
            and getattr(self.graph_memory, "eqa_stay_on_attached_view", None) is not None
            and self.graph_memory.eqa_stay_on_attached_view() is True
        ):
            oid_fn = getattr(self.graph_memory, "eqa_attached_target_obs_id", None)
            spent_fn = getattr(self.graph_memory, "eqa_obs_look_spent", None)
            oid = oid_fn() if callable(oid_fn) else None
            spent = bool(oid is not None and callable(spent_fn) and spent_fn(oid))
            try:
                covered = self.graph_memory._graph_covers_relevant_objects()
            except Exception:
                covered = True
            if covered and not spent:
                return False
        if target_point is not None:
            if habitat_nav_would_be_noop(self.robot, target_point):
                return True
        if getattr(self, "_eqa_explore_uncovered_habitat_frontier", False) and self.graph_memory is not None:
            try:
                if not self.graph_memory._graph_covers_relevant_objects():
                    return True
            except Exception:
                return True
        last_nav = getattr(self, "_last_nav_attempt", None)
        if last_nav is not None and not last_nav.finished and float(last_nav.dist_m) < 0.05:
            return True
        if target_point is None:
            return True
        return False

    def _siglip_text_match(self, text: str) -> tuple[float, np.ndarray] | None:
        """Open-vocab visual grounding via the voxel map's SigLIP features.

        Returns ``(max_cosine_similarity, xyz)`` for the best-matching observed point, or
        ``None`` if no features exist yet. Decouples grounding from the VLM caption labels.
        """
        vm = getattr(self, "voxel_map", None)
        if vm is None or not hasattr(vm, "find_alignment_over_model"):
            return None
        try:
            alignments = vm.find_alignment_over_model(text)
            if alignments is None:
                return None
            points, _, _, _ = vm.semantic_memory.get_pointcloud()
            if points is None:
                return None
            a = alignments.detach().cpu().squeeze()
            idx = int(a.argmax())
            return float(a.max()), points[idx].detach().cpu().numpy()
        except Exception:
            return None

    def _siglip_obs_id_for_text(self, text: str) -> int | None:
        """Best-aligned observation id for *text* via the voxel map's SigLIP features.

        Voxel ``obs_count`` is a frame index; graph obs ids diverge after instance
        merges, so map through ``resolve_voxel_frame_to_graph_obs_id``.
        """
        vm = getattr(self, "voxel_map", None)
        if vm is None or not hasattr(vm, "find_obs_id_for_text"):
            return None
        try:
            oid = vm.find_obs_id_for_text(text)
            if oid is None:
                return None
            if hasattr(oid, "item"):
                oid = oid.item()
            voc = int(oid)
        except Exception:
            return None
        gm = getattr(self, "graph_memory", None)
        mapper = getattr(gm, "resolve_voxel_frame_to_graph_obs_id", None) if gm is not None else None
        if callable(mapper):
            try:
                mapped = mapper(voc, vm)
                if mapped is not None:
                    return int(mapped)
            except (TypeError, ValueError):
                pass
        return voc

    def _siglip_visual_find(self, text: str, max_n: int = 4) -> list[tuple[float, int]]:
        """Top-k DynaMem retrieve: phrase → scored graph observation ids.

        Uses ``find_all_images`` (not argmax). SigLIP only proposes RGB; Qwen still
        looks. Falls back to ``find_obs_id_for_text`` when the voxel map is empty.
        """
        vm = getattr(self, "voxel_map", None)
        gm = getattr(self, "graph_memory", None)
        if vm is None or gm is None:
            return []
        ranked: list[tuple[float, int]] = []
        seen: set[int] = set()

        def _add(sim: float, oid: int | None) -> None:
            if oid is None:
                return
            try:
                graph_oid = int(oid)
            except (TypeError, ValueError):
                return
            usable = getattr(gm, "_obs_usable_for_eqa_image", None)
            if callable(usable) and not usable(graph_oid):
                return
            if graph_oid in seen:
                return
            seen.add(graph_oid)
            ranked.append((float(sim), graph_oid))

        def _map_voxel(voc: int, xyz: Any | None) -> int | None:
            mapper = getattr(gm, "resolve_voxel_frame_to_graph_obs_id", None)
            if callable(mapper):
                try:
                    mapped = mapper(int(voc), vm)
                    if mapped is not None:
                        return int(mapped)
                except (TypeError, ValueError):
                    pass
            nearest = getattr(gm, "nearest_graph_obs_to_xyz", None)
            if callable(nearest) and xyz is not None:
                try:
                    mapped = nearest(xyz)
                    if mapped is not None:
                        return int(mapped)
                except (TypeError, ValueError):
                    pass
            return None

        enc = getattr(vm, "encoder", None)
        if enc is not None and hasattr(vm, "find_all_images"):
            try:
                ids, points, aligns = vm.find_all_images(
                    text,
                    min_point_num=20,
                    max_img_num=max(int(max_n), 4),
                )
            except Exception:
                ids, points, aligns = None, None, None
            if ids is not None:
                from emet.memory.graph_eqa.graph_eqa_siglip import flatten_find_all_images

                for sim, voc, xyz in flatten_find_all_images(ids, points, aligns):
                    _add(sim, _map_voxel(int(voc), xyz))
        if not ranked:
            voc = self._siglip_obs_id_for_text(text)
            if voc is not None:
                _add(max(0.25, 0.21), voc)
        ranked.sort(key=lambda t: -t[0])
        limit = max(int(max_n), 1)
        return ranked[:limit]

    def _siglip_guided_frontier(self, text: str) -> np.ndarray | None:
        """Intelligent exploration: head toward the frontier nearest the most query-aligned
        observed point (SigLIP), so the robot moves toward visually-similar regions instead
        of revisiting seen views. Caption-independent, so it works even when the object was
        seen but mislabeled. Returns a nav waypoint ``[x, y, 1.0]`` or ``None``.
        """
        sig = self._siglip_text_match(text)
        if sig is None:
            return None
        _sim, xyz = sig
        gm = getattr(self, "graph_memory", None)
        frontier_nodes = [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)] if gm is not None else []
        best = None
        best_d = float("inf")
        for n in frontier_nodes:
            d = float(np.linalg.norm(np.asarray(n.xyz[:2], dtype=float) - np.asarray(xyz[:2], dtype=float)))
            if d < best_d:
                best_d, best = d, n
        if best is not None:
            return np.array([float(best.xyz[0]), float(best.xyz[1]), 1.0], dtype=float)
        # No frontier graph nodes: fall back to a keyword/SigLIP-biased exploration sample.
        if hasattr(self, "space") and hasattr(self.space, "sample_frontier"):
            fr = self.space.sample_frontier(
                self.planner, self._planning_base_xyt(self.robot.get_base_pose()), text=text
            )
            if fr is not None:
                return np.array([float(fr[0]), float(fr[1]), 1.0], dtype=float)
        return None

    def _vlm_frontier_choice(
        self,
        question: str,
        *,
        max_candidates: int = 6,
        current_room: str | None = None,
        room_policy: str = "canonical",
        leave_hint: bool = False,
    ) -> np.ndarray | None:
        """Ask the EQA VLM which frontier view best helps answer the question.

        Samples up to ``max_candidates`` **reachable** frontier nodes that have RGB
        (ranked by region utility), attaches graph ``room=`` / ``near=`` context,
        and asks which image is most useful for the Question. Returns ``[x,y,1]``.
        """
        gm = getattr(self, "graph_memory", None)
        if gm is None or gm.eqa_client is None:
            return None
        from emet.memory.graph_eqa.spatial.frontier_regions import (
            frontier_region_utility,
            region_from_node,
        )
        from emet.memory.graph_eqa.spatial.room_labels import coerce_room_label

        robot = getattr(self, "robot", None)
        habitat = robot is not None and is_habitat_robot_client(robot)
        robot_xy = robot_planar_xy(robot) if robot is not None else (0.0, 0.0)
        blocked = set(getattr(self, "_habitat_blocked_goals", None) or set())
        recent = list(getattr(self, "_habitat_recent_goals", None) or [])
        grid_m = explore_grid_resolution_m(self)
        policy = str(room_policy or "canonical").strip().lower()

        def _frontier_room_near(n: Any) -> tuple[str, list[str]]:
            xy = (float(n.xyz[0]), float(n.xyz[1]))
            room = "unknown"
            room_fn = getattr(gm, "graph_room_at_robot", None)
            if callable(room_fn):
                try:
                    room = coerce_room_label(room_fn(xy), room_policy=policy)
                except Exception:
                    room = "unknown"
            near: list[str] = []
            scored_labs: list[tuple[float, str]] = []
            for node in list(getattr(gm, "_nodes", None) or []):
                if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                    continue
                try:
                    nxy = (float(node.xyz[0]), float(node.xyz[1]))
                    d = float(np.hypot(nxy[0] - xy[0], nxy[1] - xy[1]))
                except Exception:
                    continue
                if d > 2.5:
                    continue
                for lab in list(getattr(node, "labels", None) or [])[:2]:
                    s = str(lab).strip()
                    if s and s.lower() != "frontier":
                        scored_labs.append((d, s))
            scored_labs.sort(key=lambda t: t[0])
            for _d, s in scored_labs:
                if s not in near:
                    near.append(s)
                if len(near) >= 3:
                    break
            return room, near

        scored: list[tuple[float, float, Any, Any, np.ndarray, str, list[str]]] = []
        for n in gm.get_nodes():
            if not getattr(n, "is_frontier", False):
                continue
            obs = gm._observation_by_id(int(n.obs_id))
            if obs is None or obs.rgb is None:
                continue
            raw = np.array([float(n.xyz[0]), float(n.xyz[1]), 1.0], dtype=float)
            waypoint = raw
            if habitat:
                key = goal_key_xy(raw)
                if key in blocked:
                    continue
                resolved = apply_habitat_nav_resolution(robot, raw)
                if resolved is None:
                    blocked.add(key)
                    continue
                eff_key = goal_key_xy(resolved)
                if eff_key in blocked or habitat_nav_would_be_noop(robot, resolved):
                    blocked.add(key)
                    blocked.add(eff_key)
                    continue
                waypoint = resolved
            util = frontier_region_utility(
                region_from_node(n),
                robot_xy,
                grid_resolution_m=grid_m,
                recent=recent,
            )
            froom, fnear = _frontier_room_near(n)
            dist = float(np.hypot(float(waypoint[0]) - robot_xy[0], float(waypoint[1]) - robot_xy[1]))
            scored.append((util, dist, n, obs, waypoint, froom, fnear))
        if not scored:
            return None
        scored.sort(key=lambda t: (-t[0], t[1]))
        candidates = scored[: max(1, int(max_candidates))]
        lines: list[str] = []
        for i, (_u, _d, n, _obs, _wp, froom, fnear) in enumerate(candidates, start=1):
            near_bit = f" near={fnear}" if fnear else ""
            lines.append(f"Image {i}: unexplored room={froom}{near_bit} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f})")
        ctx: list[str] = []
        if current_room:
            ctx.append(f"Current place: {current_room}")
        if leave_hint:
            ctx.append(
                "Current place looks unhelpful for the question — prefer a view that "
                "could lead somewhere more informative."
            )
        ctx_block = ("\n".join(ctx) + "\n") if ctx else ""
        directive = (
            "You are exploring a home. Each image is an unexplored direction "
            "(with graph room/nearby-object context). Which image would best help "
            "determine the answer to the question? Reply with ONLY the image number.\n"
            f"{ctx_block}"
            f"Question: {question}\n" + "\n".join(lines)
        )
        images = [
            Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB") for _u, _d, _n, obs, _wp, _fr, _fn in candidates
        ]
        try:
            reply = gm.eqa_client([directive, *images])
        except Exception:
            return None
        pick = _parse_image_pick(reply, len(candidates))
        if pick is None:
            return None
        return np.asarray(candidates[pick][4], dtype=float).reshape(-1)[:3].copy()

    def _graph_dedup_skips(self, label: str, xyz: np.ndarray) -> bool:
        """Skip adding a node if a compatible label already exists near this XY.

        Uses :func:`~emet.memory.graph_eqa.graph_stats.labels_compatible_for_dedup` so
        open-vocab drift (``mug`` vs ``coffee cup``) does not bypass XY dedup.
        """
        if self._graph_dedup_xy_m <= 0 or self.graph_memory is None:
            return False
        from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

        for n in self.graph_memory.get_nodes():
            if getattr(n, "is_viewpoint", False) or getattr(n, "is_frontier", False):
                continue
            if not n.labels:
                continue
            if not labels_compatible_for_dedup(label, str(n.labels[0])):
                continue
            if float(np.linalg.norm(n.xyz[:2] - xyz[:2])) < self._graph_dedup_xy_m:
                return True
        return False

    def run_eqa_one_iter(
        self,
        question: str,
        max_movement_step: int = 5,
        *,
        skip_perception_prelude: bool = False,
        allow_navigation: bool = True,
    ) -> tuple[str, str, list[Image.Image], bool]:
        """One EQA iteration using graph memory instead of voxel map.

        When *skip_perception_prelude* is True, skip the head sweep / look-around before the LLM call
        (used on follow-up EQA iterations after navigation so we do not re-run perception every step).
        When *allow_navigation* is False, return after the VLM answer without frontier chase
        (question-bank / post-explore scoring).
        """
        answer_output = None
        # After explore-loop (``_fast_explore_lookaround``), skip another ~30--60s head sweep.
        if not self._realtime_updates and not skip_perception_prelude:
            if getattr(self, "_fast_explore_lookaround", False):
                self.robot.look_front()
                self.robot.switch_to_navigation_mode()
            else:
                self.robot.look_front()
                self.look_around()
                self.robot.look_front()
                self.robot.switch_to_navigation_mode()

        if self.graph_memory is not None and hasattr(self, "_sync_graph_frontier_nodes"):
            self._sync_graph_frontier_nodes()

        try:
            q_preview = question if isinstance(question, str) else str(question)[:80]
            logger.info(f"EQA query_answer start for {q_preview!r}")
            t_qa0 = time.monotonic()
            (
                reasoning,
                answer,
                confidence,
                confidence_reasoning,
                target_point,
                relevant_images,
            ) = self.graph_memory.query_answer(
                question,
                self._planning_base_xyt(self.robot.get_base_pose()),
                self.planner,
            )
            logger.info(
                f"EQA query_answer done wall_s={time.monotonic() - t_qa0:.1f} "
                f"confidence={confidence} answer={(answer or '')[:120]!r}"
            )
        except Exception as e:
            reasoning = f"Error: {e}"
            answer = "Unknown"
            confidence = False
            confidence_reasoning = str(e)
            target_point = None
            if self.graph_memory is not None:
                self.graph_memory._append_eqa_history(
                    "Answer:Unknown\nReasoning:"
                    + reasoning
                    + "\nConfidence:False\nAction:\nConfidence_reasoning:"
                    + confidence_reasoning
                )
            if hasattr(self, "space") and hasattr(self.space, "sample_frontier"):
                target_point = self.space.sample_frontier(
                    self.planner,
                    self._planning_base_xyt(self.robot.get_base_pose()),
                    text=question,
                )
            relevant_images = []

        confidence_text = "I am confident with the answer" if confidence else "I am NOT confident with the answer"
        reasoning_output = (
            "\n#### Reasoning for the answer: " + reasoning
            if confidence
            else "\n#### Reasoning for the confidence: " + confidence_reasoning
        )
        answer_output = (
            "#### **Question:** "
            + question
            + "\n#### **Answer:** "
            + answer
            + "\n#### **Confidence:** "
            + confidence_text
            + reasoning_output
        )
        self._rerun_monologue_base = answer_output
        self._rerun_refresh_monologue_panel()
        if self.graph_memory is not None and visualizer_is_enabled(self.rerun_visualizer):
            log_vlm_context_to_visualizer(self.rerun_visualizer, self.graph_memory)
        if relevant_images and hasattr(self, "_patch_images"):
            self.rerun_visualizer.log_custom_2d_image(
                "/observation_similar_to_text", self._patch_images(relevant_images)
            )
        elif relevant_images:
            self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", relevant_images)

        if confidence:
            discord_text = answer
        else:
            short_cr = (confidence_reasoning or reasoning or "").strip()
            if len(short_cr) > 280:
                short_cr = short_cr[:277] + "..."
            discord_text = f"{answer} I am not fully confident yet; {short_cr}" if short_cr else answer
        discord_text += "\nI also provide relevant images here."

        if confidence or not allow_navigation:
            return answer, discord_text, relevant_images, confidence

        stay_fn = getattr(self.graph_memory, "eqa_should_stay_on_attached_view", None)
        if callable(stay_fn):
            stay = bool(stay_fn(answer=answer, confidence=confidence))
        else:
            stay = (
                self.graph_memory is not None
                and getattr(self.graph_memory, "eqa_stay_on_attached_view", None) is not None
                and self.graph_memory.eqa_stay_on_attached_view() is True
            )
        if (
            not stay
            and self.graph_memory is not None
            and getattr(self.graph_memory, "eqa_stay_on_attached_view", None) is not None
            and self.graph_memory.eqa_stay_on_attached_view()
        ):
            oid = self.graph_memory.eqa_attached_target_obs_id()
            logger.info(
                "EQA nav: releasing attached-view stay (unknown/uncovered/spent) obs_id=%s",
                oid,
            )
        approaching_find = False
        if stay:
            oid = self.graph_memory.eqa_attached_target_obs_id()
            if oid is not None and self.graph_memory.last_eqa_look_obs_id is None:
                self.graph_memory.last_eqa_look_obs_id = int(oid)
            approach = None
            fn = getattr(self.graph_memory, "eqa_approach_attached_find", None)
            if callable(fn):
                try:
                    raw = fn(self._planning_base_xyt(self.robot.get_base_pose()))
                    if raw is not None:
                        arr = np.asarray(raw, dtype=float).reshape(-1)
                        if arr.size >= 2 and np.isfinite(arr[:2]).all():
                            approach = np.array([float(arr[0]), float(arr[1]), 1.0], dtype=float)
                except (TypeError, ValueError):
                    approach = None
            if approach is None:
                logger.info("EQA nav: FIND view already Image 1 (obs_id=%s); stay for close-up", oid)
                return answer, discord_text, relevant_images, confidence
            target_point = approach
            approaching_find = True
            logger.info(
                "EQA nav: approach attached FIND obs_id=%s target=(%.2f, %.2f)",
                oid,
                float(target_point[0]),
                float(target_point[1]),
            )

        # Coverage: while the question-relevant objects have NOT been observed yet, prefer an
        # unexplored, question-matched frontier over revisiting the VLM's already-seen
        # "Navigate to Image N" target. The VLM anchors on objects it has already seen and
        # rarely sends the robot into new rooms, so targets it never observes (e.g. a basket
        # in an unexplored room) stay unanswerable. Once those objects are in the graph (or
        # the VLM is confident) we follow its inspection target.
        # Use blocked/recent-aware pick (Habitat navmesh + MuJoCo sample_frontier).
        if (
            not approaching_find
            and not confidence
            and self.graph_memory is not None
            and getattr(self, "_eqa_explore_when_uncovered", False)
        ):
            try:
                covered = self.graph_memory._graph_covers_relevant_objects()
            except Exception:
                covered = True
            if not covered:
                candidates: list[np.ndarray | None] = []
                if self._vlm_frontier_scoring:
                    candidates.append(self._vlm_frontier_choice(question))
                candidates.append(self._siglip_guided_frontier(question))
                candidates.append(self._best_frontier_point_from_graph(question))
                frontier_pt = pick_uncovered_explore_target(
                    self,
                    question=question,
                    candidates=candidates,
                    blocked=self._habitat_blocked_goals,
                    recent_goals=self._habitat_recent_goals,
                )
                if frontier_pt is not None:
                    target_point = frontier_pt

        if not approaching_find and self._habitat_should_prefer_frontier(
            confidence=confidence, target_point=target_point
        ):
            frontier_pt = pick_habitat_exploration_target(
                self,
                question=question,
                blocked=self._habitat_blocked_goals,
                recent_goals=self._habitat_recent_goals,
            )
            if frontier_pt is not None:
                logger.info(
                    f"EQA habitat: frontier explore target ({float(frontier_pt[0]):.2f}, {float(frontier_pt[1]):.2f})"
                )
                target_point = frontier_pt

        if not approaching_find and target_point is None and not confidence:
            target_point = pick_uncovered_explore_target(
                self,
                question=question,
                blocked=self._habitat_blocked_goals,
                recent_goals=self._habitat_recent_goals,
            )
        if (
            not approaching_find
            and target_point is None
            and not confidence
            and hasattr(self, "space")
            and hasattr(self.space, "sample_frontier")
        ):
            frontier = self.space.sample_frontier(
                self.planner,
                self._planning_base_xyt(self.robot.get_base_pose()),
                text=question,
            )
            if frontier is not None:
                target_point = np.array([float(frontier[0]), float(frontier[1]), 1.0], dtype=float)

        if (
            target_point is not None
            and is_habitat_robot_client(self.robot)
            and habitat_perfect_nav_enabled(self.parameters)
        ):
            resolved = apply_habitat_nav_resolution(self.robot, target_point)
            if resolved is None:
                self._habitat_blocked_goals.add(goal_key_xy(target_point))
                alt = pick_uncovered_explore_target(
                    self,
                    question=question,
                    blocked=self._habitat_blocked_goals,
                    recent_goals=self._habitat_recent_goals,
                )
                target_point = alt
            else:
                eff_key = goal_key_xy(resolved)
                if stay and (eff_key in self._habitat_blocked_goals or habitat_nav_would_be_noop(self.robot, resolved)):
                    logger.info("EQA habitat: already at FIND/readout view; stay for close-up")
                    return answer, discord_text, relevant_images, confidence
                if eff_key in self._habitat_blocked_goals or habitat_nav_would_be_noop(self.robot, resolved):
                    self._habitat_blocked_goals.add(eff_key)
                    self._habitat_blocked_goals.add(goal_key_xy(target_point))
                    alt = pick_uncovered_explore_target(
                        self,
                        question=question,
                        blocked=self._habitat_blocked_goals,
                        recent_goals=self._habitat_recent_goals,
                    )
                    if alt is None:
                        habitat_random_walk_step(self.robot)
                        habitat_body_scan(self.robot, turns=2, on_step=self.update)
                        if hasattr(self, "_sync_graph_frontier_nodes"):
                            self._sync_graph_frontier_nodes()
                        alt = pick_uncovered_explore_target(
                            self,
                            question=question,
                            blocked=self._habitat_blocked_goals,
                            recent_goals=self._habitat_recent_goals,
                        )
                    target_point = alt
                else:
                    target_point = resolved

        if target_point is not None and hasattr(self, "navigate_to_target_pose"):
            action_obs_id = (
                int(self.graph_memory.last_eqa_action_obs_id)
                if self.graph_memory is not None and self.graph_memory.last_eqa_action_obs_id is not None
                else None
            )
            if action_obs_id is not None and self.graph_memory is not None:
                spent_fn = getattr(self.graph_memory, "eqa_obs_look_spent", None)
                spent = bool(callable(spent_fn) and spent_fn(action_obs_id) is True)
                node = next(
                    (n for n in self.graph_memory.get_nodes() if int(n.obs_id) == action_obs_id),
                    None,
                )
                failed = node is not None and int(getattr(node, "nav_failures", 0)) > 0
                if spent or failed:
                    nxt = None
                    if spent and hasattr(self.graph_memory, "next_unspent_eqa_obs_id"):
                        nxt = self.graph_memory.next_unspent_eqa_obs_id(
                            list(self.graph_memory.last_eqa_obs_ids or []),
                            skip={int(action_obs_id)},
                        )
                    if nxt is not None:
                        logger.info(
                            "EQA nav: obs_id=%s look spent; switching to unspent obs_id=%s",
                            action_obs_id,
                            nxt,
                        )
                        wp = self.graph_memory._navigation_waypoint_for_obs(
                            int(nxt),
                            self._planning_base_xyt(self.robot.get_base_pose()),
                        )
                        if wp is not None:
                            target_point = wp
                            action_obs_id = int(nxt)
                    else:
                        alt = self.graph_memory.alternate_nav_target_for_failed_action(
                            question,
                            action_obs_id,
                            self.planner,
                            self._planning_base_xyt(self.robot.get_base_pose()),
                        )
                        if alt is not None:
                            logger.info(
                                "EQA nav: avoiding re-pick of Image obs_id=%s, alternate frontier",
                                action_obs_id,
                            )
                            target_point = alt
                            action_obs_id = None

            start_pose = self._planning_base_xyt(self.robot.get_base_pose())
            if (
                is_habitat_robot_client(self.robot)
                and habitat_perfect_nav_enabled(self.parameters)
                and target_point is not None
            ):
                target_theta = float(
                    np.arctan2(
                        float(target_point[1]) - float(start_pose[1]),
                        float(target_point[0]) - float(start_pose[0]),
                    )
                )
            else:
                obstacles, _ = self.voxel_map.get_2d_map()
                target_grid = self.voxel_map.xy_to_grid_coords((float(target_point[0]), float(target_point[1])))
                if (
                    obstacles.shape[0] > int(target_grid[0])
                    and obstacles.shape[1] > int(target_grid[1])
                    and not obstacles[int(target_grid[0]), int(target_grid[1])]
                ):
                    nav_samples = self.space.sample_navigation(start_pose, self.planner, target_point)
                    target_theta = nav_samples[-1] if nav_samples is not None else None
                else:
                    target_theta = None
                if target_theta is None and target_point is not None:
                    target_theta = float(
                        np.arctan2(
                            float(target_point[1]) - float(start_pose[1]),
                            float(target_point[0]) - float(start_pose[0]),
                        )
                    )
            stuck_retries = 0
            # Per-step displacement from navigate_to_target_pose (not distance-to-goal).
            min_success_dist_m = 0.08
            for move_i in range(max_movement_step):
                logger.info(
                    "EQA nav: step %d/%d target=(%.2f, %.2f)",
                    move_i + 1,
                    max_movement_step,
                    float(target_point[0]),
                    float(target_point[1]),
                )
                start_pose = self._planning_base_xyt(self.robot.get_base_pose())
                self.update()
                finished = self.navigate_to_target_pose(
                    target_point,
                    start_pose,
                    target_theta,
                    target_obs_id=action_obs_id,
                )
                nav_res = getattr(self, "_last_nav_attempt", None)
                if self.graph_memory is not None:
                    # Nav counters / ledger dual-write are owned by
                    # DynamemController._log_nav_attempt → sync_nav_attempt_to_ledger.
                    # Fallback only when navigate_to_target_pose published no result.
                    if nav_res is None:
                        self.graph_memory.record_nav_attempt(
                            action_obs_id,
                            success=False,
                            note="no_nav",
                            dist_m=0.0,
                        )
                    self.graph_memory.append_nav_outcome_to_last_history(
                        dist_m=float(nav_res.dist_m) if nav_res else 0.0,
                        success=bool(getattr(nav_res, "success", False) if nav_res else False),
                        note=(nav_res.note if nav_res else "no_nav"),
                    )
                if finished.finished:
                    if getattr(self, "_lazy_graph_mode", False):
                        self._commit_lazy_graph_arrival(
                            action_obs_id=action_obs_id,
                            target_point=target_point,
                        )
                    break
                if nav_res is not None and (
                    nav_res.note.startswith("already_at_goal")
                    or (not nav_res.finished and float(nav_res.dist_m) < min_success_dist_m)
                ):
                    self._habitat_blocked_goals.add(goal_key_xy(target_point))
                    stuck_retries += 1
                    if (
                        stuck_retries >= 2
                        and is_habitat_robot_client(self.robot)
                        and habitat_perfect_nav_enabled(self.parameters)
                    ):
                        alt = pick_habitat_exploration_target(
                            self,
                            question=question,
                            blocked=self._habitat_blocked_goals,
                            recent_goals=self._habitat_recent_goals,
                        )
                        if alt is not None and goal_key_xy(alt) != goal_key_xy(target_point):
                            logger.info(
                                f"EQA habitat: re-pick after stuck nav ({float(alt[0]):.2f}, {float(alt[1]):.2f})"
                            )
                            target_point = alt
                            target_theta = float(
                                np.arctan2(
                                    float(target_point[1]) - float(start_pose[1]),
                                    float(target_point[0]) - float(start_pose[0]),
                                )
                            )
                            stuck_retries = 0
                            continue
                        habitat_random_walk_step(self.robot)
                        break
            if self.graph_memory is not None and hasattr(self, "_sync_graph_frontier_nodes"):
                self._sync_graph_frontier_nodes()

        return answer, discord_text, relevant_images, confidence

    def run_eqa(
        self,
        question: str,
        max_planning_steps: int = 5,
        *,
        max_movement_step: int = 5,
        allow_navigation: bool = True,
        trace_meta: dict[str, Any] | None = None,
    ) -> tuple[str, list[Image.Image]]:
        """Run EQA until confident or max steps, using graph memory.

        Set ``allow_navigation=False`` (and typically ``max_planning_steps=1``) for
        post-explore question banks: answer from current memory without frontier chase.
        With ``allow_navigation=True`` and ``max_planning_steps>1``, the final step
        skips frontier chase so the loop can return after the last VLM call.

        When ``eqa.agentic_verify`` (or ``EMET_EQA_AGENTIC_VERIFY=1``) is set, uses the
        unified agentic explore/navigate/verify/answer loop instead.
        """
        from emet.memory.graph_eqa import agentic_verify_enabled, run_agentic_eqa

        effective_trace_meta = dict(getattr(self, "_eqa_trace_meta", None) or {})
        effective_trace_meta.update(dict(trace_meta or {}))
        if self.graph_memory is not None and effective_trace_meta:
            question_id = effective_trace_meta.get("question_id")
            if question_id is None:
                question_id = effective_trace_meta.get("qid")
            session_id = effective_trace_meta.get("session_id")
            if session_id is None:
                session_id = effective_trace_meta.get("episode_id")
            bind_context = getattr(self.graph_memory, "bind_episode_context", None)
            if callable(bind_context):
                bind_context(question_id=question_id, session_id=session_id)
        if agentic_verify_enabled(self):
            return run_agentic_eqa(
                self,
                question,
                trace_meta=effective_trace_meta,
            )

        import time as _time

        self._eqa_question = question
        self._habitat_blocked_goals = set()
        self._habitat_recent_goals = []
        answer = ""
        confidence = False
        discord_text = ""
        relevant_images: list[Image.Image] = []
        stall_patience = int(self.parameters.get("eqa_stall_patience", 4) or 0)
        # Wall-clock cap so a stuck nav/VLM loop cannot hold the GPU overnight.
        # EMET_EQA_QUESTION_TIMEOUT_S overrides; 0 disables.
        env_to = os.environ.get("EMET_EQA_QUESTION_TIMEOUT_S")
        if env_to is not None and str(env_to).strip() != "":
            question_timeout_s = float(env_to)
        else:
            question_timeout_s = float(self.parameters.get("eqa_question_timeout_s", 900) or 0)
        t_eqa0 = _time.monotonic()
        prev_node_count = -1
        prev_answer: str | None = None
        stall = 0
        for step in range(max_planning_steps):
            if question_timeout_s > 0 and (_time.monotonic() - t_eqa0) >= question_timeout_s:
                logger.warning(
                    "EQA question wall-clock timeout after %.0fs (limit=%.0fs) at step %d/%d",
                    _time.monotonic() - t_eqa0,
                    question_timeout_s,
                    step,
                    max_planning_steps,
                )
                if not discord_text:
                    discord_text = (
                        f"Answer:Unknown\nEQA timed out after {question_timeout_s:.0f}s (partial answer={answer!r})"
                    )
                break
            q_preview = question if isinstance(question, str) else str(question)[:80]
            logger.info(
                f"EQA planning step {step + 1}/{max_planning_steps} for {q_preview!r} "
                f"(allow_navigation={allow_navigation})"
            )
            if step > 0:
                self.update()
            # Multi-step: skip frontier chase on the *last* planning step so we return
            # the best answer without hanging after the final VLM call. Single-step with
            # allow_navigation=True still navigates (legacy one-shot explore). Question
            # banks pass allow_navigation=False explicitly.
            if not allow_navigation:
                nav_this_step = False
            elif max_planning_steps <= 1:
                nav_this_step = True
            else:
                nav_this_step = step < max_planning_steps - 1
            # Answer-only banks must not touch the robot (look_front / EGL) while the
            # VLM is loading or generating — MuJoCo+Qwen on one GPU made vision
            # prefill look hung until STALE_KILL (~30 min, no log growth).
            answer, discord_text, relevant_images, confidence = self.run_eqa_one_iter(
                question,
                max_movement_step=max_movement_step,
                skip_perception_prelude=(step > 0) or (not allow_navigation),
                allow_navigation=nav_this_step,
            )
            if confidence:
                break
            if stall_patience > 0 and self.graph_memory is not None:
                # Never early-stop on a repeated Yes/No while question objects are still
                # uncovered — absence is not evidence; keep exploring frontiers.
                covers = getattr(self.graph_memory, "_graph_covers_relevant_objects", None)
                uncovered = bool(callable(covers) and not covers())
                if uncovered:
                    stall = 0
                    prev_node_count = len(self.graph_memory.get_nodes())
                    prev_answer = self.graph_memory.last_eqa_parsed[1]
                else:
                    node_count = len(self.graph_memory.get_nodes())
                    cur_answer = self.graph_memory.last_eqa_parsed[1]
                    if node_count <= prev_node_count and cur_answer and cur_answer == prev_answer:
                        stall += 1
                    else:
                        stall = 0
                    prev_node_count = node_count
                    prev_answer = cur_answer
                    if stall >= stall_patience:
                        logger.info(
                            "EQA early stop after %d/%d steps: graph stalled at answer %r",
                            step + 1,
                            max_planning_steps,
                            cur_answer,
                        )
                        break
            if self.graph_memory is not None and hasattr(self, "_sync_graph_frontier_nodes"):
                self._sync_graph_frontier_nodes()
            try:
                from emet.llms.graph_eqa_vlm import trim_shared_graph_eqa_vlm_cache

                trim_shared_graph_eqa_vlm_cache()
            except Exception:
                pass
        if not relevant_images:
            relevant_images = []
        # Terminal + TTS feedback (CLI users otherwise see no reply; parent DynamemController.run_eqa does this for voxel EQA).
        print("\n--- GraphEQA answer ---\n" + discord_text.strip() + "\n---\n", flush=True)
        if confidence:
            try:
                # Async TTS — never use say_sync here (can block forever if sim is gone).
                self.robot.say("The answer to " + question + " is " + answer)
            except Exception:
                pass
        return discord_text, relevant_images


# Alias for compatibility with EQA executor
RobotAgentGraphEQA = GraphEQAController
