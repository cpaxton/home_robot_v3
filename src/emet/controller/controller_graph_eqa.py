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

import numpy as np
from PIL import Image

from emet.controller.controller_dynamem import DynamemController
from emet.core.parameters import Parameters
from emet.core.robot import AbstractRobotClient
from emet.memory.graph_eqa import GraphEQAMemory, SensorGraphBuilder
from emet.memory.graph_eqa.instance_observations import (
    DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M,
)


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

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters | dict,
        semantic_sensor=None,
        save_rerun: bool = False,
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
        perception_client=None,
        graph_instance_dedup_xy_m: float | None = None,
    ):
        # Instance graph: YoloE + SparseVoxelMap Frame masks; voxel ``run_eqa`` off (no per-frame VLM list_objects).
        # Legacy ``--no-instance-graph``: voxel list_objects + VLM / voxel labels for graph nodes.
        voxel_eqa = not use_instance_graph
        super().__init__(
            robot=robot,
            parameters=parameters,
            semantic_sensor=semantic_sensor,
            save_rerun=save_rerun,
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
            defer_eqa_vllm=True,
        )
        self.graph_memory = GraphEQAMemory(
            parameters=parameters,
            log_dir="graph_eqa_log",
            defer_llm_clients=True,
        )
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
            if voxel_pickle.exists() and vm is not None and hasattr(vm, "read_from_pickle"):
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

        from emet.memory.graph_eqa.graph_object_fusion.setup import attach_graph_object_fusion

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
        # Experiment flag: let the EQA VLM pick the exploration frontier from candidate
        # views (instead of the SigLIP-nearest heuristic). Off by default for both
        # controllers; enable per-run with EMET_VLM_FRONTIER_SCORING=1.
        self._vlm_frontier_scoring = os.environ.get(
            "EMET_VLM_FRONTIER_SCORING", ""
        ).strip().lower() in ("1", "true", "yes", "on")

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

        Lets image selection surface the actual view of the target object (caption-independent)
        instead of whatever furniture is in the most recent frames.
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
            return int(oid)
        except Exception:
            return None

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
        frontier_nodes = (
            [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)] if gm is not None else []
        )
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

    def _vlm_frontier_choice(self, question: str) -> np.ndarray | None:
        """Ask the EQA VLM which frontier view most likely leads to the question objects.

        Uses actual reasoning over candidate frontier images (<=6) instead of the
        SigLIP-nearest heuristic. Returns a nav waypoint ``[x, y, 1.0]`` or ``None``
        (no frontiers, no client, or unparseable reply).
        """
        gm = getattr(self, "graph_memory", None)
        if gm is None or gm.eqa_client is None:
            return None
        candidates = []
        for n in gm.get_nodes():
            if not getattr(n, "is_frontier", False):
                continue
            obs = gm._observation_by_id(int(n.obs_id))
            if obs is None or obs.rgb is None:
                continue
            candidates.append((n, obs))
            if len(candidates) >= 6:
                break
        if not candidates:
            return None
        lines = [
            f"Image {i}: unexplored direction at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f})"
            for i, (n, _obs) in enumerate(candidates, start=1)
        ]
        directive = (
            "You are exploring a home to answer a question. Each image shows an "
            "unexplored direction. Which image is most likely to lead to what the "
            "question asks about? Reply with ONLY the image number.\n"
            f"Question: {question}\n" + "\n".join(lines)
        )
        images = [Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB") for _n, obs in candidates]
        try:
            reply = gm.eqa_client([directive, *images])
        except Exception:
            return None
        pick = _parse_image_pick(reply, len(candidates))
        if pick is None:
            return None
        node = candidates[pick][0]
        return np.array([float(node.xyz[0]), float(node.xyz[1]), 1.0], dtype=float)

    def _graph_dedup_skips(self, label: str, xyz: np.ndarray) -> bool:
        """Skip adding a node if we already have the same label near this XY (v1 merge)."""
        if self._graph_dedup_xy_m <= 0:
            return False
        lb = label.strip().lower()
        for n in self.graph_memory.get_nodes():
            if not n.labels:
                continue
            nl = (n.labels[0] or "").strip().lower()
            if nl != lb:
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
    ) -> tuple[str, str, list[Image.Image], bool]:
        """One EQA iteration using graph memory instead of voxel map.

        When *skip_perception_prelude* is True, skip the head sweep / look-around before the LLM call
        (used on follow-up EQA iterations after navigation so we do not re-run perception every step).
        """
        answer_output = None
        if not self._realtime_updates and not skip_perception_prelude:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        if self.graph_memory is not None and hasattr(self, "_sync_graph_frontier_nodes"):
            self._sync_graph_frontier_nodes()

        try:
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
        except Exception as e:
            reasoning = f"Error: {e}"
            answer = "Unknown"
            confidence = False
            confidence_reasoning = str(e)
            target_point = None
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
            discord_text = (
                f"{answer} I am not fully confident yet; {short_cr}" if short_cr else answer
            )
        discord_text += "\nI also provide relevant images here."

        if confidence:
            return answer, discord_text, relevant_images, confidence

        # Coverage: while the question-relevant objects have NOT been observed yet, prefer an
        # unexplored, question-matched frontier over revisiting the VLM's already-seen
        # "Navigate to Image N" target. The VLM anchors on objects it has already seen and
        # rarely sends the robot into new rooms, so targets it never observes (e.g. a basket
        # in an unexplored room) stay unanswerable. Once those objects are in the graph (or
        # the VLM is confident) we follow its inspection target.
        if (
            not confidence
            and self.graph_memory is not None
            and getattr(self, "_eqa_explore_when_uncovered", False)
        ):
            try:
                covered = self.graph_memory._graph_covers_relevant_objects()
            except Exception:
                covered = True
            if not covered:
                # VLM frontier pick (experiment, EMET_VLM_FRONTIER_SCORING=1), then
                # SigLIP-guided exploration (toward visually-similar regions), then
                # the keyword-matched frontier-node heuristic.
                frontier_pt = self._vlm_frontier_choice(question) if self._vlm_frontier_scoring else None
                if frontier_pt is None:
                    frontier_pt = self._siglip_guided_frontier(question)
                if frontier_pt is None:
                    frontier_pt = self._best_frontier_point_from_graph(question)
                if frontier_pt is not None:
                    target_point = frontier_pt

        if target_point is None and not confidence:
            target_point = self._best_frontier_point_from_graph(question)
        if target_point is None and not confidence and hasattr(self, "space") and hasattr(
            self.space, "sample_frontier"
        ):
            frontier = self.space.sample_frontier(
                self.planner,
                self._planning_base_xyt(self.robot.get_base_pose()),
                text=question,
            )
            if frontier is not None:
                target_point = np.array([float(frontier[0]), float(frontier[1]), 1.0], dtype=float)

        if target_point is not None and hasattr(self, "navigate_to_target_pose"):
            start_pose = self._planning_base_xyt(self.robot.get_base_pose())
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
            for _ in range(max_movement_step):
                start_pose = self._planning_base_xyt(self.robot.get_base_pose())
                self.update()
                if self.navigate_to_target_pose(target_point, start_pose, target_theta):
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
    ) -> tuple[str, list[Image.Image]]:
        """Run EQA until confident or max steps, using graph memory."""
        self._eqa_question = question
        answer = ""
        confidence = False
        discord_text = ""
        relevant_images: list[Image.Image] = []
        for step in range(max_planning_steps):
            if step > 0:
                self.update()
            answer, discord_text, relevant_images, confidence = self.run_eqa_one_iter(
                question,
                max_movement_step=max_movement_step,
                skip_perception_prelude=(step > 0),
            )
            if confidence:
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
        print("\n--- GraphEQA answer ---\n" + discord_text.strip() + "\n---\n")
        if confidence:
            try:
                self.robot.say("The answer to " + question + " is " + answer)
            except Exception:
                pass
        return discord_text, relevant_images


# Alias for compatibility with EQA executor
RobotAgentGraphEQA = GraphEQAController
