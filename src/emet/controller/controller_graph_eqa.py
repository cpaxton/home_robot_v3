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


import numpy as np
from PIL import Image

from emet.controller.controller_dynamem import DynamemController
from emet.core.parameters import Parameters
from emet.core.robot import AbstractRobotClient
from emet.memory.graph_eqa import GraphEQAMemory, SensorGraphBuilder
from emet.memory.graph_eqa.instance_observations import (
    DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M,
)


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
        )
        self.graph_memory = GraphEQAMemory(
            parameters=parameters,
            log_dir="graph_eqa_log",
            defer_llm_clients=True,
        )
        if graph_memory_input_path:
            from emet.memory.backend import get_memory_backend

            backend = get_memory_backend(
                "graph_eqa",
                graph_memory=self.graph_memory,
                voxel_map=getattr(self, "voxel_map", None),
            )
            backend.load(graph_memory_input_path)

        self.use_instance_graph = use_instance_graph
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

        self.use_sensor_perception = use_sensor_perception
        dev = self.device if self.device in ("cuda", "mps") else "cuda"
        self.sensor_builder = SensorGraphBuilder(
            perception_client=perception_client,
            use_voxel_fallback=True,
            device=dev,
            cpu_only=self.cpu_only,
            parameters=parameters,
        )

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

    def update(self) -> None:
        """Step collector and feed the graph memory with the new observation."""
        super().update()
        obs = self.robot.get_observation()
        from emet.memory.graph_eqa.dynamem_graph_hooks import update_graph_memory_from_dynamem_observation

        update_graph_memory_from_dynamem_observation(
            graph_memory=self.graph_memory,
            robot=self.robot,
            voxel_map=self.voxel_map,
            detection_model=self.detection_model,
            sensor_builder=self.sensor_builder,
            use_instance_graph=self.use_instance_graph,
            use_sensor_perception=self.use_sensor_perception,
            dedup_skips=self._graph_dedup_skips,
            obs=obs,
        )

    def run_eqa_one_iter(self, question: str, max_movement_step: int = 5) -> tuple[str, str, list[Image.Image], bool]:
        """One EQA iteration using graph memory instead of voxel map."""
        answer_output = None
        if not self._realtime_updates:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

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
                self.robot.get_base_pose(),
                self.planner,
            )
        except Exception as e:
            reasoning = f"Error: {e}"
            answer = "Unknown"
            confidence = False
            confidence_reasoning = str(e)
            target_point = None
            if hasattr(self, "space") and hasattr(self.space, "sample_frontier"):
                target_point = self.space.sample_frontier(self.planner, self.robot.get_base_pose(), text=None)
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
        self.rerun_visualizer.log_text("robot_monologue", answer_output)
        if relevant_images and hasattr(self, "_patch_images"):
            self.rerun_visualizer.log_custom_2d_image(
                "/observation_similar_to_text", self._patch_images(relevant_images)
            )
        elif relevant_images:
            self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", relevant_images)

        discord_text = (
            answer + ". I believe this answer is correct because " + reasoning
            if confidence
            else "I am not confident to answer the question because " + confidence_reasoning
        )
        discord_text += "\nI also provide relevant images here."

        if confidence:
            return answer, discord_text, relevant_images, confidence

        if target_point is not None and hasattr(self, "navigate_to_target_pose"):
            start_pose = self.robot.get_base_pose()
            obstacles, _ = self.voxel_map.get_2d_map()
            target_grid = self.voxel_map.xy_to_grid_coords((float(target_point[0]), float(target_point[1])))
            if (
                obstacles.shape[0] > int(target_grid[0])
                and obstacles.shape[1] > int(target_grid[1])
                and not obstacles[int(target_grid[0]), int(target_grid[1])]
            ):
                target_theta = self.space.sample_navigation(start_pose, self.planner, target_point)[-1]
            else:
                target_theta = None
            for _ in range(max_movement_step):
                start_pose = self.robot.get_base_pose()
                self.update()
                if self.navigate_to_target_pose(target_point, start_pose, target_theta):
                    break

        return answer, discord_text, relevant_images, confidence

    def run_eqa(self, question: str, max_planning_steps: int = 5) -> tuple[str, list[Image.Image]]:
        """Run EQA until confident or max steps, using graph memory."""
        answer = ""
        confidence = False
        discord_text = ""
        relevant_images: list[Image.Image] = []
        for _ in range(max_planning_steps):
            answer, discord_text, relevant_images, confidence = self.run_eqa_one_iter(question)
            if confidence:
                break
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
