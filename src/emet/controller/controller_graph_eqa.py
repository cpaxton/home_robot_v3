# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# GraphEQA agent: uses graph-based memory for EQA while reusing DynaMem-style
# voxel map for navigation and exploration. Re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480).

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from emet.controller.controller_dynamem import DynamemController
from emet.core.robot import AbstractRobotClient
from emet.memory.graph_eqa import GraphEQAMemory
from emet.core.parameters import Parameters


class GraphEQAController(DynamemController):
    """
    Robot controller that uses GraphEQA (graph-based) memory for EQA instead of
    the voxel map. Keeps the same voxel map for navigation and exploration;
    feeds the graph memory on each update and uses it in run_eqa.
    """

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, dict],
        semantic_sensor=None,
        save_rerun: bool = False,
        use_instance_memory: bool = False,
        realtime_updates: bool = False,
        re: int = 3,
        manip_port: int = 5557,
        log: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        mllm: bool = False,
        manipulation_only: bool = False,
        cpu_only: bool = False,
        eqa: bool = True,  # force True for GraphEQA
        **kwargs,
    ):
        super().__init__(
            robot=robot,
            parameters=parameters,
            semantic_sensor=semantic_sensor,
            save_rerun=save_rerun,
            use_instance_memory=use_instance_memory,
            realtime_updates=realtime_updates,
            re=re,
            manip_port=manip_port,
            log=log,
            server_ip=server_ip,
            mllm=mllm,
            manipulation_only=manipulation_only,
            cpu_only=cpu_only,
            eqa=True,  # voxel map still does list_objects for labels
            **kwargs,
        )
        self.graph_memory = GraphEQAMemory(
            parameters=parameters,
            log_dir="graph_eqa_log",
        )

    def update(self) -> None:
        """Step collector and feed the graph memory with the new observation."""
        super().update()
        obs = self.robot.get_observation()
        rgb = obs.rgb
        if obs.camera_pose is None:
            return
        camera_xyz = np.array(obs.camera_pose[:3, 3], dtype=float)
        labels = ["object"]
        if getattr(self.voxel_map, "image_descriptions", None) and len(self.voxel_map.image_descriptions) > 0:
            labels = self.voxel_map.image_descriptions[-1][0]
        self.graph_memory.add_observation(rgb, camera_xyz, labels)

    def run_eqa_one_iter(
        self, question: str, max_movement_step: int = 5
    ) -> Tuple[str, str, List[Image.Image], bool]:
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
                target_point = self.space.sample_frontier(
                    self.planner, self.robot.get_base_pose(), text=None
                )
            relevant_images = []

        confidence_text = (
            "I am confident with the answer" if confidence
            else "I am NOT confident with the answer"
        )
        reasoning_output = (
            "\n#### Reasoning for the answer: " + reasoning
            if confidence
            else "\n#### Reasoning for the confidence: " + confidence_reasoning
        )
        answer_output = (
            "#### **Question:** " + question
            + "\n#### **Answer:** " + answer
            + "\n#### **Confidence:** " + confidence_text
            + reasoning_output
        )
        self.rerun_visualizer.log_text("robot_monologue", answer_output)
        if relevant_images and hasattr(self, "_patch_images"):
            self.rerun_visualizer.log_custom_2d_image(
                "/observation_similar_to_text", self._patch_images(relevant_images)
            )
        elif relevant_images:
            self.rerun_visualizer.log_custom_2d_image(
                "/observation_similar_to_text", relevant_images
            )

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
            target_grid = self.voxel_map.xy_to_grid_coords(
                (float(target_point[0]), float(target_point[1]))
            )
            if (
                obstacles.shape[0] > int(target_grid[0])
                and obstacles.shape[1] > int(target_grid[1])
                and not obstacles[int(target_grid[0]), int(target_grid[1])]
            ):
                target_theta = self.space.sample_navigation(
                    start_pose, self.planner, target_point
                )[-1]
            else:
                target_theta = None
            for _ in range(max_movement_step):
                start_pose = self.robot.get_base_pose()
                self.update()
                if self.navigate_to_target_pose(target_point, start_pose, target_theta):
                    break

        return answer, discord_text, relevant_images, confidence

    def run_eqa(
        self, question: str, max_planning_steps: int = 5
    ) -> Tuple[str, List[Image.Image]]:
        """Run EQA until confident or max steps, using graph memory."""
        for _ in range(max_planning_steps):
            answer, discord_text, relevant_images, confidence = self.run_eqa_one_iter(
                question
            )
            if confidence:
                break
        if not relevant_images:
            relevant_images = []
        return discord_text, relevant_images


# Alias for compatibility with EQA executor
RobotAgentGraphEQA = GraphEQAController
