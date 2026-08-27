# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Classic voxel query_answer, image ranking, frontiers, and active-view selection."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from torch import Tensor

from emet.llms.vllm_factory import dynamem_vllm_call
from emet.utils.morphology import get_edges
from emet.utils.voxel import scatter3d

logger = logging.getLogger(__name__)

DEBUG_SUBDIR = "debug"


class DynamemVoxelEQAMixin:
    """Classic voxel query_answer, image ranking, frontiers, and active-view selection."""

    def extract_relevant_objects(self, question: str):
        """
        Parsed the question and extract few keywords for DynaMem voxel map to select relevant images
        """
        if self._question != question:
            self._question = question
            # The cached question is not the same as the question provided
            prompt = """
                Assume there is an agent doing Question Answering in an environment.
                When it receives a question, you need to tell the agent few objects (preferably 1-3) it needs to pay special attention to.
                Example:
                    Where is the pen?
                    pen

                    Is there grey cloth on cloth hanger?
                    gery cloth,cloth hanger
            """
            messages = [prompt, self._question]
            self.relevant_objects = dynamem_vllm_call(
                self.image_description_client,
                messages,
                system_prompt="",
                max_new_tokens=64,
            ).split(",")
            print("relevant objects to look at", self.relevant_objects)
            self.history_outputs = []

    def log_text(self, commands):
        """
        Log the text input and image input into some files for debugging and visualization
        """
        desc_dir = os.path.join(self.log, DEBUG_SUBDIR, str(len(self.image_descriptions)))
        if not os.path.exists(desc_dir):
            os.makedirs(desc_dir)
            input_texts = ""
            for command in commands:
                input_texts += command + "\n"
            with open(os.path.join(desc_dir, "input.txt"), "w") as file:
                file.write(input_texts)

    def parse_answer(self, answer_outputs: str):
        """
        Parse the output of LLM text into reasoning, answer, confidence, action, confidence_reasoning
        """

        # Log LLM output
        desc_dir = os.path.join(self.log, DEBUG_SUBDIR, str(len(self.image_descriptions)))
        os.makedirs(desc_dir, exist_ok=True)
        with open(os.path.join(desc_dir, "output.txt"), "w") as file:
            file.write(answer_outputs)

        # Answer outputs in the format "Caption: Reasoning: Answer: Confidence: Action: Confidence_reasoning:"
        def extract_between(text, start, end):
            try:
                return text.split(start, 1)[1].split(end, 1)[0].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        def extract_after(text, start):
            try:
                return text.split(start, 1)[1].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        reasoning = extract_between(answer_outputs, "reasoning:", "answer:")
        answer = extract_between(answer_outputs, "answer:", "confidence:")
        confidence_text = extract_between(answer_outputs, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "")
        action = extract_between(answer_outputs, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(answer_outputs, "confidence_reasoning:")

        return reasoning, answer, confidence, action, confidence_reasoning

    def query_answer(self, question: str, xyt, planner):
        """
        Util function to prompt mLLM to provide answer output, and process the raw answer output into robot's next step.
        """

        # Extract keywords from the question
        self.extract_relevant_objects(question)

        # messages = [{"type": "text", "text": "Question: " + question}]
        commands: list[Any] = ["Question: " + question]
        # messages.append({"type": "text", "text": "HISTORY: "})
        commands.append("HISTORY: ")
        for i, history_output in enumerate(self.history_outputs):
            # messages.append({"type": "text", "text": "Iteration_" + str(i) + ":" + history_output})
            commands.append("Iteration_" + str(i) + ":" + history_output)
        # messages.append({"role": "user", "content": [{"type": "input_text", "text": question}]})

        # Select the task relevant images with DynaMem
        img_idx = 0
        all_obs_ids = set()

        for relevant_object in self.relevant_objects:
            # Limit the total number of images to 6
            image_ids, _, _ = self.find_all_images(
                relevant_object,
                min_similarity_threshold=0.12,
                max_img_num=6 // len(self.relevant_objects),
                min_point_num=40,
            )
            for obs_id in image_ids:
                obs_id = int(obs_id) - 1
                all_obs_ids.add(obs_id)

        all_obs_ids = list(all_obs_ids)  # type: ignore

        # Prepare the visual clues (image descriptions)
        selected_images, action_prompt = self.get_image_descriptions_str(xyt, planner, all_obs_ids)
        commands.append(action_prompt)
        self.log_text(commands)
        relevant_images = []

        for obs_id in all_obs_ids:
            rgb = np.copy(self.observations[obs_id].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Log the input images
            desc_dir = os.path.join(self.log, DEBUG_SUBDIR, str(len(self.image_descriptions)))
            os.makedirs(desc_dir, exist_ok=True)
            image.save(os.path.join(desc_dir, str(img_idx) + ".jpg"))
            img_idx += 1

            commands.append(image)
            relevant_images.append(image)

        # Extract answers
        from emet.llms.graph_eqa_vlm import _eqa_system_prompt

        raw_answer_outputs = dynamem_vllm_call(
            self.eqa_client,
            commands,
            system_prompt=_eqa_system_prompt(self.parameters),
            max_new_tokens=self._eqa_max_tokens,
        )
        self._last_eqa_raw = raw_answer_outputs
        answer_outputs = raw_answer_outputs.replace("*", "").replace("/", "").replace("#", "").lower()

        print(commands)
        print(answer_outputs)

        (
            reasoning,
            answer,
            confidence,
            action,
            confidence_reasoning,
        ) = self.parse_answer(answer_outputs)

        # If the robot is not confident, it should plan exploration
        if not confidence:
            action = selected_images[int(action) - 1]
            rgb = np.copy(self.observations[action - 1].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Cache conversations between the robot and the mLLM for the next iteration of question answering planning
            self.history_outputs.append(
                "Answer:"
                + answer
                + "\nReasoning:"
                + reasoning
                + "\nConfidence:"
                + str(confidence)
                + "\nAction:"
                + "Navigate to Image with objects "
                + str(self.image_descriptions[action - 1][0])
                + " with grid coord "
                + str(self.image_descriptions[action - 1][1])
                + "\nConfidence reasoning:"
                + confidence_reasoning
            )
        else:
            action = None

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            self.get_target_point_from_image_id(action, xyt, planner) if action is not None else None,
            relevant_images,
        )

    def get_active_image_descriptions(self):
        """Return descriptions still tied to at least one voxel (obs ids are 1-indexed)."""
        if self.voxel_pcd._points is None:
            return None

        obs_ids = self.voxel_pcd._obs_counts
        xyz, _, _, _ = self.voxel_pcd.get_pointcloud()
        xyz = ((xyz / self.grid_resolution) + self.grid_origin + 0.5).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        max_height = int(self.obs_max_height / self.grid_resolution)
        grid_size = self.grid_size + [max_height]
        obs_ids = obs_ids[:, None]

        history_ids = scatter3d(xyz, obs_ids, grid_size, "max")
        history = torch.max(history_ids, dim=-1).values
        history = torch.from_numpy(maximum_filter(history.float().numpy(), size=5))
        history[0:35, :] = history.max().item()
        history[-35:, :] = history.max().item()
        history[:, 0:35] = history.max().item()
        history[:, -35:] = history.max().item()

        selected_images = torch.unique(history).int()
        return (
            history,
            selected_images,
            [self.image_descriptions[selected_image.item() - 1] for selected_image in selected_images],
        )

    def get_image_descriptions_str(self, xyt, planner, obs_ids):
        """
        Select visual clues of all active images (images still associated with some voxel points in the voxel map)
        """
        (
            _,
            selected_images,
            image_descriptions,
        ) = self.get_active_image_descriptions()
        frontier_ids = list(self.get_frontier_ids(xyt, planner))
        options = ""
        if len(image_descriptions) > 0:
            for i, (cluster, grid_coord) in enumerate(image_descriptions):
                index = selected_images[i]
                cluster_string = ""
                for ob in cluster:
                    cluster_string += ob + ", "
                cluster_string = cluster_string[:-2] + ";"
                # Indicate the grid coord this image describes to avoid redundant exploration.
                cluster_string += " This image is taken at grid coords " + str(grid_coord)
                # If we have already send the raw image observation to LLM.
                if index in obs_ids:
                    cluster_string += (
                        " This observation description is associated with Image " + str(obs_ids.index(index) + 1) + ";"
                    )
                # If this image corresponds to an unexplored frontier
                if index in frontier_ids:
                    cluster_string += " This observation description corresponds to unexplored space;"
                options += f"{i + 1}. {cluster_string}\n"
        return selected_images, "IMAGE_DESCRIPTIONS: " + options

    def get_target_point_from_image_id(self, image_id: int, xyt, planner):
        """
        When the robot is not confident with the answer, mLLM will output an image id indicating a rough direction for the robot to take the next step.
        This function selects the target point's xy coordinate based on the image id provided.
        """

        # history output by get_active_descriptions output a history id map considering history id of the floor point
        # history_soft output by get_2d_map output a history id map excluding history id of the floor point
        # Therefore, history is generally used to select active image observations while history_soft is generally used to determine unexplored frontier
        (
            history,
            _,
            _,
        ) = self.get_active_image_descriptions()
        obstacles, explored = self.get_2d_map()
        outside_frontier = self.get_outside_frontier(xyt, planner)
        unexplored_frontier = outside_frontier & ~explored
        # Navigation priority: unexplored frontier > obstalces > others
        if torch.sum((history == image_id) & unexplored_frontier) > 0:
            print("unexplored frontier")
            image_coord = (
                ((history == image_id) & unexplored_frontier).nonzero(as_tuple=False).median(dim=0).values.int()
            )
        elif torch.sum((history == image_id) & obstacles) > 0:
            print("obstacles")
            image_coord = ((history == image_id) & obstacles).nonzero(as_tuple=False).median(dim=0).values.int()
        else:
            print("others")
            image_coord = (history == image_id).nonzero(as_tuple=False).median(dim=0).values.int()
        xy = self.grid_coords_to_xy(image_coord)
        return torch.Tensor([xy[0], xy[1], 1])

    def get_frontier_ids(self, xyt, planner):
        """
        This function figures out which of images correspond to an unexplored frontier.
        """
        (
            history,
            _,
            _,
        ) = self.get_active_image_descriptions()
        outside_frontier = self.get_outside_frontier(xyt, planner)
        _, explored = self.get_2d_map()
        unexplored_frontier = outside_frontier & ~explored
        history = np.ma.masked_array(history, ~unexplored_frontier)
        return np.unique(history)

    def list_objects_in_an_image(self, image: torch.Tensor | Image.Image | np.ndarray, max_tries: int = 3):
        """
        Extract visual clues (a list of featured objects) from the image observation and add the clues to a list
        """
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, Tensor):
                _image = image.cpu().numpy()
            else:
                _image = image
            pil_image = Image.fromarray(_image)

        prompt = "List representative objects in the image (excluding floor and wall) Limit your answer in 10 words. E.G.: a table,chairs,doors"
        messages = [pil_image, prompt]

        # self.obs_count inherited from voxel_dynamem
        objects = []
        for attempt in range(max_tries):
            try:
                object_names = dynamem_vllm_call(
                    self.image_description_client,
                    messages,
                    system_prompt="",
                    max_new_tokens=32,
                )
                objects = object_names.split(",")[:5]
            except Exception as e:
                objects = []
                logger.debug(
                    "list_objects_in_an_image VL call failed (attempt %s/%s): %s",
                    attempt + 1,
                    max_tries,
                    e,
                )
                continue
            else:
                break

        obs_ids = self.voxel_pcd._obs_counts
        xyz, _, _, _ = self.voxel_pcd.get_pointcloud()
        grid_coord = [0, 0]
        try:
            if (
                xyz is not None
                and getattr(xyz, "numel", lambda: len(xyz))() > 0
                and obs_ids is not None
                and getattr(obs_ids, "numel", lambda: int(np.size(obs_ids)))() > 0
            ):
                # Empty clouds happen on first frames / failed depth — do not call .max() on empty.
                max_id = obs_ids.max()
                sel = xyz[obs_ids == max_id]
                if sel.numel() > 0:
                    xy = torch.mean(sel, dim=0)[:2].int().cpu().numpy()
                    grid_coord = list(self.xy_to_grid_coords(xy))
                    for i in range(len(grid_coord)):
                        grid_coord[i] = int(grid_coord[i])
        except Exception as e:
            logger.warning(f"list_objects_in_an_image: grid coord from cloud failed ({e})")
            grid_coord = [0, 0]

        if len(objects) == 0:
            self.image_descriptions.append((["object"], grid_coord))
        else:
            self.image_descriptions.append((objects, grid_coord))

        print(objects)

    def get_reachable_map(self, xyt, planner, *, local_fallback_cells: int = 8):
        """Boolean grid of planner-reachable free cells from ``xyt``.

        When flood-fill fails (start cell buried under dilation / pose noise), fall
        back to a **local** free-explored disk around the robot — never the entire
        explored mask (that puts frontier centroids mid-room).
        """
        obstacles, explored = self.get_2d_map()
        if len(xyt) == 3:
            xyt = xyt[:2]
        start_pt = planner.to_pt(xyt)
        reachable_points = planner.get_reachable_points(start_pt)
        reachable_map = torch.zeros_like(obstacles, dtype=torch.bool)
        if reachable_points:
            reachable_xs, reachable_ys = zip(*reachable_points, strict=False)
            reachable_xs = torch.tensor(reachable_xs, device=obstacles.device, dtype=torch.long)
            reachable_ys = torch.tensor(reachable_ys, device=obstacles.device, dtype=torch.long)
            reachable_map[reachable_xs, reachable_ys] = True
            return reachable_map

        # Local disk of explored free space around the start (grid coords).
        free = (~obstacles & explored).to(torch.bool)
        if isinstance(start_pt, (list, tuple)):
            si, sj = int(start_pt[0]), int(start_pt[1])
        else:
            si, sj = int(start_pt[0]), int(start_pt[1])
        h, w = free.shape
        r = max(1, int(local_fallback_cells))
        i0, i1 = max(0, si - r), min(h, si + r + 1)
        j0, j1 = max(0, sj - r), min(w, sj + r + 1)
        yy, xx = torch.meshgrid(
            torch.arange(i0, i1, device=free.device),
            torch.arange(j0, j1, device=free.device),
            indexing="ij",
        )
        disk = (yy - si).to(torch.float32).pow(2) + (xx - sj).to(torch.float32).pow(2) <= float(r * r)
        reachable_map[i0:i1, j0:j1] = free[i0:i1, j0:j1] & disk
        return reachable_map

    def get_outside_frontier(self, xyt, planner):
        """
        This function selects the edges of currently reachable space.
        """
        reachable_map = self.get_reachable_map(xyt, planner)
        edges = get_edges(reachable_map)
        return edges & ~reachable_map
