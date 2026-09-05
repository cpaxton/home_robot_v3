# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Open-vocab localize_text, SigLIP alignment, and MLLM visual grounding."""

from __future__ import annotations

import logging
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

DEBUG_SUBDIR = "debug"


class DynamemVoxelLocalizeMixin:
    """Open-vocab localize_text, SigLIP alignment, and MLLM visual grounding."""

    def find_alignment_over_model(self, queries: str):
        if getattr(self, "encoder", None) is None:
            return None
        clip_text_tokens = self.encoder.encode_text(queries).cpu()
        points, features, weights, _ = self.semantic_memory.get_pointcloud()
        if points is None:
            return None
        features = F.normalize(features, p=2, dim=-1).cpu()
        point_alignments = clip_text_tokens.float() @ features.float().T

        # print(point_alignments.shape)
        return point_alignments

    def find_alignment_for_text(self, text: str):
        points, features, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text)
        if alignments is None or points is None:
            return None
        alignments = alignments.cpu()
        return points[alignments.argmax(dim=-1)].detach().cpu()

    def find_obs_id_for_text(self, text: str):
        alignments = self.find_alignment_over_model(text)
        if alignments is None:
            return None
        obs_counts = self.semantic_memory._obs_counts
        return obs_counts[alignments.cpu().argmax(dim=-1)].detach().cpu()

    def verify_point(
        self,
        text: str,
        point: torch.Tensor | np.ndarray,
        distance_threshold: float = 0.1,
        similarity_threshold: float = 0.21,
    ):
        """
        Running visual grounding is quite time consuming.
        Thus, sometimes if the point has very high cosine similarity with text query, we might opt not to run visual grounding again.
        This function evaluates the cosine similarity.
        """
        if isinstance(point, np.ndarray):
            point = torch.from_numpy(point)
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        if points is None:
            return False
        distances = torch.linalg.norm(point - points.detach().cpu(), dim=-1)
        if torch.min(distances) > distance_threshold:
            print("Points are so far from other points!")
            return False
        alignments = self.find_alignment_over_model(text)
        if alignments is None:
            return False
        alignments = alignments.detach().cpu()[0]
        near = distances <= distance_threshold
        if torch.count_nonzero(near) == 0:
            return False
        if torch.max(alignments[near]) < similarity_threshold:
            print("Points close to the point are not similar to the text!")
        return torch.max(alignments[near]) >= similarity_threshold

    def localize_text(self, text, debug=True, return_debug=False):
        if self.mllm:
            return self.localize_with_mllm(text, debug=debug, return_debug=return_debug)
        else:
            return self.localize_with_feature_similarity(text, debug=debug, return_debug=return_debug)

    def find_all_images(
        self,
        text: str,
        min_similarity_threshold: float | None = None,
        min_point_num: int = 100,
        max_img_num: int | None = 3,
    ):
        """
        Select all images with high pixel similarity with text (by identifying whether points in this image are relevant objects)

        Args:
            min_similarity_threshold: Make sure every point with similarity greater than this value would be considered as the relevant objects
            min_point_num: Make sure we select at least these many points as relevant images.
            max_img_num: The maximum number of images we want to identify as relevant objects.
        """
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text)
        if points is None or alignments is None:
            return (
                torch.tensor([], dtype=torch.long),
                torch.zeros(0, 3),
                torch.tensor([]),
            )
        points = points.cpu()
        alignments = alignments.cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts.cpu()

        num_points = alignments.numel()
        if num_points == 0:
            return (
                torch.tensor([], dtype=obs_counts.dtype, device=obs_counts.device),
                torch.zeros(0, points.size(1), device=points.device),
                torch.tensor([], dtype=alignments.dtype, device=alignments.device),
            )
        idx = min(min_point_num, num_points)
        turning_point = (
            min(min_similarity_threshold, alignments[torch.argsort(alignments)[-idx]].item())
            if min_similarity_threshold is not None
            else alignments[torch.argsort(alignments)[-idx]].item()
        )
        mask = alignments >= turning_point
        obs_counts = obs_counts[mask]
        alignments = alignments[mask]
        points = points[mask]

        unique_obs_counts, inverse_indices = torch.unique(obs_counts, return_inverse=True)

        points_with_max_alignment = torch.zeros((len(unique_obs_counts), points.size(1)))
        max_alignments = torch.zeros(len(unique_obs_counts))

        for i in range(len(unique_obs_counts)):
            # Get indices of elements belonging to the current cluster
            indices_in_cluster = (inverse_indices == i).nonzero(as_tuple=True)[0]
            if len(indices_in_cluster) <= 2:
                continue

            # Extract the alignments and points for the current cluster
            cluster_alignments = alignments[indices_in_cluster].squeeze()
            cluster_points = points[indices_in_cluster]

            # Find the point with the highest alignment in the cluster
            max_alignment_idx_in_cluster = cluster_alignments.argmax()
            point_with_max_alignment = cluster_points[max_alignment_idx_in_cluster]

            # Store the result
            points_with_max_alignment[i] = point_with_max_alignment
            max_alignments[i] = cluster_alignments.max()

        # Only use clusters we actually filled (skip zero-alignment clusters from skipped small clusters)
        valid = max_alignments > 0
        if not valid.any():
            return (
                torch.tensor([], dtype=obs_counts.dtype, device=obs_counts.device),
                torch.zeros(0, points.size(1), device=points.device),
                torch.tensor([], dtype=alignments.dtype, device=alignments.device),
            )
        valid_obs = unique_obs_counts[valid]
        valid_points = points_with_max_alignment[valid]
        valid_alignments = max_alignments[valid]

        if max_img_num is not None:
            top_k = min(max_img_num, len(valid_alignments))
        else:
            top_k = len(valid_alignments)
        top_alignments, top_indices = torch.topk(valid_alignments, k=top_k, dim=0, largest=True, sorted=True)
        top_points = valid_points[top_indices]
        top_obs_counts = valid_obs[top_indices]

        sorted_obs_counts, sorted_indices = torch.sort(top_obs_counts, descending=False)
        sorted_points = top_points[sorted_indices]
        top_alignments = top_alignments[sorted_indices]

        return sorted_obs_counts, sorted_points, top_alignments

    def llm_locator(self, image_ids: torch.Tensor | np.ndarray | list, text: str):
        """
        Prompting the mLLM to select the images containing objects of interest.

        Input:
            image_ids: a series of images you want to send to mLLM
            text: text query

        Return
        """
        user_messages = []
        for obs_id in image_ids:
            obs_id = int(obs_id) - 1
            rgb = np.copy(self.observations[obs_id].rgb.numpy())
            depth = self.observations[obs_id].depth
            rgb[depth > 2.5] = [0, 0, 0]
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            user_messages.append(image)
        user_messages.append("The object you need to find is " + text)

        response = self.gpt_client(user_messages)
        return self.parse_localization_response(response)

    def parse_localization_response(self, response: str):
        """
        Parse the output of GPT4o to extract the selected image's id
        """
        try:
            # Use regex to locate the 'Images:' section, allowing for varying whitespace and line breaks
            images_section_match = re.search(r"Images:\s*([\s\S]+)", response, re.IGNORECASE)
            if not images_section_match:
                raise ValueError("The 'Images:' section is missing.")

            # Extract the content after 'Images:'
            images_content = images_section_match.group(1).strip()

            # Check if the content is 'None' (case-insensitive)
            if images_content.lower() == "none":
                return None

            # Use regex to find all numbers, regardless of separators like commas, periods, or spaces
            numbers = re.findall(r"\d+", images_content)

            if not numbers:
                raise ValueError("No numbers found in the 'Images:' section.")

            # Convert all found numbers to integers
            numbers = [int(num) for num in numbers]

            # Return all numbers as a list if multiple numbers are found
            if len(numbers) > 0:
                return numbers[-1]
            else:
                return None

        except Exception as e:
            # Handle any exceptions and optionally log the error message
            print(f"Error: {e}")
            return None

    def localize_with_mllm(self, text: str, debug=True, return_debug=False):
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text)
        if alignments is None or points is None:
            msg = "Map has no points yet; run exploration first."
            if not debug:
                return None
            if not return_debug:
                return None, msg
            return None, msg, None, None
        alignments = alignments.cpu()
        point = points[alignments.argmax(dim=-1)].detach().cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts
        image_id = obs_counts[alignments.argmax(dim=-1)].detach().cpu()
        debug_text = ""
        target_point = None

        image_ids, points, alignments = self.find_all_images(
            # text, min_similarity_threshold=0.12, max_img_num=3
            text,
            max_img_num=3,
        )
        n_candidates = len(image_ids) if hasattr(image_ids, "__len__") else image_ids.numel()
        if n_candidates == 0:
            target_id = None
        else:
            target_id = self.llm_locator(image_ids, text)
            # LLM returns 1-based index; validate before converting to 0-based
            if target_id is not None and (target_id < 1 or target_id > n_candidates):
                logger.warning(
                    "llm_locator returned out-of-range image id %s (have %d candidates); ignoring.",
                    target_id,
                    n_candidates,
                )
                target_id = None

        if target_id is None:
            # Single candidate: treat as identified (LLM may have said "None" due to format)
            if n_candidates == 1:
                target_id = 1
                target_id -= 1  # 0-based
                target_point = points[target_id]
                image_id = image_ids[target_id]
                point = points[target_id]
                debug_text += "#### - An image is identified (single candidate)\n"
            else:
                debug_text += "#### - Cannot verify whether this instance is the target. **😞** \n"
                image_id = None
                point = None
        else:
            target_id -= 1  # 1-based -> 0-based index into candidates
            target_point = points[target_id]
            image_id = image_ids[target_id]
            point = points[target_id]
            debug_text += "#### - An image is identified \n"

        if image_id is not None:
            rgb = self.observations[image_id - 1].rgb
            pose = self.observations[image_id - 1].camera_pose
            depth = self.observations[image_id - 1].depth
            K = self.observations[image_id - 1].camera_K

            res = self.detection_model.compute_obj_coord(text, rgb, depth, K, pose)
            if res is not None:
                target_point = res
            else:
                target_point = point

        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, image_id, point

    def localize_with_feature_similarity(
        self, text, similarity_threshold: float = 0.14, debug=True, return_debug=False
    ):
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text)
        if alignments is None or points is None:
            self._last_localize_stats = {"query": text, "max_cosine": None, "yoloe_hit": False}
            msg = "Map has no points yet; run exploration first."
            if not debug:
                return None
            if not return_debug:
                return None, msg
            return None, msg, None, None
        alignments = alignments.cpu()
        point = points[alignments.argmax(dim=-1)].detach().cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts
        obs_id = obs_counts[alignments.argmax(dim=-1)].detach().cpu()
        debug_text = ""
        target_point = None
        max_cosine = float(alignments.max().item()) if alignments.numel() else None

        if obs_id <= 0 or obs_id > len(self.observations):
            res = None
        else:
            rgb = self.observations[obs_id - 1].rgb
            pose = self.observations[obs_id - 1].camera_pose
            depth = self.observations[obs_id - 1].depth
            K = self.observations[obs_id - 1].camera_K

            rgb_t = rgb if isinstance(rgb, torch.Tensor) else torch.as_tensor(rgb)
            rgb_np = rgb_t.detach().cpu().numpy()
            bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            debug_dir = os.path.join(self.log, DEBUG_SUBDIR)
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir, "rgb" + text + "_" + str(obs_id.item() - 1) + ".png"),
                bgr,
            )

            det = self.detection_model
            res = det.compute_obj_coord(text, rgb_t, depth, K, pose) if det is not None else None

        self._last_localize_stats = {
            "query": text,
            "max_cosine": max_cosine,
            "yoloe_hit": res is not None,
            "source_obs_id": int(obs_id),
        }
        if res is not None:
            target_point = res
            debug_text += "#### - Object is detected in observations . **😃** Directly navigate to it.\n"
        else:
            cosine_similarity_check = max_cosine is not None and max_cosine > 0.21
            if cosine_similarity_check:
                target_point = point

                debug_text += "#### - The point has high cosine similarity. **😃** Directly navigate to it.\n"
            else:
                debug_text += "#### - Cannot verify whether this instance is the target. **😞** \n"
        print("--------------------------------")
        print(debug_text)
        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, obs_id, point
