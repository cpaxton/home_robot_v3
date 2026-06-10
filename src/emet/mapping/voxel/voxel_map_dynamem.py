# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import torch

from emet.motion import Footprint

from .voxel import SparseVoxelMapProxy
from .voxel_dynamem import SparseVoxelMap
from .voxel_map import SparseVoxelMapNavigationSpace as SparseVoxelMapNavigationSpaceBase


class SparseVoxelMapNavigationSpace(SparseVoxelMapNavigationSpaceBase):
    # Used for making sure we do not divide by zero anywhere
    tolerance: float = 1e-8

    def __init__(
        self,
        voxel_map: SparseVoxelMap | SparseVoxelMapProxy,
        step_size: float = 0.1,
        rotation_step_size: float = 0.5,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
        dilate_frontier_size: int = 12,
        dilate_obstacle_size: int = 2,
        extend_mode: str = "separate",
    ):
        super().__init__(
            voxel_map=voxel_map,
            robot=None,
            step_size=step_size,
            rotation_step_size=rotation_step_size,
            use_orientation=use_orientation,
            orientation_resolution=orientation_resolution,
            dilate_frontier_size=dilate_frontier_size,
            dilate_obstacle_size=dilate_obstacle_size,
            extend_mode=extend_mode,
        )
        self.create_collision_masks(orientation_resolution)
        self.traj = None

    def create_collision_masks(self, orientation_resolution: int):
        """Create a set of orientation masks

        Args:
            orientation_resolution: number of bins to break it into
        """
        self._footprint = Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)
        self._orientation_resolution = 64
        self._oriented_masks = []

        for i in range(orientation_resolution):
            theta = i * 2 * np.pi / orientation_resolution
            mask = self._footprint.get_rotated_mask(self.voxel_map.grid_resolution, angle_radians=theta)
            # Footprint returns numpy; store as tensor for get_oriented_mask / collision checks
            mask_t = torch.from_numpy(np.asarray(mask)).bool() if not hasattr(mask, "cuda") else mask
            self._oriented_masks.append(mask_t)

    def compute_theta(self, cur_x, cur_y, end_x, end_y):
        theta = 0
        if end_x == cur_x and end_y >= cur_y:
            theta = np.pi / 2
        elif end_x == cur_x and end_y < cur_y:
            theta = -np.pi / 2
        else:
            theta = np.arctan((end_y - cur_y) / (end_x - cur_x))
            if end_x < cur_x:
                theta = theta + np.pi
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < -np.pi:
                theta = theta + 2 * np.pi
        return theta

    def sample_target_point(
        self, start: torch.Tensor, point: torch.Tensor, planner, exploration: bool = False
    ) -> np.ndarray | None:
        """Sample a position near the mask and return.

        Args:
            look_at_any_point(bool): robot should look at the closest point on target mask instead of average pt
        """

        obstacles, explored = self.voxel_map.get_2d_map()

        # Extract edges from our explored mask
        start_pt = planner.to_pt(start)
        reachable_points = planner.get_reachable_points(start_pt)
        if len(reachable_points) == 0:
            print("No target point find, maybe no point is reachable")
            return None
        reachable_xs, reachable_ys = zip(*reachable_points, strict=False)
        # # type: ignore comments used to bypass mypy check
        reachable_xs = torch.tensor(reachable_xs)  # type: ignore
        reachable_ys = torch.tensor(reachable_ys)  # type: ignore
        reachable = torch.empty(obstacles.shape, dtype=torch.bool).fill_(False)
        reachable[reachable_xs, reachable_ys] = True

        obstacles, explored = self.voxel_map.get_2d_map()
        reachable = reachable & ~obstacles

        target_x, target_y = planner.to_pt(point)

        xs, ys = torch.where(reachable)
        if len(xs) < 1:
            print("No target point find, maybe no point is reachable")
            return None
        dist_to_goal = torch.linalg.norm(
            (torch.stack([xs, ys], dim=-1) - torch.tensor([target_x, target_y], dtype=torch.float32)).float(),
            dim=-1,
        )
        selected_targets = torch.stack([xs, ys], dim=-1)[dist_to_goal.topk(k=len(xs), largest=False).indices]

        px = float(point[0].item() if hasattr(point[0], "item") else point[0])
        py = float(point[1].item() if hasattr(point[1], "item") else point[1])
        # Tight scenes: strict standoff can leave no valid goal; relax standoff for object navigation only.
        standoffs = [0.35, 0.24, 0.14, 0.08] if not exploration else [0.35]
        obs_h, obs_w = int(obstacles.shape[0]), int(obstacles.shape[1])

        for min_standoff in standoffs:
            for selected_target in selected_targets:
                sx_i, sy_i = int(selected_target[0]), int(selected_target[1])
                selected_x, selected_y = planner.to_xy([sx_i, sy_i])
                theta = self.compute_theta(selected_x, selected_y, px, py)

                if not self.is_valid(np.array([selected_x, selected_y, theta])):
                    continue

                dist_xy = float(np.hypot(selected_x - px, selected_y - py))
                if dist_xy <= min_standoff:
                    continue

                ok = True
                if dist_xy <= 0.5:
                    step_i = int(np.sign(target_x - sx_i)) if target_x != sx_i else 0
                    step_j = int(np.sign(target_y - sy_i)) if target_y != sy_i else 0
                    ni, nj = sx_i + step_i, sy_i + step_j
                    if 0 <= ni < obs_h and 0 <= nj < obs_w and bool(obstacles[ni, nj]):
                        ok = False

                if ok:
                    return np.array([selected_x, selected_y, theta])

        return None

    def sample_exploration(self, xyt, planner, text=None, debug=False):
        """
        Sample an exploration target
        """
        obstacles, explored, history_soft = self.voxel_map.get_2d_map(return_history_id=True, kernel=5)
        outside_frontier = self.voxel_map.get_outside_frontier(xyt, planner)

        time_heuristics = self._time_heuristic(history_soft, outside_frontier, debug=debug)

        keyword_weight = self._frontier_keyword_weight()
        keywords: list[str] = []
        if text and keyword_weight > 0:
            from emet.memory.graph_eqa.frontier_nodes import exploration_keywords_from_text

            keywords = exploration_keywords_from_text(text)

        # TODO: Find good alignment heuristic, we have found few candidates but none of them has satisfactory performance

        ######################################
        # Candidate 1: Borrow the idea from https://arxiv.org/abs/2310.10103
        # for i, (cluster, _) in enumerate(image_descriptions):
        #   cluser_string = ""
        #   for ob in cluster:
        #       cluser_string += ob + ", "
        #   options += f"{i+1}. {cluser_string[:-2]}\n"

        # if positive:
        #     messages = f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I search next if I try to {task}?"
        #     choices = self.positive_score_client.sample(messages, n_samples=num_samples)
        # else:
        #     messages = f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I avoid spending time searching if I try to {task}?"
        #     choices = self.negative_score_client.sample(messages, n_samples=num_samples)

        # answers = []
        # reasonings = []
        # for choice in choices:
        #     complete_response = choice.lower()
        #     reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
        #     # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
        #     if len(complete_response.split("answer:")) > 1:
        #          answer = complete_response.split("answer:")[1].split("\n")[0]
        #          # Separate the answers by commas
        #          answers.append([int(x) for x in answer.split(",")])
        #      else:
        #          answers.append([])
        #      reasonings.append(reasoning)

        # # Flatten answers
        # flattened_answers = [item for sublist in answers for item in sublist]
        # filtered_flattened_answers = [
        #     x for x in flattened_answers if x >= 1 and x <= len(image_descriptions)
        # ]
        # # Aggregate into counts and normalize to probabilities
        # answer_counts = {
        #     x: filtered_flattened_answers.count(x) / len(answers)
        #     for x in set(filtered_flattened_answers)
        # }
        ######################################
        # Candidate 2: Naively use semantic feature alignment
        # def get_2d_alignment_heuristics(self, text: str, debug: bool = False):
        # if self.semantic_memory._points is None:
        #     return None
        # # Convert metric measurements to discrete
        # # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        # xyz, _, _, _ = self.semantic_memory.get_pointcloud()
        # xyz = xyz.detach().cpu()
        # if xyz is None:
        #     xyz = torch.zeros((0, 3))

        # device = xyz.device
        # xyz = ((xyz / self.grid_resolution) + self.grid_origin).long()
        # xyz[xyz[:, -1] < 0, -1] = 0

        # # Crop to robot height
        # min_height = int(self.obs_min_height / self.grid_resolution)
        # max_height = int(self.obs_max_height / self.grid_resolution)
        # grid_size = self.grid_size + [max_height]

        # # Mask out obstacles only above a certain height
        # obs_mask = xyz[:, -1] < max_height
        # xyz = xyz[obs_mask, :]
        # alignments = self.find_alignment_over_model(text)[0].detach().cpu()
        # alignments = alignments[obs_mask][:, None]

        # alignment_heuristics = scatter3d(xyz, alignments, grid_size, "max")
        # alignment_heuristics = torch.max(alignment_heuristics, dim=-1).values
        # alignment_heuristics = torch.from_numpy(
        #     maximum_filter(alignment_heuristics.numpy(), size=5)
        # )

        from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

        unexplored = _as_bool_numpy(outside_frontier) & ~_as_bool_numpy(explored)

        alignments_heuristics = None
        total_heuristics = np.asarray(time_heuristics, dtype=np.float64)
        if keywords:
            from emet.memory.graph_eqa.frontier_nodes import keyword_score_map

            kw_scores = keyword_score_map(
                unexplored,
                getattr(self.voxel_map, "image_descriptions", None),
                keywords,
            )
            if float(kw_scores.max()) > 0:
                kw_norm = kw_scores / float(kw_scores.max())
                alignments_heuristics = kw_norm
                total_heuristics = total_heuristics + keyword_weight * kw_norm

        # SigLIP activation sampling: bias frontiers toward unexplored cells adjacent to
        # observations whose SigLIP features align with the query. Complements caption keyword
        # overlap (which is partial and misses mislabeled objects) and works even when keyword
        # extraction is empty. No-ops without a live SigLIP encoder (clean GraphEQA baseline).
        sig_norm = self._siglip_activation_map(text, unexplored)
        if sig_norm is not None:
            siglip_weight = self._frontier_siglip_weight()
            if siglip_weight > 0:
                alignments_heuristics = (
                    sig_norm if alignments_heuristics is None else np.maximum(alignments_heuristics, sig_norm)
                )
                total_heuristics = total_heuristics + siglip_weight * sig_norm

        rounded_heuristics = np.ceil(total_heuristics * 200) / 200
        max_heuristic = rounded_heuristics.max()
        indices = np.column_stack(np.where(rounded_heuristics == max_heuristic))
        closest_index = np.argmin(np.linalg.norm(indices - np.asarray(planner.to_pt(xyt)), axis=-1))
        index = indices[closest_index]
        if debug:
            from matplotlib import pyplot as plt

            plt.subplot(221)
            plt.imshow(obstacles.int() * 5 + outside_frontier.int() * 10)
            plt.subplot(222)
            plt.imshow(explored.int() * 5)
            plt.subplot(223)
            plt.imshow(total_heuristics)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.subplot(224)
            plt.imshow(history_soft)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.show()
        return index, time_heuristics, alignments_heuristics, total_heuristics

    def _siglip_activation_map(self, text, frontier_mask, radius_cells: int = 10, min_similarity: float = 0.20):
        """2D SigLIP-activation heuristic over the frontier (``None`` if unavailable).

        Scatters per-observed-point cosine similarity to *text* onto the 2D grid, spreads it to
        the ``radius_cells`` neighborhood (so unexplored frontier cells next to a strong match are
        boosted), keeps only the most-aligned region, and normalizes to ``[0, 1]``. Caption-
        independent, so it heads toward a basket that was captioned "decorative plant".

        Gated by ``min_similarity``: SigLIP cosine has a high floor (most points ~0.1), so without
        an absolute threshold a max-normalize turns floor noise into a spurious bias that perturbs
        the trajectory. The heuristic stays inert (returns ``None``) unless something genuinely
        matches near the frontier, so it only redirects exploration on a real visual hint.
        """
        if not text:
            return None
        vm = self.voxel_map
        if getattr(vm, "encoder", None) is None:
            return None
        if not hasattr(vm, "find_alignment_over_model"):
            return None
        try:
            alignments = vm.find_alignment_over_model(text)
        except Exception:
            return None
        if alignments is None:
            return None
        points, _, _, _ = vm.semantic_memory.get_pointcloud()
        if points is None:
            return None
        a = alignments.detach().cpu().squeeze().reshape(-1).numpy()
        pts = points.detach().cpu().numpy()
        if pts.shape[0] == 0 or pts.shape[0] != a.shape[0]:
            return None

        from scipy.ndimage import maximum_filter

        from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

        res = float(vm.grid_resolution)
        origin = np.asarray(vm.grid_origin, dtype=float).reshape(-1)
        mask = _as_bool_numpy(frontier_mask)
        h, w = mask.shape
        gi = np.floor(pts[:, 0] / res + origin[0] + 0.5).astype(np.int64)
        gj = np.floor(pts[:, 1] / res + origin[1] + 0.5).astype(np.int64)
        valid = (gi >= 0) & (gi < h) & (gj >= 0) & (gj < w)
        gi, gj, av = gi[valid], gj[valid], a[valid].astype(np.float32)
        if av.size == 0:
            return None
        score = np.zeros((h, w), dtype=np.float32)
        np.maximum.at(score, (gi, gj), av)
        score = maximum_filter(score, size=int(radius_cells))
        score = score * mask
        peak = float(score.max())
        # Absolute gate: only bias the trajectory when a genuine match sits near the frontier;
        # below the floor this is noise and should not perturb keyword/time exploration.
        if peak < float(min_similarity):
            return None
        # Focus on the most-aligned frontier region (the high similarity floor would otherwise
        # light up the whole boundary).
        score[score < 0.6 * peak] = 0.0
        return score / peak

    def _frontier_siglip_weight(self) -> float:
        vm = self.voxel_map
        params = getattr(vm, "parameters", None)
        if params is not None and hasattr(params, "get"):
            blk = params.get("graph_eqa_frontier_nodes")
            if not isinstance(blk, dict):
                eqa = params.get("graph_eqa")
                if isinstance(eqa, dict):
                    blk = eqa.get("frontier_nodes")
            if isinstance(blk, dict) and blk.get("siglip_score_weight") is not None:
                return max(0.0, float(blk["siglip_score_weight"]))
        # Default: match the keyword weight so SigLIP and caption keywords contribute equally.
        return self._frontier_keyword_weight()

    def _frontier_keyword_weight(self) -> float:
        vm = self.voxel_map
        params = getattr(vm, "parameters", None)
        if params is None:
            return 1.0
        blk = None
        if hasattr(params, "get"):
            blk = params.get("graph_eqa_frontier_nodes")
            if not isinstance(blk, dict):
                eqa = params.get("graph_eqa")
                if isinstance(eqa, dict):
                    blk = eqa.get("frontier_nodes")
        if isinstance(blk, dict) and blk.get("keyword_score_weight") is not None:
            return max(0.0, float(blk["keyword_score_weight"]))
        return 1.0

    def _time_heuristic(self, history_soft, outside_frontier, time_smooth=0.1, time_threshold=10, debug=False):
        frontier = np.asarray(outside_frontier, dtype=bool)
        if frontier.any():
            history_soft = np.ma.masked_array(history_soft, ~frontier)
        else:
            # No frontier cells (e.g. empty map); avoid masking the entire grid.
            history_soft = np.asarray(history_soft, dtype=float)
        time_heuristics = history_soft.max() - history_soft
        time_heuristics[history_soft < 1] = float("inf")
        time_heuristics = 1 / (1 + np.exp(-time_smooth * (time_heuristics - time_threshold)))
        index = np.unravel_index(np.argmax(time_heuristics), history_soft.shape)
        # return index
        # debug = True
        if debug:
            # plt.clf()
            plt.title("time")
            plt.imshow(history_soft)
            plt.scatter(index[1], index[0], s=15, c="r")
            plt.show()
        return time_heuristics

    def to_pt(self, xy: tuple[float, float]) -> tuple[int, int]:
        """Converts a point from continuous, world xy coordinates to grid coordinates.

        Args:
            xy: The point in continuous xy coordinates.

        Returns:
            The point in discrete grid coordinates.
        """
        # # type: ignore to bypass mypy checking
        xy = np.array([xy[0], xy[1]], dtype=float)  # type: ignore
        pt = self.voxel_map.xy_to_grid_coords(xy)  # type: ignore
        if pt is None:
            # Base pose can be outside the allocated grid (world vs map frame); snap for planning.
            pt_t = self.voxel_map.grid.xy_to_grid_coords_clamped(torch.tensor(xy, dtype=torch.float32))
            pt = pt_t.detach().cpu().numpy()
        return int(pt[0]), int(pt[1])

    def to_xy(self, pt: tuple[int, int]) -> tuple[float, float]:
        """Converts a point from grid coordinates to continuous, world xy coordinates.

        Args:
            pt: The point in grid coordinates.

        Returns:
            The point in continuous xy coordinates.
        """
        # # type: ignore to bypass mypy checking
        pt = np.array([pt[0], pt[1]])  # type: ignore
        xy = self.voxel_map.grid_coords_to_xy(pt)  # type: ignore
        return float(xy[0]), float(xy[1])

    def sample_navigation(self, start, planner, point, mode="navigation"):
        plt.clf()
        if point is None:
            start_pt = self.to_pt(start)
            return None
        goal = self.sample_target_point(start, point, planner, exploration=mode != "navigation")
        print("point:", point, "goal:", goal)
        obstacles, explored = self.voxel_map.get_2d_map()
        plt.imshow(obstacles)
        start_pt = self.to_pt(start)
        plt.scatter(start_pt[1], start_pt[0], s=15, c="b")
        point_pt = self.to_pt(point)
        plt.scatter(point_pt[1], point_pt[0], s=15, c="r")
        if goal is not None:
            goal_pt = self.to_pt(goal)
            plt.scatter(goal_pt[1], goal_pt[0], s=10, c="g")
        # plt.show()
        return goal

    def sample_frontier(self, planner, start_pose=None, text=None):
        if start_pose is None:
            start_pose = [0, 0, 0]
        (
            index,
            time_heuristics,
            alignments_heuristics,
            total_heuristics,
        ) = self.sample_exploration(
            start_pose,
            planner,
            text=text,
            debug=False,
        )

        obstacles, explored = self.voxel_map.get_2d_map()
        return self.voxel_map.grid_coords_to_xyt(torch.tensor([index[0], index[1]]))
