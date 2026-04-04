# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


import numpy as np
import torch

from emet.core.parameters import Parameters
from emet.mapping.instance import Instance
from emet.utils.memory import get_path_to_debug


class SceneGraph:
    """Compute a very simple scene graph. Use it to extract relationships between instances."""

    def __init__(self, parameters: Parameters, instances: list[Instance]):
        self.parameters = parameters
        self.instances = instances
        self.relationships: list[tuple[int, int, str]] = []
        self.update(instances)

    def update(self, instances):
        """Extract pairwise symbolic spatial relationship between instances using heurisitcs"""
        self.relationships: list[tuple[int, int, str]] = []
        self.instances = instances
        for idx_a, ins_a in enumerate(instances):
            for idx_b, ins_b in enumerate(instances):
                if idx_a == idx_b:
                    continue
                # Use list indices for position lookups; relationships store global_id for API
                if self.near(idx_a, idx_b) and (ins_b.global_id, ins_a.global_id, "near") not in self.relationships:
                    self.relationships.append((ins_a.global_id, ins_b.global_id, "near"))

                if (self.on(idx_a, idx_b)) and ((ins_b.global_id, ins_a.global_id, "on") not in self.relationships):
                    self.relationships.append((ins_a.global_id, ins_b.global_id, "on"))
            if self.on_floor(idx_a):
                self.relationships.append((ins_a.global_id, "floor", "on"))

    def get_matching_relations(
        self,
        id0: int | str | None,
        id1: int | str | None,
        relation: str | None,
    ) -> list[tuple[int, int, str]]:
        """Get all relationships between two instances.

        Args:
            id0: The first instance id
            id1: The second instance id
            relation: The relationship between the two instances

        Returns:
            List of relationships in the form (idx_a, idx_b, relation)
        """
        if isinstance(id1, Instance):
            id1 = id1.global_id
        if isinstance(id0, Instance):
            id0 = id0.global_id
        return [
            rel
            for rel in self.relationships
            if (id0 is None or rel[0] == id0)
            and (id1 is None or rel[1] == id1)
            and (rel[2] == relation or relation is None)
        ]

    def _index_for_global_id(self, gid: int | str) -> int | None:
        """Return list index for an instance with the given global_id, or None."""
        if gid == "floor":
            return None
        for i, inst in enumerate(self.instances):
            if getattr(inst, "global_id", None) == gid:
                return i
        return None

    def get_ins_center_pos(self, idx: int):
        """Get the center of an instance by list index (not global_id)."""
        return torch.mean(self.instances[idx].point_cloud, axis=0)

    def get_instance_image(self, idx: int) -> np.ndarray:
        """Get a viewable image from tensorized instances"""
        return (
            (self.instances[idx].get_best_view().cropped_image * self.instances[idx].get_best_view().mask / 255.0)
            .detach()
            .cpu()
            .numpy()
        )

    def get_relationships(self, debug: bool = False) -> list[tuple[int, int, str]]:
        """Return the relationships between instances.

        Args:
            debug: If True, show the relationships in a matplotlib window

        Returns:
            List of relationships in the form (idx_a, idx_b, relation)
        """
        # show symbolic relationships
        if debug:
            for id_a, id_b, rel in self.relationships:
                print(id_a, id_b, rel)
                i_a = self._index_for_global_id(id_a)
                i_b = self._index_for_global_id(id_b) if id_b != "floor" else None
                if i_a is None:
                    continue
                img_a = self.get_instance_image(i_a)
                img_b = np.zeros_like(img_a) if i_b is None else self.get_instance_image(i_b)

                import matplotlib

                matplotlib.use("TkAgg")
                import matplotlib.pyplot as plt

                plt.subplot(1, 2, 1)
                plt.imshow(img_a)
                plt.title("Instance A is " + rel)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(img_b)
                plt.title("Instance B")
                plt.axis("off")
                # plt.show()
                plt.savefig(get_path_to_debug(f"scene_graph_{id_a}_{id_b}_{rel}.png"))

        # Return the detected relationships in list form
        return self.relationships

    def near(self, ins_a, ins_b):
        dist = torch.pairwise_distance(self.get_ins_center_pos(ins_a), self.get_ins_center_pos(ins_b)).item()
        if dist < self.parameters["scene_graph"]["max_near_distance"]:
            return True
        return False

    def on(self, ins_a, ins_b):
        """On is defined as near and above, within some tolerance"""
        if self.near(ins_a, ins_b):
            z_dist = self.get_ins_center_pos(ins_a)[2] - self.get_ins_center_pos(ins_b)[2]
            if (
                z_dist < self.parameters["scene_graph"]["max_on_height"]
                and z_dist > self.parameters["scene_graph"]["min_on_height"]
            ):
                return True
        return False

    def on_floor(self, ins_a):
        """Check if an instance is on the floor"""
        pos = self.get_ins_center_pos(ins_a)
        if pos[2] < self.parameters["scene_graph"]["max_on_height"] and pos[2] > 0:
            return True
        return False
