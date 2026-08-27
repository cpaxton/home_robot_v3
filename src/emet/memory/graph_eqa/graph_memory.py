# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Graph-based EQA memory: re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480). Object-centric scene graph + task-relevant
# images for embodied question answering. No code copied from closed-source repos.

"""Graph-based EQA memory facade.

``GraphEQAMemory`` owns a :class:`~emet.memory.graph_eqa.store.GraphStore` and a
``WorldEvidenceStore``. Mutate / query / nav methods live in ``ingest`` / ``eqa`` /
``spatial`` and are bound onto this class. Callers keep importing from this module.

Package layout lives in ``docs/graph_memory.md``.
"""

from __future__ import annotations

from dataclasses import replace as replace

from emet.memory.graph_eqa import graph_init
from emet.memory.graph_eqa._bind import bind_module_methods
from emet.memory.graph_eqa.count_mcq import CountTarget
from emet.memory.graph_eqa.eqa import graph_answer, graph_eqa_obs, graph_hypotheses, graph_nav, graph_prompt
from emet.memory.graph_eqa.eqa.graph_eqa_siglip import SIGLIP_CONFIRM_THRESHOLD, SIGLIP_PRESENT_THRESHOLD
from emet.memory.graph_eqa.geom import _inside_bounds, _near, _node_is_room, _on, _on_floor
from emet.memory.graph_eqa.ingest import graph_mutate
from emet.memory.graph_eqa.labels import (
    consolidate_relevant_keywords,
    countable_primary_label_matches,
    distinctive_choice_tokens,
    finder_label_texts,
    format_graph_node_candidates,
    heuristic_relevant_objects,
    heuristic_relevant_phrases,
    label_matches_relevant_object,
    labels_are_semantic_graph_hypothesis,
    location_mcq_landmark_phrases,
    node_display_name,
    parse_eqa_action,
    question_stem_for_keywords,
)
from emet.memory.graph_eqa.spatial import graph_rooms
from emet.memory.graph_eqa.store import GraphStore, attach_store_accessors
from emet.memory.graph_eqa.types import (
    GT_BODY_DESC_PREFIX,
    GraphNavigationSample,
    GraphNode,
    GraphObservation,
    NavHypothesis,
    RelationBelief,
    VerifyResult,
    is_ground_truth_node,
)

__all__ = [
    "GraphEQAMemory",
    "GraphStore",
    "CountTarget",
    "GT_BODY_DESC_PREFIX",
    "GraphNavigationSample",
    "GraphNode",
    "GraphObservation",
    "NavHypothesis",
    "RelationBelief",
    "SIGLIP_CONFIRM_THRESHOLD",
    "SIGLIP_PRESENT_THRESHOLD",
    "VerifyResult",
    "_inside_bounds",
    "_near",
    "_node_is_room",
    "_on",
    "_on_floor",
    "consolidate_relevant_keywords",
    "countable_primary_label_matches",
    "distinctive_choice_tokens",
    "finder_label_texts",
    "format_graph_node_candidates",
    "heuristic_relevant_objects",
    "heuristic_relevant_phrases",
    "is_ground_truth_node",
    "label_matches_relevant_object",
    "labels_are_semantic_graph_hypothesis",
    "location_mcq_landmark_phrases",
    "node_display_name",
    "parse_eqa_action",
    "question_stem_for_keywords",
    "replace",
]


class GraphEQAMemory:
    """
    Graph-based semantic memory for Embodied Question Answering (EQA).

    Maintains an object-centric scene graph (nodes = objects/regions with labels and
    3D positions; edges = spatial relations). Uses the same EQA query contract as
    the DynaMem voxel map: query_answer(question, xyt, planner) returns
    (reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images).

    Optional **Dynagraph** behavior (parameters ``dynagraph_merge_xy_m``,
    ``dynagraph_staleness_horizon``): spatial merge of nodes with the same primary
    label within XY distance, and ``maintain(current_step)`` to drop stale nodes.
    """

    def __init__(self, *args, **kwargs):
        self.store = GraphStore()
        graph_init.init_memory(self, *args, **kwargs)

    def add_observation(self, *args, **kwargs):
        return graph_mutate.add_observation(self, *args, **kwargs)

    def query_answer(self, *args, **kwargs):
        return graph_answer.query_answer(self, *args, **kwargs)

    def hypothesize_nav_targets(self, *args, **kwargs):
        return graph_hypotheses.hypothesize_nav_targets(self, *args, **kwargs)


attach_store_accessors(GraphEQAMemory)

for _mod in (
    graph_init,
    graph_mutate,
    graph_rooms,
    graph_eqa_obs,
    graph_hypotheses,
    graph_prompt,
    graph_nav,
    graph_answer,
):
    bind_module_methods(
        GraphEQAMemory,
        _mod,
        skip=frozenset({"add_observation", "query_answer", "hypothesize_nav_targets"}),
    )
