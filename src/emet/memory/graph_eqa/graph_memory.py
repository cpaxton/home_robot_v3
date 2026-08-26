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
# Graph-based EQA memory: re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480). Object-centric scene graph + task-relevant
# images for embodied question answering. No code copied from closed-source repos.

"""Graph-based EQA memory: thin facade over mixin modules.

Implementation lives in ``graph_types.py`` and
``graph_{init,mutate,rooms,eqa_obs,hypotheses,prompt,nav,answer}.py``.
Callers and tests keep importing from this module.
"""

from __future__ import annotations

from dataclasses import replace

from emet.memory.graph_eqa.graph_answer import GraphAnswerMixin
from emet.memory.graph_eqa.graph_answer_trace import GraphAnswerTraceMixin
from emet.memory.graph_eqa.graph_eqa_obs import GraphEqaObsMixin
from emet.memory.graph_eqa.graph_hypotheses import GraphHypothesisMixin
from emet.memory.graph_eqa.graph_init import GraphInitMixin
from emet.memory.graph_eqa.graph_mutate import GraphMutateMixin
from emet.memory.graph_eqa.graph_nav import GraphNavMixin
from emet.memory.graph_eqa.graph_prompt import GraphPromptMixin
from emet.memory.graph_eqa.graph_rooms import GraphRoomsMixin
from emet.memory.graph_eqa.graph_types import (
    _ACTION_LOOK_RE,
    _ACTION_READ_RE,
    _COUNT_LEADING_WORDS,
    _COUNT_PHRASE_ALIASES,
    _COUNT_QUANTITY_WRAPPERS,
    _COUNT_ROOM_PHRASES,
    _COUNT_SCOPE_PREPOSITION_RE,
    _COUNT_TARGET_BOUNDARY_RE,
    _COUNT_WORD_ALIASES,
    _GRAPH_CANDIDATE_COUNT_DISCLAIMER,
    _LANDMARK_GENERIC_TOKENS,
    _OBJECT_LABEL_ALIASES,
    _QUESTION_LANDMARK_BOOST,
    _QUESTION_STOPWORDS,
    _QUESTION_VERB_FILLERS,
    _RECALL_SOURCE_TIER,
    _ROOM_WORDS,
    _WEAK_SIGLIP_FIND_TOKENS,
    GT_BODY_DESC_PREFIX,
    SIGLIP_CONFIRM_THRESHOLD,
    SIGLIP_PRESENT_THRESHOLD,
    CountTarget,
    GraphNavigationSample,
    GraphNode,
    GraphObservation,
    NavHypothesis,
    RelationBelief,
    VerifyResult,
    _collapse_count_nodes_spatially,
    _count_phrase_matches,
    _count_room_scope_tokens,
    _count_target_from_stem,
    _count_tokens,
    _count_word_forms,
    _count_word_matches,
    _inside_bounds,
    _location_mcq_weak_tokens,
    _near,
    _node_is_room,
    _object_match_tokens,
    _on,
    _on_floor,
    _strip_count_wrappers,
    consolidate_relevant_keywords,
    countable_primary_label_matches,
    distinctive_choice_tokens,
    finder_label_texts,
    format_graph_node_candidates,
    heuristic_relevant_objects,
    heuristic_relevant_phrases,
    is_ground_truth_node,
    label_matches_relevant_object,
    labels_are_semantic_graph_hypothesis,
    location_mcq_landmark_phrases,
    node_display_name,
    parse_eqa_action,
    question_stem_for_keywords,
)

__all__ = [
    "GraphEQAMemory",
    "replace",
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
    "_ACTION_LOOK_RE",
    "_ACTION_READ_RE",
    "_COUNT_LEADING_WORDS",
    "_COUNT_PHRASE_ALIASES",
    "_COUNT_QUANTITY_WRAPPERS",
    "_COUNT_ROOM_PHRASES",
    "_COUNT_SCOPE_PREPOSITION_RE",
    "_COUNT_TARGET_BOUNDARY_RE",
    "_COUNT_WORD_ALIASES",
    "_GRAPH_CANDIDATE_COUNT_DISCLAIMER",
    "_LANDMARK_GENERIC_TOKENS",
    "_OBJECT_LABEL_ALIASES",
    "_QUESTION_LANDMARK_BOOST",
    "_QUESTION_STOPWORDS",
    "_QUESTION_VERB_FILLERS",
    "_RECALL_SOURCE_TIER",
    "_ROOM_WORDS",
    "_WEAK_SIGLIP_FIND_TOKENS",
    "_collapse_count_nodes_spatially",
    "_count_phrase_matches",
    "_count_room_scope_tokens",
    "_count_target_from_stem",
    "_count_tokens",
    "_count_word_forms",
    "_count_word_matches",
    "_inside_bounds",
    "_location_mcq_weak_tokens",
    "_near",
    "_node_is_room",
    "_object_match_tokens",
    "_on",
    "_on_floor",
    "_strip_count_wrappers",
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
]


class GraphEQAMemory(
    GraphInitMixin,
    GraphMutateMixin,
    GraphRoomsMixin,
    GraphEqaObsMixin,
    GraphHypothesisMixin,
    GraphPromptMixin,
    GraphNavMixin,
    GraphAnswerTraceMixin,
    GraphAnswerMixin,
):
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
