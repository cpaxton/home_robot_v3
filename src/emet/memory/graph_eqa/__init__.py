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
# GraphEQA memory: a re-implementation of graph-based EQA memory inspired by
# GraphEQA (Saxena et al., https://arxiv.org/abs/2412.14480). No code copied
# from the closed-source graph_eqa repository.

"""Graph memory public API.

Import product types from this package, not from mixin / facade modules::

    from emet.memory.graph_eqa import (
        GraphEQAMemory,
        AgenticEQAExecutor,
        NavHypothesis,
        run_agentic_eqa,
    )

``graph_memory``, ``agentic_eqa``, and ``agentic_*.py`` / ``graph_*.py`` are
implementation. Callers outside this package should not import them. Submodule
paths stay importable so ``mock.patch`` and in-package code keep working.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emet.memory.graph_eqa.agentic_config import question_has_mcq_options, question_is_locate
    from emet.memory.graph_eqa.agentic_eqa import (
        AgenticEQAExecutor,
        AgenticEQAResult,
        AgenticState,
        agentic_verify_enabled,
        build_agentic_eqa_executor,
        run_agentic_eqa,
        run_agentic_eqa_result,
    )
    from emet.memory.graph_eqa.attempt_ledger import AttemptRecord
    from emet.memory.graph_eqa.graph_memory import (
        GraphEQAMemory,
        GraphNode,
        GraphObservation,
        NavHypothesis,
        RelationBelief,
        VerifyResult,
        label_matches_relevant_object,
        labels_are_semantic_graph_hypothesis,
    )
    from emet.memory.graph_eqa.human_answer import (
        HumanEQAResult,
        format_eqa_tool_response,
        format_human_eqa_answer,
    )
    from emet.memory.graph_eqa.pretty_print import format_graph_edges_only, format_scene_graph_pretty
    from emet.memory.graph_eqa.sensor_graph_builder import SensorGraphBuilder

# name -> (relative module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    # Product
    "AgenticEQAExecutor": (".agentic_eqa", "AgenticEQAExecutor"),
    "AgenticEQAResult": (".agentic_eqa", "AgenticEQAResult"),
    "AgenticState": (".agentic_eqa", "AgenticState"),
    "AttemptRecord": (".attempt_ledger", "AttemptRecord"),
    "GraphEQAMemory": (".graph_memory", "GraphEQAMemory"),
    "GraphNode": (".graph_memory", "GraphNode"),
    "GraphObservation": (".graph_memory", "GraphObservation"),
    "HumanEQAResult": (".human_answer", "HumanEQAResult"),
    "NavHypothesis": (".graph_memory", "NavHypothesis"),
    "RelationBelief": (".graph_memory", "RelationBelief"),
    "SensorGraphBuilder": (".sensor_graph_builder", "SensorGraphBuilder"),
    "VerifyResult": (".graph_memory", "VerifyResult"),
    "agentic_verify_enabled": (".agentic_eqa", "agentic_verify_enabled"),
    "build_agentic_eqa_executor": (".agentic_eqa", "build_agentic_eqa_executor"),
    "format_eqa_tool_response": (".human_answer", "format_eqa_tool_response"),
    "format_graph_edges_only": (".pretty_print", "format_graph_edges_only"),
    "format_human_eqa_answer": (".human_answer", "format_human_eqa_answer"),
    "format_scene_graph_pretty": (".pretty_print", "format_scene_graph_pretty"),
    "label_matches_relevant_object": (".graph_memory", "label_matches_relevant_object"),
    "labels_are_semantic_graph_hypothesis": (".graph_memory", "labels_are_semantic_graph_hypothesis"),
    "question_has_mcq_options": (".agentic_config", "question_has_mcq_options"),
    "question_is_locate": (".agentic_config", "question_is_locate"),
    "run_agentic_eqa": (".agentic_eqa", "run_agentic_eqa"),
    "run_agentic_eqa_result": (".agentic_eqa", "run_agentic_eqa_result"),
    # Eval / ingest helpers already imported from this package (compat)
    "DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M": (".instance_observations", "DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M"),
    "build_ground_truth_graph_from_session": (".sim_ground_truth_graph", "build_ground_truth_graph_from_session"),
    "compare_graph_to_placements_report": (".mujoco_align", "compare_graph_to_placements_report"),
    "count_ground_truth_nodes": (".sim_ground_truth_graph", "count_ground_truth_nodes"),
    "deduplicate_placements": (".sim_ground_truth_graph", "deduplicate_placements"),
    "frame_instances_to_labels_xyz": (".instance_observations", "frame_instances_to_labels_xyz"),
    "ground_truth_alignment_report": (".sim_ground_truth_graph", "ground_truth_alignment_report"),
    "label_for_detection_category": (".instance_observations", "label_for_detection_category"),
    "labels_from_extract_response": (".sensor_graph_builder", "labels_from_extract_response"),
    "nearest_gt_for_node": (".mujoco_align", "nearest_gt_for_node"),
    "parse_comma_separated_labels": (".sensor_graph_builder", "parse_comma_separated_labels"),
    "parse_graph_object_json": (".sensor_graph_builder", "parse_graph_object_json"),
    "populate_graph_memory_from_placements": (".sim_ground_truth_graph", "populate_graph_memory_from_placements"),
    "read_sim_object_placements": (".sim_ground_truth_graph", "read_sim_object_placements"),
    "score_nodes_vs_gt": (".mujoco_align", "score_nodes_vs_gt"),
    "short_labels_from_voxel_descriptions": (".sensor_graph_builder", "short_labels_from_voxel_descriptions"),
    "upsert_graph_memory_from_placements": (".sim_ground_truth_graph", "upsert_graph_memory_from_placements"),
    "world_xyz_median_from_depth": (".sensor_graph_builder", "world_xyz_median_from_depth"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    # Do not cache: ``mock.patch`` on the implementation module must still win.
    return getattr(import_module(mod_name, __name__), attr)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
