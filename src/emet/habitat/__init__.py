# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Habitat EQA harness helpers (no habitat-sim import in this package)."""

from emet.habitat.config import default_habitat_eqa_data_dir, default_hm3d_scene_dir
from emet.habitat.datasets import HMEQAQuestion, load_hmeqa_questions, load_scene_init_poses
from emet.habitat.metrics import (
    EpisodeMetrics,
    extract_mcq_letter_from_raw_eqa,
    grade_mcq_answer,
    write_episode_jsonl,
)

__all__ = [
    "HMEQAQuestion",
    "EpisodeMetrics",
    "default_habitat_eqa_data_dir",
    "default_hm3d_scene_dir",
    "extract_mcq_letter_from_raw_eqa",
    "grade_mcq_answer",
    "load_hmeqa_questions",
    "load_scene_init_poses",
    "write_episode_jsonl",
]
