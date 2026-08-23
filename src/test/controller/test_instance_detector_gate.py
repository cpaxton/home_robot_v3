# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Object-detector load policy for manipulation_only + instance-graph stacks."""

from __future__ import annotations


def _skip_object_detector(*, manipulation_only: bool, eqa: bool, use_instance_memory: bool) -> bool:
    """Mirror of the gate in DynamemController.create_obstacle_map."""
    return not use_instance_memory and (manipulation_only or eqa)


def test_manipulation_only_skips_detector_without_instance_memory():
    assert _skip_object_detector(manipulation_only=True, eqa=False, use_instance_memory=False) is True


def test_instance_graph_loads_detector_under_manipulation_only():
    """Habitat HM-EQA and other nav-only stacks still need YoloE when instance graph is on."""
    assert _skip_object_detector(manipulation_only=True, eqa=False, use_instance_memory=True) is False


def test_voxel_eqa_without_instances_skips_detector():
    assert _skip_object_detector(manipulation_only=False, eqa=True, use_instance_memory=False) is True


def test_full_graph_eqa_stack_loads_detector():
    assert _skip_object_detector(manipulation_only=False, eqa=False, use_instance_memory=True) is False
