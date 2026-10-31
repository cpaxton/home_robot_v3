# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Experimental schema descriptions; no change to valid answers or actions."""

from copy import deepcopy


def clarify_planner_schema(schema):
    """Restate the reference policy next to its response fields, without forcing it.

    This only annotates a copied schema for a frozen-input experiment. In
    particular, empty plans remain legal and the model still decides to stop.
    """
    result = deepcopy(schema)
    properties = result.get("properties", {})
    if not {"steps", "answer", "scene_graph_description"} <= properties.keys():
        raise ValueError("Expected the reference planner response schema")
    properties["steps"]["description"] = (
        "The next useful observation-gathering action you choose when more evidence is needed. "
        "An empty list means you choose to stop gathering evidence. Use only the supplied action IDs."
    )
    properties["scene_graph_description"]["description"] = (
        "Briefly summarize task-relevant graph evidence. Labels are learned predictions; "
        "an unlisted object is unknown, not proven absent. Do not enumerate the graph or invent IDs."
    )
    if "image_description" in properties:
        properties["image_description"]["description"] = (
            "Briefly describe task-relevant visible evidence. A target missing from these views "
            "does not establish absence elsewhere in the environment."
        )
    reference = properties["answer"].get("$ref", "")
    if not reference.startswith("#/$defs/"):
        raise ValueError("Expected a local answer definition")
    answer = result["$defs"][reference.removeprefix("#/$defs/")]["properties"]
    answer["explanation_ans"]["description"] = "Brief evidence for your current answer, including what remains unknown."
    answer["explanation_conf"]["description"] = "Briefly explain whether the observed evidence suffices to answer."
    answer["is_confident"]["description"] = (
        "Your decision that observed evidence suffices and further observation would not help. "
        "Missing detections alone do not establish scene-wide absence."
    )
    return result
