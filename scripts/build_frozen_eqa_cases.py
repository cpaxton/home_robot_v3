#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Export fixed sensor snapshots without reading scorer labels or choosing views by answer."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from emet.eval.grapheqa_sensor import uniform_image_evidence


def callable_frontier_projection(context):
    """Migrate older recorded projections; only remove isolated, non-callable nodes.

    This matches the corrected native export. Never discard connected evidence,
    change object geometry, or fabricate a missing relation in a cached graph.
    """
    result = copy.deepcopy(context)
    allowed = set(result["actions"]["frontiers"])
    graph = result["scene_graph"]
    removed = {node["id"] for node in graph["nodes"] if node.get("name") == "frontier" and node["id"] not in allowed}
    if any(edge["source"] in removed or edge["target"] in removed for edge in graph.get("links", [])):
        raise ValueError("Replay native mapping instead of dropping a connected non-callable frontier")
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in removed]
    if {node["id"] for node in graph["nodes"] if node.get("name") == "frontier"} != allowed:
        raise ValueError("Recorded frontier registry and projection disagree")
    return result, sorted(removed)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    spec = yaml.safe_load(args.spec.read_text())
    questions = {case["id"]: case for case in json.loads(args.questions.read_text())}
    args.output.mkdir(parents=True, exist_ok=False)
    cases, provenance = [], []
    for item in spec["cases"]:
        source = args.runs / item["source"]
        frames_path = source / "frames.jsonl"
        frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
        frames = [frame for frame in frames if frame["sequence"] <= item["capture"]]
        if not frames or frames[-1]["sequence"] != item["capture"]:
            raise ValueError(f"Missing declared capture: {item['id']}")
        records = []
        for frame in frames:
            if not frame["odometry"].get("ok"):
                raise ValueError("Frozen pose labels require accepted sensor odometry")
            records.append(
                {
                    "path": str(source / f"frame-{frame['sequence']:05d}.png"),
                    "label": f"sequence={frame['sequence']} timestamp_ns={frame['timestamp_ns']} "
                    f"estimated_camera_to_local_world={json.dumps(frame['odometry']['camera_pose'])}",
                }
            )
        images = uniform_image_evidence(records, spec["images"])
        context_path = source / item["context"]
        context = json.loads(context_path.read_text())
        if f"timestamp_ns={frames[-1]['timestamp_ns']}" not in context["agent_state"]:
            raise ValueError("Captured graph context does not match the declared assessment timestamp")
        context, removed = callable_frontier_projection(context)
        destination = args.output / item["id"]
        destination.mkdir()
        hashes = []
        for index, image in enumerate(images):
            original = Path(image["path"])
            target = destination / f"image-{index}.png"
            shutil.copyfile(original, target)
            image["path"] = str(target.relative_to(args.output))
            hashes.append(sha256(target))
        question = questions[item["question_source_id"]]
        cases.append(
            {
                "id": item["id"],
                "question": question["question"],
                "choices": question["choices"],
                "images": images,
                "context": context,
            }
        )
        provenance.append(
            {
                **item,
                "source_frames_sha256": sha256(frames_path),
                "source_context_sha256": sha256(context_path),
                "image_sha256": hashes,
                "removed_isolated_frontiers": removed,
            }
        )
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate frozen case ID")
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "fixed_development_snapshots_not_representative_task_accuracy",
                "spec_sha256": sha256(args.spec),
                "question_fixture_sha256": sha256(args.questions),
                "driver_sha256": sha256(Path(__file__)),
                "selection": "current_then_uniform_history",
                "scorer_labels_read": False,
                "cases": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(cases)} frozen snapshots to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True, help="Existing label-free question/choice fixture")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
