#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Small sensor-only GraphEQA interaction qualification with a supplied model.

This qualifies the external adaptation, not a matched active emet comparison.
Gold labels are used only after all episodes, and incomplete rows are retained.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from probe_grapheqa_sensor_only import LABEL_PATH, NATIVE_IMAGE, UPSTREAM_REV, habitat_sensor_config, sha256
from run_shared_room_pairs import assert_frozen_source, git

from emet.eval.grapheqa_native_client import NativeSensorGraph
from emet.eval.grapheqa_navigation import HabitatMotionExecutor, ObservedSpaceNavigation, SensorOnlyHabitat
from emet.eval.grapheqa_sensor import learned_label_map
from emet.habitat.config import hm3d_scene_glb_path, questions_csv_path
from emet.habitat.datasets import load_hmeqa_questions, load_scene_init_poses
from emet.perception.detection.yoloe import YoloEPerception
from emet.utils.job_registry import validated_gpu_lock_fd


def run(args):
    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Run through emet jobs --gpu-exclusive")
    root = Path(__file__).resolve().parents[1]
    revision, fork_revision = git(root, "rev-parse", "HEAD"), git(args.fork, "rev-parse", "HEAD")
    assert_frozen_source(root, revision)
    assert_frozen_source(args.fork, fork_revision)
    if not args.question_ids or len(set(args.question_ids)) != len(args.question_ids):
        raise ValueError("Use explicit unique development question IDs")
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise RuntimeError("Keep the shared 20 GiB disk reserve")
    labels_path = args.fork / LABEL_PATH
    expected = git(args.fork, "show", f"{UPSTREAM_REV}:{LABEL_PATH}").encode()
    if labels_path.read_bytes().strip() != expected.strip():
        raise RuntimeError("Fixed upstream perception vocabulary changed")
    for path in (root / "yoloe-v8l-seg.pt", root / "mobileclip_blt.ts"):
        if not path.is_file():
            raise FileNotFoundError(f"Cached learned-perception weight required: {path}")
    sys.path.insert(0, str(args.fork))
    from graph_eqa.utils.active_eval import run_active_episode
    from graph_eqa.utils.model_api import with_model_config
    from omegaconf import OmegaConf

    cfg = with_model_config(OmegaConf.create({"vlm": {"use_image": True, "add_history": True}}), args.model_config)
    OmegaConf.resolve(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    labels = yaml.safe_load(labels_path.read_text())
    entries = [entry for entry in labels["label_names"] if entry["label"] not in labels["invalid_labels"]]
    detector = YoloEPerception(class_list=[entry["name"] for entry in entries], device="cuda", confidence_threshold=0.1)
    ids = [entry["label"] for entry in entries]
    manifest = {
        "mode": "active_external_adaptation_qualification_not_paired_emet_acceptance",
        "source_commit": revision,
        "fork_commit": fork_revision,
        "model_config_sha256": sha256(args.model_config),
        "driver_sha256": sha256(Path(__file__)),
        "questions_sha256": sha256(questions_csv_path()),
        "question_ids": args.question_ids,
        "max_decisions": args.max_decisions,
        "native_image": NATIVE_IMAGE,
        "pose_source": "rgbd_odometry_command_initialized",
        "simulator_pose_input": False,
        "semantic_sensor": False,
        "annotation_enrichment": False,
        "navmesh_routing": False,
        "renderer_collision_filter": "Habitat internal navmesh motor filter; never a planner input",
        "perception": {
            "provider": "yoloe-v8l-seg",
            "confidence": 0.1,
            "checkpoint_sha256": sha256(root / "yoloe-v8l-seg.pt"),
            "text_encoder_sha256": sha256(root / "mobileclip_blt.ts"),
            "vocabulary_sha256": sha256(labels_path),
        },
        "status": "running",
        "planned_episodes": len(args.question_ids),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    questions, spawns = load_hmeqa_questions(), load_scene_init_poses()
    rows = []
    for qid in args.question_ids:
        assert_frozen_source(root, revision)
        assert_frozen_source(args.fork, fork_revision)
        case = args.output / f"q{qid}"
        case.mkdir()
        question = questions[qid]
        sensor, graph = None, None
        row = {"question_id": qid, "complete": False}
        started = time.monotonic()
        try:
            sensor = SensorOnlyHabitat(
                habitat_sensor_config(hm3d_scene_glb_path(question.scene)), spawns[(question.scene, question.floor)]
            )
            nav = ObservedSpaceNavigation()
            graph = NativeSensorGraph(
                fork=args.fork,
                contract=root / "src/emet/eval/grapheqa_sensor.py",
                labels=labels_path,
                inputs=case / "inputs",
                output=case / "native",
                image=NATIVE_IMAGE,
            )
            latest = {}

            def ingest(frame):
                if frame["camera_pose"] is None:
                    raise RuntimeError(f"RGB-D tracking failed: {frame['odometry']}")
                semantic, instances, _ = detector.predict(frame["rgb"], draw_instance_predictions=False)
                packet = {key: frame[key] for key in ("rgb", "depth", "camera_K", "camera_pose")}
                packet.update(
                    timestamp_ns=np.array(frame["timestamp_ns"], dtype=np.int64),
                    learned_labels=learned_label_map(semantic, instances, ids),
                )
                latest["reply"] = graph.step(packet, nav.frontiers(), num_labels=labels["total_semantic_labels"])
                path = case / f"frame-{frame['sequence']:05d}.png"
                Image.fromarray(frame["rgb"]).save(path)
                latest["image"] = path
                with (case / "frames.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({key: frame[key] for key in ("sequence", "timestamp_ns", "collided", "odometry")})
                        + "\n"
                    )

            nav.update(sensor.last)
            ingest(sensor.last)
            # Match the qualified acquisition sequence: 45-degree scan and
            # bounded real floor inspection; all frames enter learned mapping.
            for action in ["turn_left"] * 9 + ["look_down"] * 3:
                frame = sensor.step(action)
                nav.update(frame)
                ingest(frame)
            executor = HabitatMotionExecutor(sensor, nav, ingest)

            def observe():
                frame = sensor.capture()
                nav.update(frame)
                ingest(frame)
                context = latest["reply"]["context"]
                if context is None:
                    raise RuntimeError("Native graph has no qualified current agent state")
                return {"timestamp_ns": frame["timestamp_ns"], "image": latest["image"], "context": context}

            def execute(target, position):
                object_target = target in {item["id"] for item in latest["reply"]["context"]["actions"]["objects"]}
                route, reason = nav.plan(position, object_target=object_target)
                if not route:
                    return {
                        "success": False,
                        "reason": reason,
                        "execution_started": False,
                        "current_pose": nav.pose.tolist(),
                    }
                return {**executor.execute(route), "execution_started": True}

            outcome = run_active_episode(
                cfg.vlm,
                question.question,
                question.choices,
                observe,
                execute,
                case / "controller",
                max_decisions=args.max_decisions,
            )
            row.update(complete=True, **outcome, primitive_count=executor.primitive_count)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for resource in (graph, sensor):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        row.update(complete=False, cleanup_error=f"{type(exc).__name__}: {exc}")
            row["wall_s"] = time.monotonic() - started
            rows.append(row)
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    # Scorer-side only, after every controller request and motion has completed.
    for row in rows:
        row["correct"] = row.get("answer") == questions[row["question_id"]].answer_letter if row["complete"] else None
    (args.output / "scored_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest.update(
        status="complete" if all(row["complete"] for row in rows) else "incomplete_or_failed",
        completed_episodes=sum(row["complete"] for row in rows),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-ids", nargs="+", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, default=4)
    raise SystemExit(run(parser.parse_args()))
