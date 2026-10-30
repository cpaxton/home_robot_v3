#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Qualify learned RGB-D -> native Hydra, without oracle semantic/navigation inputs.

Run with the Habitat Python under an ``emet jobs`` GPU lock. This deliberately
does not run GraphEQA's answer/planning loop or claim an EQA baseline score.
The native-worker subcommand runs in a CPU-only, network-isolated container.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

# The native container mounts just this script and the lightweight contract;
# it must not import the host's agent, simulator, datasets, or model clients.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/emet/eval"))
from grapheqa_sensor import command_camera_pose, learned_label_map, validate_packet  # noqa: E402

UPSTREAM_REV = "d3fbbf14bdb05631a1633556d7d3056601374cd3"
NATIVE_IMAGE = "sha256:1cab965fa1815e43854248d62bbf910454b12c791584df95111f3b87d0fabde3"
LABEL_PATH = "cfg/label_spaces/hm3d_label_space.yaml"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def native_worker(args):
    import hydra_python as hydra
    from hydra_python._hydra_bindings import PythonConfig
    from scipy.spatial.transform import Rotation

    labels = yaml.safe_load(args.labels.read_text())
    native_configs = [
        hydra.get_config_path() / "habitat" / f"{ns}_config.yaml" for ns in ("frontend", "backend", "reconstruction")
    ]
    configs = PythonConfig()
    for namespace in ("frontend", "backend", "reconstruction"):
        configs.add_file(hydra.get_config_path() / "habitat" / f"{namespace}_config.yaml", config_ns=namespace)
    configs.add_file(args.labels)
    # These are the overrides in the pinned upstream load_configs("habitat").
    configs.add_yaml(
        yaml.safe_dump(
            {
                "frontend": {"type": "FrontendModule", "objects": {"bounding_box_type": "AABB"}},
                "backend": {"type": "BackendModule"},
                "lcd": {"lcd_use_bow_vectors": False},
                "reconstruction": {
                    "type": "ReconstructionModule",
                    "show_stats": False,
                    "pose_graphs": {"make_pose_graph": True},
                },
            }
        )
    )
    paths = sorted(args.inputs.glob("frame-*.npz"))
    cadence = yaml.safe_load(native_configs[2].read_text())["num_poses_per_update"]
    if len(paths) < cadence:
        raise ValueError(f"Need at least {cadence} fresh frames for the native graph-update cadence")
    with np.load(paths[0], allow_pickle=False) as first:
        validate_packet(first, num_labels=labels["total_semantic_labels"])
        height, width = first["depth"].shape
        k = first["camera_K"]
        camera = {
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
            "width": width,
            "height": height,
        }
    hydra.set_glog_level(0, 0)
    config = hydra.PipelineConfig(configs)
    config.enable_reconstruction = True
    config.label_names = {entry["label"]: entry["name"] for entry in labels["label_names"]}
    colormap = hydra.SegmentationColormap.from_names(names=list(config.label_names.values()))
    colormap.fill_label_space(config.label_space)
    config.logs.log_dir = str(args.output)
    pipeline = hydra.HydraPipeline(config, robot_id=0, config_verbosity=0, freeze_global_info=False)
    pipeline.init(configs, hydra.create_camera(camera))
    receipts, previous_stamp = [], 0
    for path in paths:
        with np.load(path, allow_pickle=False) as packet:
            validate_packet(packet, num_labels=labels["total_semantic_labels"])
            stamp = int(packet["timestamp_ns"])
            if stamp <= previous_stamp:
                raise ValueError("Sensor timestamps must strictly increase")
            previous_stamp = stamp
            if packet["depth"].shape != (height, width) or not np.array_equal(packet["camera_K"], k):
                raise ValueError("Camera calibration changed within probe")
            pose = packet["camera_pose"]
            quaternion = np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1)
            started = time.monotonic()
            updated = pipeline.step(
                stamp, pose[:3, 3].copy(), quaternion, packet["depth"], packet["learned_labels"], packet["rgb"]
            )
            expected_update = (len(receipts) + 1) % cadence == 0
            receipts.append(
                {
                    "frame": path.name,
                    "graph_updated": bool(updated),
                    "expected_graph_update": expected_update,
                    "wall_s": time.monotonic() - started,
                }
            )
            # Native step returns false between scheduled graph updates, even
            # though it integrates those RGB-D frames into the reconstruction.
            if bool(updated) != expected_update:
                raise RuntimeError(f"Unexpected native update receipt for {path.name}: {receipts[-1]}")
    pipeline.save()
    graph_path = args.output / "graph.json"
    pipeline.graph.save(str(graph_path), False)
    graph = json.loads(graph_path.read_text())
    result = {
        "status": "passed_graph_probe",
        "frames": receipts,
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "nodes_by_layer": dict(Counter(str(node["layer"]) for node in graph["nodes"])),
        "mesh_vertices": int(pipeline.graph.mesh.get_vertices().shape[1]),
        "hydra_commit": subprocess.check_output(
            ["git", "-C", "/catkin_ws/src/hydra", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_sha256": {path.name: sha256(path) for path in native_configs + [args.labels]},
    }
    if not result["graph_nodes"] or not result["mesh_vertices"]:
        raise RuntimeError(f"Empty native reconstruction: {result}")
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


def capture(args):
    from run_shared_room_pairs import assert_frozen_source

    from emet.habitat.config import hm3d_scene_glb_path
    from emet.habitat.datasets import load_hmeqa_questions, load_scene_init_poses
    from emet.perception.detection.yoloe import YoloEPerception
    from emet.utils.job_registry import validated_gpu_lock_fd

    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Run under emet jobs --gpu-exclusive; this probe renders and runs learned perception")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert_frozen_source(root, revision)
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise RuntimeError("Keep the shared 20 GiB disk reserve")
    labels_path = args.upstream / LABEL_PATH
    expected = subprocess.check_output(["git", "-C", str(args.upstream), "show", f"{UPSTREAM_REV}:{LABEL_PATH}"])
    if labels_path.read_bytes() != expected:
        raise ValueError("Upstream label vocabulary differs from the pinned source")
    labels = yaml.safe_load(expected)
    invalid = set(labels["invalid_labels"])
    entries = [entry for entry in labels["label_names"] if entry["label"] not in invalid]
    ids = [entry["label"] for entry in entries]
    vocabulary = [entry["name"] for entry in entries]
    # The selected row and spawn are renderer-side only. No gold answer, label,
    # scene metadata, or formatted/annotated question enters the mapper process.
    question = load_hmeqa_questions()[args.question_id]
    spawn = load_scene_init_poses()[(question.scene, question.floor)]
    scene = hm3d_scene_glb_path(question.scene)
    args.output.mkdir(exist_ok=False)
    inputs, native = args.output / "inputs", args.output / "native"
    inputs.mkdir()
    native.mkdir()
    checkpoint = Path.cwd() / "yoloe-v8l-seg.pt"
    if not checkpoint.is_file() or not (Path.cwd() / "mobileclip_blt.ts").is_file():
        raise FileNotFoundError("Use the cached YOLOE and MobileCLIP weights; this probe must not download models")
    manifest = {
        "status": "running",
        "qualification": "mapper_only_no_active_eqa_score",
        "source_commit": revision,
        "driver_sha256": sha256(Path(__file__).resolve()),
        "contract_sha256": sha256(root / "src/emet/eval/grapheqa_sensor.py"),
        "upstream_commit": UPSTREAM_REV,
        "native_image": NATIVE_IMAGE,
        "question_id": args.question_id,
        "frames": args.frames,
        "perception": {
            "provider": "yoloe-v8l-seg",
            "checkpoint_sha256": sha256(checkpoint),
            "text_encoder_sha256": sha256(Path.cwd() / "mobileclip_blt.ts"),
            "vocabulary_sha256": sha256(labels_path),
            "confidence": 0.1,
        },
        "sensor": {"width": 640, "height": 480, "hfov_deg": 120, "height_m": 1.5, "tilt_deg": -30},
        "pose_source": "in_place_turn_command_odometry_initial_local_origin",
        "motion": "stationary_then_10_degree_left_turns",
        "semantic_sensor": False,
        "navmesh_routing": False,
        "simulator_pose_input": False,
        "answer_input": False,
        "enrichment_input": False,
        "packet_sha256": {},
    }
    write_json(args.output / "manifest.json", manifest)
    try:
        render_and_detect(scene, spawn, args.frames, labels, ids, vocabulary, inputs, manifest, YoloEPerception)
        manifest["packet_sha256"] = {path.name: sha256(path) for path in sorted(inputs.glob("*.npz"))}
        write_json(args.output / "manifest.json", manifest)
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--cpuset-cpus",
            "12",
            "--memory",
            "6g",
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-e",
            "MPLCONFIGDIR=/tmp/matplotlib",
        ]
        for source, target, readonly in [
            (Path(__file__).resolve(), "/work/probe.py", True),
            (root / "src/emet/eval/grapheqa_sensor.py", "/work/grapheqa_sensor.py", True),
            (labels_path, "/work/labels.yaml", True),
            (inputs, "/inputs", True),
            (native, "/output", False),
        ]:
            command.extend(
                ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]
            )
        command.extend(
            [
                "--entrypoint",
                "/bin/bash",
                NATIVE_IMAGE,
                "-c",
                "source /opt/ros/noetic/setup.bash; source /catkin_ws/devel/setup.bash; "
                "exec /root/miniforge3/envs/grapheqa/bin/python /work/probe.py "
                "native-worker --inputs /inputs --labels /work/labels.yaml --output /output",
            ]
        )
        with (args.output / "native.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        result = json.loads((native / "result.json").read_text())
        manifest["status"] = result["status"]
        manifest["native_result"] = result
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def habitat_sensor_config(scene):
    """Construct only RGB-D sensors; do not request a semantic mesh or sensor."""
    import habitat_sim
    import magnum as mn

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(scene)
    config.enable_physics = False
    config.load_semantic_mesh = False
    config.use_semantic_textures = False
    sensors = []
    for uuid, kind in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = uuid, kind
        sensor.resolution = [480, 640]
        sensor.position = mn.Vector3(0, 1.5, 0)
        sensor.orientation = mn.Vector3(float(np.deg2rad(-30)), 0, 0)
        sensor.hfov = 120.0
        sensors.append(sensor)
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = sensors
    # No route oracle or global bounds are requested; this probe only turns.
    agent_config.action_space = {
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=10.0))
    }
    return habitat_sim.Configuration(config, [agent_config])


def render_and_detect(scene, spawn, frames, labels, ids, vocabulary, inputs, manifest, detector_type):
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    from PIL import Image

    detector = detector_type(class_list=vocabulary, device="cuda", confidence_threshold=0.1)
    started = time.monotonic()
    with habitat_sim.Simulator(habitat_sensor_config(scene)) as sim:
        state = habitat_sim.AgentState()
        state.position = [spawn.x, spawn.y, spawn.z]
        state.rotation = quat_from_angle_axis(spawn.heading, np.array([0.0, 1.0, 0.0]))
        sim.initialize_agent(0, state)
        focal = 320.0 / np.tan(np.deg2rad(60.0))
        k = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1]], dtype=np.float64)
        for index in range(frames):
            # Sensor data only: never inspect semantic_scene, agent state,
            # pathfinder, scene bounds, or object annotations after spawning.
            raw = sim.get_sensor_observations() if index == 0 else sim.step("turn_left")
            stamp = time.monotonic_ns()
            rgb = np.ascontiguousarray(raw["rgb"][..., :3], dtype=np.uint8)
            depth = np.asarray(raw["depth"], dtype=np.float32)
            semantic, instances, _ = detector.predict(rgb, draw_instance_predictions=False)
            packet = {
                "rgb": rgb,
                "depth": depth,
                "camera_K": k,
                "camera_pose": command_camera_pose(yaw_deg=index * 10, tilt_deg=-30, sensor_height=1.5),
                "timestamp_ns": np.array(stamp, dtype=np.int64),
                "learned_labels": learned_label_map(semantic, instances, ids),
            }
            validate_packet(packet, num_labels=labels["total_semantic_labels"])
            np.savez_compressed(inputs / f"frame-{index:03d}.npz", **packet)
            Image.fromarray(rgb).save(inputs / f"frame-{index:03d}.png")
            print(f"captured frame {index}: {np.unique(packet['learned_labels']).tolist()}", flush=True)
    manifest["capture_and_perception_wall_s"] = time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    host = sub.add_parser("capture")
    host.add_argument("--upstream", required=True, type=Path)
    host.add_argument("--output", required=True, type=Path)
    host.add_argument("--question-id", type=int, default=2)
    host.add_argument("--frames", type=int, choices=range(1, 37), default=5)
    worker = sub.add_parser("native-worker")
    worker.add_argument("--inputs", required=True, type=Path)
    worker.add_argument("--labels", required=True, type=Path)
    worker.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "native-worker":
        native_worker(args)
    else:
        capture(args)


if __name__ == "__main__":
    main()
