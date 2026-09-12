#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Hold a real robot asset at a known view; isolate rendering and learned perception.

No bridge, navigation, physics stepping, graph, or VLM router. Ground-truth
segmentation is used only for visibility and scoring, never detector input.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml
from PIL import Image

from emet.robots import get_robot_spec
from emet.simulation.head_look_action import apply_head_to_robosuite
from emet.simulation.mujoco_server import _load_default_scene_with_robot
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import head_camera_geomgroup_mask


def setup_scene(config):
    spec = get_robot_spec(config["robot"])
    model = _load_default_scene_with_robot(config["robot"])
    if model is None:
        raise ValueError("robot asset could not be loaded")
    data = mujoco.MjData(model)
    base = model.body(spec.base_link_name)
    joint = int(model.body_jntadr[base.id])
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("stationary fixture requires a free-root robot asset")
    adr = int(model.jnt_qposadr[joint])
    x, y, yaw = config["base_xyt"]
    data.qpos[adr : adr + 2] = [x, y]
    data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    apply_head_to_robosuite(spec, model, data, config["head_pan"], config["head_tilt"])
    # Kinematic hold: set position-actuated joints to their control targets.
    for aid in range(model.nu):
        if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[aid, 0])
            if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                data.qpos[model.jnt_qposadr[jid]] = data.ctrl[aid]
    table = model.body("table")
    model.body_pos[table.id] = config["table_position"]
    model.geom_size[int(table.geomadr[0])] = config["table_size"]
    for target in config["targets"]:
        body = model.body(target["body"])
        qadr = int(model.jnt_qposadr[int(body.jntadr[0])])
        data.qpos[qadr : qadr + 3] = target["position"]
        gid = int(body.geomadr[0])
        model.geom_size[gid] = target["size"]
        model.geom_rgba[gid] = target["rgba"]
    mujoco.mj_forward(model, data)
    return spec, model, data


def run(config, output, learned=False):
    output.mkdir(parents=True, exist_ok=True)
    spec, model, data = setup_scene(config)
    height, width = config["image_size"]
    camera = model.camera(config["camera"]).id
    option = mujoco.MjvOption()
    option.geomgroup[:] = head_camera_geomgroup_mask(model, spec.base_link_name)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        overview = mujoco.MjvCamera()
        overview.lookat[:] = [0, -0.3, 0.8]
        overview.distance, overview.azimuth, overview.elevation = 4.5, 90, -55
        renderer.update_scene(data, camera=overview)
        Image.fromarray(renderer.render()).save(output / "overview.png")
    Image.fromarray(rgb).save(output / "rgb.png")
    focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
    pose[:3, 3] = data.cam_xpos[camera]
    yy, xx = np.indices(depth.shape)
    rays = np.stack([(xx - width / 2) / focal, (yy - height / 2) / focal, np.ones_like(xx)], axis=-1)
    world_xyz = (rays * depth[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    np.savez_compressed(
        output / "frame.npz", rgb=rgb, depth=depth, camera_K=K, camera_pose=pose, segmentation=segmentation
    )
    report = {
        "fixture": config,
        "mjcf": spec.mjcf_path,
        "mode": "kinematic_stationary",
        "bridge_tested": False,
        "camera_pose": pose.tolist(),
        "targets": [],
    }
    for target in config["targets"]:
        gid = int(model.body(target["body"]).geomadr[0])
        mask = (segmentation[..., 0] == gid) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        cam_xyz = np.linalg.solve(pose, [*target["position"], 1])[:3]
        uv = K @ cam_xyz
        report["targets"].append(
            {
                "query": target["query"],
                "geometry_id": gid,
                "visible_pixels": int(mask.sum()),
                "pixel_xy": (uv[:2] / uv[2]).tolist(),
                "camera_z_m": float(cam_xyz[2]),
            }
        )
    if learned:
        import torch

        from emet.controller.controller_lazy_graph import LazyGraphController
        from emet.memory.graph_eqa.graph_object_fusion.attach import fusion_config_from_sources
        from emet.memory.graph_eqa.ingest.instance_observations import filter_detections_for_graph_admission
        from emet.perception.detection.yoloe import get_shared_yoloe_perception
        from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

        detector = get_shared_yoloe_perception(confidence_threshold=0.05, device="cuda", size="l")
        agent = LazyGraphController.__new__(LazyGraphController)
        agent.detection_model = detector
        agent.voxel_map = SimpleNamespace(min_depth=0.25, max_depth=2.5)
        frame = SimpleNamespace(rgb=rgb, depth=depth, full_world_xyz=world_xyz)
        report["detections"] = []
        queries = [t["query"] for t in config["targets"]] + config.get("negative_queries", [])
        fusion = fusion_config_from_sources(parameters={})
        report["detector_admission_threshold"] = fusion.admission.instance_min_confidence
        report["localization_tolerance_m"] = 0.15
        for query in queries:
            detected, detections = agent._detect_query_frame(frame, query)
            np.savez_compressed(output / f"detector_masks_{len(report['detections'])}.npz", masks=detected.instance)
            admitted, _ = filter_detections_for_graph_admission(detections, config=fusion)
            target = next((t for t in config["targets"] if t["query"] == query), None)
            errors = (
                [float(np.linalg.norm(np.asarray(d["xyz"]) - target["position"])) for d in admitted]
                if target is not None
                else []
            )
            passed = bool(errors and min(errors) <= 0.15) if target else not admitted
            report["detections"].append(
                {
                    "query": query,
                    "predictions": detections,
                    "admitted_count": len(admitted),
                    "centroid_errors_m": errors,
                    "passed": passed,
                }
            )
        # Dense features are the learned representation used by voxel retrieval;
        # this isolates one-frame retrieval, not voxel fusion or graph search.
        encoder = get_shared_mask_siglip_encoder()
        with torch.no_grad():
            _, features = encoder.run_mask_siglip(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float(), (120, 160))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            report["dense_retrieval"] = []
            for query in queries:
                text_feature = torch.nn.functional.normalize(encoder.encode_text(query).float().reshape(-1), dim=0)
                scores = (features @ text_feature.to(features.device)).squeeze().cpu().numpy()
                iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
                py, px = int((iy + 0.5) * height / scores.shape[0]), int((ix + 0.5) * width / scores.shape[1])
                report["dense_retrieval"].append(
                    {
                        "query": query,
                        "max_cosine": float(scores[iy, ix]),
                        "pixel_xy": [px, py],
                        "candidate_xyz": world_xyz[py, px].tolist(),
                    }
                )
            np.savez_compressed(output / "dense_features.npz", features=features.cpu().numpy())
        report["detector_pass"] = all(row["passed"] for row in report["detections"])
    report["visibility_pass"] = all(t["visible_pixels"] >= 100 for t in report["targets"])

    def serialize(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    (output / "result.json").write_text(json.dumps(report, indent=2, default=serialize) + "\n")
    print(json.dumps(report, default=serialize), flush=True)
    return 0 if report["visibility_pass"] and report.get("detector_pass", True) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ovmm/stationary_rby1.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learned", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(yaml.safe_load(args.config.read_text()), args.output_dir, args.learned))
