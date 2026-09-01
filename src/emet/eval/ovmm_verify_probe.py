# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Live Robocasa viewpoint + current-view YOLOE / dense SigLIP.

Spawns PickPlace, aims from a navigable floor pose (look_front, yaw, back up),
and scores phrases on the head RGB. This is **not** AgenticEQA and does not
teleport onto GT body origins.

GT selection lives in :mod:`emet.eval.ovmm_probe_targets`. Offline dumped-map
queries live in :mod:`emet.eval.ovmm_map_probe`.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from emet.eval.ovmm_probe_targets import (
    DEFAULT_OBJECT_BODY,
    pick_view_targets,
    placement_catalog,
    resolve_phrases,
)
from emet.memory.graph_eqa.agentic.config import SIGLIP_IMAGE_PRESENT_THRESHOLD
from emet.memory.graph_eqa.eval.sim_ground_truth_graph import read_sim_object_placements

DEFAULT_SIM = "configs/sim/robocasa_pick_place_rby1.yaml"
DEFAULT_STANDOFF_M = 1.6
YOLOE_HIT_MIN = 0.10

# Re-export for callers that historically imported selection from this module.
pick_drive_targets = pick_view_targets


def standoff_xyt(
    object_xyz: np.ndarray | list[float],
    robot_xy: np.ndarray | list[float],
    *,
    standoff_m: float = DEFAULT_STANDOFF_M,
) -> np.ndarray:
    """World ``(x, y, yaw)`` a fixed distance from the object, facing it.

    Approach is along the vector from the object toward the current robot XY so
    the goal stays on the occupied (usually spawn) side of the counter. The live
    probe uses relative yaw + backup instead of teleporting to this pose.
    """
    obj = np.asarray(object_xyz, dtype=np.float64).reshape(-1)
    rob = np.asarray(robot_xy, dtype=np.float64).reshape(-1)
    delta = rob[:2] - obj[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        approach = np.array([0.0, 1.0], dtype=np.float64)
    else:
        approach = delta / dist
    xy = obj[:2] + approach * float(standoff_m)
    yaw = math.atan2(float(obj[1]) - float(xy[1]), float(obj[0]) - float(xy[0]))
    return np.array([float(xy[0]), float(xy[1]), float(yaw)], dtype=np.float64)


def backup_distance_m(
    object_xyz: np.ndarray | list[float],
    robot_xy: np.ndarray | list[float],
    *,
    min_standoff_m: float = DEFAULT_STANDOFF_M,
) -> float:
    """Meters to reverse along current heading so planar range is at least ``min_standoff_m``.

    Zero when already far enough. Used after yaw-to-face so ``move_base_to([-d,0,0], relative=True)``
    backs away from the object instead of driving into the counter.
    """
    obj = np.asarray(object_xyz, dtype=np.float64).reshape(-1)
    rob = np.asarray(robot_xy, dtype=np.float64).reshape(-1)
    dist = float(np.linalg.norm(rob[:2] - obj[:2]))
    if dist >= float(min_standoff_m):
        return 0.0
    return float(min_standoff_m) - dist


def interpret_scores(
    *,
    yoloe_score: float | None,
    yoloe_bbox: tuple[int, int, int, int] | None,
    dense_max: float | None,
    crop_siglip: float | None = None,
    yoloe_hit_min: float = YOLOE_HIT_MIN,
    siglip_present_min: float = SIGLIP_IMAGE_PRESENT_THRESHOLD,
) -> dict[str, Any]:
    """Gate a single phrase the way live find does: YOLOE conf or image-space SigLIP PRESENT.

    ``verified`` is True if either channel clears its bar (YOLOE ≥ 0.10, dense SigLIP ≥ 0.12).
    """
    yoloe_hit = yoloe_bbox is not None and yoloe_score is not None and float(yoloe_score) >= float(yoloe_hit_min)
    siglip_present = dense_max is not None and float(dense_max) >= float(siglip_present_min)
    return {
        "yoloe_score": None if yoloe_score is None else float(yoloe_score),
        "yoloe_bbox": list(yoloe_bbox) if yoloe_bbox is not None else None,
        "yoloe_hit": bool(yoloe_hit),
        "dense_siglip": None if dense_max is None else float(dense_max),
        "crop_siglip": None if crop_siglip is None else float(crop_siglip),
        "siglip_present": bool(siglip_present),
        "verified": bool(yoloe_hit or siglip_present),
    }


def score_view(
    rgb: np.ndarray,
    phrases: list[str],
    detector: Any,
    encoder: Any,
) -> list[dict[str, Any]]:
    """Run YOLOE + dense SigLIP for each phrase on one RGB frame."""
    from emet.eval.presence_verifiers import (
        dense_siglip_patch_similarities,
        detector_crop_evidence,
    )

    rows: list[dict[str, Any]] = []
    for phrase in phrases:
        ev = detector_crop_evidence(detector, encoder, rgb, phrase)
        dense = dense_siglip_patch_similarities(encoder, rgb, phrase)
        dense_max = float(np.max(dense[0])) if dense is not None else None
        row = interpret_scores(
            yoloe_score=ev.score,
            yoloe_bbox=ev.bbox_xyxy,
            dense_max=dense_max,
            crop_siglip=ev.crop_siglip_sim,
        )
        row["query"] = phrase
        rows.append(row)
    return rows


def _as_uint8_rgb(rgb: Any) -> np.ndarray | None:
    """Convert a robot RGB observation to HWC uint8, or None if unusable."""
    if rgb is None:
        return None
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _robot_xyt(robot: Any) -> np.ndarray:
    """Base ``(x, y, yaw)`` in MuJoCo world frame when the client supports it."""
    getter = getattr(robot, "get_base_pose_world", None)
    if callable(getter):
        pose = getter(timeout=10.0)
        if pose is not None:
            arr = np.asarray(pose, dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                return arr[:3].copy()
            if arr.size >= 2:
                return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)
    pose = robot.get_base_pose(timeout=10.0)
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size >= 3:
        return arr[:3].copy()
    return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)


def _robot_xy(robot: Any) -> np.ndarray:
    """World-frame base XY (see :func:`_robot_xyt`)."""
    return _robot_xyt(robot)[:2]


def _look_front(robot: Any, *, timeout: float = 10.0) -> None:
    """Pitch the head/torso to the mapping look and wait for a fresh RGB."""
    from emet.controller.controller_dynamem import DYNAMEM_HEAD_SETTLE_S

    look = getattr(robot, "look_front", None)
    if callable(look):
        look(blocking=True, timeout=timeout)
    wait = getattr(robot, "wait_for_obs", None)
    if callable(wait):
        wait(timeout=timeout)
    time.sleep(DYNAMEM_HEAD_SETTLE_S)


def _aim_at(
    robot: Any,
    target_xyz: np.ndarray | list[float],
    *,
    min_standoff_m: float = DEFAULT_STANDOFF_M,
    allow_backup: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Face ``target_xyz`` from the current floor pose; optionally back up if too close.

    Uses relative yaw then relative ``-x`` (robot forward) so we stay on the spawn-side
    floor. Does not teleport to the GT body origin (cabinets sit inside fixtures).
    """
    from emet.agent.face_toward import yaw_to_face_xy

    tgt = np.asarray(target_xyz, dtype=np.float64).reshape(-1)
    _look_front(robot, timeout=min(timeout, 10.0))
    pose = _robot_xyt(robot)
    delta_yaw, _bearing = yaw_to_face_xy(pose, tgt[:2])
    move = getattr(robot, "move_base_to", None)
    yaw_ok = False
    if callable(move):
        yaw_ok = bool(
            move(
                np.array([0.0, 0.0, float(delta_yaw)], dtype=np.float64),
                relative=True,
                blocking=True,
                timeout=timeout,
            )
        )
    backup = backup_distance_m(tgt, pose[:2], min_standoff_m=min_standoff_m) if allow_backup else 0.0
    backup_ok = True
    if backup > 0.05 and callable(move):
        backup_ok = bool(
            move(
                np.array([-float(backup), 0.0, 0.0], dtype=np.float64),
                relative=True,
                blocking=True,
                timeout=timeout,
            )
        )
    _look_front(robot, timeout=min(timeout, 10.0))
    after = _robot_xyt(robot)
    dist = float(np.linalg.norm(after[:2] - tgt[:2]))
    return {
        "ok": bool(yaw_ok and backup_ok),
        "backup_m": float(backup),
        "base_xy": [float(after[0]), float(after[1])],
        "dist_to_target_m": dist,
    }


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write a PNG for later eyeballing of what the head camera actually saw."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def _capture_rgb(robot: Any) -> np.ndarray | None:
    """Latest head RGB as uint8 HWC, after a short wait_for_obs."""
    wait = getattr(robot, "wait_for_obs", None)
    if callable(wait):
        wait(timeout=10.0)
    obs = robot.get_observation()
    if obs is None:
        return None
    return _as_uint8_rgb(getattr(obs, "rgb", None))


def _append_view(
    *,
    report: dict[str, Any],
    views_dir: Path,
    view_id: str,
    meta: dict[str, Any],
    robot: Any,
    phrases: list[str],
    detector: Any,
    encoder: Any,
) -> None:
    """Capture, save PNG, score phrases, append one row to ``report["views"]``."""
    rgb = _capture_rgb(robot)
    row: dict[str, Any] = {"view": view_id, **meta}
    if rgb is None:
        row["error"] = "no_rgb"
        row["queries"] = []
        report["views"].append(row)
        return
    png = views_dir / f"{view_id}.png"
    _save_rgb(png, rgb)
    row["rgb"] = str(png)
    row["queries"] = score_view(rgb, phrases, detector, encoder)
    n_ok = sum(1 for q in row["queries"] if q.get("verified"))
    print(f"  view={view_id} verified={n_ok}/{len(phrases)}", flush=True)
    report["views"].append(row)


def run_verify_probe(
    *,
    sim: str | Path = DEFAULT_SIM,
    queries: list[str] | tuple[str, ...] | None = None,
    object_body: str = DEFAULT_OBJECT_BODY,
    standoff_m: float = DEFAULT_STANDOFF_M,
    out_dir: Path | str | None = None,
    port_offset: int | None = None,
    include_counter: bool = False,
    cpu_only: bool = False,
) -> dict[str, Any]:
    """Spawn rby1 PickPlace, aim from the floor, score YOLOE + SigLIP on head RGB.

    Writes ``report.json`` and ``views/*.png`` under ``out_dir`` (default
    ``~/runs/emet/ovmm_verify_probe/<stamp>/``). Skips sugar-cube ``obj_main`` in
    favor of a preferred kitchen body when one exists.
    """
    from dataclasses import replace

    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.eval.presence_verifiers import YoloEPresenceDetector
    from emet.eval.sim_eval_session import benchmark_sim_server, connect_benchmark_robot
    from emet.perception.detection.yoloe import get_shared_yoloe_perception
    from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

    os.environ.setdefault("EMET_SIM_NAV_TELEPORT", "1")
    os.environ.setdefault("EMET_SKIP_HEAD_SWEEP", "1")
    os.environ.setdefault("EMET_ALLOW_SDPA_ATTN", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")

    repo = Path(__file__).resolve().parents[3]
    sim_path = Path(sim)
    if not sim_path.is_absolute():
        sim_path = repo / sim_path
    sim_cfg = load_sim_launch_config_from_path(str(sim_path))
    offset = int(port_offset) if port_offset is not None else int(520 + (os.getpid() % 40) * 2)
    sim_cfg = replace(sim_cfg, port_offset=offset, headless=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(out_dir).expanduser() if out_dir else Path.home() / "runs" / "emet" / "ovmm_verify_probe" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    views_dir = dest / "views"

    device = "cpu" if cpu_only else "cuda"
    print("OVMM verify-probe: loading YOLOE + SigLIP…", flush=True)
    encoder = get_shared_mask_siglip_encoder(version="so400m", device=device, feature_matching_threshold=0.14)
    yoloe = get_shared_yoloe_perception(confidence_threshold=0.1, device=device, size="l")
    detector = YoloEPresenceDetector(model=yoloe)
    print("OVMM verify-probe: perception ready", flush=True)

    report: dict[str, Any] = {
        "sim": str(sim_path),
        "robot": str(getattr(sim_cfg, "robot", "")),
        "port_offset": offset,
        "standoff_m": float(standoff_m),
        "object_body": object_body,
        "views": [],
    }
    t0 = time.monotonic()
    robot = None
    log_path = dest / "mujoco_server.stderr"
    with log_path.open("w", encoding="utf-8") as log_fh:
        with benchmark_sim_server(sim_cfg, repo=repo, cpu_only=cpu_only, server_stderr=log_fh) as _server:
            robot = connect_benchmark_robot(sim_cfg, offset)
            try:
                sess = robot.get_emet_session() or {}
                placements = read_sim_object_placements(sess) or {}
                catalog = placement_catalog(placements)
                targets = pick_view_targets(placements, object_body=object_body, include_counter=include_counter)
                obj_tgt = next((t for t in targets if t.get("id") == "object"), None)
                obj_cat = None if obj_tgt is None else str(obj_tgt.get("cat") or "")
                phrases = resolve_phrases(queries, placements, object_body=object_body, object_cat=obj_cat)
                report["phrases"] = phrases
                report["gt_object_cat"] = None
                if object_body in placements:
                    report["gt_object_cat"] = str(placements[object_body].get("cat") or "")
                report["find_object"] = obj_tgt
                report["targets"] = targets
                report["placements"] = catalog
                print(
                    f"OVMM verify-probe: {len(placements)} placements, "
                    f"obj_main={report['gt_object_cat']!r}, find={obj_cat!r}, "
                    f"targets={[t['id'] for t in targets]}",
                    flush=True,
                )

                _look_front(robot)
                spawn_xy = _robot_xy(robot)
                _append_view(
                    report=report,
                    views_dir=views_dir,
                    view_id="spawn",
                    meta={"base_xy": [float(spawn_xy[0]), float(spawn_xy[1])]},
                    robot=robot,
                    phrases=phrases,
                    detector=detector,
                    encoder=encoder,
                )
                for tgt in targets:
                    nav = _aim_at(
                        robot,
                        tgt["xyz"],
                        min_standoff_m=standoff_m,
                        allow_backup=not bool(tgt.get("yaw_only")),
                    )
                    _append_view(
                        report=report,
                        views_dir=views_dir,
                        view_id=str(tgt["id"]),
                        meta={"target": tgt, "nav": nav},
                        robot=robot,
                        phrases=phrases,
                        detector=detector,
                        encoder=encoder,
                    )
            finally:
                stop = getattr(robot, "stop", None)
                if callable(stop):
                    stop()

    report["wall_s"] = round(time.monotonic() - t0, 2)
    out_json = dest / "report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["out"] = str(out_json)
    print(json.dumps({k: report[k] for k in ("sim", "gt_object_cat", "wall_s", "out")}, indent=2), flush=True)
    return report
