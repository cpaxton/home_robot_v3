# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Recompute the frozen three-scene sensor gate; never score EQA task success."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RUNTIME_REVISION = "818066c9b75f0b75a38773d78691b7e98c046bdb"
EXPECTED_BUDGET = {"max_primitives": 200, "max_seconds": 240, "max_legs": 2, "leg_distance_m": 0.75}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_case(path):
    manifest = read(path / "manifest.json")
    rows = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    if not (
        manifest["status"] == "completed"
        and manifest.get("completed_legs") == 2
        and manifest["source_commit"] == RUNTIME_REVISION
        and manifest["budgets"] == EXPECTED_BUDGET
        and manifest["frontier_policy"] == "motion_effort"
        and manifest["acquisition_seed"] == 0
        and manifest["model_calls"] == 0
        and manifest["scorer_read"] is False
    ):
        raise ValueError(f"Incomplete or mismatched collection: {path}")
    if (
        len(rows) != manifest["frame_count"]
        or [r["sequence"] for r in rows] != list(range(1, len(rows) + 1))
        or any(not r["odometry"]["ok"] or r["collided"] for r in rows)
        or sum(r["command"] is not None for r in rows) != manifest["attempted_primitives"]
        or manifest["attempted_primitives"] > 200
        or manifest["wall_s"] > 240
    ):
        raise ValueError(f"Invalid frame, tracking, collision or action ledger: {path}")
    for row in rows:
        prefix = path / f"frame-{row['sequence']:05d}"
        if (
            sha256(prefix.with_suffix(".png")) != row["rgb_sha256"]
            or sha256(prefix.with_suffix(".npz")) != row["depth_sha256"]
        ):
            raise ValueError(f"Changed sensor packet: {prefix}")
    initial = next(e for e in events if e["stage"] == "initial_scan_complete")
    expected = ["turn_left"] * 9 + ["look_down"] * 3 + ["look_up"] * 3 + ["turn_left"] * 63
    if initial["sequence"] != 79 or [r["command"]["action"] for r in rows[1:79]] != expected:
        raise ValueError(f"Initial scan is incomplete: {path}")
    if sha256(path / "mapping_rng_state.npy") != manifest["mapping_rng_sha256"]:
        raise ValueError(f"Mapping RNG artifact changed: {path}")
    residuals = []
    for leg in range(2):
        started, finished, scan = (
            next(e for e in events if e["stage"] == stage and e.get("leg") == leg)
            for stage in ("motion_started", "motion_finished", "arrival_scan_complete")
        )
        receipt = finished["receipt"]
        start_xy = np.asarray(rows[started["sequence"] - 1]["odometry"]["camera_pose"])[:2, 3]
        route = np.asarray(started["route"])
        planned_distance = float(np.linalg.norm(np.diff(np.vstack((start_xy, route[:, :2])), axis=0), axis=1).sum())
        end, goal = np.asarray(receipt["current_pose"]), np.asarray(receipt["goal_pose"])
        residual = float(np.linalg.norm(end[:2] - goal[:2]))
        yaw = float(abs(np.arctan2(np.sin(end[2] - goal[2]), np.cos(end[2] - goal[2]))))
        if not (
            receipt["success"]
            and 0 < planned_distance <= 0.75 + 1e-8
            and residual <= 0.025
            and yaw <= np.deg2rad(3)
            and np.allclose(goal, route[-1])
        ):
            raise ValueError(f"Failed measured route: {path}, leg {leg}")
        if scan["sequence"] - finished["sequence"] != 19 or any(
            r["command"] != {"action": "turn_left", "amount": None}
            for r in rows[scan["sequence"] - 18 : scan["sequence"]]
        ):
            raise ValueError(f"Arrival scan is incomplete: {path}, leg {leg}")
        residuals.append(
            {
                "leg": leg,
                "planned_distance_m": planned_distance,
                "position_residual_m": residual,
                "yaw_residual_rad": yaw,
            }
        )
    return {
        "case": path.name,
        "status": "passed",
        "frames": len(rows),
        "attempted_primitives": manifest["attempted_primitives"],
        "wall_s": manifest["wall_s"],
        "estimated_camera_path_m": manifest["estimated_camera_path_m"],
        "legs": residuals,
        "manifest_sha256": sha256(path / "manifest.json"),
        "frames_sha256": sha256(path / "frames.jsonl"),
    }


def validate(root):
    cases = [check_case(root / "motion-effort" / case) for case in ("q2", "q6", "q12")]
    fixed = read(root / "fixed-turn-q6/result.json")
    if not (
        fixed["status"] == "passed_navigation_gate"
        and fixed["requested_turn_deg"] == 180
        and len(fixed["tests"]) == 3
        and all(t["success"] for t in fixed["tests"])
    ):
        raise ValueError("Fixed half-turn / unknown-goal control failed")
    oracle = read(root / "scorer-only/q6-calibration-r2/result.json")
    if not (
        oracle["status"] == "matched_renderer_replay"
        and oracle["matched_rgb_frames"] == oracle["planned_frames"] == 176
        and oracle["native_raster_center_max_ray_error"] < 1e-4
        and all(t["max_true_base_displacement_m"] == 0 for t in oracle["turn_segments"])
    ):
        raise ValueError("Scorer-only renderer/calibration control failed")
    preflight_rows = []
    preflight_log = root / "standard-client-preflight-r2.log"
    for line in preflight_log.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "ok" in item:
            preflight_rows.append(item)
    if len(preflight_rows) != 1 or not preflight_rows[0]["ok"] or preflight_rows[0]["rgb_shape"] != [480, 640, 3]:
        raise ValueError("Ordinary-client full-import renderer preflight failed")
    return {
        "scope": "sensor/navigation gate, not EQA or OVMM task performance",
        "status": "passed",
        "runtime_source_commit": RUNTIME_REVISION,
        "budgets": EXPECTED_BUDGET,
        "cases": cases,
        "fixed_turn_control": fixed,
        "scorer_only_calibration": oracle,
        "ordinary_client_preflight": preflight_rows[0],
        "ordinary_client_preflight_log_sha256": sha256(preflight_log),
        "validator_sha256": sha256(Path(__file__)),
        "figures": {p.name: sha256(p) for p in sorted((root / "figures").glob("*")) if p.suffix in (".png", ".pdf")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: all three complete histories, six measured arrivals, full scans, fixed half-turn, and renderer oracle")
