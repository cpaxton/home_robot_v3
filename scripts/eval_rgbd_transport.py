#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Replay one recorded camera pair across original/wire RGB and depth profiles.

Uses identical captures, known-model prior, seed pixels and tracker thresholds.
No scene labels, object poses, command execution or sensor retiming are used.
This isolates image transport from acquisition/registration in a diagnostic.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import label

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.multiview_rgbd_tracking import MultiViewRgbdTracker, SynchronizedRgbdView
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.core.rgbd_registration import decode_registration


def replay(recording):
    recording = Path(recording)
    record = json.loads((recording / "result.json").read_text())
    rows = record["input_captures"]
    if len(rows) < 2:
        raise ValueError("two_recorded_captures_required")
    if not record.get("captures") or rows[0]["observation_id"] != record["captures"][0]["observation_id"]:
        raise ValueError("reference_capture_not_retained_in_bounded_archive")
    frames = []
    for index, metadata in enumerate(rows[:2]):
        wire = np.load(recording / f"input-{index:03}.npz")
        archive = json.loads((recording / f"registration-{index:03}.json").read_text())
        packet, timing = archive["registration"], archive["image_timing"]
        decoded = decode_registration(packet)
        if (
            packet["clock_id"] != timing["clock_id"]
            or packet["rgb_interval_ns"] != timing["capture_interval_ns"]
            or packet["depth_interval_ns"] != timing["depth_capture_interval_ns"]
            or packet["rgb_interval_ns"] != packet["depth_interval_ns"]
            or abs(packet["rgb_interval_ns"][0] - metadata["earliest"] * 1e9) > 2
            or abs(packet["rgb_interval_ns"][1] - metadata["latest"] * 1e9) > 2
        ):
            raise ValueError("rgbd_registration_capture_mismatch")
        k, shape = decoded["rgb_k"], decoded["rgb"].shape
        rotated_k = np.array([[k[1, 1], 0, shape[0] - 1 - k[1, 2]], [0, k[0, 0], k[0, 2]], [0, 0, 1]])
        np.testing.assert_allclose(rotated_k, archive["camera_K"], rtol=0, atol=1e-9)
        frames.append(
            {
                "wire_rgb": wire["rgb"],
                "wire_depth": wire["depth"],
                "original_rgb": np.rot90(decoded["rgb"], -1),
                "original_depth": np.rot90(decoded["registered_depth"], -1),
                "pose": np.asarray(archive["camera_pose"]),
                "k": rotated_k,
                "capture": CaptureInterval(**metadata),
                "registration": decoded,
            }
        )
    result = {
        "schema_version": 1,
        "scope": "same-capture transport ablation; no task or ownership qualification",
        "capture_gap_s": frames[1]["capture"].latest - frames[0]["capture"].earliest,
        "profiles": {},
        "small_wire_components": [],
    }
    for rgb_key in ("wire_rgb", "original_rgb"):
        for depth_key in ("wire_depth", "original_depth"):
            clock = [frames[0]["capture"].received_monotonic]
            views = []
            for f in frames:
                rgb, depth, world = capture_arrays(
                    SimpleNamespace(rgb=f[rgb_key], depth=f[depth_key], camera_K=f["k"], camera_pose=f["pose"])
                )
                views.append(SynchronizedRgbdView(rgb, depth, world, f["capture"]))
            ref, current = views
            seed = np.load(recording / "reference.npz")["mask"]
            name = f"{rgb_key}+{depth_key}"
            tracker = None
            try:
                tracker = MultiViewRgbdTracker(
                    PrimitiveModel("cylinder", (0.02, 0.04)),
                    reference_view="head",
                    reference=ref,
                    mask=seed,
                    scene_epoch="recorded-transport",
                    object_ref="operator-red-instance",
                    component_policy="complete_prior_union",
                    appearance_model="neutral_specular",
                    allow_clipped_views=True,
                    clock=lambda: clock[0],
                )
                if name == "wire_rgb+wire_depth":
                    mask = tracker._candidate_mask(
                        current.rgb,
                        current.depth,
                        current.world,
                        centers=tracker.latest.fit.poses[:, :3, 3],
                        displacement=tracker.speed * result["capture_gap_s"] + tracker.margin,
                        allow_image_boundary=True,
                        merge_reachable=True,
                    )
                    parts, count = label(mask)
                    for part in range(1, count + 1):
                        pixels = np.argwhere(parts == part)
                        if len(pixels) < 10:
                            result["small_wire_components"].append(
                                [
                                    {
                                        "yx": [int(y), int(x)],
                                        "wire_rgb": frames[1]["wire_rgb"][y, x].tolist(),
                                        "original_rgb": frames[1]["original_rgb"][y, x].tolist(),
                                        "wire_depth_m": float(frames[1]["wire_depth"][y, x]),
                                        "original_depth_m": float(frames[1]["original_depth"][y, x]),
                                    }
                                    for y, x in pixels
                                ]
                            )
                clock[0] = current.capture.received_monotonic
                fit = tracker.update({"head": current}, scene_epoch="recorded-transport")
                result["profiles"][name] = {"accepted": True, "fit": fit.as_dict()}
            except ObservationUnavailable as error:
                result["profiles"][name] = {"accepted": False, "reason": str(error)}
    depth_error = np.abs(frames[1]["wire_depth"] - frames[1]["original_depth"])
    result["wire_depth_max_error_m"] = float(depth_error.max())
    result["wire_rgb_changed_pixels"] = int(np.any(frames[1]["wire_rgb"] != frames[1]["original_rgb"], axis=-1).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.recording)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "small_wire_components"}))


if __name__ == "__main__":
    main()
