"""Measure physical joint/hand speeds from recordings, independently of video fps."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.profiles import PROFILES

SPEED_BINS = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, float("inf"))


def joint_speed_statistics(times, joints, speed_limits):
    """Time-weighted finite-interval statistics; limits are configured, not measured."""
    times, joints, limits = (np.asarray(x, float) for x in (times, joints, speed_limits))
    if (
        times.ndim != 1
        or len(times) < 2
        or limits.ndim != 1
        or joints.ndim != 2
        or joints.shape != (len(times), len(limits))
        or not len(limits)
        or any(not np.isfinite(x).all() for x in (times, joints, limits))
        or np.any(np.diff(times) <= 0)
        or np.any(limits <= 0)
    ):
        raise ValueError("finite increasing times, complete joints and positive speed limits required")
    dt = np.diff(times)
    speeds = abs(np.diff(joints, axis=0)) / dt[:, None]
    exposure, _ = np.histogram((speeds / limits).reshape(-1), bins=SPEED_BINS, weights=np.repeat(dt, len(limits)))
    return {
        "record_interval_median_s": float(np.median(dt)),
        "duration_s": float(dt.sum()),
        "joint_minimum_rad": joints.min(axis=0).tolist(),
        "joint_maximum_rad": joints.max(axis=0).tolist(),
        "peak_joint_speed_rad_s": speeds.max(axis=0).tolist(),
        "mean_absolute_joint_speed_rad_s": (np.sum(speeds * dt[:, None], axis=0) / dt.sum()).tolist(),
        "configured_profile_speed_rad_s": limits.tolist(),
        "speed_bin_joint_seconds": exposure.tolist(),
        "speed_bin_joint_time_fraction": (exposure / exposure.sum()).tolist(),
    }


def audit(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError("fresh speed audit directory required")
    output.mkdir(parents=True)
    result = {
        "format": "recorded-motion-speed-v1",
        "hardware_used": False,
        "interpretation": "Recorded finite-interval speeds; profile caps are not verified hardware ratings.",
        "absolute_speed_ratio_bin_edges": [*SPEED_BINS[:-1], "inf"],
        "families": {},
    }
    for path in sorted(directory.glob("*/report.json")):
        report = json.loads(path.read_text())
        rows = []
        for row in report["rows"]:
            if not row.get("recording"):
                continue
            recording = path.parent / row["recording"]
            if hashlib.sha256(recording.read_bytes()).hexdigest() != row["assets"][recording.name]:
                raise ValueError("recording checksum mismatch")
            with np.load(recording, allow_pickle=False) as data:
                stats = joint_speed_statistics(data["time_s"], data["joints"], PROFILES["innate_mars"].speed)
                velocity = np.diff(data["hand"][:, :3], axis=0) / np.diff(data["time_s"])[:, None]
                stats["peak_hand_speed_m_s"] = float(np.linalg.norm(velocity, axis=1).max())
            rows.append({"case": row["case"], "success": row["success"], **stats})
        if not rows:
            result["families"][path.parent.name] = {
                "cases": [],
                "no_recordings": True,
                "rejected_cases": len(report["rows"]),
            }
            continue
        exposure = np.sum([r["speed_bin_joint_seconds"] for r in rows], axis=0)
        result["families"][path.parent.name] = {
            "source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "active_arm": path.parent.name != "base",
            "cases": rows,
            "median_case_peak_joint_speed_rad_s": float(np.median([max(r["peak_joint_speed_rad_s"]) for r in rows])),
            "median_case_peak_hand_speed_m_s": float(np.median([r["peak_hand_speed_m_s"] for r in rows])),
            "speed_bin_joint_time_fraction": (exposure / exposure.sum()).tolist(),
        }
    save(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.recordings, args.output)
