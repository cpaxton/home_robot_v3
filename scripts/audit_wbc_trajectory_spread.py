"""Measure within-episode pose spread, motion duty and video padding."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.trajectory_metrics import trajectory_spread
from emet.visualization.curriculum_record import sha


def audit(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError("fresh trajectory audit output required")
    output.mkdir(parents=True)
    report = {
        "format": "curriculum-motion-audit-v1",
        "source": "recorded active-target timeline; failures retained",
        "families": {},
        "cases": [],
        "hardware_used": False,
    }
    for path in sorted(directory.glob("*/report.json")):
        rows = json.loads(path.read_text())["rows"]
        family = path.parent.name
        metrics = []
        receipt = json.loads((directory / (family + "-video.json")).read_text())
        for row in rows:
            if not row.get("recording"):
                continue
            recording = path.parent / row["recording"]
            if sha(recording) != row["assets"][recording.name]:
                raise ValueError("changed recording")
            with np.load(recording, allow_pickle=False) as data:
                value = trajectory_spread(
                    data["time_s"],
                    data["hand_target"],
                    data["base_target"],
                    hand_mask=family != "base",
                    base_mask=row["case"].get("base_mask", [True] * 3),
                )
                value["measured_joint_span_rad"] = np.ptp(data["joints"], axis=0).tolist()
            value.update(
                case=row["case"],
                family=family,
                success=row["success"],
                video_padding_s=max(0, receipt["duration_s"] - value["duration_s"]),
            )
            metrics.append(value)
            report["cases"].append(value)
        duration = sum(r["duration_s"] for r in metrics)
        moving = sum(r["reference_motion_s"] for r in metrics)
        summary = {
            "cases": len(metrics),
            "physical_duration_s": duration,
            "reference_motion_s": moving,
            "reference_motion_fraction": moving / duration,
            "useful_speed_motion_fraction": sum(r["useful_speed_motion_s"] for r in metrics) / duration,
            "video_motion_fraction": moving / (receipt["duration_s"] * len(rows)),
            "median_duration_s": float(np.median([r["duration_s"] for r in metrics])),
            "median_motion_s": float(np.median([r["reference_motion_s"] for r in metrics])),
            "median_video_padding_s": float(np.median([r["video_padding_s"] for r in metrics])),
        }
        if receipt.get("shorter_recordings") == "replay_success_with_visible_reset":
            summary["video_motion_fraction"] = None
            summary["video_presentation"] = "successful records replay with visible reset; physical duty unchanged"
        for name in ["world_hand", "base_relative_hand"]:
            if family == "base":
                summary[name] = None
                continue
            keys = [
                "maximum_excursion_m",
                "path_length_m",
                "rotation_excursion_rad",
                "rotation_travel_rad",
                "peak_linear_speed_m_s",
            ]
            summary[name] = {
                k: {
                    "p10": float(np.quantile([r[name][k] for r in metrics], 0.1)),
                    "median": float(np.median([r[name][k] for r in metrics])),
                    "p90": float(np.quantile([r[name][k] for r in metrics], 0.9)),
                }
                for k in keys
            }
        report["families"][family] = summary
    save(output / "report.json", report)
    save(
        output / "summary.json",
        {"format": report["format"], "source": report["source"], "families": report["families"]},
    )
    lines = [
        "# Within-trajectory motion audit",
        "",
        "Global reset-pose coverage is not per-episode movement.",
        "",
        "| Family | Target moving | Video moving | Median duration | Base-relative hand excursion |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, s in report["families"].items():
        excursion = (
            f"{s['base_relative_hand']['maximum_excursion_m']['median'] * 1000:.1f} mm"
            if s["base_relative_hand"]
            else "inactive"
        )
        video = f"{s['video_motion_fraction']:.1%}" if s["video_motion_fraction"] is not None else "labelled replays"
        lines.append(
            f"| {name} | {s['reference_motion_fraction']:.1%} | {video} | {s['median_duration_s']:.2f} s | {excursion} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.recordings, args.output)


if __name__ == "__main__":
    main()
