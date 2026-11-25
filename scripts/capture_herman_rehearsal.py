"""Capture existing ROS observations over SSH for an offline rehearsal. No motion."""

import argparse
import base64
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from emet.robots.innate_mars.passive_ssh import subscriber_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", default="jetson1@192.168.1.43")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--astra-device", type=Path, help="Record a simultaneous local Astra RGB clip")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output directory required; never overwrite a captured fixture")
    if not 3 <= args.seconds <= 20:
        parser.error("duration must be 3..20 s")
    command = subscriber_command(args.ssh, program="rehearsal_capture_ros", arguments=["--seconds", str(args.seconds)])
    if args.astra_device:
        args.output.mkdir(parents=True)
        astra_path = args.output / "astra-rgb.mp4"

        def capture_astra():
            started = time.time_ns()
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-f",
                    "v4l2",
                    "-i",
                    str(args.astra_device),
                    "-t",
                    str(args.seconds + 4),
                    "-vf",
                    "scale=960:-2",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    str(astra_path),
                ],
                capture_output=True,
                text=True,
                timeout=args.seconds + 15,
            )
            if result.returncode or not astra_path.is_file() or not astra_path.stat().st_size:
                raise RuntimeError("Astra RGB capture failed: " + result.stderr[-1500:])
            return {
                "path": astra_path.name,
                "sha256": hashlib.sha256(astra_path.read_bytes()).hexdigest(),
                "host_start_ns": started,
                "host_end_ns": time.time_ns(),
                "frame_sync_verified": False,
                "base_transform_calibrated": False,
            }

        with ThreadPoolExecutor(max_workers=1) as pool:
            astra_future = pool.submit(capture_astra)
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.seconds + 30)
            astra = astra_future.result()
    else:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.seconds + 30)
        astra = None
    records = [
        line[len("EMET_REHEARSAL ") :] for line in result.stdout.splitlines() if line.startswith("EMET_REHEARSAL ")
    ]
    if result.returncode or len(records) != 1:
        raise RuntimeError("passive capture failed: " + result.stderr[-1500:])
    report = json.loads(records[0])
    if astra is None:
        args.output.mkdir(parents=True)
    else:
        report["astra_rgb"] = astra
    (args.output / "raw-capture.json").write_text(records[0] + "\n")
    for name, sample in report["samples"].items():
        if isinstance(sample, dict) and "jpeg_base64" in sample:
            data = base64.b64decode(sample.pop("jpeg_base64"), validate=True)
            path = args.output / (name + ".jpg")
            path.write_bytes(data)
            sample["image_asset"] = {"path": path.name, "sha256": hashlib.sha256(data).hexdigest()}
    description = report["samples"].pop("robot_description", None)
    if description:
        path = args.output / "robot_description.urdf"
        path.write_text(description)
        report["robot_description"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": report["counts"],
                "transforms": {
                    k: "available" if "error" not in v else v["error"] for k, v in report["transforms"].items()
                },
                "hardware_motion_commands_sent": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
