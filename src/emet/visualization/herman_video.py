"""Render source-bound Herman recordings to a comparison MP4; no physics steps."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.visualization.herman_rehearsal import Replay

TARGET = (50, 230, 255)
ACTUAL = (255, 170, 70)
TEXT = (220, 233, 245)
BACKGROUND = (15, 24, 34)


def pose_rotations(values):
    columns = np.asarray(values)[..., 3:].reshape(-1, 3, 2)
    return np.concatenate([columns, np.cross(columns[:, :, 0], columns[:, :, 1])[:, :, None]], axis=2)


def measurements(replay, index):
    qpos, times = replay.frames(index)
    goals = replay.targets[index][:, 3:]
    target = pose_rotations(goals)
    vectors = Rotation.from_matrix(target[0].T @ target).as_rotvec()
    largest = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
    axis = largest / np.linalg.norm(largest) if np.linalg.norm(largest) > 1e-8 else np.array([1, 0, 0])
    data = mujoco.MjData(replay.model)
    actual, position_errors, rotation_errors = [], [], []
    for q, goal, desired in zip(qpos, goals, target, strict=True):
        data.qpos[:] = q
        mujoco.mj_forward(replay.model, data)
        hand = data.body("ee_link")
        rotation = hand.xmat.copy().reshape(3, 3)
        actual.append(Rotation.from_matrix(target[0].T @ rotation).as_rotvec() @ axis)
        position_errors.append(float(np.linalg.norm(hand.xpos.copy() - goal[:3])))
        rotation_errors.append(float(Rotation.from_matrix(desired @ rotation.T).magnitude()))
    result = {
        "times": times,
        "target_angle": np.rad2deg(vectors @ axis),
        "actual_angle": np.rad2deg(actual),
        "position_mm": np.array(position_errors) * 1000,
        "rotation_deg": np.rad2deg(rotation_errors),
    }
    joints = replay.joint_outputs[index]
    if joints:
        origin = joints["measured_joints"][0]
        result.update({key + "_deg": np.rad2deg(value - origin) for key, value in joints.items()})
    return result


def text(image, value, position, scale=0.55, color=TEXT):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def chart(image, stats, index, x, duration):
    if "reference_joints_deg" in stats:
        for row, (joint, label) in enumerate([(2, "Elbow"), (3, "Pitch"), (4, "Roll")]):
            top, bottom = 577 + row * 43, 614 + row * 43
            left, right = x + 68, x + 492
            bound = max(
                5.0,
                max(
                    float(np.max(abs(stats[k][:, joint])))
                    for k in ("reference_joints_deg", "measured_joints_deg", "actuator_targets_deg")
                )
                * 1.1,
            )
            middle = (top + bottom) / 2
            cv2.rectangle(image, (left, top), (right, bottom), (44, 61, 78), 1)
            text(image, label, (x + 3, top + 12), 0.35)
            text(image, f"+/-{bound:.0f}", (x + 3, bottom - 2), 0.3)
            for key, color in [
                ("reference_joints_deg", TARGET),
                ("actuator_targets_deg", (150, 160, 170)),
                ("measured_joints_deg", ACTUAL),
            ]:
                values = stats[key][: index + 1, joint]
                xx = left + (right - left) * stats["times"][: index + 1] / duration
                yy = middle - values / bound * ((bottom - top) / 2 - 2)
                points = np.column_stack([xx, yy]).astype(np.int32)
                if len(points) >= 2:
                    cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
        return
    left, right, top, bottom = x + 42, x + 492, 577, 700
    bound = max(4.5, float(np.max(abs(stats["actual_angle"]))) * 1.1)
    cv2.rectangle(image, (left, top), (right, bottom), (44, 61, 78), 1)
    mid = (top + bottom) // 2
    cv2.line(image, (left, mid), (right, mid), (44, 61, 78), 1)
    for key, color in [("target_angle", TARGET), ("actual_angle", ACTUAL)]:
        values = stats[key][: index + 1]
        xx = left + (right - left) * stats["times"][: index + 1] / duration
        yy = mid - values / bound * ((bottom - top) / 2 - 4)
        points = np.column_stack([xx, yy]).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
    text(image, f"+{bound:.1f}", (x + 2, top + 12), 0.35)
    text(image, f"-{bound:.1f}", (x + 2, bottom), 0.35)
    text(image, "0 s", (left, bottom + 17), 0.35)
    text(image, f"{duration:.1f} s", (right - 40, bottom + 17), 0.35)


def render_video(directory, output, *, fps=20):
    output = Path(output)
    if output.exists() or output.with_suffix(".partial.mp4").exists():
        raise ValueError("fresh video output required")
    if not 1 <= fps <= 30:
        raise ValueError("video fps must be 1..30")
    replay = Replay(directory)
    selected = list(range(1, len(replay.entries)))
    if not 1 <= len(selected) <= 3:
        replay.close()
        raise ValueError("select a rehearsal containing one to three controller recordings")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.mp4")
    process = None
    try:
        stats = [measurements(replay, i) for i in selected]
        duration = max(float(s["times"][-1]) for s in stats)
        width, height = 512 * len(selected), 768
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        with output.with_suffix(".ffmpeg.log").open("w") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            cache = {}
            frame_count = int(round(duration * fps)) + 1
            for tick in range(frame_count):
                now = tick / fps
                canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
                text(canvas, "HERMAN | OFFLINE RECORDED WAVE COMPARISON", (18, 27), 0.75)
                text(
                    canvas,
                    "Synthetic staged start. No hardware motion. Elbow + wrist wave; base/gripper targets fixed."
                    if replay.report["proposals"][0].get("gesture") == "elbow_wrist_wave"
                    else "Synthetic staged start. No hardware motion. Small wrist-only reference; base target fixed.",
                    (18, 51),
                    0.52,
                )
                text(
                    canvas,
                    f"Time {now:05.2f} s | cyan: reference | orange: measured | grey: actuator target | inset: hand",
                    (18, 76),
                    0.52,
                )
                for column, (index, values) in enumerate(zip(selected, stats, strict=True)):
                    frame = min(
                        len(values["times"]) - 1, max(0, int(np.searchsorted(values["times"], now, side="right") - 1))
                    )
                    entry = replay.entries[index]
                    label = entry["label"].split(" \u00b7 ")[0]
                    x = column * 512
                    ended = now > values["times"][-1] + 1 / fps
                    text(canvas, label, (x + 15, 110), 0.65)
                    if cache.get(index, (None,))[0] != frame:
                        main = cv2.resize(replay.render(index, frame, "orbit", 225, -15), (512, 384))
                        detail = cv2.resize(replay.render(index, frame, "hand", 270, -15), (176, 132))
                        cache[index] = frame, main, detail
                    _, main, detail = cache[index]
                    canvas[130:514, x : x + 512] = main
                    canvas[382:514, x + 336 : x + 512] = detail
                    text(canvas, "Hand close-up", (x + 340, 396), 0.35)
                    text(
                        canvas,
                        f"Hand error: {values['position_mm'][frame]:.2f} mm / {values['rotation_deg'][frame]:.2f} deg",
                        (x + 15, 537),
                        0.47,
                    )
                    status = (
                        f"RECORD ENDED at {values['times'][-1]:.2f}s - last active pose frozen"
                        if ended
                        else ("Initial hold (2 s)" if now < 2 else "Recorded execution")
                    )
                    text(canvas, status, (x + 15, 557), 0.4, (255, 120, 100) if ended else TEXT)
                    chart(canvas, values, frame, x, duration)
                    text(
                        canvas,
                        "Joint travel from start (deg); FK witness is not an actor input."
                        if "reference_joints_deg" in values
                        else "Hand rotation about wave axis (degrees)",
                        (x + 12, 740),
                        0.35,
                    )
                text(
                    canvas,
                    "Fault stop was confirmed separately; post-stop motion is not in these recordings.",
                    (18, 760),
                    0.47,
                )
                process.stdin.write(canvas.tobytes())
                if tick % 100 == 0:
                    print(f"VIDEO {tick}/{frame_count}", flush=True)
            process.stdin.close()
            if process.wait(timeout=60):
                raise RuntimeError("ffmpeg failed; see log")
        temporary.replace(output)
        receipt = {
            "format": "herman-replay-video-v1",
            "video": output.name,
            "fps": fps,
            "frames": frame_count,
            "duration_s": frame_count / fps,
            "resolution": [width, height],
            "hardware_motion_commands_sent": 0,
            "simulation_steps": 0,
            "motion_magnified": False,
            "ended_recordings": "last active frame frozen; not simulated stopping",
            "source_report_sha256": replay.inventory["report.json"],
            "source_artifacts_sha256": hashlib.sha256((Path(directory) / "artifacts.json").read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_files": {
                str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [__file__, Path(__file__).with_name("herman_rehearsal.py")]
            },
        }
        output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
        return receipt
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        replay.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    output = args.output or args.rehearsal.with_name(args.rehearsal.name + "-comparison.mp4")
    render_video(args.rehearsal, output, fps=args.fps)


if __name__ == "__main__":
    main()
