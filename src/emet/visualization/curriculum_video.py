"""Render actual teacher recordings as real-time 4x4 videos; never step physics."""

import argparse
import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.visualization.curriculum_cases import FAMILIES, TITLES
from emet.visualization.curriculum_record import sha

CELL_W, CELL_H, SCENE_H, HEADER = 384, 288, 240, 80
BG, TEXT, CYAN, ORANGE = (16, 24, 34), (225, 235, 245), (60, 230, 250), (255, 170, 70)


def label(image, value, x, y, scale=0.36, color=TEXT):
    cv2.putText(image, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def encoder(path, width, height, fps):
    return subprocess.Popen(
        [
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
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def pose_rotation(pose):
    c = np.asarray(pose)[3:].reshape(3, 2)
    return np.column_stack([c, np.cross(c[:, 0], c[:, 1])])


def segment(scene, start, end, color, width=0.0015, arrow=False):
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("curriculum marker budget exhausted")
    geom = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.asarray(start), np.eye(3).ravel(), color)
    mujoco.mjv_connector(geom, kind, width, np.asarray(start), np.asarray(end))
    scene.ngeom += 1


def axes(scene, position, rotation, color):
    for k in range(3):
        segment(scene, position, position + rotation[:, k] * 0.045, color, arrow=True)


def frame_index(times, now):
    # Physics timestamps accumulate floating-point roundoff. At native 20 Hz,
    # an exact display tick must not turn into a repeated/then-skipped state.
    return max(0, min(len(times) - 1, int(np.searchsorted(times, now + 1e-9, side="right") - 1)))


def tracking_readout(values, index, family):
    if family == "base":
        delta = values["base_target"][index] - values["base"][index]
        angle = float(abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))))
        return "BASE", float(np.linalg.norm(delta[:2])) * 1000, float(np.rad2deg(angle))
    hand, goal = values["hand"][index], values["hand_target"][index]
    position = float(np.linalg.norm(hand[:3] - goal[:3])) * 1000
    angle = float(Rotation.from_matrix(pose_rotation(hand) @ pose_rotation(goal).T).magnitude())
    return "EE", position, float(np.rad2deg(angle))


def motion_readout(values, index):
    """Measured finite-interval speeds, in deg/s and cm/s, without interpolation."""
    if index == 0:
        return None
    dt = float(values["time_s"][index] - values["time_s"][index - 1])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive recorded interval required for measured speed")
    joints = (values["joints"][index] - values["joints"][index - 1]) / dt
    hand = (values["hand"][index, :3] - values["hand"][index - 1, :3]) / dt
    return float(np.rad2deg(np.max(abs(joints)))), float(100 * np.linalg.norm(hand))


def render_cell(directory, row, index, path, *, fps, frame_count, loop_shorter=False):
    process = encoder(path, CELL_W, CELL_H, fps)
    renderer = None
    motion = {"recorded": False, "joint_span_deg": row.get("measured_joint_span_deg")}
    try:
        if row.get("recording"):
            for name, digest in row["assets"].items():
                if sha(directory / name) != digest:
                    raise ValueError("recording changed before rendering")
            with np.load(directory / row["recording"], allow_pickle=False) as archive:
                values = {k: archive[k].copy() for k in archive.files}
            with tempfile.NamedTemporaryFile(suffix=".mjb", dir=directory) as temporary:
                with gzip.open(directory / row["model"], "rb") as source:
                    shutil.copyfileobj(source, temporary)
                temporary.flush()
                if sha(temporary.name) != row["model_raw_sha256"]:
                    raise ValueError("expanded model differs from recorded model")
                model = mujoco.MjModel.from_binary_path(temporary.name)
            if values["qpos"].shape != (len(values["time_s"]), model.nq):
                raise ValueError("recorded qpos layout differs from model")
            data = mujoco.MjData(model)
            model.vis.headlight.active = 1
            model.vis.headlight.ambient[:] = 0.4
            model.vis.headlight.diffuse[:] = 0.7
            renderer = mujoco.Renderer(model, height=SCENE_H, width=CELL_W)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0.07, 0, 0.16]
            camera.distance = (
                0.70 if directory.name.startswith("arm_splines") or directory.name == "dense_arm" else 0.90
            )
            camera.azimuth, camera.elevation = 225, -25
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            motion.update(
                recorded=True,
                base_xy_span_m=np.ptp(values["base"][:, :2], axis=0).tolist(),
                base_yaw_span_rad=float(np.ptp(np.unwrap(values["base"][:, 2]))),
            )
        cached = None
        previous = None
        for tick in range(frame_count):
            now = tick / fps
            canvas = np.full((CELL_H, CELL_W, 3), BG, dtype=np.uint8)
            case = row["case"]
            split = case.get("split", "unknown").upper()
            suffix = case.get("posture", case.get("mode", case.get("layout", "")))
            label(canvas, f"{index + 1:02d} {split} | {suffix}", 8, 15)
            label(canvas, f"seed {case['seed']}", 220, 15, 0.30)
            if not row.get("recording"):
                label(canvas, "REJECTED / NO ROLLOUT", 15, 120, 0.5, (255, 100, 100))
                label(canvas, str(row.get("code", "not admitted"))[:55], 10, 145, 0.3)
            else:
                duration = float(values["time_s"][-1])
                looping = bool(loop_shorter and row["success"] and duration > 0)
                replay_number = int(now // duration) + 1 if looping else 1
                local_time = now % duration if looping else now
                selected = frame_index(values["time_s"], local_time)
                if previous != selected:
                    data.qpos[:] = values["qpos"][selected]
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    goal = values["hand_target"][selected]
                    body = data.body("ee_link")
                    axes(renderer.scene, body.xpos.copy(), body.xmat.copy().reshape(3, 3), [1, 0.65, 0.25, 1])
                    axes(renderer.scene, goal[:3], pose_rotation(goal), [0.2, 0.9, 1, 0.85])
                    base = values["base_target"][selected]
                    point = np.r_[base[:2], 0.025]
                    # The ground goal can sit underneath the chassis. A pole
                    # and elevated heading arrow preserve its XY/yaw identity
                    # while keeping it visible in the fixed world camera.
                    above = point + [0, 0, 0.33]
                    segment(renderer.scene, point, above, [0.3, 0.55, 1, 0.8])
                    segment(
                        renderer.scene,
                        above,
                        above + np.r_[0.10 * np.cos(base[2]), 0.10 * np.sin(base[2]), 0],
                        [0.3, 0.55, 1, 1],
                        arrow=True,
                    )
                    cached = renderer.render().copy()
                    previous = selected
                canvas[24:264] = cached
                speeds = motion_readout(values, selected)
                if speeds is not None:
                    cv2.rectangle(canvas, (0, 24), (CELL_W, 43), BG, -1)
                    label(canvas, f"Measured joint max {speeds[0]:.1f} deg/s | hand {speeds[1]:.1f} cm/s", 8, 38, 0.33)
                ended = not looping and now > values["time_s"][-1] + 1 / fps
                kind, position, angle = tracking_readout(values, selected, directory.name)
                detail = f"{local_time:04.1f}s {kind} {position:.1f}mm / {angle:.1f}deg"
                label(canvas, detail, 8, 281, 0.34)
                status = (
                    ("END " + str(row.get("code", "")))
                    if ended
                    else ("PASS" if row["success"] else "FAIL " + str(row.get("code", "see report")))
                )
                if replay_number > 1:
                    status = f"REPLAY {replay_number} PASS"
                label(canvas, status[:30], 220, 281, 0.27, (100, 235, 150) if row["success"] else (255, 120, 100))
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait(timeout=60):
            raise RuntimeError("tile encoder failed")
        return motion
    finally:
        if process.poll() is None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=10)
        if renderer is not None:
            renderer.close()


def render(output, family, *, fps=20, loop_shorter=False):
    # imwrite may otherwise start a full OpenCV worker pool after rendering,
    # exhausting a bounded service's task limit before ffmpeg can be spawned.
    cv2.setNumThreads(1)
    if family not in FAMILIES or type(fps) is not int or not 1 <= fps <= 20:
        raise ValueError("known family and fps in 1..20 required")
    output = Path(output)
    directory = output / family
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("format") != "curriculum-recordings-v1" or not report.get("complete") or len(report["rows"]) != 16:
        raise ValueError("complete sixteen-case recording required")
    path = output / (family + ".mp4")
    if path.exists():
        raise ValueError("fresh grid video required")
    duration = max((r.get("duration_s", 0) for r in report["rows"]), default=0)
    if duration <= 0:
        raise ValueError("no actual recordings available")
    frame_count = int(np.ceil(duration * fps)) + 1
    tiles = []
    motion = []
    try:
        for index, row in enumerate(report["rows"]):
            tile = directory / f"tile-{index:02d}.mp4"
            tiles.append(tile)
            motion.append(
                render_cell(directory, row, index, tile, fps=fps, frame_count=frame_count, loop_shorter=loop_shorter)
            )
            print("RENDER", family, index, flush=True)
        head = np.full((HEADER, CELL_W * 4, 3), BG, dtype=np.uint8)
        label(head, "MARS | " + TITLES[family], 15, 26, 0.65)
        label(
            head,
            f"ACTUAL ANALYTIC-TEACHER EXECUTIONS | {fps} fps | 1x simulation time | 16 distinct case/seed identities | no hardware motion",
            15,
            49,
            0.43,
        )
        label(
            head,
            (
                "Cyan: hand target | orange: measured | blue: base. Short successful records REPLAY with a visible reset; failures stay frozen."
                if loop_shorter
                else "Cyan: hand target | orange: measured hand | blue: base target. Shorter records freeze at their final recorded state. Failures retained."
            ),
            15,
            69,
            0.39,
        )
        header = directory / "header.png"
        cv2.imwrite(str(header), cv2.cvtColor(head, cv2.COLOR_RGB2BGR))
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
        for tile in tiles:
            command += ["-threads", "1", "-i", str(tile)]
        command += ["-loop", "1", "-i", str(header)]
        layout = "|".join(f"{i % 4 * CELL_W}_{i // 4 * CELL_H}" for i in range(16))
        filters = f"{''.join(f'[{i}:v]' for i in range(16))}xstack=inputs=16:layout={layout}[tiles];[tiles]pad={4 * CELL_W}:{4 * CELL_H + HEADER}:0:{HEADER}:color=0x101822[padded];[padded][16:v]overlay=0:0:shortest=1[out]"
        partial = output / (family + ".partial.mp4")
        command += [
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-t",
            str(frame_count / fps),
            "-r",
            str(fps),
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
            str(partial),
        ]
        subprocess.run(command, check=True)
        partial.replace(path)
        webm = path.with_suffix(".webm")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-threads",
                "1",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libvpx-vp9",
                "-threads",
                "1",
                "-row-mt",
                "0",
                "-deadline",
                "realtime",
                "-cpu-used",
                "6",
                "-crf",
                "34",
                "-b:v",
                "0",
                str(webm),
            ],
            check=True,
        )
        poster = path.with_suffix(".jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-threads",
                "1",
                "-ss",
                str(min(5, duration / 2)),
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-threads",
                "1",
                str(poster),
            ],
            check=True,
        )
        receipt = {
            "format": "curriculum-video-v1",
            "family": family,
            "title": TITLES[family],
            "controller": "analytic_teacher",
            "fps": fps,
            "frames": frame_count,
            "duration_s": frame_count / fps,
            "resolution": [4 * CELL_W, 4 * CELL_H + HEADER],
            "time_scale": 1,
            "shorter_recordings": "replay_success_with_visible_reset" if loop_shorter else "freeze_final_state",
            "hardware_used": False,
            "simulation_steps_during_render": 0,
            "source_report_sha256": sha(report_path),
            "producer_sha256": sha(__file__),
            "motion": motion,
            "summary": report["summary"],
            "assets": {p.name: sha(p) for p in (path, webm, poster)},
        }
        save(output / (family + "-video.json"), receipt)
        return receipt
    finally:
        for tile in tiles:
            tile.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--loop-shorter",
        action="store_true",
        help="replay shorter successful records, clearly labelled; failed records freeze",
    )
    args = parser.parse_args()
    render(args.output, args.family, fps=args.fps, loop_shorter=args.loop_shorter)


if __name__ == "__main__":
    main()
