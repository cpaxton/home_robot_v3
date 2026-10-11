#!/usr/bin/env python3
"""Record native Stretch actuator motion, explicitly not a placement benchmark.

CPU render example:
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 \
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  PYTHONPATH=src python scripts/render_robot_model_diagnostic.py --output /tmp/stretch-model
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
from OpenGL import GL

from emet.eval.episode_video import write_rgb_sequence_mp4
from emet.simulation.mujoco_server import _load_default_scene_with_robot
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators
from emet.visualization.manip_video import overlay_manip_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = _load_default_scene_with_robot("stretch")
    if model is None:
        raise RuntimeError("Stretch model unavailable")
    data = mujoco.MjData(model)
    if model.nkey:
        # Merged-scene key_qpos includes different freejoint offsets. Preserve
        # scene poses and initialize the robot from its named actuator controls.
        data.ctrl[:] = model.key_ctrl[0]
        snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    targets = {"lift": .7, "arm": .30, "wrist_pitch": 0., "gripper": .03}
    for name, target in targets.items():
        data.ctrl[model.actuator(name).id] = target
    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0, -.25, .65]
        camera.distance, camera.azimuth, camera.elevation = 2.4, 145, -20
        for frame in range(96):
            for _ in range(max(1, round(1 / (12 * model.opt.timestep)))):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=camera)
            clean = renderer.render()
            if frame in (0, 95):
                cv2.imwrite(str(args.output / f"{frame:04d}.png"), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
            frames.append(overlay_manip_frame(clean, title="Stretch | model motion diagnostic",
                action="lift + extend", goal="inspect connected arm and floor",
                flags="MUJOCO ACTUATORS | not a placement test", overlay_style="border"))
    video = write_rgb_sequence_mp4(frames, args.output / "stretch-model-motion.mp4", fps=12)
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": 1, "robot": "stretch",
        "track": "model_actuation_diagnostic", "placement_tested": False, "renderer": renderer_name,
        "commanded_actuators": targets, "initial_qpos": initial_qpos.tolist(), "final_qpos": data.qpos.tolist(),
        "video": str(video)}, indent=2) + "\n")
    print(video)


if __name__ == "__main__":
    main()
