#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
import random
import sys
import time

import click
import numpy as np
import zmq

# Robocasa is imported lazily when --use-robocasa is used, to avoid loading robosuite/numba
# on every server start (and to avoid numba init failures when not using Robocasa).
model_generation_wizard = None
_ROBOCASA_IMPORT_FAILED = True

import emet.utils.logger as logger
from emet.simulation.scene_resolution import (
    build_zmq_environment,
    default_packaged_stretch_scene_xml_path,
    resolve_merged_physics_scene,
    scene_source_basename_from_path,
)
from emet.utils.port_utils import kill_processes_on_port

default_scene_xml_path = default_packaged_stretch_scene_xml_path()

# Stretch-specific server (MujocoZmqServer) and motion/pinocchio deps are imported only when
# --robot stretch, so that emet serve mujoco --robot rby1 works without pinocchio/hppfcl.


def _get_stretch_server():
    """Lazy import of the Stretch MuJoCo ZMQ server (avoids loading pinocchio for non-stretch robots)."""
    from emet.simulation.mujoco_server_stretch import MujocoZmqServer

    return MujocoZmqServer


def _load_default_scene_with_robot(robot_key: str):
    """Backward-compatible wrapper; see :func:`emet.simulation.scene_resolution.load_default_scene_with_robot`."""
    from emet.simulation.scene_resolution import load_default_scene_with_robot

    return load_default_scene_with_robot(robot_key)


@click.command()
@click.option(
    "--port-offset",
    default=0,
    type=int,
    help="Add this to all port numbers (e.g. 100 → 4501,4502,4503,4504). Use when default ports are in use.",
)
@click.option("--send_port", default=None, help="Port to send messages to clients (default 4401 + port-offset)")
@click.option("--recv_port", default=None, help="Port to receive messages from clients (default 4402 + port-offset)")
@click.option("--send_state_port", default=None, help="Port for state-only messages (default 4403 + port-offset)")
@click.option("--send_servo_port", default=None, help="Port for visual servoing images (default 4404 + port-offset)")
@click.option("--use_remote_computer", default=True, help="Whether to use a remote computer")
@click.option("--verbose", default=False, help="Whether to print verbose messages", is_flag=True)
@click.option("--image_scaling", default=1.0, help="Scaling factor for images")
@click.option("--ee_image_scaling", default=1.0, help="Scaling factor for end-effector images")
@click.option("--depth_scaling", default=0.001, help="Scaling factor for depth images")
@click.option(
    "--scene_path",
    default=None,
    help=(
        "Stretch: MJCF path (default: packaged scene.xml). "
        "Non-stretch (e.g. rby1): optional path to a merged MJCF (e.g. MolmoSpaces scene + robot); "
        "if omitted, uses the default table scene + robot."
    ),
)
@click.option(
    "--use-robocasa",
    "--use_robocasa",
    default=False,
    help="Use robocasa for generating a scene",
    is_flag=True,
)
@click.option(
    "--robocasa-task",
    "--robocasa_task",
    default="PickPlaceCounterToCabinet",
    help="Robocasa task name (e.g. PickPlaceCounterToCabinet, OpenCabinet). Use --list-robocasa-tasks to show all.",
)
@click.option("--robocasa-style", "--robocasa_style", type=int, default=1, help="Robocasa style to generate")
@click.option(
    "--robocasa-layout",
    "--robocasa_layout",
    type=int,
    default=1,
    help="Robocasa layout to generate",
)
@click.option(
    "--show-viewer-ui",
    "--show_viewier_ui",
    default=False,
    help="Show the MuJoCo passive viewer (Stretch always; rby1 / merged MJCF when not --headless)",
    is_flag=True,
)
@click.option("--headless", default=False, help="Run the simulation headless", is_flag=True)
@click.option(
    "--no-cameras",
    "--no_cameras",
    default=False,
    is_flag=True,
    help="Disable camera rendering (use on WSL/headless when EGL camera init hangs)",
)
@click.option(
    "--use-glx",
    "--use_glx",
    default=False,
    is_flag=True,
    help="Use GLX instead of EGL for rendering (use with Xvfb on WSL to get camera images)",
)
@click.option(
    "--molmospaces-session-scene",
    default=None,
    help="Internal: scene id for emet_session when using MolmoSpaces merge via CLI.",
)
@click.option(
    "--molmospaces-session-split",
    default=None,
    help="Internal: split for emet_session (train/val/test).",
)
@click.option(
    "--molmospaces-session-index",
    default=None,
    type=int,
    help="Internal: scene index for emet_session.",
)
@click.option("--seed", default=0, help="Seed for the simulation")
@click.option(
    "--steps",
    default=None,
    type=int,
    metavar="N",
    help="Exit after N MuJoCo physics steps (debug; non-stretch / rby1 server).",
)
@click.option(
    "--debug-molmospaces-spawn",
    default=False,
    is_flag=True,
    help="Verbose MolmoSpaces spawn placement logs and contact summary after base move.",
)
@click.option(
    "--robocasa-write-to-xml",
    default=False,
    help="Write the generated scene to an xml file",
    is_flag=True,
)
@click.option(
    "--list-robocasa-tasks",
    "list_robocasa_tasks",
    is_flag=True,
    help="Print registered Robocasa task names and exit (use with --use-robocasa env).",
)
@click.option(
    "--robot",
    default="stretch",
    help=(
        "Robot to simulate. Default Stretch uses merged table + StretchRobosuiteZmqServer. "
        "Use --stretch-legacy or Robocasa for the legacy StretchMujocoSimulator stack. "
        "'rby1' / 'galaxea_r1' / 'innate_mars' load the default table + generic ZMQ sim. "
        "Robosuite-native names (PandaOmron, Tiago, GR1) keep the robosuite robot in Robocasa."
    ),
)
@click.option(
    "--stretch-legacy",
    "stretch_legacy",
    default=False,
    is_flag=True,
    help=(
        "Stretch only: use the legacy StretchMujocoSimulator ZMQ stack (packaged scene.xml or "
        "Robocasa-generated MJCF) instead of the default merged table + StretchRobosuiteZmqServer."
    ),
)
@click.option(
    "--kinematic-sim",
    "kinematic_sim",
    default=False,
    is_flag=True,
    help=(
        "Kinematic sim: mj_forward-only stepping with base pose snaps (merged MJCF path). "
        "Stretch legacy stack: threads the same mode into StretchMujocoSimulator. "
        "Default Stretch (merged table) uses RobosuiteZmqServer kinematics."
    ),
)
def main(
    port_offset: int,
    send_port: int | None,
    recv_port: int | None,
    send_state_port: int | None,
    send_servo_port: int | None,
    use_remote_computer: bool,
    verbose: bool,
    image_scaling: float,
    ee_image_scaling: float,
    depth_scaling: float,
    scene_path: str | None,
    use_robocasa: bool,
    robocasa_task: str,
    robocasa_style: int,
    robocasa_layout: int,
    robocasa_write_to_xml: bool,
    show_viewer_ui: bool,
    headless: bool = False,
    no_cameras: bool = False,
    use_glx: bool = False,
    seed: int = 0,
    list_robocasa_tasks: bool = False,
    robot: str = "stretch",
    molmospaces_session_scene: str | None = None,
    molmospaces_session_split: str | None = None,
    molmospaces_session_index: int | None = None,
    steps: int | None = None,
    debug_molmospaces_spawn: bool = False,
    kinematic_sim: bool = False,
    stretch_legacy: bool = False,
):
    # Use EGL for offscreen rendering when headless (or no DISPLAY) to avoid GLFW X11 assertion
    # (_glfwGrabErrorHandlerX11: errorHandler == NULL) in headless/CI/no-window-manager setups
    if "MUJOCO_GL" not in os.environ and (headless or not os.environ.get("DISPLAY")):
        os.environ["MUJOCO_GL"] = "egl"

    # Fail fast if ROS or another layout shadows ``cv2`` (missing resize/imencode breaks image threads).
    from emet.utils.opencv_import import assert_cv2_is_real_opencv

    assert_cv2_is_real_opencv()

    from emet.simulation.molmospaces_config import ensure_molmospaces_assets_dir_env

    ensure_molmospaces_assets_dir_env()

    if debug_molmospaces_spawn:
        os.environ["EMET_MOLMOSPACES_SPAWN_DEBUG"] = "1"

    if list_robocasa_tasks:
        try:
            from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

            tasks = sorted(ALL_KITCHEN_ENVIRONMENTS)
            print("Robocasa tasks (use with --robocasa-task <name>):")
            for t in tasks:
                print(f"  {t}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Could not list Robocasa tasks: {e}")
            sys.exit(1)

    # MuJoCo is optional in pyproject unless ``sim`` extra is installed (avoid long import chains).
    from emet.utils.mujoco_import import assert_mujoco_available

    assert_mujoco_available()

    scene_model = None
    objects_info = None

    from emet.utils.port_utils import get_ports

    ports = get_ports(port_offset)
    send_port = send_port if send_port is not None else ports.send
    recv_port = recv_port if recv_port is not None else ports.recv
    send_state_port = send_state_port if send_state_port is not None else ports.state
    send_servo_port = send_servo_port if send_servo_port is not None else ports.servo

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    zmq_environment = build_zmq_environment(
        molmospaces_session_scene=molmospaces_session_scene,
        molmospaces_session_split=molmospaces_session_split,
        molmospaces_session_index=molmospaces_session_index,
        use_robocasa=use_robocasa,
        robocasa_task=robocasa_task,
        robocasa_style=robocasa_style,
        robocasa_layout=robocasa_layout,
    )

    scene_source_basename = scene_source_basename_from_path(scene_path)

    use_stretch = robot.lower() in ("stretch", "hello_stretch", "hellostretch")
    stretch_legacy_stack = use_stretch and (stretch_legacy or use_robocasa)

    if use_robocasa:
        # Lazy import so we only load robosuite/numba when actually using Robocasa.
        wizard = model_generation_wizard
        if wizard is None:
            try:
                from emet.simulation.stretch_mujoco.robocasa_gen import (
                    model_generation_wizard as _wizard,
                )

                wizard = _wizard
                globals()["model_generation_wizard"] = _wizard
                globals()["_ROBOCASA_IMPORT_FAILED"] = False
            except Exception as e:
                wizard = None
                err_type = type(e).__name__
                err_msg = str(e) if e else "unknown"
                logger.error(
                    "\n" + "=" * 60 + "\n"
                    "  Robocasa scene generation could not be loaded.\n"
                    "  You passed --use-robocasa but robosuite/robocasa failed to import.\n\n"
                    f"  {err_type}: {err_msg}\n\n"
                    f"  (Python: {sys.executable})\n\n"
                    "  If you see 'initialization of _internal failed' (numba), run:\n"
                    "    uv sync   (or: emet sync)\n"
                    "  to install a numba version compatible with numpy in this project.\n\n"
                    "  Otherwise ensure Robocasa is installed:\n"
                    "    1. emet install sim   (clones third_party/robosuite and robocasa)\n"
                    "    2. emet sync --extra sim   (installs into the project env)\n"
                    "  Then run: emet serve mujoco --use-robocasa\n" + "=" * 60,
                )
                sys.exit(1)
        if wizard is None:
            logger.error(
                "\n" + "=" * 60 + "\n"
                "  Robocasa scene generation is not installed.\n"
                "  You passed --use-robocasa but robocasa is missing or failed to load in this env.\n\n"
                "  From the project root, run:\n"
                "    1. emet install sim        (clones third_party/robosuite and robocasa)\n"
                "    2. emet sync --extra sim   (installs into the project env; use same env as 'emet serve')\n"
                "  Then run: emet serve mujoco --use-robocasa\n\n"
                "  Run 'emet' from the project directory so it uses the project .venv (or activate that venv first).\n"
                + "="
                * 60,
            )
            sys.exit(1)
        try:
            scene_model, scene_xml, objects_info = wizard(
                task=robocasa_task,
                style=robocasa_style,
                layout=robocasa_layout,
                write_to_file=scene_path,
                robot=robot,
            )
        except FileNotFoundError as e:
            msg = (
                "\n" + "=" * 60 + "\n"
                "  Robocasa kitchen assets are missing or incomplete.\n"
                "  Scene generation failed when loading:\n"
                "    {}\n\n"
                "  Download the full fixture pack (~10GB) by running from the project root:\n"
                "    ./scripts/install_simulation.sh\n"
                "  (do not use -n; when prompted, allow the asset download).\n\n"
                "  Or run only the asset download:\n"
                "    uv run python -m robocasa.scripts.download_kitchen_assets\n"
                "  and choose to download 'fixtures_lw' (lightwheel fixtures).\n\n"
                "  See docs/simulation.md for details.\n" + "=" * 60
            )
            logger.error(msg.format(e.filename))
            sys.exit(1)

    if stretch_legacy_stack and use_stretch and (scene_path is None or len(str(scene_path).strip()) == 0):
        scene_path = default_scene_xml_path

    # Only relevant when we fall back to generated/default scenes, not when --scene_path is set (e.g. MolmoSpaces).
    if _ROBOCASA_IMPORT_FAILED and not (scene_path and str(scene_path).strip()):
        logger.warning(
            "Robocasa scene generation (--use-robocasa) is not available. "
            "Using default scene. To enable: emet install sim  then  uv sync --extra sim  (or: emet sync -e sim)",
        )

    # Free server ports so we can bind (e.g. kill previous mujoco_server).
    for p in (send_port, recv_port, send_state_port, send_servo_port):
        if kill_processes_on_port(p):
            logger.warning(f"Freed port {p} (killed previous process).")
    time.sleep(0.5)

    physics_mode = "kinematic" if kinematic_sim else "dynamic"

    if use_stretch and not stretch_legacy_stack:
        from emet.simulation.stretch_robosuite_server import StretchRobosuiteZmqServer
        from emet.simulation.stretch_zmq_spec import get_stretch_robosuite_mjcf_spec

        robot_spec = get_stretch_robosuite_mjcf_spec()
        try:
            loaded = resolve_merged_physics_scene(
                robot_key="stretch",
                scene_path=scene_path,
                use_robocasa=False,
                wizard_scene_model=None,
                wizard_scene_xml=None,
                wizard_objects_info=None,
                zmq_environment=zmq_environment,
                scene_source_basename=scene_source_basename,
            )
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(str(e))
            sys.exit(1)
        scene_model = loaded.scene_model

        try:
            if "MUJOCO_GL" not in os.environ and not use_glx and not show_viewer_ui:
                os.environ["MUJOCO_GL"] = "egl"

            server = StretchRobosuiteZmqServer(
                robot_spec=robot_spec,
                send_port=send_port,
                recv_port=recv_port,
                send_state_port=send_state_port,
                send_servo_port=send_servo_port,
                use_remote_computer=use_remote_computer,
                verbose=verbose,
                scene_xml=None,
                scene_model=scene_model,
                environment=loaded.zmq_environment,
                scene_source_basename=loaded.scene_source_basename,
                max_sim_steps=steps,
                debug_molmospaces_spawn=debug_molmospaces_spawn,
                scene_disk_path=loaded.scene_disk_path,
                physics_mode=physics_mode,
            )
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                logger.error(f"\nPort {send_port} is already in use. Use --port-offset or emet kill-mujoco-server.\n")
            raise

        try:
            try:
                server.start(robocasa=False, headless=headless, show_viewer_ui=show_viewer_ui)
            finally:
                try:
                    server.stop()
                except Exception:
                    pass
        except KeyboardInterrupt:
            try:
                server.stop()
            except Exception:
                pass

    elif stretch_legacy_stack:
        MujocoZmqServer = _get_stretch_server()
        try:
            server = MujocoZmqServer(
                send_port,
                recv_port,
                send_state_port,
                send_servo_port,
                use_remote_computer=use_remote_computer,
                verbose=verbose,
                image_scaling=image_scaling,
                ee_image_scaling=ee_image_scaling,
                depth_scaling=depth_scaling,
                scene_path=scene_path,
                scene_model=scene_model,
                objects_info=objects_info,
                no_cameras=no_cameras,
                environment=zmq_environment,
                scene_source_basename=scene_source_basename,
                physics_mode=physics_mode,
            )
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                logger.error(
                    f"\nPort {send_port} (or another server port) is already in use.\n\n"
                    f"Option 1 – free the port:\n"
                    f"  kill $(lsof -t -i:{send_port})\n"
                    f"  # or: pkill -f mujoco_server\n\n"
                    f"Option 2 – use different ports (e.g. 4501–4504):\n"
                    f"  emet serve mujoco --port-offset 100\n\n"
                    f"Option 3 – stop the server then retry:\n"
                    f"  emet kill-mujoco-server\n",
                )
            raise

        try:
            server.start(
                show_viewer_ui=show_viewer_ui,
                robocasa=use_robocasa,
                headless=headless,
                use_glx=use_glx,
            )
        except KeyboardInterrupt:
            if hasattr(server, "robot_sim") and server.robot_sim is not None:
                server.robot_sim.stop()
    else:
        # Non-stretch robot: use RobosuiteZmqServer
        from emet.robots import ROBOT_REGISTRY
        from emet.simulation.robosuite_server import RobosuiteZmqServer

        robot_key = robot.lower().replace("-", "_")
        try:
            loaded = resolve_merged_physics_scene(
                robot_key=robot_key,
                scene_path=scene_path,
                use_robocasa=use_robocasa,
                wizard_scene_model=scene_model,
                wizard_scene_xml=scene_xml,
                wizard_objects_info=objects_info,
                zmq_environment=zmq_environment,
                scene_source_basename=scene_source_basename,
            )
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(str(e))
            sys.exit(1)

        if robot_key in ROBOT_REGISTRY:
            import importlib

            mod = importlib.import_module(ROBOT_REGISTRY[robot_key])
            backend_cls = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and hasattr(attr, "get_spec") and attr_name != "RobotBackend":
                    backend_cls = attr
                    break
            if backend_cls is None:
                logger.error(f"No RobotBackend found in {ROBOT_REGISTRY[robot_key]}")
                sys.exit(1)
            robot_spec = backend_cls().get_spec()
        else:
            logger.error(
                f"Unknown robot '{robot}'. Known robots: {list(ROBOT_REGISTRY.keys())}.\n"
                f"Robosuite-native robots can also be used directly."
            )
            sys.exit(1)

        try:
            # RobosuiteZmqServer renders from multiple ZMQ threads; EGL avoids GLFW issues headless.
            # With --show-viewer-ui, leave MUJOCO_GL unset (or set MUJOCO_GL=glfw) so the passive viewer can open a window.
            if "MUJOCO_GL" not in os.environ and not use_glx and not show_viewer_ui:
                os.environ["MUJOCO_GL"] = "egl"

            server = RobosuiteZmqServer(
                robot_spec=robot_spec,
                send_port=send_port,
                recv_port=recv_port,
                send_state_port=send_state_port,
                send_servo_port=send_servo_port,
                use_remote_computer=use_remote_computer,
                verbose=verbose,
                scene_xml=loaded.scene_xml if use_robocasa else None,
                scene_model=None if use_robocasa else loaded.scene_model,
                environment=loaded.zmq_environment,
                scene_source_basename=loaded.scene_source_basename,
                max_sim_steps=steps,
                debug_molmospaces_spawn=debug_molmospaces_spawn,
                scene_disk_path=loaded.scene_disk_path,
                physics_mode=physics_mode,
            )
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                logger.error(f"\nPort {send_port} is already in use. Use --port-offset or emet kill-mujoco-server.\n")
            raise

        try:
            try:
                server.start(robocasa=use_robocasa, headless=headless, show_viewer_ui=show_viewer_ui)
            finally:
                try:
                    server.stop()
                except Exception:
                    pass
        except KeyboardInterrupt:
            try:
                server.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
