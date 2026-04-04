#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""
Load the Stretch MuJoCo model and optionally start the simulation.

Requires: pip install -e ".[sim]"  (mujoco, etc.)

Run: python examples/load_stretch_mujoco.py
     python examples/load_stretch_mujoco.py --run   # Start simulation with viewer
     python examples/load_stretch_mujoco.py --run --headless  # Headless mode
"""

import argparse
import sys


def test_load_model():
    """Load the MuJoCo model without starting the simulation."""
    try:
        import mujoco
    except ImportError:
        print('Error: mujoco not installed. Run: pip install -e ".[sim]"')
        return False

    from emet.utils.assets import get_mujoco_models_path

    models_path = get_mujoco_models_path()
    scene_xml = models_path / "scene.xml"

    if not scene_xml.exists():
        print(f"Error: scene.xml not found at {scene_xml}")
        return False

    print(f"Loading MuJoCo model from {scene_xml}")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    _ = mujoco.MjData(model)

    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print("✓ Model loaded successfully")
    return True


def run_simulation(headless: bool = False):
    """Start the Stretch MuJoCo simulation."""
    try:
        from emet.simulation.stretch_mujoco import StretchMujocoSimulator
        from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
    except ImportError as e:
        print(f'Error: Could not import simulation. Run: pip install -e ".[sim]"\n  {e}')
        return

    print("Starting Stretch MuJoCo simulation...")
    print("  (Press Ctrl+C to stop)\n")

    # No cameras in headless mode - renderers require OpenGL/DISPLAY
    cameras = [] if headless else [StretchCameras.cam_d435i_rgb]

    sim = StretchMujocoSimulator(
        scene_xml_path=None,  # Use default
        cameras_to_use=cameras,
    )
    sim.start(headless=headless)

    try:
        while sim.is_running():
            status = sim.pull_status()
            if status:
                b = status.base
                print(f"\rBase: x={b.x:.2f} y={b.y:.2f} θ={b.theta:.2f} rad", end="")
            import time

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sim.stop()
        print("Simulation stopped.")


def main():
    parser = argparse.ArgumentParser(description="Load Stretch MuJoCo model")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start the simulation (otherwise just load the model)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulation without GUI (only with --run)",
    )
    args = parser.parse_args()

    if args.run:
        run_simulation(headless=args.headless)
    else:
        success = test_load_model()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
