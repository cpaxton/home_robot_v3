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
Simple script to start the Stretch MuJoCo simulation.

Requires: pip install -e ".[sim]"

Run: python examples/run_stretch_simulation.py
     python examples/run_stretch_simulation.py --headless
     python examples/run_stretch_simulation.py --headless --cameras  # Show camera imagery
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Run Stretch MuJoCo simulation")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (no viewer window)",
    )
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="Show camera imagery (requires OpenCV window, not headless)",
    )
    args = parser.parse_args()

    if args.cameras and args.headless:
        print("Warning: --cameras has no effect in headless mode")

    try:
        from emet.simulation.stretch_mujoco import StretchMujocoSimulator
        from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
    except ImportError as e:
        print("Error: Could not import simulation modules.")
        print('Install with: pip install -e ".[sim]"')
        print(f"  {e}")
        sys.exit(1)

    # No cameras in headless mode - renderers require OpenGL/DISPLAY
    cameras = StretchCameras.all() if (args.cameras and not args.headless) else []

    print("Starting Stretch MuJoCo simulation...")
    print("  (Press Ctrl+C to stop)\n")

    sim = StretchMujocoSimulator(
        scene_xml_path=None,  # Use default
        cameras_to_use=cameras,
    )
    sim.start(headless=args.headless)

    try:
        if args.cameras and not args.headless:
            import cv2

            while sim.is_running():
                camera_data = sim.pull_camera_data()
                if camera_data:
                    for cam in cameras:
                        img = camera_data.get_camera_data(cam)
                        if img is not None:
                            cv2.imshow(cam.name, img)
                if cv2.waitKey(10) == ord("q"):
                    break
            cv2.destroyAllWindows()
        else:
            while sim.is_running():
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sim.stop()
        print("Simulation stopped.")


if __name__ == "__main__":
    main()
