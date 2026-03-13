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
Test that Stretch asset paths resolve correctly.

Run: python examples/load_stretch_assets.py
Or:  uv run python examples/load_stretch_assets.py
"""

from emet.utils.assets import get_assets_root, get_mujoco_models_path, get_robot_assets_path


def main():
    print("=== Stretch Asset Path Test ===\n")

    assets_root = get_assets_root()
    print(f"Assets root: {assets_root}")
    print(f"  Exists: {assets_root.exists()}\n")

    robot_path = get_robot_assets_path()
    print(f"Robot assets: {robot_path}")
    print(f"  Exists: {robot_path.exists()}\n")

    models_path = get_mujoco_models_path()
    print(f"MuJoCo models: {models_path}")
    print(f"  Exists: {models_path.exists()}")

    # Check key files
    scene_xml = models_path / "scene.xml"
    stretch_xml = models_path / "stretch.xml"
    assets_dir = models_path / "assets"

    print("\nKey files:")
    print(f"  scene.xml:     {scene_xml.exists()}")
    print(f"  stretch.xml:   {stretch_xml.exists()}")
    print(f"  assets/:       {assets_dir.exists()}")

    if assets_dir.exists():
        mesh_count = len(list(assets_dir.glob("*.obj"))) + len(list(assets_dir.glob("*.stl")))
        print(f"  mesh files:    {mesh_count}")

    print("\n✓ Asset paths OK" if scene_xml.exists() and stretch_xml.exists() else "\n✗ Some assets missing")


if __name__ == "__main__":
    main()
