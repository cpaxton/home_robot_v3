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
Create minimal MuJoCo placeholder model.xml for any fixtures/accessories path
that is referenced in the registry but missing on disk. Lets emet serve mujoco
--use-robocasa work when the full lightwheel fixture pack was not extracted
(e.g. zip had a different structure or some assets are missing).
"""

import os
import sys

try:
    import robocasa
    import yaml
except ImportError:
    print(
        "Need robocasa and pyyaml. Run from project env: uv run python scripts/ensure_robocasa_placeholders.py",
        file=sys.stderr,
    )
    sys.exit(1)

PLACEHOLDER_XML = """<mujoco model="placeholder">
  <compiler angle="radian"/>
  <default>
    <default class="visual">
      <geom conaffinity="0" contype="0" group="1"/>
    </default>
    <default class="collision">
      <geom group="0" rgba="0.6 0.5 0.3 0.5"/>
    </default>
  </default>
  <worldbody>
    <body>
      <site rgba="1 1 1 1" size="0.01" pos="0 0 0" name="ext_p0"/>
      <site rgba="1 0 0 1" size="0.01" pos="0 0 0" name="ext_px"/>
      <site rgba="0 1 0 1" size="0.01" pos="0 0 0" name="ext_py"/>
      <site rgba="0 0 1 1" size="0.01" pos="0 0 0" name="ext_pz"/>
      <site rgba="1 1 1 1" size="0.01" pos="0 0 0" name="int_p0"/>
      <site rgba="1 1 0 1" size="0.01" pos="0 0 0" name="int_px"/>
      <site rgba="0 1 1 1" size="0.01" pos="0 0 0" name="int_py"/>
      <site rgba="1 0 1 1" size="0.01" pos="0 0 0" name="int_pz"/>
      <body name="object">
        <geom class="collision" type="box" size="0.06 0.04 0.02" pos="0 0 0" friction="0.5 0.005 0.0001"/>
        <geom class="visual" type="box" size="0.06 0.04 0.02" pos="0 0 0" rgba="0.72 0.53 0.34 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    assets_root = os.path.join(robocasa.__path__[0], "models", "assets")
    registry_dir = os.path.join(assets_root, "fixtures", "fixture_registry")
    if not os.path.isdir(registry_dir):
        print("Registry dir not found:", registry_dir, file=sys.stderr)
        sys.exit(1)

    paths_to_ensure = set()
    for fname in os.listdir(registry_dir):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(registry_dir, fname)
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            continue
        for _key, val in data.items():
            if isinstance(val, dict) and "xml" in val:
                xml = val["xml"].strip()
                if xml.startswith("fixtures/"):
                    paths_to_ensure.add(xml)

    created = 0
    for xml_path in sorted(paths_to_ensure):
        # path is relative to assets_root, e.g. fixtures/accessories/knife_blocks/light_wood_3
        model_path = os.path.join(assets_root, xml_path, "model.xml")
        if os.path.isfile(model_path):
            continue
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            f.write(PLACEHOLDER_XML)
        print("Created placeholder:", model_path)
        created += 1

    if created:
        print("Created", created, "placeholder(s). Run: emet serve mujoco --robot rby1 --use-robocasa")
    else:
        print("All accessory paths already exist. Nothing to do.")


if __name__ == "__main__":
    main()
