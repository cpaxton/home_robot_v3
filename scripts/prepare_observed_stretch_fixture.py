#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Provision the declared development scene and robot-only MJCF for camera trials.

This constructs simulator initial conditions; generated scene/object state must
not become runtime observed evidence. All launch/manifests are JSON.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from emet.utils.assets import get_mujoco_models_path


def prepare(output, *, port_offset=3020):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = get_mujoco_models_path()
    hashes = {}
    for name in ("stretch.xml", "docking_station.xml", "scene_environment.xml", "scene.xml"):
        source = assets / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        xml = ET.parse(source)
        compiler = xml.getroot().find("compiler")
        if compiler is not None:
            compiler.set("assetdir", str((assets / "assets").resolve()))
        if name == "scene_environment.xml":
            for body, position in [("object1", "-.02 -.55 .6"), ("object2", "-.02 -.70 .6")]:
                element = xml.getroot().find(f"worldbody/body[@name='{body}']")
                if element is None:
                    raise ValueError("development_fixture_source_changed")
                element.set("pos", position)
        xml.write(output / name, encoding="unicode")
    config = {
        "kind": "default_mujoco",
        "robot": "stretch",
        "headless": True,
        "scene_path": str(output / "scene.xml"),
        "port_offset": port_offset,
        "seed": 0,
        "no_cameras": False,
    }
    (output / "sim.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "tier": "development_fixture_not_held_out",
        "source_sha256": hashes,
        "sim_config": "sim.json",
        "robot_only_model": "stretch.xml",
        "note": "Initial-condition construction and independent evaluator inputs; not runtime object observations.",
    }
    (output / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-offset", type=int, default=3020)
    args = parser.parse_args()
    if args.port_offset < 0:
        parser.error("--port-offset must be nonnegative")
    print(json.dumps(prepare(args.output, port_offset=args.port_offset), indent=2))


if __name__ == "__main__":
    main()
