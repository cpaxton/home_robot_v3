"""Immutable regression evidence; frozen TAMP cases are never training resets."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robot"
FROZEN = ASSETS / "rl_manipulation" / "stretch_frozen"
CASES = ("left", "right")
SOURCE_SHA = "0e1e2e37351103733766b80fdf0c5dfc311deebd"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_case(source, target):
    """Archive exact evidence; rewrite only asset lookup at model load time."""
    import shutil

    source, target = Path(source), Path(target)
    if json.loads((source / "protocol.json").read_text())["source_sha"] != SOURCE_SHA:
        raise ValueError("frozen source revision mismatch")
    target.mkdir(parents=True, exist_ok=True)
    files = [
        *sorted((source / "fixture").glob("*.xml")),
        source / "fixture" / "fixture.json",
        source / "execution.json",
        source / "protocol.json",
        source / "summary.json",
    ]
    manifest = {"source_sha": SOURCE_SHA, "source": str(source), "split": "frozen_regression", "files": {}}
    for path in files:
        relative = str(path.relative_to(source))
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(path):
            raise ValueError(f"refusing to overwrite frozen evidence: {relative}")
        shutil.copy2(path, destination)
        manifest["files"][relative] = sha256(destination)
    # Verify rebased assets are byte-identical to those used by the archived model.
    xml = ET.parse(source / "fixture/stretch.xml").getroot()
    old_assets = Path(xml.find("compiler").get("assetdir"))
    manifest["assets"] = {}
    for mesh in xml.findall("asset/mesh"):
        name = mesh.get("file")
        if sha256(old_assets / name) != sha256(ASSETS / "assets" / name):
            raise ValueError(f"frozen mesh differs: {name}")
        manifest["assets"][name] = sha256(ASSETS / "assets" / name)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_case(case):
    if case not in CASES:
        raise ValueError("unknown frozen case")
    directory = FROZEN / case
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source_sha"] != SOURCE_SHA or manifest["split"] != "frozen_regression":
        raise ValueError("frozen provenance mismatch")
    for relative, expected in manifest["files"].items():
        if sha256(directory / relative) != expected:
            raise ValueError(f"frozen fixture checksum mismatch: {relative}")
    for relative, expected in manifest["assets"].items():
        if sha256(ASSETS / "assets" / relative) != expected:
            raise ValueError(f"frozen mesh checksum mismatch: {relative}")
    return directory, manifest


def scene_xml(case):
    directory, _ = verify_case(case)
    root = ET.Element("mujoco", model=f"frozen_stretch_{case}")
    for name in ("stretch.xml", "docking_station.xml", "scene_environment.xml"):
        included = ET.parse(directory / "fixture" / name).getroot()
        for child in included:
            if child.tag == "compiler":
                child.set("assetdir", str(ASSETS / "assets"))
            existing = root.find(child.tag)
            if existing is not None and child.tag in {
                "asset",
                "worldbody",
                "contact",
                "actuator",
                "sensor",
                "equality",
                "tendon",
            }:
                existing.extend(list(child))
            else:
                root.append(child)
    return root


def load_case(case):
    return mujoco.MjModel.from_xml_string(ET.tostring(scene_xml(case), encoding="unicode"))


def evidence(case):
    directory, manifest = verify_case(case)
    protocol = json.loads((directory / "protocol.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    return {
        "case": case,
        "source_sha": manifest["source_sha"],
        "goals": protocol["goals"],
        "fixture": json.loads((directory / "fixture/fixture.json").read_text()),
        "events": execution["events"],
        "translation_limit_m": 0.02,
        "rotation_limit_rad": 0.1,
        "complete_physics_snapshot": False,
    }
