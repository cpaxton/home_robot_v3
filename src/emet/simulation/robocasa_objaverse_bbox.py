# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Post-process Robocasa objaverse object MJCFs to add ``reg_bbox`` geoms."""

from __future__ import annotations

import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from emet.simulation.robocasa_assets_check import robocasa_package_dir

logger = logging.getLogger(__name__)

_OBJAVERSE_SENTINEL = ("apple", "apple_0", "model.xml")


def objaverse_dir(robocasa_pkg: Path | None = None) -> Path:
    pkg = robocasa_pkg or robocasa_package_dir()
    return pkg / "models" / "assets" / "objects" / "objaverse"


def _sentinel_model(robocasa_pkg: Path | None = None) -> Path | None:
    obj_dir = objaverse_dir(robocasa_pkg)
    sentinel = obj_dir.joinpath(*_OBJAVERSE_SENTINEL)
    return sentinel if sentinel.is_file() else None


def _mjcf_has_region_default(model_xml: Path) -> bool:
    try:
        root = ET.parse(model_xml).getroot()
    except (ET.ParseError, OSError):
        return False
    for top in root.findall("default"):
        if top.get("class") == "region":
            return True
        for child in top.findall("default"):
            if child.get("class") == "region":
                return True
    return False


def _inject_region_default_classes(model_xml: Path) -> bool:
    """Insert ``default class=region/collision`` block required by robosuite MJCF loading."""
    try:
        tree = ET.parse(model_xml)
    except (ET.ParseError, OSError):
        return False
    root = tree.getroot()
    if _mjcf_has_region_default(model_xml):
        return False
    worldbody = root.find("worldbody")
    if worldbody is None:
        return False

    default_elem = ET.Element("default")
    region_default = ET.SubElement(default_elem, "default", {"class": "region"})
    ET.SubElement(
        region_default,
        "geom",
        {"group": "1", "conaffinity": "0", "contype": "0", "rgba": "0 1 0 0"},
    )
    collision_default = ET.SubElement(default_elem, "default", {"class": "collision"})
    ET.SubElement(
        collision_default,
        "geom",
        {"group": "0", "rgba": "0.5 0 0 0.5", "density": "1000.0"},
    )
    root.insert(list(root).index(worldbody), default_elem)
    tree.write(model_xml, encoding="utf-8", xml_declaration=False)
    return True


def ensure_objaverse_mjcf_defaults(objaverse_root: Path) -> int:
    """Add region/collision default classes to objaverse ``model.xml`` files. Returns count updated."""
    updated = 0
    for model_xml in objaverse_root.rglob("model.xml"):
        if "reg_bbox" not in model_xml.read_text(encoding="utf-8"):
            continue
        if _inject_region_default_classes(model_xml):
            updated += 1
    return updated


def objaverse_reg_bbox_present(robocasa_pkg: Path | None = None) -> bool:
    """True when objaverse models are post-processed (reg_bbox + MJCF default classes)."""
    sentinel = _sentinel_model(robocasa_pkg)
    if sentinel is None:
        return False
    try:
        text = sentinel.read_text(encoding="utf-8")
    except OSError:
        return False
    return "reg_bbox" in text and _mjcf_has_region_default(sentinel)


def _calc_bbox_script(robocasa_pkg: Path) -> Path | None:
    script = robocasa_pkg.parent / "robocasa" / "scripts" / "asset_scripts" / "calc_object_bb_reg.py"
    return script if script.is_file() else None


def ensure_objaverse_reg_bbox(robocasa_pkg: Path | None = None) -> bool:
    """Run Robocasa ``calc_object_bb_reg`` on objaverse if the zip was extracted without bbox geoms."""
    pkg = robocasa_pkg or robocasa_package_dir()
    obj_dir = objaverse_dir(pkg)
    if not obj_dir.is_dir():
        return False

    sentinel = _sentinel_model(pkg)
    if sentinel is not None and "reg_bbox" in sentinel.read_text(encoding="utf-8"):
        if not _mjcf_has_region_default(sentinel):
            n_defaults = ensure_objaverse_mjcf_defaults(obj_dir)
            if n_defaults:
                logger.info("Added MJCF default classes to %d objaverse model.xml file(s).", n_defaults)
        if objaverse_reg_bbox_present(pkg):
            return True

    script = _calc_bbox_script(pkg)
    if script is None:
        logger.warning("calc_object_bb_reg.py not found under %s", pkg.parent)
        return False

    logger.info(
        "Objaverse models lack reg_bbox geoms; running %s (one-time, may take several minutes)...",
        script.name,
    )
    proc = subprocess.run(
        [sys.executable, str(script), "--folder", str(obj_dir)],
        cwd=str(pkg.parent),
        check=False,
    )
    if proc.returncode != 0:
        logger.error("calc_object_bb_reg failed with exit code %s", proc.returncode)
        return False

    n_defaults = ensure_objaverse_mjcf_defaults(obj_dir)
    if n_defaults:
        logger.info("Added MJCF default classes to %d objaverse model.xml file(s).", n_defaults)

    ok = objaverse_reg_bbox_present(pkg)
    if ok:
        logger.info("Objaverse reg_bbox processing complete.")
    return ok
