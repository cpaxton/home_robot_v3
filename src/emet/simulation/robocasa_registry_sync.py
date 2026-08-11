# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Sync LightWheel fixture directories into Robocasa ``fixture_registry/*.yaml``.

The ``fixtures_lw`` zip adds mesh folders (e.g. ``fixtures/sinks/Sink025/``) but does not
always update the registry YAML. Kitchen style files reference ``Sink025`` via the registry,
so we add ``<name>: {xml: fixtures/<category>/<name>}`` entries when ``model.xml`` exists.

Accessory styles (utensil racks, knife blocks, plants, …) live under
``objects/lightwheel/<category>/`` and need the same treatment — kitchen style YAMLs
reference ids like ``UtensilRack009`` that are absent from the slim vendored registries.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from emet.simulation.robocasa_assets_check import robocasa_package_dir
from emet.utils.logger import Logger

logger = Logger(__name__)

# Registry YAML stems required for default kitchen layouts (often deleted by fixtures_lw zip).
REQUIRED_REGISTRY_STEMS: tuple[str, ...] = (
    "fridge_bottom_freezer",
    "fridge_french_door",
    "fridge_side_by_side",
    "stove_wide",
    "dish_rack",
    "toaster_oven",
    "stand_mixer",
    "electric_kettle",
)

_REGISTRY_GIT_PATH = "robocasa/models/assets/fixtures/fixture_registry/"

# registry yaml stem -> assets subdirectory under fixtures/
_REGISTRY_TO_FOLDER: dict[str, str] = {
    "sink": "sinks",
    "stove": "stoves",
    "stove_wide": "stoves",
    "microwave": "microwaves",
    "oven": "ovens",
    "dishwasher": "dishwashers",
    "coffee_machine": "coffee_machines",
    "hood": "hoods",
    "stovetop": "stovetops",
    "toaster": "toasters",
    "toaster_oven": "toaster_ovens",
    "blender": "blenders",
    "stand_mixer": "stand_mixers",
    "electric_kettle": "electric_kettles",
    "dish_rack": "dish_racks",
    "fridge_bottom_freezer": "fridges",
    "fridge_side_by_side": "fridges",
    "fridge_french_door": "fridges",
}

# registry yaml stem -> subdirectory under objects/lightwheel/ (when names differ).
_REGISTRY_TO_LIGHTWHEEL_OBJECTS: dict[str, str] = {
    "paper_towel": "paper_towel_holder",
    "utensil_rack": "utensil_rack",
    "utensil_set": "utensil_set",
    "knife_block": "knife_block",
    "plant": "plant",
    "flower_vase": "flower_vase",
    "jar": "jar",
    "digital_scale": "digital_scale",
    "stool": "stool",
    "dish_rack": "dish_rack",
    "soap_dispenser": "soap_dispenser",
    "fruit_bowl": "fruit_bowl",
    "glass_cup": "glass_cup",
    "tiered_basket": "tiered_basket",
    "cinnamon": "cinnamon",
    "paprika": "paprika",
    "oil_bottle": "oil_and_vinegar_bottle",
    "vinegar_bottle": "oil_and_vinegar_bottle",
    "salt_shaker": "salt_and_pepper_shaker",
    "pepper_shaker": "salt_and_pepper_shaker",
}

# Only auto-register PascalCase LightWheel-style ids (Sink025, Stove074, CoffeeMachine067).
_LW_MODEL_NAME = re.compile(r"^[A-Z][A-Za-z0-9]+[0-9]{2,3}([_][A-Za-z0-9]+)*$")
_CABINET_DOOR_PANEL = re.compile(r"^CabinetDoorPanel\d{3}$")
_CABINET_HANDLE = re.compile(r"^CabinetHandle\d{3}$")


def _infer_folder_from_registry(registry_path: Path, fixtures_root: Path) -> str | None:
    stem = registry_path.stem
    if stem in _REGISTRY_TO_FOLDER:
        folder = _REGISTRY_TO_FOLDER[stem]
        if (fixtures_root / folder).is_dir():
            return folder
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except OSError:
        return None
    for key, val in data.items():
        if key == "default" or not isinstance(val, dict):
            continue
        xml = val.get("xml")
        if isinstance(xml, str) and xml.startswith("fixtures/"):
            parts = xml.split("/")
            if len(parts) >= 3:
                return parts[1]
    return None


def _load_registry(registry_path: Path) -> dict | None:
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except OSError:
        return None
    return data if isinstance(data, dict) else None


def _write_registry_if_changed(registry_path: Path, data: dict, added: int) -> int:
    if not added:
        return 0
    registry_path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return added


def _sync_one_registry(registry_path: Path, fixtures_root: Path) -> int:
    folder_name = _infer_folder_from_registry(registry_path, fixtures_root)
    if folder_name is None:
        return 0
    category_dir = fixtures_root / folder_name
    if not category_dir.is_dir():
        return 0

    data = _load_registry(registry_path)
    if data is None:
        return 0

    added = 0
    for child in sorted(category_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name in data or name == "default":
            continue
        if not (child / "model.xml").is_file():
            continue
        if not _LW_MODEL_NAME.match(name):
            continue
        data[name] = {"xml": f"fixtures/{folder_name}/{name}"}
        added += 1

    return _write_registry_if_changed(registry_path, data, added)


def _infer_lightwheel_object_folder(registry_path: Path, objects_root: Path) -> str | None:
    """Return ``objects/lightwheel/<folder>`` name for this registry, if present."""
    stem = registry_path.stem
    mapped = _REGISTRY_TO_LIGHTWHEEL_OBJECTS.get(stem, stem)
    if (objects_root / mapped).is_dir():
        return mapped

    data = _load_registry(registry_path)
    if not data:
        return None
    for key, val in data.items():
        if key == "default" or not isinstance(val, dict):
            continue
        xml = val.get("xml")
        if isinstance(xml, str) and xml.startswith("objects/lightwheel/"):
            parts = xml.split("/")
            if len(parts) >= 3 and (objects_root / parts[2]).is_dir():
                return parts[2]
    return None


def _sync_one_registry_from_objects(registry_path: Path, objects_root: Path) -> int:
    """Register ``objects/lightwheel/<cat>/<PascalCaseNN>/model.xml`` into a registry YAML.

    Also upgrades existing PascalCase entries that still point at legacy
    ``fixtures/accessories/...`` paths when a LightWheel mesh exists — those legacy
    models often only have ``ext_`` sites, while this fork's Fixture sizing reads
    ``reg_`` geoms (needed for floor-mounted stools).
    """
    folder_name = _infer_lightwheel_object_folder(registry_path, objects_root)
    if folder_name is None:
        return 0
    category_dir = objects_root / folder_name
    if not category_dir.is_dir():
        return 0

    data = _load_registry(registry_path)
    if data is None:
        return 0

    added = 0
    for child in sorted(category_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name == "default":
            continue
        if not (child / "model.xml").is_file():
            continue
        if not _LW_MODEL_NAME.match(name):
            continue
        lw_xml = f"objects/lightwheel/{folder_name}/{name}"
        existing = data.get(name)
        if isinstance(existing, dict):
            cur_xml = existing.get("xml")
            if isinstance(cur_xml, str) and cur_xml.startswith("objects/lightwheel/"):
                continue
            if isinstance(cur_xml, str) and cur_xml.startswith("fixtures/"):
                data[name] = {**existing, "xml": lw_xml}
                added += 1
            continue
        if name in data:
            continue
        data[name] = {"xml": lw_xml}
        added += 1

    return _write_registry_if_changed(registry_path, data, added)


def fixture_registry_dir(robocasa_pkg: Path | None = None) -> Path:
    pkg = robocasa_pkg or robocasa_package_dir()
    return pkg / "models" / "assets" / "fixtures" / "fixture_registry"


def missing_required_registry_stems(robocasa_pkg: Path | None = None) -> list[str]:
    """Return required registry YAML stems that are missing on disk."""
    reg_dir = fixture_registry_dir(robocasa_pkg)
    return [stem for stem in REQUIRED_REGISTRY_STEMS if not (reg_dir / f"{stem}.yaml").is_file()]


def _robocasa_git_root(robocasa_pkg: Path | None = None) -> Path | None:
    pkg = robocasa_pkg or robocasa_package_dir()
    root = pkg.parent
    if (root / ".git").is_dir():
        return root
    return None


def restore_fixture_registry_from_vcs(robocasa_pkg: Path | None = None) -> int:
    """Restore vendored ``fixture_registry/*.yaml`` from the robocasa git checkout.

    The ``fixtures_lw`` zip often replaces ``fixture_registry/`` with a slim subset and
    deletes per-type files (e.g. ``fridge_bottom_freezer.yaml``). Returns the number of
    required stems that were missing before restore and exist after.
    """
    pkg = robocasa_pkg or robocasa_package_dir()
    missing_before = missing_required_registry_stems(pkg)
    if not missing_before:
        return 0

    git_root = _robocasa_git_root(pkg)
    if git_root is None:
        logger.warning(
            "Robocasa fixture_registry incomplete (%s missing) but %s is not a git checkout; "
            "cannot auto-restore. Re-clone third_party/robocasa or copy registry YAML from upstream.",
            ", ".join(missing_before),
            git_root or pkg.parent,
        )
        return 0

    subprocess.run(
        ["git", "-C", str(git_root), "checkout", "--", _REGISTRY_GIT_PATH],
        check=True,
        capture_output=True,
        text=True,
    )
    missing_after = missing_required_registry_stems(pkg)
    restored = len(missing_before) - len(missing_after)
    if restored:
        logger.info(
            "Restored %d Robocasa fixture_registry YAML file(s) from git (%s still missing).",
            restored,
            ", ".join(missing_after) if missing_after else "none",
        )
    return restored


def _sync_cabinet_panels_and_handles(fixtures_root: Path) -> int:
    """Register mesh cabinet door panels / handles into ``cabinet.yaml``.

    Kitchen styles pass ``CabinetDoorPanel003`` / ``CabinetHandle014`` as config ids.
    Those must appear in the registry with ``panel_type`` / ``handle_type`` set to the
    same id so :mod:`robocasa.models.fixtures.cabinets` can resolve the mesh path.
    """
    registry_path = fixtures_root / "fixture_registry" / "cabinet.yaml"
    if not registry_path.is_file():
        return 0
    data = _load_registry(registry_path)
    if data is None:
        return 0

    added = 0
    panels_dir = fixtures_root / "cabinets" / "cabinet_panels"
    if panels_dir.is_dir():
        for child in sorted(panels_dir.iterdir()):
            if not child.is_dir() or not _CABINET_DOOR_PANEL.match(child.name):
                continue
            if not (child / "model.xml").is_file():
                continue
            if child.name in data:
                continue
            data[child.name] = {"panel_type": child.name}
            added += 1

    handles_dir = fixtures_root / "handles"
    if handles_dir.is_dir():
        for child in sorted(handles_dir.iterdir()):
            if not child.is_dir() or not _CABINET_HANDLE.match(child.name):
                continue
            if not (child / "model.xml").is_file():
                continue
            if child.name in data:
                continue
            data[child.name] = {"handle_type": child.name}
            added += 1

    return _write_registry_if_changed(registry_path, data, added)


def sync_lightwheel_registry(robocasa_pkg: Path | None = None) -> int:
    """Update all ``fixture_registry/*.yaml`` files. Returns total entries added.

    Syncs both ``fixtures/<category>/`` meshes (Sink025, …) and
    ``objects/lightwheel/<category>/`` accessory meshes (UtensilRack009, …),
    plus cabinet door panel / handle mesh ids referenced by kitchen styles.
    """
    pkg = robocasa_pkg or robocasa_package_dir()
    fixtures_root = pkg / "models" / "assets" / "fixtures"
    objects_root = pkg / "models" / "assets" / "objects" / "lightwheel"
    registry_dir = fixtures_root / "fixture_registry"
    if not registry_dir.is_dir():
        return 0
    total = 0
    for reg in sorted(registry_dir.glob("*.yaml")):
        total += _sync_one_registry(reg, fixtures_root)
        if objects_root.is_dir():
            total += _sync_one_registry_from_objects(reg, objects_root)
    total += _sync_cabinet_panels_and_handles(fixtures_root)
    return total


def ensure_robocasa_fixture_registry(robocasa_pkg: Path | None = None) -> bool:
    """Restore VCS registry layout if needed, sync LW meshes, return layout + Sink025 ok."""
    pkg = robocasa_pkg or robocasa_package_dir()
    restore_fixture_registry_from_vcs(pkg)
    sync_lightwheel_registry(pkg)
    return not missing_required_registry_stems(pkg) and _sink025_registered(pkg)


def _sink025_registered(pkg: Path) -> bool:
    sink_reg = pkg / "models" / "assets" / "fixtures" / "fixture_registry" / "sink.yaml"
    if not sink_reg.is_file():
        return False
    try:
        data = yaml.safe_load(sink_reg.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    return "Sink025" in data


def ensure_lightwheel_registry(robocasa_pkg: Path | None = None) -> bool:
    """Restore registry YAML if LW zip wiped them, sync meshes; True when Sink025 registered."""
    pkg = robocasa_pkg or robocasa_package_dir()
    restore_fixture_registry_from_vcs(pkg)
    if _sink025_registered(pkg):
        return True
    sink_mesh = pkg / "models" / "assets" / "fixtures" / "sinks" / "Sink025" / "model.xml"
    if sink_mesh.is_file():
        sync_lightwheel_registry(pkg)
        return _sink025_registered(pkg)
    return False
