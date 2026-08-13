# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

from pathlib import Path

from emet.simulation.robocasa_assets_check import (
    fixture_registry_layout_ok,
    format_robocasa_assets_incomplete_message,
    lightwheel_registry_ok,
    robocasa_kitchen_assets_complete,
    robocasa_package_dir,
)
from emet.simulation.robocasa_registry_sync import (
    REQUIRED_REGISTRY_STEMS,
    missing_required_registry_stems,
)


def test_format_robocasa_assets_incomplete_mentions_registry_restore():
    msg = format_robocasa_assets_incomplete_message(
        detail="Missing file: .../fixture_registry/fridge_bottom_freezer.yaml",
        missing_registry=True,
    )
    assert "fixture_registry" in msg
    assert "fridge_bottom_freezer" not in msg or "fixtures_lw" in msg
    assert "git -C third_party/robocasa checkout" in msg
    assert "sync_robocasa_lightwheel_registry" in msg


def test_format_robocasa_assets_incomplete_mentions_fixtures_lw():
    msg = format_robocasa_assets_incomplete_message(
        detail='Did not find style that matches "Sink025" for fixture type "sink"',
        missing_lightwheel=True,
    )
    assert "fixtures_lw" in msg
    assert "download_robocasa_assets.py" in msg
    assert "Sink025" in msg


def test_sync_lightwheel_registry_adds_sink025_when_mesh_present():
    from emet.simulation.robocasa_assets_check import robocasa_package_dir
    from emet.simulation.robocasa_registry_sync import ensure_lightwheel_registry

    pkg = robocasa_package_dir()
    if not (pkg / "models/assets/fixtures/sinks/Sink025/model.xml").is_file():
        return
    assert ensure_lightwheel_registry(pkg)
    text = (pkg / "models/assets/fixtures/fixture_registry/sink.yaml").read_text(encoding="utf-8")
    assert "Sink025:" in text


def test_sync_lightwheel_registry_adds_object_lightwheel_accessories(tmp_path: Path):
    """Kitchen styles reference UtensilRack009 under objects/lightwheel, not fixtures/."""
    from emet.simulation.robocasa_registry_sync import sync_lightwheel_registry

    pkg = tmp_path / "robocasa"
    reg = pkg / "models/assets/fixtures/fixture_registry"
    rack = pkg / "models/assets/objects/lightwheel/utensil_rack/UtensilRack009"
    reg.mkdir(parents=True)
    rack.mkdir(parents=True)
    (rack / "model.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (reg / "utensil_rack.yaml").write_text(
        "default:\n  name: utensil_rack\nUtensilRack004:\n  xml: fixtures/accessories/utensil_racks/metal\n",
        encoding="utf-8",
    )
    assert sync_lightwheel_registry(pkg) >= 1
    text = (reg / "utensil_rack.yaml").read_text(encoding="utf-8")
    assert "UtensilRack009:" in text
    assert "objects/lightwheel/utensil_rack/UtensilRack009" in text
    assert "UtensilRack004:" in text  # existing entries preserved


def test_sync_upgrades_legacy_fixture_pascalcase_to_lightwheel_object(tmp_path: Path):
    """Stool009→rattan_stool has no reg_bbox; LW Stool009 does — upgrade for floor attach."""
    from emet.simulation.robocasa_registry_sync import sync_lightwheel_registry

    pkg = tmp_path / "robocasa"
    reg = pkg / "models/assets/fixtures/fixture_registry"
    stool = pkg / "models/assets/objects/lightwheel/stool/Stool009"
    reg.mkdir(parents=True)
    stool.mkdir(parents=True)
    (stool / "model.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (reg / "stool.yaml").write_text(
        "default:\n  name: chair\n  size: [null, null, 1.0]\n"
        "Stool009:\n  xml: fixtures/accessories/stools/rattan_stool\n",
        encoding="utf-8",
    )
    assert sync_lightwheel_registry(pkg) >= 1
    data = __import__("yaml").safe_load((reg / "stool.yaml").read_text(encoding="utf-8"))
    assert data["Stool009"]["xml"] == "objects/lightwheel/stool/Stool009"


def test_sync_cabinet_panels_and_handles(tmp_path: Path):
    from emet.simulation.robocasa_registry_sync import sync_lightwheel_registry

    pkg = tmp_path / "robocasa"
    fixtures = pkg / "models/assets/fixtures"
    reg = fixtures / "fixture_registry"
    panel = fixtures / "cabinets/cabinet_panels/CabinetDoorPanel003"
    handle = fixtures / "handles/CabinetHandle014"
    reg.mkdir(parents=True)
    panel.mkdir(parents=True)
    handle.mkdir(parents=True)
    (panel / "model.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (handle / "model.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (reg / "cabinet.yaml").write_text(
        "default:\n  texture: textures/flat/white.png\n"
        "CabinetDoorPanel047:\n  panel_type: slab\n",
        encoding="utf-8",
    )
    assert sync_lightwheel_registry(pkg) >= 2
    data = __import__("yaml").safe_load((reg / "cabinet.yaml").read_text(encoding="utf-8"))
    assert data["CabinetDoorPanel003"] == {"panel_type": "CabinetDoorPanel003"}
    assert data["CabinetHandle014"] == {"handle_type": "CabinetHandle014"}
    assert data["CabinetDoorPanel047"]["panel_type"] == "slab"


def test_fixture_registry_layout_requires_fridge_stems():
    pkg = robocasa_package_dir()
    if not pkg.is_dir():
        return
    missing = missing_required_registry_stems(pkg)
    if "fridge_bottom_freezer" in missing:
        assert not fixture_registry_layout_ok(pkg)
        msg = format_robocasa_assets_incomplete_message(missing_registry=True)
        assert "fridge_bottom_freezer" not in msg
        assert "fixture_registry" in msg
    else:
        assert fixture_registry_layout_ok(pkg)
        for stem in ("fridge_bottom_freezer", "fridge_french_door", "fridge_side_by_side"):
            assert stem in REQUIRED_REGISTRY_STEMS


def test_format_robocasa_assets_incomplete_mentions_objaverse_bbox():
    msg = format_robocasa_assets_incomplete_message(missing_objaverse_bbox=True)
    assert "reg_bbox" in msg
    assert "process_robocasa_objaverse_reg_bbox" in msg


def test_basic_fixtures_present_accepts_current_sentinel():
    pkg = robocasa_package_dir()
    if not pkg.is_dir():
        return
    counter = pkg / "models/assets/fixtures/counters/counter/model.xml"
    if counter.is_file():
        from emet.simulation.robocasa_assets_check import basic_fixtures_present

        assert basic_fixtures_present(pkg)


def test_robocasa_kitchen_assets_complete_matches_sentinel_when_present():
    root = Path(__file__).resolve().parents[3]
    pkg = root / "third_party" / "robocasa" / "robocasa"
    if not pkg.is_dir():
        return
    from emet.simulation.robocasa_objaverse_bbox import objaverse_reg_bbox_present

    if lightwheel_registry_ok(pkg) and fixture_registry_layout_ok(pkg) and objaverse_reg_bbox_present(pkg):
        assert robocasa_kitchen_assets_complete(pkg)
