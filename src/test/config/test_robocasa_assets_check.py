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
