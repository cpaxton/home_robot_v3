# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Repair LightWheel registry aliases that point at missing base meshes."""

from __future__ import annotations

import yaml


def test_sync_repairs_sink025_alias_to_the_lightwheel_mesh(tmp_path):
    from emet.simulation.robocasa_registry_sync import _sink025_registered, sync_lightwheel_registry

    pkg = tmp_path / "robocasa"
    fixtures = pkg / "models" / "assets" / "fixtures"
    (fixtures / "sinks" / "Sink025").mkdir(parents=True)
    (fixtures / "sinks" / "Sink025" / "model.xml").write_text("<mujoco/>")
    registry = fixtures / "fixture_registry"
    registry.mkdir(parents=True)
    (registry / "sink.yaml").write_text(
        "Sink025:\n  xml: fixtures/sinks/white_sink\nwhite_sink:\n  xml: fixtures/sinks/white_sink\n"
    )

    # The alias looks registered, but its mesh is absent from the installed packs.
    assert not _sink025_registered(pkg)

    sync_lightwheel_registry(pkg)

    data = yaml.safe_load((registry / "sink.yaml").read_text())
    assert data["Sink025"]["xml"] == "fixtures/sinks/Sink025"
    assert _sink025_registered(pkg)
