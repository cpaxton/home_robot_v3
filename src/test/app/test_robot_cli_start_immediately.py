# Copyright (c) Hello Robot, Inc. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

from emet.app.robot_cli import create_robot_client_from_cli


def test_create_robot_client_passes_start_immediately_to_galaxea():
    captured: dict = {}

    class FakeBackend:
        def create_client(self, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        def get_spec(self):
            return MagicMock()

    fake_mod = MagicMock()
    fake_mod.GalaxeaR1Backend = FakeBackend

    with patch("emet.app.robot_cli.importlib.import_module", return_value=fake_mod):
        with patch("emet.app.robot_cli.ROBOT_REGISTRY", {"galaxea_r1": "fake.mod"}):
            create_robot_client_from_cli(
                "galaxea_r1",
                "127.0.0.1",
                start_immediately=True,
                enable_rerun_server=False,
            )
    assert captured.get("start_immediately") is True
