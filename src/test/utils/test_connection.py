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

import json

from emet.utils import connection as conn_mod


def test_save_connection_writes_profile_and_robot_ip(tmp_path, monkeypatch):
    stretch = tmp_path / "stretch"
    monkeypatch.setattr(conn_mod, "_STRETCH_DIR", str(stretch))
    monkeypatch.setattr(conn_mod, "_CONNECTION_FILE", str(stretch / "connection.json"))
    monkeypatch.setattr(conn_mod, "_ROBOT_IP_FILE", str(stretch / "robot_ip.txt"))

    name = conn_mod.save_connection(
        host=" 192.168.1.9 ",
        user=" robot ",
        password="secret",
        name="mars",
        workspace=" ~/innate-os/ros2_ws ",
        emet_dir=" ~/emet ",
        robot=" innate_mars ",
    )
    assert name == "mars"
    data = json.loads((stretch / "connection.json").read_text())
    assert data["active"] == "mars"
    profile = data["connections"]["mars"]
    assert profile == {
        "host": "192.168.1.9",
        "user": "robot",
        "password": "secret",
        "workspace": "~/innate-os/ros2_ws",
        "emet_dir": "~/emet",
        "robot": "innate_mars",
    }
    assert (stretch / "robot_ip.txt").read_text() == "192.168.1.9"


def test_save_connection_no_active_skips_robot_ip_sync(tmp_path, monkeypatch):
    stretch = tmp_path / "stretch"
    monkeypatch.setattr(conn_mod, "_STRETCH_DIR", str(stretch))
    monkeypatch.setattr(conn_mod, "_CONNECTION_FILE", str(stretch / "connection.json"))
    monkeypatch.setattr(conn_mod, "_ROBOT_IP_FILE", str(stretch / "robot_ip.txt"))

    conn_mod.save_connection(host="10.0.0.1", name="a", set_active=False)
    assert not (stretch / "robot_ip.txt").exists()
