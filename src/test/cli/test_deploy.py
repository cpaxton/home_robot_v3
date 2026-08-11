# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Unit tests for robot deploy helpers and ``emet deploy llm`` CLI."""

import pytest
from click.testing import CliRunner

from emet.deploy import (
    build_remote_bridge_import_verify_cmd,
    build_stretch_bridge_start_remote_cmd,
    get_deploy_spec,
    normalize_deploy_robot,
    resolve_deploy_robot,
)
from emet.deploy_llm import CALIBAN_ORIN_VRAM_GIB, LLM_PROFILES, deploy_llm


def test_remote_bridge_import_verify_cmd_mars():
    cmd = build_remote_bridge_import_verify_cmd(
        remote_emet="~/emet",
        remote_ws="~/innate-os/ros2_ws",
        robot="innate_mars",
    )
    assert "~/emet/emet_core" in cmd
    assert "innate_mars_bridge.ros.camera" in cmd
    assert "emet.utils.image" in cmd
    assert "emet.core.server" in cmd
    assert "colcon" not in cmd
    assert '-c "import innate_mars_bridge' in cmd
    assert "-c 'import" not in cmd


def test_remote_bridge_import_verify_cmd_stretch():
    cmd = build_remote_bridge_import_verify_cmd(
        remote_emet="~/emet",
        remote_ws="~/ament_ws",
        robot="stretch",
    )
    assert "stretch_ros2_bridge.ros.camera" in cmd
    assert "emet.core.server" in cmd
    assert "innate_mars" not in cmd


def test_normalize_and_specs():
    assert normalize_deploy_robot("hello_stretch") == "stretch"
    assert normalize_deploy_robot("mars") == "innate_mars"
    assert get_deploy_spec("stretch").bridge_pkg == "stretch_ros2_bridge"
    assert get_deploy_spec("innate_mars").bridge_pkg == "innate_mars_bridge"
    assert get_deploy_spec("stretch").default_workspace == "~/ament_ws"
    assert get_deploy_spec("innate_mars").default_workspace == "~/innate-os/ros2_ws"


def test_resolve_deploy_robot_defaults_and_workspace_hint(tmp_path, monkeypatch):
    from emet.utils import connection as conn_mod

    stretch = tmp_path / "stretch"
    monkeypatch.setattr(conn_mod, "_STRETCH_DIR", str(stretch))
    monkeypatch.setattr(conn_mod, "_CONNECTION_FILE", str(stretch / "connection.json"))
    monkeypatch.setattr(conn_mod, "_ROBOT_IP_FILE", str(stretch / "robot_ip.txt"))
    assert resolve_deploy_robot(None, host="10.0.0.1") == "stretch"
    assert resolve_deploy_robot(None, host="10.0.0.1", workspace="~/innate-os/ros2_ws") == "innate_mars"
    assert resolve_deploy_robot("stretch") == "stretch"


def test_resolve_deploy_robot_from_connection(tmp_path, monkeypatch):
    from emet.utils import connection as conn_mod

    stretch = tmp_path / "stretch"
    monkeypatch.setattr(conn_mod, "_STRETCH_DIR", str(stretch))
    monkeypatch.setattr(conn_mod, "_CONNECTION_FILE", str(stretch / "connection.json"))
    monkeypatch.setattr(conn_mod, "_ROBOT_IP_FILE", str(stretch / "robot_ip.txt"))

    conn_mod.save_connection(
        host="10.0.0.9",
        user="hello-robot",
        name="stretch",
        robot="stretch",
        workspace="~/ament_ws",
        set_active=True,
    )
    assert resolve_deploy_robot(None) == "stretch"
    conn_mod.save_connection(
        host="10.0.0.8",
        user="jetson1",
        name="mars",
        robot="innate_mars",
        workspace="~/innate-os/ros2_ws",
        set_active=True,
    )
    assert resolve_deploy_robot(None) == "innate_mars"
    assert resolve_deploy_robot(None, connection_name="stretch") == "stretch"


def test_resolve_deploy_robot_errors_without_robot_field(tmp_path, monkeypatch):
    from emet.utils import connection as conn_mod

    stretch = tmp_path / "stretch"
    monkeypatch.setattr(conn_mod, "_STRETCH_DIR", str(stretch))
    monkeypatch.setattr(conn_mod, "_CONNECTION_FILE", str(stretch / "connection.json"))
    monkeypatch.setattr(conn_mod, "_ROBOT_IP_FILE", str(stretch / "robot_ip.txt"))

    conn_mod.save_connection(host="10.0.0.7", user="jetson1", name="legacy", set_active=True)
    with pytest.raises(SystemExit, match="no robot"):
        resolve_deploy_robot(None)


def test_resolve_deploy_robot_host_override_requires_robot(tmp_path, monkeypatch):
    from emet.utils import connection as conn_mod

    stretch = tmp_path / "stretch"
    monkeypatch.setattr(conn_mod, "_STRETCH_DIR", str(stretch))
    monkeypatch.setattr(conn_mod, "_CONNECTION_FILE", str(stretch / "connection.json"))
    monkeypatch.setattr(conn_mod, "_ROBOT_IP_FILE", str(stretch / "robot_ip.txt"))

    conn_mod.save_connection(
        host="10.0.0.9",
        user="hello-robot",
        name="stretch",
        robot="stretch",
        set_active=True,
    )
    with pytest.raises(SystemExit, match="differs from active"):
        resolve_deploy_robot(None, host="10.0.0.8")
    assert resolve_deploy_robot("innate_mars", host="10.0.0.8") == "innate_mars"


def test_stretch_bridge_start_uses_fuser_not_pkill():
    cmd = build_stretch_bridge_start_remote_cmd()
    assert "fuser -k 4401/tcp" in cmd
    assert "pkill" not in cmd
    assert "stretch_ros2_bridge" in cmd
    assert "/tmp/emet-stretch-bridge.log" in cmd


def test_caliban_orin_vram_constant() -> None:
    assert CALIBAN_ORIN_VRAM_GIB >= 60
    assert "unified-7b" in LLM_PROFILES


def test_deploy_llm_help_lists_profiles_and_vram() -> None:
    from emet.cli import main

    r = CliRunner().invoke(main, ["deploy", "llm", "--help"])
    assert r.exit_code == 0, r.output
    assert "unified-7b" in r.output
    assert "dual-2b" in r.output
    assert "64" in r.output or "60" in r.output


def test_deploy_group_help_lists_llm_and_robot() -> None:
    from emet.cli import main

    r = CliRunner().invoke(main, ["deploy", "--help"])
    assert r.exit_code == 0, r.output
    assert "llm" in r.output
    assert "--robot" in r.output
    assert "stretch" in r.output


def test_connect_use_sets_active(tmp_path, monkeypatch) -> None:
    from emet.cli import main
    from emet.utils import connection as conn_mod

    stretch = tmp_path / "stretch"
    monkeypatch.setattr(conn_mod, "_STRETCH_DIR", str(stretch))
    monkeypatch.setattr(conn_mod, "_CONNECTION_FILE", str(stretch / "connection.json"))
    monkeypatch.setattr(conn_mod, "_ROBOT_IP_FILE", str(stretch / "robot_ip.txt"))

    conn_mod.save_connection(host="10.0.0.1", user="u", name="a", robot="stretch", set_active=True)
    conn_mod.save_connection(host="10.0.0.2", user="u", name="b", robot="innate_mars", set_active=False)
    r = CliRunner().invoke(main, ["connect", "use", "b"])
    assert r.exit_code == 0, r.output
    assert conn_mod.get_active_connection()["host"] == "10.0.0.2"


def test_deploy_llm_requires_host(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EMET_LLM_HOST", raising=False)
    monkeypatch.delenv("EMET_CALIBAN_HOST", raising=False)
    rc = deploy_llm(host=None, profile="unified-7b", root=tmp_path)
    assert rc == 1


def test_deploy_llm_missing_script_returns_error(tmp_path) -> None:
    rc = deploy_llm(host="orin-host", profile="unified-7b", root=tmp_path)
    assert rc == 1
