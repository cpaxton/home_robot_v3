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
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from emet.config.sim_launch_config import (
    SimLaunchDefaultMujoco,
    SimLaunchMolmospaces,
    SimLaunchRobocasa,
    apply_sim_launch_cli_overrides,
    build_sim_launch_config_from_serve_cli,
    decode_sim_launch_config,
    load_sim_launch_config_from_path,
    load_sim_launch_from_agent_yaml,
    resolve_sim_launch_for_agent,
    validate_sim_launch_serve_combo,
)
from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv


def test_load_default_table_yaml():
    cfg = load_sim_launch_config_from_path("configs/sim/default_table_rby1.yaml")
    assert isinstance(cfg, SimLaunchDefaultMujoco)
    assert cfg.robot == "rby1"
    assert cfg.headless is True
    assert cfg.physics_mode == "kinematic"
    argv = prepare_mujoco_server_argv(cfg)
    assert "--robot" in argv and "rby1" in argv
    assert "--headless" in argv
    assert "--kinematic-sim" in argv


def test_load_default_table_stretch_yaml():
    cfg = load_sim_launch_config_from_path("configs/sim/default_table_stretch.yaml")
    assert isinstance(cfg, SimLaunchDefaultMujoco)
    assert cfg.robot == "stretch"
    assert cfg.physics_mode == "kinematic"
    assert cfg.stretch_legacy is False
    argv = prepare_mujoco_server_argv(cfg)
    assert "--kinematic-sim" in argv
    assert "--stretch-legacy" not in argv


def test_prepare_argv_stretch_legacy():
    cfg = SimLaunchDefaultMujoco(robot="stretch", stretch_legacy=True, headless=True)
    argv = prepare_mujoco_server_argv(cfg)
    assert "--stretch-legacy" in argv


def test_load_robocasa_yaml():
    cfg = load_sim_launch_config_from_path("configs/sim/robocasa_pick_place.yaml")
    assert isinstance(cfg, SimLaunchRobocasa)
    argv = prepare_mujoco_server_argv(cfg)
    assert "--use-robocasa" in argv
    assert "PickPlaceCounterToCabinet" in argv


def test_decode_inline_molmospaces():
    raw = {"kind": "molmospaces", "scene": "ithor", "split": "val", "index": 2, "headless": True}
    cfg = decode_sim_launch_config(raw)
    assert isinstance(cfg, SimLaunchMolmospaces)
    assert cfg.split == "val" and cfg.index == 2


def test_agent_yaml_sim_inline(tmp_path: Path):
    p = tmp_path / "agent.yaml"
    p.write_text(
        textwrap.dedent(
            """
            encoder: siglip
            sim:
              kind: default_mujoco
              robot: galaxea_r1
              headless: true
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = load_sim_launch_from_agent_yaml(str(p))
    assert isinstance(cfg, SimLaunchDefaultMujoco)
    assert cfg.robot == "galaxea_r1"


def test_agent_yaml_sim_config_path_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI --sim-config wins over agent sim_config key."""
    sim1 = tmp_path / "sim_a.yaml"
    sim1.write_text("kind: default_mujoco\nrobot: rby1\nheadless: true\n", encoding="utf-8")
    sim2 = tmp_path / "sim_b.yaml"
    sim2.write_text("kind: default_mujoco\nrobot: stretch\nheadless: true\n", encoding="utf-8")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        yaml.dump({"encoder": "x", "sim_config": str(sim1)}),
        encoding="utf-8",
    )
    cfg = resolve_sim_launch_for_agent(
        agent_config_path=str(agent),
        sim_config_cli=str(sim2),
        port_offset_cli=0,
    )
    assert isinstance(cfg, SimLaunchDefaultMujoco)
    assert cfg.robot == "stretch"


def test_port_offset_override_from_cli(tmp_path: Path):
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        yaml.dump({"sim": {"kind": "default_mujoco", "robot": "rby1", "port_offset": 0}}),
        encoding="utf-8",
    )
    cfg = resolve_sim_launch_for_agent(
        agent_config_path=str(agent),
        sim_config_cli=None,
        port_offset_cli=50,
    )
    assert cfg.port_offset == 50


def test_resolve_raises_without_sim_when_no_fallback(tmp_path: Path):
    agent = tmp_path / "agent.yaml"
    agent.write_text(yaml.dump({"encoder": "siglip"}), encoding="utf-8")
    with pytest.raises(ValueError, match="No sim launch configuration"):
        resolve_sim_launch_for_agent(
            agent_config_path=str(agent),
            sim_config_cli=None,
            port_offset_cli=0,
            default_mujoco_table_if_missing=False,
        )


def test_resolve_default_mujoco_fallback(tmp_path: Path):
    agent = tmp_path / "agent.yaml"
    agent.write_text(yaml.dump({"encoder": "siglip"}), encoding="utf-8")
    cfg = resolve_sim_launch_for_agent(
        agent_config_path=str(agent),
        sim_config_cli=None,
        port_offset_cli=7,
        default_mujoco_table_if_missing=True,
        default_robot="galaxea_r1",
        default_headless=True,
    )
    assert isinstance(cfg, SimLaunchDefaultMujoco)
    assert cfg.robot == "galaxea_r1"
    assert cfg.headless is True
    assert cfg.port_offset == 7


def test_build_sim_launch_from_serve_cli_matches_molmospaces():
    cfg = build_sim_launch_config_from_serve_cli(
        molmospaces_scene="ithor",
        molmospaces_split="val",
        molmospaces_index=3,
        molmospaces_install=False,
        use_robocasa=False,
        scene_path=None,
        robot="rby1",
        headless=True,
        show_viewer_ui=False,
        no_cameras=False,
        use_glx=False,
        seed=1,
        steps=None,
        debug_molmospaces_spawn=False,
        port_offset=0,
        robocasa_task="",
    )
    assert isinstance(cfg, SimLaunchMolmospaces)
    assert cfg.scene == "ithor" and cfg.split == "val" and cfg.index == 3


def test_validate_sim_launch_serve_combo_molmo_vs_robocasa():
    with pytest.raises(ValueError, match="Cannot combine"):
        validate_sim_launch_serve_combo(molmospaces_scene="ithor", scene_path=None, use_robocasa=True)


def test_apply_sim_physics_mode_override():
    base = SimLaunchDefaultMujoco(robot="rby1", physics_mode="dynamic")
    out = apply_sim_launch_cli_overrides(base, physics_mode="kinematic")
    assert isinstance(out, SimLaunchDefaultMujoco)
    assert out.physics_mode == "kinematic"


def test_capture_merge_sim_cli_kinematic():
    from emet.app.capture_sim_dataset_episode import _merge_sim_cli

    base = SimLaunchDefaultMujoco(robot="rby1", physics_mode="dynamic")
    out = _merge_sim_cli(
        base,
        port_offset=0,
        headless=True,
        seed=0,
        robot=None,
        show_viewer_ui=False,
        kinematic_sim=True,
    )
    assert out.physics_mode == "kinematic"


def test_apply_sim_seed_on_existing_default():
    base = SimLaunchDefaultMujoco(robot="stretch", headless=False, seed=0)
    out = apply_sim_launch_cli_overrides(base, seed=99)
    assert isinstance(out, SimLaunchDefaultMujoco)
    assert out.seed == 99
    assert out.robot == "stretch"


def test_scene_resolution_module_used_by_cli_helpers() -> None:
    from emet.simulation.scene_resolution import build_zmq_environment, resolve_merged_physics_scene

    assert (
        build_zmq_environment(
            molmospaces_session_scene=None,
            molmospaces_session_split=None,
            molmospaces_session_index=None,
            use_robocasa=True,
            robocasa_task="T",
            robocasa_style=1,
            robocasa_layout=1,
        )["kind"]
        == "robocasa"
    )
    loaded = resolve_merged_physics_scene(
        robot_key="rby1",
        scene_path=None,
        use_robocasa=True,
        wizard_scene_model=None,
        wizard_scene_xml="<mujoco/>",
        wizard_objects_info=None,
        zmq_environment=None,
        scene_source_basename=None,
    )
    assert loaded.scene_xml == "<mujoco/>"
