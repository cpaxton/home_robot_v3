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
    decode_sim_launch_config,
    load_sim_launch_config_from_path,
    load_sim_launch_from_agent_yaml,
    resolve_sim_launch_for_agent,
)
from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv


def test_load_default_table_yaml():
    cfg = load_sim_launch_config_from_path("configs/sim/default_table_rby1.yaml")
    assert isinstance(cfg, SimLaunchDefaultMujoco)
    assert cfg.robot == "rby1"
    assert cfg.headless is True
    argv = prepare_mujoco_server_argv(cfg)
    assert "--robot" in argv and "rby1" in argv
    assert "--headless" in argv


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
