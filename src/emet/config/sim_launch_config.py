# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""YAML-driven MuJoCo / Robocasa / MolmoSpaces launch configs for ``emet serve mujoco`` and ``emet run agent --start-sim``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import draccus
import yaml

from emet.utils.config import resolve_config_yaml_path

SimLaunchKind = Literal["default_mujoco", "robocasa", "molmospaces"]


@dataclass
class SimLaunchCommon:
    """Flags shared by all ``emet.simulation.mujoco_server`` backends."""

    port_offset: int = 0
    headless: bool = False
    show_viewer_ui: bool = False
    no_cameras: bool = False
    use_glx: bool = False
    seed: int = 0
    steps: int | None = None
    debug_molmospaces_spawn: bool = False
    verbose: bool = False
    use_remote_computer: bool = True


@dataclass
class SimLaunchDefaultMujoco(SimLaunchCommon):
    """Default packaged table scene + robot (or an explicit merged MJCF path)."""

    kind: str = "default_mujoco"
    robot: str = "rby1"
    scene_path: str | None = None


@dataclass
class SimLaunchRobocasa(SimLaunchCommon):
    """Robocasa-generated kitchen scene."""

    kind: str = "robocasa"
    robot: str = "PandaOmron"
    robocasa_task: str = "PickPlaceCounterToCabinet"
    robocasa_style: int = 1
    robocasa_layout: int = 1
    robocasa_write_to_xml: bool = False


@dataclass
class SimLaunchMolmospaces(SimLaunchCommon):
    """MolmoSpaces scene merged with a mobile robot (via emet-molmospaces wrapper)."""

    kind: str = "molmospaces"
    robot: str = "rby1"
    scene: str = "ithor"
    split: str = "train"
    index: int = 0
    molmospaces_install: bool = False


SimLaunchConfig = SimLaunchDefaultMujoco | SimLaunchRobocasa | SimLaunchMolmospaces


_KIND_TO_TYPE: dict[str, type[SimLaunchConfig]] = {
    "default_mujoco": SimLaunchDefaultMujoco,
    "robocasa": SimLaunchRobocasa,
    "molmospaces": SimLaunchMolmospaces,
}


def decode_sim_launch_config(raw: dict[str, Any]) -> SimLaunchConfig:
    if not isinstance(raw, dict):
        raise ValueError("sim launch config must be a mapping")
    kind = str(raw.get("kind", "default_mujoco")).strip().lower()
    cls = _KIND_TO_TYPE.get(kind)
    if cls is None:
        raise ValueError(f"unknown sim launch kind {kind!r}; expected one of {sorted(_KIND_TO_TYPE)}")
    return draccus.decode(cls, raw)


def load_sim_launch_config_from_path(path: str) -> SimLaunchConfig:
    """Load a standalone sim YAML (``kind:`` required)."""
    full = Path(resolve_config_yaml_path(path))
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return decode_sim_launch_config(raw)


def load_sim_launch_from_agent_yaml(agent_config_path: str) -> SimLaunchConfig | None:
    """Return sim config from ``sim:`` inline or ``sim_config:`` path in a dynav-style agent YAML."""
    full = Path(resolve_config_yaml_path(agent_config_path))
    with full.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    inline = raw.get("sim")
    if isinstance(inline, dict):
        return decode_sim_launch_config(inline)
    path_key = raw.get("sim_config")
    if path_key is None or path_key is False:
        return None
    if not isinstance(path_key, str) or not str(path_key).strip():
        return None
    # Resolve relative to cwd first, then agent file directory, then emet/config
    p = str(path_key).strip()
    try:
        return load_sim_launch_config_from_path(p)
    except FileNotFoundError:
        pass
    candidate = (full.parent / p).resolve()
    if candidate.is_file():
        return load_sim_launch_config_from_path(str(candidate))
    return load_sim_launch_config_from_path(p)


def resolve_sim_launch_for_agent(
    *,
    agent_config_path: str,
    sim_config_cli: str | None,
    port_offset_cli: int,
) -> SimLaunchConfig:
    """Merge agent YAML sim section with optional ``--sim-config`` and ``--port-offset`` override."""
    cfg: SimLaunchConfig | None = None
    if sim_config_cli and str(sim_config_cli).strip():
        cfg = load_sim_launch_config_from_path(str(sim_config_cli).strip())
    if cfg is None:
        cfg = load_sim_launch_from_agent_yaml(agent_config_path)
    if cfg is None:
        raise ValueError(
            "No sim launch configuration: pass --sim-config PATH, or add "
            "'sim_config: configs/sim/....yaml' or 'sim:' inline block to the agent YAML."
        )
    if port_offset_cli != 0:
        cfg.port_offset = int(port_offset_cli)
    return cfg
