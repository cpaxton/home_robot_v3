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

"""YAML-driven MuJoCo / Robocasa / MolmoSpaces launch configs for ``emet serve mujoco`` and ``emet run agent --start-sim``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import draccus
import yaml

from emet.utils.config import resolve_config_yaml_path

SimLaunchKind = Literal["default_mujoco", "robocasa", "molmospaces"]
SimPhysicsMode = Literal["dynamic", "kinematic"]


@dataclass
class SimLaunchCommon:
    """Flags shared by all ``emet.simulation.mujoco_server`` backends.

    ``physics_mode=kinematic``: non-Stretch server uses ``mj_forward`` only (pose snap, no ``mj_step``).
    Default Stretch (merged table) uses the same kinematic/dynamic mode via RobosuiteZmqServer.
    """

    port_offset: int = 0
    physics_mode: SimPhysicsMode = "dynamic"
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
    stretch_legacy: bool = False


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


def validate_sim_launch_serve_combo(
    *,
    molmospaces_scene: str | None,
    scene_path: str | None,
    use_robocasa: bool,
) -> None:
    """Same mutual-exclusion rules as ``emet serve mujoco``."""
    if molmospaces_scene and scene_path:
        raise ValueError("Use either --scene-path or --molmospaces-scene, not both.")
    if molmospaces_scene and use_robocasa:
        raise ValueError("Cannot combine --molmospaces-scene with --use-robocasa / robocasa backend.")


def build_sim_launch_config_from_serve_cli(
    *,
    molmospaces_scene: str | None,
    molmospaces_split: str,
    molmospaces_index: int,
    molmospaces_install: bool,
    use_robocasa: bool,
    scene_path: str | None,
    robot: str,
    headless: bool,
    show_viewer_ui: bool,
    no_cameras: bool,
    use_glx: bool,
    seed: int,
    steps: int | None,
    debug_molmospaces_spawn: bool,
    port_offset: int,
    robocasa_task: str,
    physics_mode: SimPhysicsMode = "dynamic",
    stretch_legacy: bool = False,
) -> SimLaunchConfig:
    """Build :class:`SimLaunchConfig` the same way ``emet serve mujoco`` does (single source of truth)."""
    validate_sim_launch_serve_combo(
        molmospaces_scene=molmospaces_scene,
        scene_path=scene_path,
        use_robocasa=use_robocasa,
    )
    common: dict[str, Any] = {
        "headless": headless,
        "show_viewer_ui": show_viewer_ui,
        "no_cameras": no_cameras,
        "use_glx": use_glx,
        "seed": seed,
        "steps": steps,
        "debug_molmospaces_spawn": debug_molmospaces_spawn,
        "port_offset": port_offset,
        "physics_mode": physics_mode,
    }
    if molmospaces_scene:
        return SimLaunchMolmospaces(
            scene=molmospaces_scene,
            split=molmospaces_split,
            index=molmospaces_index,
            molmospaces_install=molmospaces_install,
            robot=robot,
            **common,
        )
    if use_robocasa:
        return SimLaunchRobocasa(robot=robot, robocasa_task=robocasa_task or "", **common)
    return SimLaunchDefaultMujoco(robot=robot, scene_path=scene_path, stretch_legacy=stretch_legacy, **common)


def apply_sim_launch_cli_overrides(
    cfg: SimLaunchConfig,
    *,
    use_robocasa: bool = False,
    robocasa_task: str | None = None,
    scene_path: str | None = None,
    molmospaces_scene: str | None = None,
    molmospaces_split: str | None = None,
    molmospaces_index: int | None = None,
    molmospaces_install: bool | None = None,
    headless: bool | None = None,
    show_viewer_ui: bool | None = None,
    no_cameras: bool | None = None,
    use_glx: bool | None = None,
    seed: int | None = None,
    steps: int | None = None,
    debug_molmospaces_spawn: bool | None = None,
    robot: str | None = None,
    physics_mode: SimPhysicsMode | None = None,
) -> SimLaunchConfig:
    """Apply ``emet serve mujoco``-style CLI overrides onto a resolved sim config (for ``--start-sim``)."""
    ms = str(molmospaces_scene).strip() if molmospaces_scene else None
    sp = str(scene_path).strip() if scene_path else None
    if ms is not None and not ms:
        ms = None
    if sp is not None and not sp:
        sp = None

    validate_sim_launch_serve_combo(molmospaces_scene=ms, scene_path=sp, use_robocasa=use_robocasa)

    def _common_from(c: SimLaunchCommon) -> dict[str, Any]:
        return {
            "port_offset": c.port_offset,
            "physics_mode": c.physics_mode,
            "headless": c.headless,
            "show_viewer_ui": c.show_viewer_ui,
            "no_cameras": c.no_cameras,
            "use_glx": c.use_glx,
            "seed": c.seed,
            "steps": c.steps,
            "debug_molmospaces_spawn": c.debug_molmospaces_spawn,
            "verbose": c.verbose,
            "use_remote_computer": c.use_remote_computer,
        }

    merged_common = _common_from(cfg)
    if headless is not None:
        merged_common["headless"] = headless
    if show_viewer_ui is not None:
        merged_common["show_viewer_ui"] = show_viewer_ui
    if no_cameras is not None:
        merged_common["no_cameras"] = no_cameras
    if use_glx is not None:
        merged_common["use_glx"] = use_glx
    if seed is not None:
        merged_common["seed"] = int(seed)
    if steps is not None:
        merged_common["steps"] = int(steps)
    if debug_molmospaces_spawn is not None:
        merged_common["debug_molmospaces_spawn"] = bool(debug_molmospaces_spawn)
    if physics_mode is not None:
        pm = str(physics_mode).strip().lower()
        if pm not in ("dynamic", "kinematic"):
            raise ValueError(f"physics_mode must be 'dynamic' or 'kinematic', not {physics_mode!r}")
        merged_common["physics_mode"] = pm  # type: ignore[assignment]

    if ms is not None:
        if molmospaces_split is not None:
            split = str(molmospaces_split).strip() or "train"
        elif isinstance(cfg, SimLaunchMolmospaces):
            split = str(cfg.split)
        else:
            split = "train"
        if molmospaces_index is not None:
            idx = int(molmospaces_index)
        elif isinstance(cfg, SimLaunchMolmospaces):
            idx = int(cfg.index)
        else:
            idx = 0
        if molmospaces_install is not None:
            inst = bool(molmospaces_install)
        elif isinstance(cfg, SimLaunchMolmospaces):
            inst = bool(cfg.molmospaces_install)
        else:
            inst = False
        rob = robot if robot is not None else cfg.robot
        return SimLaunchMolmospaces(
            scene=ms,
            split=split,
            index=idx,
            molmospaces_install=inst,
            robot=rob,
            **merged_common,
        )

    if use_robocasa:
        task = (robocasa_task or "").strip() or (
            cfg.robocasa_task if isinstance(cfg, SimLaunchRobocasa) else "PickPlaceCounterToCabinet"
        )
        style = int(cfg.robocasa_style) if isinstance(cfg, SimLaunchRobocasa) else 1
        layout = int(cfg.robocasa_layout) if isinstance(cfg, SimLaunchRobocasa) else 1
        write_xml = bool(cfg.robocasa_write_to_xml) if isinstance(cfg, SimLaunchRobocasa) else False
        rob = robot if robot is not None else cfg.robot
        return SimLaunchRobocasa(
            robot=rob,
            robocasa_task=task,
            robocasa_style=style,
            robocasa_layout=layout,
            robocasa_write_to_xml=write_xml,
            **merged_common,
        )

    if sp is not None:
        rob = robot if robot is not None else cfg.robot
        return SimLaunchDefaultMujoco(robot=rob, scene_path=sp, **merged_common)

    if isinstance(cfg, SimLaunchDefaultMujoco):
        r = robot if robot is not None else cfg.robot
        sc = cfg.scene_path
        return SimLaunchDefaultMujoco(robot=r, scene_path=sc, **merged_common)
    if isinstance(cfg, SimLaunchRobocasa):
        r = robot if robot is not None else cfg.robot
        if robocasa_task is None:
            task = cfg.robocasa_task
        else:
            task = str(robocasa_task).strip() or cfg.robocasa_task
        return replace(
            cfg,
            robot=r,
            robocasa_task=task,
            **merged_common,
        )
    assert isinstance(cfg, SimLaunchMolmospaces)
    r = robot if robot is not None else cfg.robot
    split = str(molmospaces_split) if molmospaces_split is not None else cfg.split
    idx = int(molmospaces_index) if molmospaces_index is not None else int(cfg.index)
    inst = bool(molmospaces_install) if molmospaces_install is not None else bool(cfg.molmospaces_install)
    return replace(
        cfg,
        robot=r,
        scene=cfg.scene,
        split=split,
        index=idx,
        molmospaces_install=inst,
        **merged_common,
    )


def resolve_sim_launch_for_agent(
    *,
    agent_config_path: str,
    sim_config_cli: str | None,
    port_offset_cli: int,
    default_mujoco_table_if_missing: bool = False,
    default_robot: str = "stretch",
    default_headless: bool = False,
) -> SimLaunchConfig:
    """Merge agent YAML sim section with optional ``--sim-config`` and ``--port-offset`` override.

    When *default_mujoco_table_if_missing* is True and the agent YAML has no ``sim`` / ``sim_config``
    and ``--sim-config`` was not passed, use packaged default-table MuJoCo with *default_robot* and
    *default_headless* (same idea as ``configs/sim/default_table_*.yaml`` without a file).
    """
    cfg: SimLaunchConfig | None = None
    if sim_config_cli and str(sim_config_cli).strip():
        cfg = load_sim_launch_config_from_path(str(sim_config_cli).strip())
    if cfg is None:
        cfg = load_sim_launch_from_agent_yaml(agent_config_path)
    if cfg is None:
        if not default_mujoco_table_if_missing:
            raise ValueError(
                "No sim launch configuration: pass --sim-config PATH, add "
                "'sim_config: configs/sim/....yaml' or 'sim:' inline block to the agent YAML, "
                "or use --start-sim alone for default packaged table + same robot as the agent."
            )
        cfg = SimLaunchDefaultMujoco(
            robot=str(default_robot).strip() or "stretch",
            headless=bool(default_headless),
        )
    if port_offset_cli != 0:
        cfg.port_offset = int(port_offset_cli)
    return cfg
