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

"""Central scene resolution for ``emet serve mujoco`` (default table, custom merged MJCF, Robocasa wizard)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import emet.utils.logger as log
from emet.utils.assets import get_mujoco_models_path, get_robot_mjcf_path

logger = log.Logger(__name__)

DEFAULT_SCENE_NO_ROBOT = "scene_environment.xml"  # canonical table room; scene_default.xml aliases this


@dataclass(frozen=True)
class LoadedScene:
    """Normalized scene payload for :class:`~emet.simulation.base_mujoco_zmq_server.BaseMujocoZmqServer` / CLI."""

    scene_model: Any | None = None
    scene_xml: str | None = None
    scene_disk_path: str | None = None
    scene_source_basename: str | None = None
    zmq_environment: dict[str, Any] | None = None
    objects_info: Any = None


def default_packaged_stretch_scene_xml_path() -> str:
    return str((get_mujoco_models_path() / "scene.xml").resolve())


def build_zmq_environment(
    *,
    molmospaces_session_scene: str | None,
    molmospaces_session_split: str | None,
    molmospaces_session_index: int | None,
    use_robocasa: bool,
    robocasa_task: str,
    robocasa_style: int,
    robocasa_layout: int,
) -> dict[str, Any] | None:
    if molmospaces_session_scene:
        return {
            "kind": "molmospaces",
            "scene": molmospaces_session_scene,
            "split": molmospaces_session_split or "train",
            "index": int(0 if molmospaces_session_index is None else molmospaces_session_index),
        }
    if use_robocasa:
        return {
            "kind": "robocasa",
            "task": robocasa_task,
            "style": int(robocasa_style),
            "layout": int(robocasa_layout),
        }
    return None


def scene_source_basename_from_path(scene_path: str | None) -> str | None:
    if scene_path and str(scene_path).strip():
        return Path(str(scene_path).strip()).name
    return None


def spawn_scene_disk_path(scene_path: str | None) -> str | None:
    spath = (scene_path or "").strip()
    if spath and Path(spath).is_file():
        return str(Path(spath).resolve())
    return None


def load_default_scene_with_robot(robot_key: str) -> Any | None:
    """Merge ``scene_environment.xml`` with robot MJCF; return ``MjModel`` or ``None``."""
    import mujoco

    models_path = get_mujoco_models_path()
    scene_path = models_path / DEFAULT_SCENE_NO_ROBOT
    robot_path = get_robot_mjcf_path(robot_key)
    if not scene_path.exists() or not robot_path:
        return None
    scene_abs = str(scene_path.resolve())
    robot_abs = str(robot_path.resolve())
    meshes_dir = robot_path.parent / "meshes"
    compiler_line = ""
    if meshes_dir.is_dir():
        mesh_abs = str(meshes_dir.resolve())
        compiler_line = f'  <compiler meshdir="{mesh_abs}" angle="radian" coordinate="local" eulerseq="zyx"/>\n'
    wrapper = (
        '<?xml version="1.0"?>\n'
        '<mujoco model="default_scene_with_robot">\n'
        f"{compiler_line}"
        f'  <include file="{scene_abs}"/>\n'
        f'  <include file="{robot_abs}"/>\n'
        "</mujoco>\n"
    )
    robot_dir = str(robot_path.parent)
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="scene_robot_", dir=robot_dir)
    try:
        os.close(fd)
        Path(path).write_text(wrapper)
        model = mujoco.MjModel.from_xml_path(path)
        try:
            from emet.simulation.default_table_spawn import snap_packaged_table_robot_to_scene_floor

            snap_packaged_table_robot_to_scene_floor(model, robot_key=robot_key)
        except Exception as e:
            logger.warning("default table spawn adjust failed (%r); using MJCF defaults.", e)
        return model
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def load_merged_mjcf_from_disk(scene_path: str) -> Any:
    """Load ``MjModel`` from ``scene_path`` after MolmoSpaces asset symlink prep."""
    import mujoco

    from emet.simulation.molmospaces_config import ensure_molmo_asset_layout_symlinks

    ensure_molmo_asset_layout_symlinks()
    return mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))


def resolve_merged_physics_scene(
    *,
    robot_key: str,
    scene_path: str | None,
    use_robocasa: bool,
    wizard_scene_model: Any | None,
    wizard_scene_xml: str | None,
    wizard_objects_info: Any,
    zmq_environment: dict[str, Any] | None,
    scene_source_basename: str | None,
) -> LoadedScene:
    """Resolve ``scene_model`` / ``scene_xml`` for merged-MJCF :class:`~emet.simulation.base_mujoco_zmq_server.BaseMujocoZmqServer` (and Stretch subclass).

    When ``use_robocasa`` is true, wizard outputs are used (model and/or XML per wizard). When false,
    loads ``--scene-path`` if set, else packaged default table + robot.
    """
    disk = spawn_scene_disk_path(scene_path)
    if use_robocasa:
        return LoadedScene(
            scene_model=wizard_scene_model,
            scene_xml=wizard_scene_xml,
            scene_disk_path=disk,
            scene_source_basename=scene_source_basename,
            zmq_environment=zmq_environment,
            objects_info=wizard_objects_info,
        )
    custom = (scene_path or "").strip()
    if custom and Path(custom).is_file():
        try:
            model = load_merged_mjcf_from_disk(custom)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MJCF from --scene-path {custom}: {e}\n"
                "If this is a MolmoSpaces scene, ensure THOR objects are installed and "
                "MLSPACES_ASSETS_DIR / MLSPACES_CACHE_DIR match the env used for merge "
                "(sibling dirs; they must not be the same path). Re-run merge: "
                "emet molmospaces merge-scene or emet serve mujoco --molmospaces-scene ..."
            ) from e
        return LoadedScene(
            scene_model=model,
            scene_xml=None,
            scene_disk_path=disk,
            scene_source_basename=scene_source_basename,
            zmq_environment=zmq_environment,
            objects_info=None,
        )
    model = load_default_scene_with_robot(robot_key)
    if model is None:
        raise FileNotFoundError(
            "Default scene with robot not found (scene_environment.xml or robot MJCF missing). "
            "Use --scene-path with a merged MJCF, --use-robocasa for Robocasa-generated scenes, "
            "or run from repo root with assets."
        )
    return LoadedScene(
        scene_model=model,
        scene_xml=None,
        scene_disk_path=None,
        scene_source_basename=scene_source_basename,
        zmq_environment=zmq_environment,
        objects_info=None,
    )
