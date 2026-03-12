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
# MolmoSpaces runner entrypoint. Run with the MolmoSpaces venv:
#   MOLMOSPACES_PYTHON=.venv-molmospaces/bin/python python -m emet.simulation.molmospaces_runner serve --scene ithor --robot rby1 --viewer
# Imports of molmo_spaces and mujoco are inside the functions that need them (runner env only).

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoSpaces runner (run in .venv-molmospaces)")
    p.add_argument("command", choices=["list-scenes", "install-scene", "serve"], help="Command")
    p.add_argument("--scene", default="ithor", help="Scene name (e.g. ithor, procthor-10k)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"], help="Scene split")
    p.add_argument("--index", type=int, default=0, help="Scene index within split")
    p.add_argument("--robot", default="rby1", help="Robot ID (e.g. rby1, franka_droid)")
    p.add_argument("--headless", action="store_true", help="No GUI")
    p.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer")
    p.add_argument("--rerun", type=str, default="", metavar="PORT_OR_PATH", help="Log to rerun (port or RRD path)")
    p.add_argument("--scene-path", type=str, default="", help="Write installed scene XML to this path")
    return p.parse_args()


def _get_molmo_api():
    """Import get_scenes and install_scene from molmo_spaces (runner env only)."""
    try:
        from molmo_spaces.scenes import get_scenes
    except ImportError:
        try:
            from molmo_spaces import get_scenes
        except ImportError as err:
            raise ImportError("molmo_spaces not found. Install the MolmoSpaces venv: install.sh --molmospaces") from err
    try:
        from molmo_spaces.resource_manager import install_scene_with_objects_and_grasps_from_path
    except ImportError:
        try:
            from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
        except ImportError as err:
            raise ImportError("install_scene_with_objects_and_grasps_from_path not found in molmo_spaces") from err
    return get_scenes, install_scene_with_objects_and_grasps_from_path


def _run_list_scenes() -> int:
    try:
        get_scenes, _ = _get_molmo_api()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1
    from emet.simulation.molmospaces_config import MOLMOSPACES_SCENE_NAMES, MOLMOSPACES_SPLITS

    print("Scenes (name / splits):")
    for name in MOLMOSPACES_SCENE_NAMES:
        splits = []
        for split in MOLMOSPACES_SPLITS:
            try:
                out = get_scenes(name, split)
                if isinstance(out, dict) and split in out:
                    n: int | str = len(out[split]) if out[split] else 0
                else:
                    n = len(out) if isinstance(out, (list, tuple)) else "?"
                splits.append(f"{split}={n}")
            except Exception:
                splits.append(f"{split}=?")
        print(f"  {name}: {', '.join(splits)}")
    return 0


def _find_installed_scene_xml(scene: str, index: int) -> Path | None:
    """Locate installed scene XML under MLSPACES_ASSETS_DIR."""
    assets_dir = os.environ.get("MLSPACES_ASSETS_DIR", "")
    ad = Path(assets_dir) if assets_dir else Path.home() / ".cache" / "molmospaces" / "assets"
    candidate = ad / "scenes" / scene / f"FloorPlan{index + 1}_physics.xml"
    if candidate.exists():
        return candidate
    scene_dir = ad / "scenes" / scene
    if scene_dir.exists():
        xmls = sorted(scene_dir.rglob("*_physics.xml"))
        if xmls:
            return xmls[min(index, len(xmls) - 1)]
        xmls = sorted(scene_dir.rglob("*.xml"))
        if xmls:
            return xmls[min(index, len(xmls) - 1)]
    return None


def _run_install_scene(scene: str, split: str, index: int, scene_path_out: str) -> int:
    try:
        get_scenes, install_scene_with_objects_and_grasps_from_path = _get_molmo_api()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    scenes = get_scenes(scene, split)
    if isinstance(scenes, dict):
        paths = scenes.get(split, [])
    else:
        paths = list(scenes) if scenes else []
    if not paths or index >= len(paths):
        print(f"No scene at {scene} {split}[{index}]. Split has {len(paths)} scenes.", file=sys.stderr)
        return 1
    path = paths[index]
    install_scene_with_objects_and_grasps_from_path(path)
    if not os.environ.get("MLSPACES_ASSETS_DIR"):
        print("MLSPACES_ASSETS_DIR not set; scene installed to default location.", file=sys.stderr)
    if scene_path_out:
        candidate = _find_installed_scene_xml(scene, index)
        if candidate and candidate.exists():
            out = Path(scene_path_out)
            shutil.copy(candidate, out)
            print(f"Wrote scene XML to {out} (from {candidate})")
        else:
            print(
                f"Installed; could not resolve XML to write to {scene_path_out}. Set MLSPACES_ASSETS_DIR.",
                file=sys.stderr,
            )
    return 0


def _run_serve(
    scene: str,
    split: str,
    index: int,
    robot: str,
    headless: bool,
    viewer: bool,
    rerun: str,
    scene_path_out: str,
) -> int:
    try:
        import mujoco

        get_scenes, install_scene_with_objects_and_grasps_from_path = _get_molmo_api()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    scenes = get_scenes(scene, split)
    if isinstance(scenes, dict):
        paths = scenes.get(split, [])
    else:
        paths = list(scenes) if scenes else []
    if not paths or index >= len(paths):
        print(f"No scene at {scene} {split}[{index}]. Run install-scene first or check list-scenes.", file=sys.stderr)
        return 1
    install_scene_with_objects_and_grasps_from_path(paths[index])
    candidate = _find_installed_scene_xml(scene, index)
    if not candidate or not candidate.exists():
        print("No MJCF found after install. Set MLSPACES_ASSETS_DIR.", file=sys.stderr)
        return 1
    model_path = str(candidate)
    if scene_path_out:
        Path(scene_path_out).write_text(model_path)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    rr = None
    if rerun:
        try:
            import rerun as rr_module

            rr_module.init("emet_molmospaces")
            if rerun.isdigit():
                rr_module.connect(port=int(rerun))
            else:
                rr_module.save(rerun)
            rr = rr_module
        except Exception as e:
            print(f"Rerun logging failed: {e}", file=sys.stderr)
    if viewer and not headless:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as v:
            step = 0
            while v.is_running():
                mujoco.mj_step(model, data)
                if rr and step % 10 == 0:
                    try:
                        rr.log("world/step", rr.Scalar(step))
                    except Exception:
                        pass
                step += 1
                v.sync()
        return 0
    # Headless step loop (e.g. for benchmarking)
    step = 0
    try:
        while True:
            mujoco.mj_step(model, data)
            if rr and step % 10 == 0:
                try:
                    rr.log("world/step", rr.Scalar(step))
                except Exception:
                    pass
            step += 1
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    args = _parse()
    if args.command == "list-scenes":
        return _run_list_scenes()
    if args.command == "install-scene":
        return _run_install_scene(args.scene, args.split, args.index, args.scene_path or "")
    if args.command == "serve":
        return _run_serve(
            args.scene,
            args.split,
            args.index,
            args.robot,
            args.headless,
            args.viewer,
            args.rerun or "",
            args.scene_path or "",
        )
    return 1
