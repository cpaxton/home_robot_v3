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
# MolmoSpaces runner logic: list-scenes, install-scene, serve.
# Imports molmo_spaces and mujoco here (wrapper-only deps).

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from emet.simulation.molmospaces_config import (
    ensure_molmo_asset_layout_symlinks,
    ensure_molmospaces_assets_dir_env,
)

# Before any ``import molmo_spaces...`` (ASSETS_DIR is fixed at import time).
ensure_molmospaces_assets_dir_env()

_MOLMO_VENV_HELP = (
    "Recreate the MolmoSpaces venv: rm -rf .venv-molmospaces && ./install.sh --molmospaces -y "
    "(Python 3.11+; install emet with --no-deps, then packages/emet_molmospaces; "
    "do not pip install emet-molmospaces in the main 3.10 .venv)."
)


def _xml_path_for_scene_index(scene: str, scene_map: object, split: str, index: int) -> str | Path | None:
    """Resolve a concrete scene XML path from ``get_scenes(...)`` (current or legacy return shape).

    For **ithor**, MolmoSpaces index keys follow **FloorPlan{N}** numbering (N >= 1). Emet uses a
    **0-based scene index** that matches ``FloorPlan{index+1}_physics.xml`` and ``--molmospaces-index``,
    so we look up key ``index + 1`` in the per-split map (slot 0 is unused in upstream maps).
    """
    map_key = index
    if scene == "ithor":
        map_key = index + 1

    if isinstance(scene_map, (list, tuple)):
        if index < 0 or index >= len(scene_map):
            return None
        return scene_map[index]
    if not isinstance(scene_map, dict):
        return None
    per_split = scene_map.get(split)
    if per_split is None:
        return None
    if isinstance(per_split, (list, tuple)):
        if index < 0 or index >= len(per_split):
            return None
        return per_split[index]
    if isinstance(per_split, dict):
        entry = per_split.get(map_key)
        if entry is None:
            return None
        if isinstance(entry, dict):
            for key in ("base", "ceiling", "map"):
                v = entry.get(key)
                if v is not None:
                    return v
            for v in entry.values():
                if v is not None:
                    return v
            return None
        return entry
    return None


def _split_scene_count(scene_map: object, split: str) -> int:
    """Number of scene indices available for *split* (for error messages and guards)."""
    if isinstance(scene_map, (list, tuple)):
        return len(scene_map)
    if not isinstance(scene_map, dict):
        return 0
    per_split = scene_map.get(split)
    if per_split is None:
        return 0
    if isinstance(per_split, (list, tuple, dict)):
        return len(per_split)
    return 0


def _resource_manager_scene_source(scene: str, split: str) -> str:
    """MolmoSpaces ResourceManager ``source`` name for ``scenes`` (see molmo_spaces_constants)."""
    if scene == "ithor":
        return "ithor"
    if scene == "procthor-10k":
        return f"procthor-10k-{split}"
    if scene == "procthor-objaverse":
        return f"procthor-objaverse-{split}"
    if scene == "holodeck-objaverse":
        return f"holodeck-objaverse-{split}"
    if scene == "procthor-objaverse-debug":
        return "procthor-objaverse-debug"
    return scene


def _install_lookup_token(scene: str, index: int) -> str:
    """Token passed to ``install_scene_from_source_index`` / archive index (ithor uses FloorPlan numbering)."""
    if scene == "ithor":
        return str(index + 1)
    return str(index)


def _print_scene_missing_help(scene: str, split: str, index: int, n: int) -> None:
    assets = os.environ.get("MLSPACES_ASSETS_DIR", "") or "(default ~/.cache/molmospaces/assets or venv assets)"
    print(
        f"No scene XML available for {scene} {split}[{index}] (split lists {n} index slots; "
        "some may be empty until packages are downloaded).\n"
        f"  Set MLSPACES_ASSETS_DIR if scenes should live outside the default cache (currently: {assets}).\n"
        "  Or install this scene explicitly, then retry:\n"
        f"    emet molmospaces install-scene --scene {scene} --split {split} --index {index}\n"
        "  For iTHOR, index 0 is the first house (FloorPlan1); MolmoSpaces prints “Using SCENES_ROOT: …” "
        "when the API initializes — that directory must receive the downloaded scene package.",
        file=sys.stderr,
    )


def _try_download_scene_package(scene: str, split: str, index: int, *, install_if_missing: bool) -> bool:
    """Prompt or auto-run MolmoSpaces on-demand scene archive install. Returns True if a install was attempted."""
    env = os.environ.get("EMET_MOLMOSPACES_AUTO_INSTALL", "").strip().lower()
    if env in ("0", "no", "n", "false"):
        return False
    auto = install_if_missing or env in ("1", "yes", "y", "true")
    if not auto:
        if not sys.stdin.isatty():
            return False
        try:
            reply = input("Install scene package now? (Y/n) ")
        except EOFError:
            return False
        if reply.strip().lower() in ("n", "no"):
            return False

    try:
        from molmo_spaces.utils.lazy_loading_utils import install_scene_from_source_index
    except ImportError as e:
        print(f"Could not import install_scene_from_source_index: {e}", file=sys.stderr)
        return False

    src = _resource_manager_scene_source(scene, split)
    token = _install_lookup_token(scene, index)
    try:
        install_scene_from_source_index(src, token)
        return True
    except Exception as e:
        print(f"Scene package install failed ({src!r}, token={token!r}): {e}", file=sys.stderr)
        return False


def _resolve_scene_xml_path(
    scene: str,
    split: str,
    index: int,
    get_scenes,
    *,
    install_if_missing: bool,
) -> str | Path | None:
    """Return a concrete scene XML path, optionally downloading the scene archive first."""
    scenes = get_scenes(scene, split)
    path = _xml_path_for_scene_index(scene, scenes, split, index)
    if path is not None:
        return path
    n = _split_scene_count(scenes, split)
    _print_scene_missing_help(scene, split, index, n)
    if _try_download_scene_package(scene, split, index, install_if_missing=install_if_missing):
        scenes = get_scenes(scene, split)
        path = _xml_path_for_scene_index(scene, scenes, split, index)
    return path


def _get_molmo_api():
    """Import get_scenes and install_scene_with_objects_and_grasps_from_path from molmo_spaces."""
    try:
        from molmo_spaces.molmo_spaces_constants import get_scenes
    except ImportError:
        try:
            from molmo_spaces.scenes import get_scenes
        except ImportError:
            try:
                from molmo_spaces import get_scenes
            except ImportError as err:
                raise ImportError(
                    "Could not import get_scenes from molmo_spaces (tried molmo_spaces_constants, "
                    f"scenes, top-level). {_MOLMO_VENV_HELP}"
                ) from err

    try:
        from molmo_spaces.utils.lazy_loading_utils import (
            install_scene_with_objects_and_grasps_from_path,
        )
    except ImportError:
        try:
            from molmo_spaces.resource_manager import (
                install_scene_with_objects_and_grasps_from_path,
            )
        except ImportError as err:
            raise ImportError(
                "Could not import install_scene_with_objects_and_grasps_from_path "
                f"(tried lazy_loading_utils, resource_manager). {_MOLMO_VENV_HELP}"
            ) from err

    return get_scenes, install_scene_with_objects_and_grasps_from_path


# MolmoSpaces ``install_scene_with_objects_and_grasps_from_path(..., exclude_thor=True)`` skips
# ``../objects/thor/`` references in scene XML; iTHOR houses require those AI2-THOR assets.
_SCENES_NEEDING_THOR_OBJECT_INSTALL = frozenset({"ithor"})


def _ensure_resource_cache_ready() -> None:
    """Populate ``MLSPACES_CACHE_DIR`` so :meth:`ResourceManager.install_packages` can run.

    ``setup_resource_manager`` may skip :meth:`ResourceManager.setup` when the *assets* tree's
    version manifest matches, while ``MLSPACES_CACHE_DIR`` is empty or inconsistent. The cache
    copy of ``LOCAL_MANIFEST`` can also list a version as installed while ``_setup_source`` skips
    the download (``must_download`` false) and then fails to open the remote manifest.

    If any ``scenes`` / ``objects`` / ``grasps`` remote manifest is missing under
    ``cache_dir``, we remove the cache ``LOCAL_MANIFEST``, delete those versioned cache
    directories, reset the MolmoSpaces resource-manager singleton, and run ``setup()`` on a
    fresh manager so manifests are fetched before ``get_scenes`` / ``install_packages``.

    **Important:** ``ResourceManager.setup`` skips ``_setup_source`` when COMBINED/REMOTE manifests
    exist under ``symlink_dir/<type>/<source>``. Those paths are often **symlinks** into an *old*
    ``MLSPACES_CACHE_DIR`` (e.g. ``~/.cache/molmo-spaces-resources/...``). After resolving,
    ``parent.name == <version>`` matches and the new ``cache_dir`` never gets populated.  For any
    source whose *versioned* cache dir lacks the remote manifest, we remove the symlink install path
    ``symlink_dir/<type>/<source>`` so setup cannot skip.
    """
    import molmo_spaces.molmo_spaces_constants as mconst
    from molmo_spaces.molmo_spaces_constants import DATA_TYPE_TO_SOURCE_TO_VERSION
    from molmospaces_resources.constants import LOCAL_MANIFEST_NAME, REMOTE_MANIFEST_NAME
    from molmospaces_resources.setup_utils import _RESOURCE_MANAGERS as _rm_registries

    rm = mconst.get_resource_manager()
    cache_dir = Path(rm.cache_dir)
    cache_local = cache_dir / LOCAL_MANIFEST_NAME
    need_setup = False

    for dt in ("scenes", "objects", "grasps"):
        for _src, _ver in DATA_TYPE_TO_SOURCE_TO_VERSION.get(dt, {}).items():
            cp = rm.cache_path(dt, _src)
            if (cp / REMOTE_MANIFEST_NAME).is_file():
                continue
            need_setup = True
            if cp.exists():
                shutil.rmtree(cp, ignore_errors=True)
            inst = rm.symlink_path(dt, _src)
            try:
                if inst.is_symlink():
                    inst.unlink()
                elif inst.exists():
                    shutil.rmtree(inst, ignore_errors=True)
            except OSError:
                pass

    if not need_setup:
        return

    try:
        cache_local.unlink(missing_ok=True)
    except OSError:
        pass

    mconst._RESOURCE_MANAGER = None
    _rm_registries.clear()
    rm = mconst.get_resource_manager()
    rm.setup()


def _install_scene_with_deps(path: str | Path, scene: str) -> None:
    """Install scene package plus object/grasp archives. iTHOR must not exclude THOR object meshes."""
    _ensure_resource_cache_ready()
    _, install_fn = _get_molmo_api()
    install_fn(
        path,
        exclude_thor=(scene not in _SCENES_NEEDING_THOR_OBJECT_INSTALL),
    )
    # Molmo scene XML resolves ``../objects`` from ``scenes/<dataset>/`` to ``scenes/objects``, not
    # the real ``objects/`` tree; link so MuJoCo finds meshes (see molmospaces_config).
    ensure_molmo_asset_layout_symlinks()


def run_list_scenes() -> int:
    from emet.simulation.molmospaces_config import MOLMOSPACES_SCENE_NAMES, MOLMOSPACES_SPLITS

    try:
        get_scenes, _ = _get_molmo_api()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

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


def _scene_xml_search_roots(scene: str) -> list[Path]:
    """Directory(ies) containing scene XMLs for *scene* (match molmo_spaces_constants layout)."""
    roots: list[Path] = []
    env = os.environ.get("MLSPACES_ASSETS_DIR", "").strip()
    if env:
        roots.append(Path(env) / "scenes" / scene)
    try:
        from molmo_spaces.molmo_spaces_constants import get_scenes_root

        roots.append(get_scenes_root() / scene)
    except ImportError:
        pass
    try:
        from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

        roots.append(Path(ASSETS_DIR) / "scenes" / scene)
    except ImportError:
        pass
    roots.append(Path.home() / ".cache" / "molmospaces" / "assets" / "scenes" / scene)

    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        try:
            key = str(r.resolve())
        except (OSError, RuntimeError):
            key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _find_installed_scene_xml(scene: str, index: int) -> Path | None:
    """Locate installed scene MJCF under the same roots MolmoSpaces uses (not only ~/.cache)."""
    for scene_dir in _scene_xml_search_roots(scene):
        candidate = scene_dir / f"FloorPlan{index + 1}_physics.xml"
        if candidate.is_file():
            return candidate
        if scene_dir.is_dir():
            xmls = sorted(scene_dir.rglob("*_physics.xml"))
            if xmls:
                return xmls[min(index, len(xmls) - 1)]
            xmls = sorted(scene_dir.rglob("*.xml"))
            if xmls:
                return xmls[min(index, len(xmls) - 1)]
    return None


def _scene_xml_after_install(path_from_index: str | Path, scene: str, index: int) -> Path | None:
    """Prefer the path from ``get_scenes`` after linking (often under ASSETS_DIR); else search by index."""
    p = Path(path_from_index)
    try:
        if p.is_file():
            return p.resolve()
    except OSError:
        pass
    return _find_installed_scene_xml(scene, index)


def run_install_scene(
    scene: str, split: str, index: int, scene_path_out: str, *, install_if_missing: bool = False
) -> int:
    _ensure_resource_cache_ready()
    try:
        get_scenes, _ = _get_molmo_api()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    path = _resolve_scene_xml_path(scene, split, index, get_scenes, install_if_missing=install_if_missing)
    if path is None:
        return 1
    _install_scene_with_deps(path, scene)
    if not os.environ.get("MLSPACES_ASSETS_DIR"):
        print("MLSPACES_ASSETS_DIR not set; scene installed to default location.", file=sys.stderr)
    if scene_path_out:
        candidate = _scene_xml_after_install(path, scene, index)
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


def _get_robot_mjcf_path(robot: str) -> Path | None:
    """Resolve robot id to vendored MJCF under emet (rby1, galaxea_r1, innate_mars, …).

    Delegates to :func:`emet.utils.assets.get_robot_mjcf_path` so the registry stays in one place.
    """
    try:
        from emet.utils.assets import get_robot_mjcf_path as emet_robot_mjcf
    except ImportError:
        return None
    key = robot.lower().replace("-", "_")
    p = emet_robot_mjcf(key)
    if p is None or not p.is_file():
        return None
    return p


def _merge_robot_into_scene(scene_xml_path: Path, robot_mjcf_path: Path) -> Path:
    """Write a temporary MJCF that includes the scene and the robot. Uses a top-level wrapper so both
    are included as full models; MuJoCo merges multiple top-level includes. We write the merged file
    into the robot's directory so the robot's assetdir='meshes' and mesh paths resolve correctly.
    """
    import tempfile

    scene_abs = str(scene_xml_path.resolve())
    robot_path = robot_mjcf_path.resolve()
    robot_abs = str(robot_path)
    wrapper = f'''<?xml version="1.0"?>
<mujoco model="scene_with_robot">
  <include file="{scene_abs}"/>
  <include file="{robot_abs}"/>
</mujoco>
'''
    # Write into robot dir so included robot XML's relative paths (meshes/) resolve.
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="scene_robot_", dir=robot_path.parent)
    os.close(fd)
    Path(path).write_text(wrapper)
    return Path(path)


def run_merge_scene(
    scene: str,
    split: str,
    index: int,
    robot: str,
    output: str,
    *,
    install_if_missing: bool = False,
) -> int:
    """Install scene (if needed), merge robot MJCF, write persistent merged XML for emet serve mujoco --scene_path."""
    _ensure_resource_cache_ready()
    try:
        get_scenes, _ = _get_molmo_api()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    path = _resolve_scene_xml_path(scene, split, index, get_scenes, install_if_missing=install_if_missing)
    if path is None:
        return 1
    _install_scene_with_deps(path, scene)
    candidate = _scene_xml_after_install(path, scene, index)
    if not candidate or not candidate.exists():
        print(
            "No MJCF found after install. Scene XML is usually under MolmoSpaces ASSETS_DIR "
            "(same tree as ‘Using SCENES_ROOT: …’). Set MLSPACES_ASSETS_DIR to that assets root "
            "if installs are not found under the default package path.",
            file=sys.stderr,
        )
        return 1
    robot_mjcf = _get_robot_mjcf_path(robot)
    if robot_mjcf is None:
        print(
            f"No bundled MJCF merge for robot '{robot}'. "
            "Use rby1, galaxea_r1, innate_mars, stretch, or merge manually.",
            file=sys.stderr,
        )
        return 1
    merged_path: Path | None = None
    try:
        merged_path = _merge_robot_into_scene(candidate, robot_mjcf)
        out = Path(output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(merged_path, out)
    except Exception as e:
        print(f"merge-scene failed: {e}", file=sys.stderr)
        return 1
    finally:
        if merged_path is not None and merged_path.exists():
            try:
                merged_path.unlink(missing_ok=True)
            except Exception:
                pass
    print(f"Merged MJCF: {out}")
    print(
        f"Run server: emet serve mujoco --robot {robot} --scene_path {out} [--headless]\n"
        f"Run agent:  emet run agent --robot-ip 127.0.0.1 --robot {robot}",
    )
    return 0


def run_serve(
    scene: str,
    split: str,
    index: int,
    robot: str,
    headless: bool,
    viewer: bool,
    rerun: str,
    scene_path_out: str,
    *,
    install_if_missing: bool = False,
) -> int:
    _ensure_resource_cache_ready()
    try:
        import mujoco

        get_scenes, _ = _get_molmo_api()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    path = _resolve_scene_xml_path(scene, split, index, get_scenes, install_if_missing=install_if_missing)
    if path is None:
        return 1
    _install_scene_with_deps(path, scene)
    candidate = _scene_xml_after_install(path, scene, index)
    if not candidate or not candidate.exists():
        print(
            "No MJCF found after install. Scene XML is usually under MolmoSpaces ASSETS_DIR "
            "(same tree as ‘Using SCENES_ROOT: …’). Set MLSPACES_ASSETS_DIR to that assets root "
            "if installs are not found under the default package path.",
            file=sys.stderr,
        )
        return 1
    scene_path = candidate
    model_path = str(scene_path)
    robot_mjcf = _get_robot_mjcf_path(robot)
    merged_path: Path | None = None
    if robot_mjcf is not None:
        try:
            merged_path = _merge_robot_into_scene(scene_path, robot_mjcf)
            model_path = str(merged_path)
        except Exception as e:
            print(f"Could not merge robot {robot} into scene: {e}", file=sys.stderr)
            if merged_path and merged_path.exists():
                merged_path.unlink(missing_ok=True)
    if scene_path_out:
        Path(scene_path_out).write_text(model_path)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    if merged_path is not None and merged_path.exists():
        try:
            merged_path.unlink(missing_ok=True)
        except Exception:
            pass
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


def main_runner(argv: list[str] | None = None) -> int:
    """Dispatch to list-scenes, install-scene, merge-scene, or serve. argv defaults to sys.argv[1:]."""
    import argparse

    p = argparse.ArgumentParser(description="MolmoSpaces wrapper (list-scenes, install-scene, merge-scene, serve)")
    p.add_argument("command", choices=["list-scenes", "install-scene", "merge-scene", "serve"])
    p.add_argument("--scene", default="ithor")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--robot", default="stretch", help="Robot ID (default: stretch)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--rerun", type=str, default="")
    p.add_argument("--scene-path", type=str, default="")
    p.add_argument("--output", "-o", type=str, default="", help="Output MJCF path (required for merge-scene)")
    p.add_argument(
        "--install-if-missing",
        action="store_true",
        help="Download/link the scene archive without prompting (for install-scene, merge-scene, serve).",
    )
    args = p.parse_args(argv)

    if args.command == "list-scenes":
        return run_list_scenes()
    if args.command == "install-scene":
        return run_install_scene(
            args.scene,
            args.split,
            args.index,
            args.scene_path or "",
            install_if_missing=args.install_if_missing,
        )
    if args.command == "merge-scene":
        if not (args.output or "").strip():
            print("--output / -o is required for merge-scene", file=sys.stderr)
            return 1
        return run_merge_scene(
            args.scene,
            args.split,
            args.index,
            args.robot,
            args.output.strip(),
            install_if_missing=args.install_if_missing,
        )
    if args.command == "serve":
        return run_serve(
            args.scene,
            args.split,
            args.index,
            args.robot,
            args.headless,
            args.viewer,
            args.rerun or "",
            args.scene_path or "",
            install_if_missing=args.install_if_missing,
        )
    return 1
