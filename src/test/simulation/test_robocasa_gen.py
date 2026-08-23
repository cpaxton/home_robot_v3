# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
#
# Robocasa tests: lightweight import/task list (always runs), full env test,
# and emet-serve-matching wizard test (same entry point as emet serve mujoco --use-robocasa).
# Assets are ensured by conftest (downloaded if missing).
# Run: pytest src/test/simulation/test_robocasa_gen.py -v
#
# Why Stretch loads without --use-robocasa but failed with it:
# - Without --use-robocasa the server loads the default scene from a file (scene.xml).
#   That file is static and Stretch is included or the scene has no mesh-inertia issue.
# - With --use-robocasa we build one XML string: kitchen from env.sim.model.get_xml()
#   (which serializes the compiled model and can omit mesh inertia) + <include stretch>.
#   MuJoCo 2.x requires every mesh asset to have inertia (e.g. inertia="shell"). So we
#   run ensure_mesh_inertia() on the kitchen XML and on the generated Stretch XML.

import numpy as np
import pytest


def test_stretch_model_loads():
    """Stretch MuJoCo model loads (generated XML has inertia on meshes for MuJoCo 2.x).

    Uses the same generation path as emet serve mujoco --use-robocasa: get_absolute_path_stretch_xml
    writes stretch_temp_abs.xml with ensure_mesh_inertia() so mesh assets have inertia="shell".
    Without that, MjModel.from_xml_path would raise "mesh volume is too small" or
    "inertia should be specified in the mesh asset".
    """
    pytest.importorskip("mujoco")
    try:
        from emet.simulation.stretch_mujoco.utils import get_absolute_path_stretch_xml
    except Exception as e:
        pytest.skip(f"Stretch utils not available: {e}")
    path = get_absolute_path_stretch_xml(robot_pose_attrib=None)
    assert path is not None
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    assert model is not None
    assert model.nq >= 0
    assert model.nv >= 0


def test_robocasa_import_and_tasks():
    """Robocasa imports and registers kitchen tasks (always runs, no heavy assets)."""
    import robocasa  # noqa: F401
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

    tasks = list(ALL_KITCHEN_ENVIRONMENTS)
    assert len(tasks) > 0
    assert "PickPlaceCounterToCabinet" in tasks


def test_robocasa_env_sim_after_reset():
    """After robosuite.make() and reset(), env.sim must be set and yield model XML."""
    import robocasa  # noqa: F401  # register envs
    import robosuite
    from robocasa.utils.errors import PlacementError
    from robosuite import load_part_controller_config

    config = {
        "env_name": "PickPlaceCounterToCabinet",
        "robots": "PandaMobile",
        "controller_configs": load_part_controller_config(default_controller="OSC_POSE"),
        "translucent_robot": False,
        "layout_and_style_ids": [[1, 1]],
    }
    try:
        env = robosuite.make(
            **config,
            has_offscreen_renderer=False,
            render_camera=None,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
        )
        env.reset()
    except (FileNotFoundError, ValueError, TypeError) as e:
        pytest.skip(
            f"Kitchen env could not be built (asset/format mismatch): {e}. "
            "Full compatible Robocasa assets may be required."
        )
    except PlacementError as e:
        pytest.skip(
            f"Kitchen placement failed (layout/object mismatch): {e}. Full compatible Robocasa assets may be required."
        )
    except RuntimeError as e:
        if "50 times" in str(e) and "could not initialize" in str(e).lower():
            pytest.skip(
                "Kitchen reset hit max placement retries (scene built, placement failed). "
                "Full compatible Robocasa assets may be required."
            )
        raise
    assert env.sim is not None, "sim must be set after reset()"
    assert env.sim.model is not None
    xml = env.sim.model.get_xml()
    assert isinstance(xml, str) and len(xml) > 100
    env.close()


def test_robocasa_wizard_matches_emet_serve():
    """Call the same model_generation_wizard used by emet serve mujoco --use-robocasa.

    Uses the same import path and default args as mujoco_server main():
    task=PickPlaceCounterToCabinet, layout=1, style=1. On success returns (MjModel, xml, objects_info).
    Skips on placement/reset failures (same as when running the server with current assets).
    """
    try:
        from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard
    except Exception as e:
        pytest.skip(
            f"Robocasa wizard could not be imported (sim not installed?): {e}. "
            "Run: emet install sim  then  emet sync -e sim"
        )

    # Same defaults as: emet serve mujoco --use-robocasa (no --robocasa-task/layout/style)
    task = "PickPlaceCounterToCabinet"
    layout = 1
    style = 1
    write_to_file = None

    try:
        scene_model, scene_xml, objects_info = model_generation_wizard(
            task=task,
            layout=layout,
            style=style,
            write_to_file=write_to_file,
        )
    except Exception as e:
        from robocasa.utils.errors import PlacementError

        if isinstance(e, PlacementError):
            pytest.skip(
                f"Wizard placement failed (layout/object mismatch): {e}. "
                "Full compatible Robocasa assets may be required."
            )
        if isinstance(e, RuntimeError) and "50 times" in str(e) and "could not initialize" in str(e).lower():
            pytest.skip(
                "Wizard hit max placement retries (scene built, placement failed). "
                "Full compatible Robocasa assets may be required."
            )
        raise

    # Same contract as mujoco_server: server uses scene_model, scene_xml, objects_info
    import mujoco

    assert scene_model is not None
    assert isinstance(scene_model, mujoco.MjModel)
    assert isinstance(scene_xml, str)
    assert len(scene_xml) > 100
    assert isinstance(objects_info, dict)
    hint = objects_info.get("_emet_spawn_hint_xyt")
    assert hint is not None and len(hint) == 3
    bid = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if bid >= 0:
        data = mujoco.MjData(scene_model)
        mujoco.mj_forward(scene_model, data)
        pos = data.body(bid).xpos[:2]
        assert abs(float(pos[0]) - float(hint[0])) < 0.05
        assert abs(float(pos[1]) - float(hint[1])) < 0.05


def test_robocasa_obj_main_placement_is_seed_deterministic():
    """Same task/layout/style/seed → identical ``obj_main`` category and position."""
    pytest.importorskip("robocasa")
    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    kwargs = {
        "task": "PickPlaceCounterToCabinet",
        "layout": 1,
        "style": 1,
        "robot": "stretch",
        "seed": 0,
    }
    _m0, _x0, p0 = model_generation_wizard(**kwargs)
    _m1, _x1, p1 = model_generation_wizard(**kwargs)
    assert "obj_main" in p0 and "obj_main" in p1
    assert p0["obj_main"]["cat"] == p1["obj_main"]["cat"]
    np.testing.assert_allclose(p0["obj_main"]["pos"], p1["obj_main"]["pos"], rtol=0, atol=1e-6)
