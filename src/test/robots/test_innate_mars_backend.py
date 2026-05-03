# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from pathlib import Path

import pytest

from emet.robots.innate_mars import (
    INNATE_MARS_JOINT_NAMES,
    DummyInnateMarsClient,
    InnateMarsBackend,
)
from emet.utils.assets import get_robot_mjcf_path


def test_innate_mars_spec():
    b = InnateMarsBackend()
    s = b.get_spec()
    assert s.name == "innate_mars"
    assert s.dof == len(INNATE_MARS_JOINT_NAMES)
    canonical = get_robot_mjcf_path("innate_mars")
    assert canonical is not None
    assert Path(s.mjcf_path).resolve() == canonical.resolve()
    assert s.urdf_path is not None
    assert Path(s.urdf_path).resolve() == canonical.with_name("maurice.urdf").resolve()
    assert s.optional_uv_extras == ()
    assert s.dynamem_depth_source_hint == "da3"


def test_innate_mars_mjcf_registered():
    p = get_robot_mjcf_path("innate_mars")
    assert p is not None and p.exists()


def test_merge_scene_loads():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.utils.assets import get_mujoco_models_path

    scene = get_mujoco_models_path() / "scene_environment.xml"
    robot = get_robot_mjcf_path("innate_mars")
    assert scene.exists() and robot is not None
    import os
    import tempfile
    from pathlib import Path

    wrapper = (
        '<?xml version="1.0"?>\n<mujoco model="t">\n'
        f'  <include file="{scene.resolve()}"/>\n'
        f'  <include file="{robot.resolve()}"/>\n'
        "</mujoco>\n"
    )
    d = robot.parent
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="t_", dir=str(d))
    try:
        os.close(fd)
        Path(path).write_text(wrapper)
        m = mujoco.MjModel.from_xml_path(path)
        assert m.nq > 0
    finally:
        Path(path).unlink(missing_ok=True)


def test_innate_mars_merged_scene_matches_standalone_kinematics():
    """How emet serve mujoco merges scene + robot must not change head geom pose vs standalone MJCF."""
    pytest.importorskip("mujoco")
    import mujoco
    import numpy as np

    from emet.simulation.mujoco_server import _load_default_scene_with_robot

    robot = get_robot_mjcf_path("innate_mars")
    assert robot is not None
    m1 = mujoco.MjModel.from_xml_path(str(robot))
    d1 = mujoco.MjData(m1)
    mujoco.mj_forward(m1, d1)
    gid1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")

    m2 = _load_default_scene_with_robot("innate_mars")
    assert m2 is not None
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    gid2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")

    np.testing.assert_allclose(d1.geom_xpos[gid1], d2.geom_xpos[gid2], atol=1e-6)


def test_dummy_innate_mars_utils_reexport_matches_package():
    from emet.robots.innate_mars.dummy_client import DummyInnateMarsClient as Direct
    from emet.utils.dummy_innate_mars_client import DummyInnateMarsClient as ViaUtils

    assert ViaUtils is Direct


def test_innate_mars_wave_operation_smoke():
    """Plan smoke: arm wave mutates joint5 on DummyInnateMarsClient."""
    from emet.robots.innate_mars.emote_backend import InnateMarsWaveOperation

    class _Agent:
        pass

    agent = _Agent()
    agent.robot = DummyInnateMarsClient()
    op = InnateMarsWaveOperation("emote", agent)
    op.run(n_waves=1)
    q, _, _ = agent.robot.get_joint_state()
    assert q.shape[0] == len(INNATE_MARS_JOINT_NAMES)


def test_stereo_right_camera_name_from_spec_innate_mars():
    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation.stereo_camera_utils import stereo_right_camera_name_from_spec

    names = InnateMarsBackend().get_spec().camera_names
    assert stereo_right_camera_name_from_spec(list(names)) == "head_right"


def test_get_robot_spec_and_runtime_notes_innate_mars():
    from emet.robots import format_robot_runtime_notes, get_robot_spec

    s = get_robot_spec("innate_mars")
    assert s is not None
    assert s.optional_uv_extras == ()
    notes = format_robot_runtime_notes(s)
    assert notes is not None
    assert "uv sync" not in notes
    assert "depth_source=" in notes


def test_stereo_right_camera_name_from_spec_galaxea_no_pair():
    from emet.robots.galaxea_r1 import GalaxeaR1Backend
    from emet.simulation.stereo_camera_utils import stereo_right_camera_name_from_spec

    names = GalaxeaR1Backend().get_spec().camera_names
    assert stereo_right_camera_name_from_spec(list(names)) is None
