# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from pathlib import Path

from emet.robots.innate_mars import INNATE_MARS_JOINT_NAMES, InnateMarsBackend
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


def test_innate_mars_mjcf_registered():
    p = get_robot_mjcf_path("innate_mars")
    assert p is not None and p.exists()


def test_merge_scene_loads():
    import mujoco

    from emet.utils.assets import get_mujoco_models_path

    scene = get_mujoco_models_path() / "scene_default.xml"
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
    import numpy as np
    import mujoco

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
