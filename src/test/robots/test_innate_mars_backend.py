# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from emet.robots.innate_mars import INNATE_MARS_JOINT_NAMES, InnateMarsBackend
from emet.utils.assets import get_robot_mjcf_path


def test_innate_mars_spec():
    b = InnateMarsBackend()
    s = b.get_spec()
    assert s.name == "innate_mars"
    assert s.dof == len(INNATE_MARS_JOINT_NAMES)
    assert s.mjcf_path is not None


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
