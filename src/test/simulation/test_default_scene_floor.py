"""The scene owns its floor; standalone robot preview floors must not overlap it."""
from pathlib import Path

import mujoco

from emet.simulation import mujoco_server


def test_default_merge_keeps_only_environment_floor(monkeypatch, tmp_path: Path):
    (tmp_path / 'scene_environment.xml').write_text('''<mujoco><worldbody>
      <geom name="scene_floor" type="plane" size="2 2 .01"/></worldbody></mujoco>''')
    robot = tmp_path / 'robot.xml'
    robot.write_text('''<mujoco><worldbody><geom name="preview_floor" type="plane" size="2 2 .01"/>
      <body name="robot" pos="0 0 .3"><geom name="chassis" type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    original = robot.read_bytes()
    monkeypatch.setattr(mujoco_server, 'get_mujoco_models_path', lambda: tmp_path)
    monkeypatch.setattr(mujoco_server, 'get_robot_mjcf_path', lambda _: robot)
    model = mujoco_server._load_default_scene_with_robot('example')
    assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
    assert model.geom('scene_floor').id >= 0
    assert model.geom('chassis').id >= 0
    assert robot.read_bytes() == original
    assert not list(tmp_path.glob('robot_no_preview_floor_*'))
