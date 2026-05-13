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
    urdf = canonical.with_name("maurice.urdf")
    if urdf.is_file():
        assert s.urdf_path is not None
        assert Path(s.urdf_path).resolve() == urdf.resolve()
    else:
        assert s.urdf_path is None
    assert s.optional_uv_extras == ()
    assert s.dynamem_depth_source_hint == "da3"
    assert s.default_dynav_config == "dynav_innate_mars.yaml"
    assert s.planar_base_joint_names == ("base_x", "base_y", "base_yaw")
    assert s.robosuite_rgb_depth_ops == ("flipud",)


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


def test_innate_mars_pinhole_K_chain_flipud_ops():
    """Sim uses ``flipud`` on Renderer RGB/depth; intrinsics chain must match projected pixels."""
    import numpy as np

    from emet.utils.pinhole_intrinsics import chain_pinhole_K_pixel_ops

    np.random.seed(1)
    K0 = np.array([[400.0, 0.0, 320.0], [0.0, 380.0, 239.5], [0.0, 0.0, 1.0]])
    h0, w0 = 480, 640
    ops: tuple[str, ...] = ("flipud",)
    Kf, hf, wf = chain_pinhole_K_pixel_ops(K0, h0, w0, ops)
    Hm = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(h0) - 1.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(Kf, Hm @ K0, rtol=0, atol=1e-9)
    assert hf == h0 and wf == w0
    for _ in range(20):
        X, Y, Z = np.random.uniform(-0.1, 0.1), np.random.uniform(-0.08, 0.08), np.random.uniform(0.8, 2.0)
        p = K0 @ np.array([X, Y, Z])
        u0, v0, _ = p / p[2]
        p2 = Kf @ np.array([X, Y, Z])
        u2p, v2p, _ = p2 / p[2]
        # flipud maps row v → h0 - 1 - v before rot90 chaining (none here)
        v_flipped = h0 - 1.0 - v0
        assert np.hypot(u2p - u0, v2p - v_flipped) < 1e-3


def test_innate_mars_flipud_pixels_match_numpy_flipud():
    import numpy as np

    from emet.utils.pinhole_intrinsics import apply_pinhole_pixel_ops

    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    ops = ("flipud",)
    out = apply_pinhole_pixel_ops(img, ops)
    np.testing.assert_array_equal(out, np.flipud(img))


def test_innate_mars_head_stereo_cameras_match_mjcf():
    """Stereo: 60 mm baseline on head **X**, shared optics ~−head/world **Y**, matches ``innate_mars.xml`` MJCF mounts."""
    pytest.importorskip("mujoco")
    import mujoco
    import numpy as np

    from emet.robots.innate_mars import InnateMarsBackend

    def cam_look_world(d, cid: int) -> np.ndarray:
        R = np.asarray(d.cam_xmat[cid]).reshape(3, 3)
        v = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return v / np.linalg.norm(v)

    spec = InnateMarsBackend().get_spec()
    assert spec.mjcf_path is not None
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_left")
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_right")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_base")
    assert lid >= 0 and rid >= 0 and bid >= 0

    pl = np.asarray(data.cam_xpos[lid], dtype=np.float64).ravel()
    pr = np.asarray(data.cam_xpos[rid], dtype=np.float64).ravel()
    # MJCF: Rz(+π/2) remaps URDF Y offsets → −X stereo separation (~60 mm on head X).
    np.testing.assert_allclose(pl[1], pr[1], atol=1e-6)
    np.testing.assert_allclose(pl[2], pr[2], atol=1e-6)
    np.testing.assert_allclose(pr[0] - pl[0], 0.06, rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(pr - pl), 0.06, rtol=0, atol=1e-5)

    np.testing.assert_allclose(float(model.cam_pos[lid][2]), float(model.cam_pos[rid][2]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(model.cam_pos[lid][2]), -0.000275, rtol=0, atol=1e-9)

    Rl = np.asarray(data.cam_xmat[lid]).reshape(3, 3)
    Rr = np.asarray(data.cam_xmat[rid]).reshape(3, 3)
    np.testing.assert_allclose(Rl, Rr, atol=1e-6)
    np.testing.assert_allclose(model.cam_fovy[lid], model.cam_fovy[rid])
    np.testing.assert_allclose(float(model.cam_fovy[lid]), 80.0)

    # Rz(−π/2)×REP ⇒ MuJoCo −Z ∥ −head **Y** (table-forward sim hack vs raw URDF optical +Z ∥ +head **X**).
    look_h = cam_look_world(data, lid)
    look_b = cam_look_world(data, bid)
    toward_table = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    assert float(np.dot(look_h, toward_table)) > 0.97
    assert abs(float(np.dot(look_h, look_b))) < 0.1

    # Same OpenCV camera→world rotation from MuJoCo cam_xmat + diag(1,-1,-1) as RobosuiteZmqServer._camera_pose_world.
    D = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(Rl @ D, Rr @ D, atol=1e-6)


def test_head_stereo_center_rays_miss_head_geom():
    """Any visible hit along optic axis must not be head_geom in the near wedge (black mesh artifact)."""
    pytest.importorskip("mujoco")
    import mujoco
    import numpy as np

    from emet.simulation.mujoco_server import _load_default_scene_with_robot

    model = _load_default_scene_with_robot("innate_mars")
    assert model is not None
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")
    assert hid >= 0

    def prim_fwd(name: str) -> tuple[int, float]:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        R = np.asarray(data.cam_xmat[cid]).reshape(3, 3)
        pos = np.asarray(data.cam_xpos[cid], dtype=np.float64).ravel()
        look = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        look /= np.linalg.norm(look)
        gidbuf = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pos + 6e-4 * look, look, None, 1, -1, gidbuf)
        return int(gidbuf[0]) if dist >= 0 else -1, float(dist)

    for cam in ("head_left", "head_right"):
        gid, dist = prim_fwd(cam)
        if dist >= 0:
            assert gid != hid, (cam, gid, dist)
            assert dist > 0.08, (cam, dist)


def test_innate_mars_camera_arm_table_aim_heads_match_at_default_pose():
    """Head stereo aims −world **Y** (sim hack); wrist uses URDF pitch + optical→MuJoCo (world ray differs)."""
    pytest.importorskip("mujoco")
    import mujoco
    import numpy as np

    from emet.simulation.mujoco_server import _load_default_scene_with_robot

    def cam_look_world(d, m, name: str) -> np.ndarray:
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        R = np.asarray(d.cam_xmat[cid]).reshape(3, 3)
        v = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return v / np.linalg.norm(v)

    model = _load_default_scene_with_robot("innate_mars")
    assert model is not None
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    look_h = cam_look_world(data, model, "head_left")
    look_a = cam_look_world(data, model, "camera_arm")
    toward_table = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    assert float(np.dot(look_h, toward_table)) > 0.96
    assert float(np.linalg.norm(look_a)) > 0.99
    assert float(look_a[2]) < -0.35


def test_innate_mars_head_nod_montage_sequence_records_varying_images(tmp_path):
    """Local MuJoCo: sweep joint_head and write montages; pixel means should change across poses."""
    pytest.importorskip("mujoco")
    import cv2
    import numpy as np

    from emet.app.preview_robot_cameras import record_head_nod_montage_sequence

    outs = record_head_nod_montage_sequence(
        "innate_mars",
        tmp_path,
        n_frames=8,
        motion="once",
        angle_low_rad=-0.08,
        angle_high_rad=0.12,
    )
    assert len(outs) == 8

    im0 = cv2.imread(str(outs[0]))
    im_last = cv2.imread(str(outs[-1]))
    assert im0 is not None and im_last is not None and im0.shape == im_last.shape
    mse = float(np.mean((im0.astype(np.float64) - im_last.astype(np.float64)) ** 2))
    assert mse > 50.0

    bounce = record_head_nod_montage_sequence(
        "innate_mars",
        tmp_path / "bounce",
        n_frames=11,
        motion="bounce",
        angle_low_rad=-0.05,
        angle_high_rad=0.1,
    )
    assert len(bounce) == 11


def test_innate_mars_nod_montage_with_arm_joint_smoke(tmp_path):
    """Head nod + parallel joint5 sweep writes PNGs; arm labels differ across frames."""
    pytest.importorskip("mujoco")
    import cv2
    import numpy as np

    from emet.app.preview_robot_cameras import record_head_nod_montage_sequence

    outs = record_head_nod_montage_sequence(
        "innate_mars",
        tmp_path,
        n_frames=5,
        motion="once",
        angle_low_rad=-0.06,
        angle_high_rad=0.08,
        arm_joint_name="joint5",
        arm_angle_low_rad=-0.12,
        arm_angle_high_rad=0.12,
    )
    assert len(outs) == 5
    im0 = cv2.imread(str(outs[0]))
    im_mid = cv2.imread(str(outs[2]))
    assert im0 is not None and im_mid is not None
    mse = float(np.mean((im0.astype(np.float64) - im_mid.astype(np.float64)) ** 2))
    assert mse > 20.0


def test_innate_mars_joint_head_hinge_matches_mjcf_sim_nods_gaze():
    """Sim hinge +head **X**: stereo ~−**Y** gaze pitches under ``joint_head`` (URDF −Y hinge differs — see MJCF comment)."""
    pytest.importorskip("mujoco")
    import mujoco
    import numpy as np

    from emet.simulation.mujoco_server import _load_default_scene_with_robot

    model = _load_default_scene_with_robot("innate_mars")
    assert model is not None
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head")
    assert jid >= 0
    ax = np.asarray(model.jnt_axis[jid]).ravel()
    np.testing.assert_allclose(ax / np.linalg.norm(ax), np.array([1.0, 0.0, 0.0]), atol=1e-7)

    data = mujoco.MjData(model)
    qa = int(model.jnt_qposadr[jid])
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_left")
    assert cid >= 0

    def gaze() -> np.ndarray:
        m = np.asarray(data.cam_xmat[cid]).reshape(3, 3)
        v = m @ np.array([0.0, 0.0, -1.0])
        return v / np.linalg.norm(v)

    data.qpos[qa] = 0.0
    mujoco.mj_forward(model, data)
    l0 = gaze()
    toward_table = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    assert float(np.dot(l0, toward_table)) > 0.94
    assert abs(float(np.dot(ax / np.linalg.norm(ax), l0))) < 0.15
    data.qpos[qa] = 0.08
    mujoco.mj_forward(model, data)
    l1 = gaze()
    assert float(np.linalg.norm(l1 - l0)) > 0.06


def test_resolve_dynav_config_yaml_innate_mars_and_stretch():
    from emet.robots import DEFAULT_DYNAV_CONFIG_YAML, resolve_dynav_config_yaml

    assert (
        resolve_dynav_config_yaml("innate_mars", DEFAULT_DYNAV_CONFIG_YAML)
        == "dynav_innate_mars.yaml"
    )
    assert resolve_dynav_config_yaml("innate-mars", DEFAULT_DYNAV_CONFIG_YAML) == "dynav_innate_mars.yaml"
    assert (
        resolve_dynav_config_yaml("innate_mars", "dynav_innate_mars.yaml") == "dynav_innate_mars.yaml"
    )
    assert resolve_dynav_config_yaml("stretch", DEFAULT_DYNAV_CONFIG_YAML) == DEFAULT_DYNAV_CONFIG_YAML


def test_innate_mars_resolved_default_dynav_parameters_use_da3():
    from emet.core.parameters import get_parameters
    from emet.robots import DEFAULT_DYNAV_CONFIG_YAML, resolve_dynav_config_yaml

    path = resolve_dynav_config_yaml("innate_mars", DEFAULT_DYNAV_CONFIG_YAML)
    params = get_parameters(path)
    assert str(params.get("depth_source")).lower() == "da3"


def test_get_robot_spec_and_runtime_notes_innate_mars():
    from emet.robots import format_robot_runtime_notes, get_robot_spec

    s = get_robot_spec("innate_mars")
    assert s is not None
    assert s.optional_uv_extras == ()
    assert s.default_dynav_config == "dynav_innate_mars.yaml"
    notes = format_robot_runtime_notes(s)
    assert notes is not None
    assert "uv sync" not in notes
    assert "depth_source=" in notes


def test_innate_mars_planar_base_navigation_moves_toward_xyt_goal():
    """Innate Mars has slide/slide/yaw base joints (no free joint); nav must drive them, not no-op."""
    pytest.importorskip("mujoco")
    import mujoco
    import numpy as np

    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation.mujoco_server import _load_default_scene_with_robot
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    model = _load_default_scene_with_robot("innate_mars")
    if model is None:
        pytest.skip("Merged innate_mars scene not available")
    spec = InnateMarsBackend().get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
    )
    server._load_model()
    server._stabilize_physics_state_after_load()
    server._initial_xyt = server.get_base_xyt().copy()
    assert server._planar_base_velocity_actuator_ids() is not None

    xyt0 = server.get_base_xyt().copy()
    server.handle_action({"xyt": [0.25, 0.0, 0.0], "nav_relative": False})
    assert server._nav_goal_world is not None

    for _ in range(900):
        with server._mj_lock:
            server._step_base_navigation_drive()
            mujoco.mj_step(server._mjmodel, server._mjdata)
        if server._at_goal:
            break

    xyt1 = server.get_base_xyt()
    moved = float(np.hypot(xyt1[0] - xyt0[0], xyt1[1] - xyt0[1]))
    assert moved > 0.02, f"expected planar base to translate toward goal; |Δxy|={moved:.4f}"


def test_stereo_right_camera_name_from_spec_galaxea_no_pair():
    from emet.robots.galaxea_r1 import GalaxeaR1Backend
    from emet.simulation.stereo_camera_utils import stereo_right_camera_name_from_spec

    names = GalaxeaR1Backend().get_spec().camera_names
    assert stereo_right_camera_name_from_spec(list(names)) is None
