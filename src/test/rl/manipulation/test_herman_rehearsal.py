"""Captured posture admission, calibrated metric geometry and offline reset invariants."""

import ast
import base64
import copy
import json
import zlib
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.herman_rehearsal import (
    DEPTH_RULE,
    capture_audit,
    decode_array,
    decode_depth,
    joint_values,
    project_depth,
    rectified_intrinsics,
    scene_voxels,
)
from emet.rl.manipulation.herman_rehearsal_env import HermanRehearsalEnv, camera_calibration
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.reference import REFERENCE
from emet.robots.innate_mars.passive_ssh import subscriber_command

FIXTURE = Path(__file__).with_name("fixtures") / "herman_capture_v1.json"
CANONICAL_MARS = "dc7199658397a4d5e027a426cb3e936a46199b779d6bc9dc73113f400a5ca68a"


def encoded(a):
    return base64.b64encode(zlib.compress(a.tobytes())).decode()


def info(width=4, height=2):
    return {
        "width": width,
        "height": height,
        "P": [4, 0, 2, 0, 0, 4, 1, 0, 0, 0, 1, 0],
        "R": np.eye(3).ravel().tolist(),
        "frame_id": "optical",
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0},
    }


def depth(a):
    return {
        "encoding": "16SC1",
        "width": a.shape[1],
        "height": a.shape[0],
        "step": a.shape[1] * 2,
        "bigendian": a.dtype.byteorder == ">",
        "data_zlib_base64": encoded(a),
        "stamp_ns": 10**9,
        "frame_id": "optical",
    }


def test_real_capture_outside_both_published_limits_is_blocked_without_clamping():
    capture = json.loads(FIXTURE.read_text())
    before = copy.deepcopy(capture)
    audit = capture_audit(capture, REFERENCE.read_text())
    assert not audit["captured_physics_admitted"] and not audit["hardware_ready"]
    rows = {x["joint"]: x for x in audit["limit_violations"]}
    assert rows["joint1"]["outside_degrees"] == pytest.approx(1.31814891585)
    assert rows["joint2"]["outside_degrees"] == pytest.approx(10.01932079085)
    assert audit["live_urdf_vs_tf"]["ee_link"]["position_m"] < 1e-8
    assert not audit["wrist_image_available"] and not audit["wrist_optical_tf_available"]
    assert capture == before


@pytest.mark.parametrize("endian", ["<i2", ">i2"])
def test_signed_depth_requires_explicit_source_units_and_drops_invalid_values(endian):
    sample = depth(np.array([[1000, 0, -16, 2000]], dtype=endian))
    result = decode_depth(sample, DEPTH_RULE)
    np.testing.assert_allclose(result[[0], [0, 3]], [1, 2])
    assert np.isnan(result[0, 1:3]).all()
    with pytest.raises(ValueError, match="pinned publisher"):
        decode_depth(sample, {**DEPTH_RULE, "metres_per_unit": 1})
    with pytest.raises(ValueError, match="pinned publisher"):
        decode_depth(sample, {**DEPTH_RULE, "publisher_sha256": "0" * 64})


def test_depth_size_and_calibration_errors_fail_before_scene_geometry():
    with pytest.raises(ValueError, match="size mismatch"):
        decode_array(encoded(np.zeros(8, dtype="<i2")), "<i2", (1, 4))
    k, r = rectified_intrinsics(info(), 8, 4)
    np.testing.assert_allclose(k, [[8, 0, 4], [0, 8, 2], [0, 0, 1]])
    with pytest.raises(ValueError, match="cropped/binned"):
        rectified_intrinsics({**info(), "binning_x": 2}, 8, 4)
    with pytest.raises(ValueError, match="calibration"):
        rectified_intrinsics({**info(), "P": [0] * 12}, 8, 4)


def test_inverse_rectification_and_capture_time_transform_are_used():
    r = Rotation.from_euler("y", 0.2).as_matrix()
    a = np.full((2, 4), 1000, dtype="<i2")
    cal = {**info(), "R": r.ravel().tolist()}
    tf = {"position": [0.2, 0, 0.3], "quaternion_xyzw": [0, 0, 0, 1], "child": "optical"}
    cap = {
        "samples": {"depth": depth(a), "head_left_info": cal, "arm": {"stamp_ns": 10**9}},
        "image_transforms": {"depth": tf},
    }
    points, pixels, _ = project_depth(cap, stride=1)
    i = np.where((pixels == [2, 1]).all(axis=1))[0][0]
    np.testing.assert_allclose(points[i], r.T @ np.array([0, 0, 1]) + [0.2, 0, 0.3])
    cap["samples"]["arm"]["stamp_ns"] += 100_000_000
    with pytest.raises(ValueError, match="pairing"):
        project_depth(cap)


def test_only_observed_points_become_voxels_and_bad_geometry_is_rejected():
    points = np.array([[0.5, 0, 0.3], [0.501, 0, 0.3], [0, 0, 0], [-0.5, 0.2, 0.171]])
    geoms = scene_voxels(points)
    assert len(geoms) == 2
    assert all(g["half_size"] == [0.03] * 3 for g in geoms)
    with pytest.raises(ValueError, match="finite"):
        scene_voxels(np.array([[float("nan"), 0, 0]]))


def test_capture_subscriber_has_no_actuator_publisher_or_service_calls():
    from emet.robots.innate_mars import rehearsal_capture_ros

    tree = ast.parse(Path(rehearsal_capture_ros.__file__).read_text())
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "create_subscription" in calls and not calls & {"create_publisher", "create_client", "publish", "call_async"}
    assert "StrictHostKeyChecking=yes" in subscriber_command(
        "jetson1@herman", program="rehearsal_capture_ros", arguments=["--seconds", "8"]
    )
    with pytest.raises(ValueError, match="subscriber"):
        subscriber_command("jetson1@herman", program="../../other", arguments=[])


@pytest.mark.sim
def test_default_model_identity_is_unchanged_and_custom_scene_is_distinct():
    _, fingerprint = build_manipulation_model("innate_mars", mesh_storage="binary")
    assert fingerprint == CANONICAL_MARS
    geom = {"position": [1, 0, 0.3], "half_size": [0.03] * 3, "rgba": [1, 0, 0, 1]}
    model, derived = build_manipulation_model(
        "innate_mars", mesh_storage="binary", scene_geoms=[geom], hide_training_fixtures=True
    )
    assert derived != fingerprint and model.geom("observed_scene_0").id >= 0
    for name in ("object1", "object2"):
        assert model.body_gravcomp[model.body(name).id] == 1
        g = model.body_geomadr[model.body(name).id]
        assert model.geom_contype[g] == model.geom_conaffinity[g] == 0
    with pytest.raises(ValueError, match="finite bounded"):
        build_manipulation_model("innate_mars", scene_geoms=[{**geom, "half_size": [-1, 0, 0]}])


@pytest.mark.sim
def test_out_of_limits_reset_is_rejected_before_any_physics_step(monkeypatch):
    capture = json.loads(FIXTURE.read_text())
    values = joint_values(capture)

    def forbidden(*_):
        pytest.fail("physics stepped before captured posture admission")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    with pytest.raises(ValueError, match="exceeds model limits"):
        HermanRehearsalEnv(
            ManipulationConfig(robot="innate_mars", scenario="reach"),
            initial_arm=[values["joint" + str(i)] for i in range(1, 6)],
            initial_grip=values["joint6"],
            initial_head=0,
            model_factory=build_manipulation_model,
        )


@pytest.mark.sim
def test_reset_preserves_declared_gripper_and_head_without_runtime_injection():
    env = HermanRehearsalEnv(
        ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0),
        initial_arm=[0, -0.4, 0.2, 0.2, 0],
        initial_grip=0.089,
        initial_head=0.05,
        model_factory=build_manipulation_model,
    )
    try:
        env.reset(seed=430_000_000)
        assert env.control.grip_reference == pytest.approx(0.089)
        assert env.data.ctrl[env.model.actuator("joint_head").id] == pytest.approx(0.05)
        assert env.model.body("base_link").id >= 0
    finally:
        env.close()


@pytest.mark.sim
def test_calibrated_renderer_matches_opencv_pixels_including_principal_offset():
    xml = """<mujoco><asset><material name="red" rgba="1 0 0 1" emission="1"/></asset><worldbody>
    <camera name="head_left"/><geom type="sphere" size=".01" pos=".15 -.1 1" material="red" contype="0" conaffinity="0"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    cal = {
        "width": 320,
        "height": 240,
        "P": [200, 0, 220, 0, 0, 180, 95, 0, 0, 0, 1, 0],
        "R": np.eye(3).ravel().tolist(),
    }
    camera_calibration(model, {"samples": {"depth": {"width": 320, "height": 240}, "head_left_info": cal}})
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=320, height=240)
    try:
        renderer.update_scene(data, camera="head_left")
        image = renderer.render().copy()
        v, u = np.where(
            (image[:, :, 0] > 100) & (image[:, :, 0] > image[:, :, 1] * 2) & (image[:, :, 0] > image[:, :, 2] * 2)
        )
        assert len(u) > 3
        np.testing.assert_allclose([u.mean(), v.mean()], [250, 77], atol=2)
    finally:
        renderer.close()


def test_replay_inventory_detects_changed_images_and_external_paths(tmp_path):
    from emet.rl.manipulation.herman_rehearsal import seal_artifacts, verify_artifacts

    for name in ("report.json", "scene.mjb", "captured-replay.npz", "head_left.jpg"):
        (tmp_path / name).write_bytes(b"fixture")
    inventory = seal_artifacts(tmp_path)
    assert verify_artifacts(tmp_path) == inventory
    (tmp_path / "head_left.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum/path"):
        verify_artifacts(tmp_path)
    inventory["files"] = {"../external": "0" * 64}
    (tmp_path / "artifacts.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="checksum/path"):
        verify_artifacts(tmp_path)


@pytest.mark.sim
def test_calibrated_head_mount_matches_captured_optical_tf():
    from emet.rl.manipulation.herman_rehearsal import pose_matrix
    from emet.rl.manipulation.herman_rehearsal_env import scene_factory

    capture = json.loads(FIXTURE.read_text())
    capture["samples"]["depth"] = {"width": 640, "height": 480}
    model, _ = scene_factory(capture, [])("innate_mars", mesh_storage="binary")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = data.body("base_link").xpos.copy()
    cid = model.camera("head_left").id
    tf = pose_matrix(capture["transforms"]["camera_optical_frame"])
    np.testing.assert_allclose(data.cam_xpos[cid] - base, tf[:3, 3], atol=1e-9)
    rectification = np.asarray(capture["samples"]["head_left_info"]["R"]).reshape(3, 3)
    expected = tf[:3, :3] @ rectification.T @ np.diag([1, -1, -1])
    np.testing.assert_allclose(data.cam_xmat[cid].reshape(3, 3), expected, atol=1e-8)


def test_passive_capture_pairs_arm_with_delayed_depth():
    from emet.robots.innate_mars.rehearsal_capture_ros import choose_arm_sample

    arms = [{"stamp_ns": stamp} for stamp in (500_000_000, 600_000_000, 800_000_000, 950_000_000)]
    assert choose_arm_sample(arms, [590_000_000, 950_000_000], 1_000_000_000) == arms[1]
    assert choose_arm_sample(arms, [], 1_000_000_000) == arms[2]


def test_innate_080_unsigned_depth_requires_matching_publisher():
    from emet.rl.manipulation.herman_rehearsal import DEPTH_RULE_INNATE_080

    buf = np.array([[0, 750]], dtype="<u2")
    sample = {
        "encoding": "16UC1",
        "publisher_sha256": DEPTH_RULE_INNATE_080["publisher_sha256"],
        "width": 2,
        "height": 1,
        "step": 4,
        "bigendian": False,
        "data_zlib_base64": base64.b64encode(zlib.compress(buf.tobytes())).decode(),
    }
    assert np.isnan(decode_depth(sample, DEPTH_RULE_INNATE_080)[0, 0])
    assert decode_depth(sample, DEPTH_RULE_INNATE_080)[0, 1] == 0.75
    with pytest.raises(ValueError, match="pinned publisher"):
        decode_depth({**sample, "publisher_sha256": "0" * 64}, DEPTH_RULE_INNATE_080)
