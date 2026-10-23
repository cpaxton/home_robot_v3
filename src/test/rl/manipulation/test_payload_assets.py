import mujoco
import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.payload_assets import NATIVE_TABLE, provenance


@pytest.mark.sim
def test_native_objects_preserve_source_geometry_mass_inertia_and_contacts():
    source = mujoco.MjModel.from_xml_path(str(NATIVE_TABLE))
    model, fingerprint = build_manipulation_model("innate_mars", payload_asset_set="native_table")
    _, nominal_fingerprint = build_manipulation_model("innate_mars")
    assert fingerprint != nominal_fingerprint
    for name in ("object1", "object2"):
        a, b = source.body(name).id, model.body(name).id
        for field in ("mass", "inertia", "ipos", "iquat"):
            np.testing.assert_allclose(getattr(source, f"body_{field}")[a], getattr(model, f"body_{field}")[b])
        assert model.body_mass[b] == pytest.approx(0.5)
        ga, gb = source.body_geomadr[a], model.body_geomadr[b]
        for field in (
            "type",
            "size",
            "pos",
            "quat",
            "friction",
            "solref",
            "solimp",
            "condim",
            "contype",
            "conaffinity",
        ):
            np.testing.assert_allclose(getattr(source, f"geom_{field}")[ga], getattr(model, f"geom_{field}")[gb])
    assert provenance("native_table")["geometry_scale"] == 1
    assert len(provenance("native_table")["source_sha256"]) == 64


@pytest.mark.sim
@pytest.mark.parametrize("name", ["object1", "object2"])
def test_native_payload_reset_selects_requested_object_without_mass_reduction(name):
    env = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            payload_asset_set="native_table",
            payload_object=name,
            randomization_scale=0,
        )
    )
    try:
        _, info = env.reset(seed=1800006)
        assert env.payload_name == name
        assert info["randomization"]["payload_mass_kg"] == pytest.approx(0.5)
        assert info["randomization"]["asset_set"] == "native_table"
    finally:
        env.close()


def test_native_suite_has_explicit_object_coverage_and_preserves_failures():
    from emet.rl.manipulation.tamp_benchmark import make_suite, summarize

    suite = make_suite(episodes=3)
    assert len(suite["cases"]) == 6
    assert {r["payload_object"] for r in suite["cases"]} == {"object1", "object2"}
    failed = {
        "payload_object": "object1",
        "seed": 1,
        "success": False,
        "failure": "retention_lost",
        "stages": [{"stage": "transport", "execution_verified": False}],
    }
    report = summarize([failed])
    assert report["object1"]["successes"] == 0
    assert report["object1"]["verified_transport_segments"] == 0
    assert report["object1"]["failures_by_stage"] == {"transport:retention_lost": 1}


@pytest.mark.sim
@pytest.mark.parametrize("name", ["object1", "object2"])
def test_native_nominal_pinch_rejects_oversize_objects_before_motion(name):
    env = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="pregrasp",
            payload_asset_set="native_table",
            payload_object=name,
            randomization_scale=0,
            initialization_attempts=1,
        )
    )
    try:
        _, info = env.reset(seed=1800006)
        assert info["initialization_failed"]
        assert info["initialization_failure_reason"] == "nominal_pinch_clearance"
        fit = info["randomization"]["nominal_pinch_admission"]
        assert fit["open_gap_m"] < fit["object_projected_width_m"]
        qpos = env.data.qpos.copy()
        _, _, done, _, result = env.step(np.zeros(env.action_space.shape))
        assert done and result["code"] == "initialization_failed"
        np.testing.assert_array_equal(env.data.qpos, qpos)
    finally:
        env.close()


@pytest.mark.sim
def test_nominal_pinch_admission_accepts_existing_small_payload():
    from emet.rl.manipulation.payload_assets import nominal_pinch_admission
    from emet.rl.manipulation.profiles import MARS

    model, _ = build_manipulation_model("innate_mars")
    data = mujoco.MjData(model)
    from emet.rl.manipulation.control import ManipulationControl

    control = ManipulationControl(model, data, MARS)
    control.initialize(np.array([0, -0.4, 0.2, 0.2, 0]), MARS.gripper_open)
    fit = nominal_pinch_admission(model, data, MARS, model.body("object1").id, 0.03)
    assert fit["accepted"]
    assert fit["open_gap_m"] > 0.016


@pytest.mark.sim
def test_elliptic_contact_option_is_fingerprinted_without_raising_forces():
    nominal, original = build_manipulation_model("innate_mars")
    same, explicit = build_manipulation_model("innate_mars", contact_cone="pyramidal")
    elliptic, changed = build_manipulation_model("innate_mars", contact_cone="elliptic")
    assert original == explicit and changed != original
    assert int(nominal.opt.cone) == int(same.opt.cone) == 0
    assert int(elliptic.opt.cone) == 1
    np.testing.assert_array_equal(nominal.actuator_forcerange, elliptic.actuator_forcerange)
    np.testing.assert_array_equal(nominal.geom_friction, elliptic.geom_friction)
    np.testing.assert_array_equal(nominal.body_mass, elliptic.body_mass)
    np.testing.assert_array_equal(nominal.body_inertia, elliptic.body_inertia)
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", contact_cone="elliptic"))
    try:
        assert int(env.model.opt.cone) == 1
        _, info = env.reset(seed=7)
        assert info["randomization"]["contact_cone"] == "elliptic"
    finally:
        env.close()
