"""Physical end-to-end task gate, in addition to isolated trajectory tests."""

import pytest

from emet.rl.manipulation.integrated import run_case


@pytest.mark.sim
@pytest.mark.parametrize(
    "action_mode,mass,cone,balanced",
    [
        ("residual", None, "pyramidal", False),
        ("direct_velocity", None, "pyramidal", False),
        ("residual", 0.05, "elliptic", False),
        ("direct_velocity", 0.05, "elliptic", False),
        ("residual", 0.1, "elliptic", True),
        ("direct_velocity", 0.1, "elliptic", True),
    ],
)
def test_mars_box_physical_pick_transport_place(action_mode, mass, cone, balanced):
    result = run_case(
        robot="innate_mars",
        seed=1800006,
        action_mode=action_mode,
        payload_mass_kg=mass,
        contact_cone=cone,
        contact_impratio=3 if cone == "elliptic" else 1,
        balanced_grasp_setup=balanced,
    )
    assert result["success"], result
    assert result["payload"] == "object1"
    assert result["placement_xy_error_m"] <= 0.02
    assert len(result["stages"]) == 4
    for stage in result["stages"]:
        if stage["stage"] == "transport":
            assert stage["execution_verified"]
            assert stage["controller_failure"] is None
            assert stage["settled_hand_position_error"] < 0.005
            assert stage["settled_hand_orientation_error"] < 0.05
            assert stage["max_hand_tracking_position_error"] < 0.015
            assert stage["max_hand_tracking_orientation_error"] < 0.1
        else:
            assert stage["is_success"]
            assert not stage["wheel_lift"]
            assert not stage["unsafe_contact"]
            assert stage["ee_position_error"] < 0.005
            assert stage["ee_orientation_error"] < 0.05
        assert stage["max_payload_rotation"] < 0.1
        assert stage["max_payload_translation"] < 0.02
    final = result["stages"][-1]
    assert final["released"]
    assert final["support_load_n"] > 0
    assert final["settle_time"] >= 2
