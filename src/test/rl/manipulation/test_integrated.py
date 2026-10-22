"""Physical end-to-end task gate, in addition to isolated trajectory tests."""

import pytest

from emet.rl.manipulation.integrated import run_case


@pytest.mark.sim
@pytest.mark.parametrize("action_mode", ["residual", "direct_velocity"])
def test_mars_box_physical_pick_transport_place(action_mode):
    result = run_case(robot="innate_mars", seed=1800006, action_mode=action_mode)
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
