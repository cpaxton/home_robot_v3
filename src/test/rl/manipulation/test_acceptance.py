from copy import deepcopy

from emet.rl.manipulation.env import SCENARIOS
from emet.rl.manipulation.post_training import component_gates, curriculum_gates


def metrics():
    return {
        name: {
            "episodes": 100,
            "success_rate": 0.96,
            "wheel_lift_rate": 0,
            "unsafe_contact_rate": 0,
            "initialization_failure_rate": 0,
            "joint_error_mean": 0.01,
            "peak_contact_force_p95": 5,
            "peak_measured_jerk_p95": 10,
        }
        for name in SCENARIOS
    }


def test_acceptance_requires_all_categories_and_no_safety_accuracy_regression():
    teacher = metrics()
    assert not curriculum_gates(teacher, deepcopy(teacher))
    assert curriculum_gates(teacher, {})
    for key, value in (
        ("success_rate", 0.94),
        ("wheel_lift_rate", None),
        ("unsafe_contact_rate", 0.01),
        ("joint_error_mean", 0.02),
        ("peak_measured_jerk_p95", 11),
        ("episodes", 99),
    ):
        learned = deepcopy(teacher)
        learned["carry"][key] = value
        assert curriculum_gates(teacher, learned), (key, value)


def test_component_gates_do_not_ignore_valid_paths_that_fail_execution():
    report = {
        "cases": [
            {
                "scenario": scenario,
                "waypoints": [[0], [1]],
                "execution_verified": True,
                "controller_failure": None,
                "contact_metrics": {"wheel_lift": False},
            }
            for scenario in ("reach", "carry")
        ]
    }
    assert not component_gates(report, report)
    assert component_gates(report, {"cases": []})
    learned = deepcopy(report)
    learned["cases"][1]["execution_verified"] = False
    assert component_gates(report, learned)
    learned["cases"][1]["execution_verified"] = True
    learned["cases"][1]["contact_metrics"]["wheel_lift"] = None
    assert component_gates(report, learned)
