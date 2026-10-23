import pytest

from emet.rl.manipulation.reward_audit import audit


@pytest.mark.sim
def test_goal_gated_objective_prefers_reaching_over_idle_on_known_counterexample():
    report = audit(seeds=(9100002,))
    results = {(r["goal_gated_settling"], r["discount"]): r for r in report["comparisons"]}
    assert results[False, 0.99]["teacher_preferred_count"] == 0
    assert results[True, 0.99]["teacher_preferred_count"] == 1
    assert results[True, 0.9975]["teacher_preferred_count"] == 1
    for row in report["episodes"]:
        assert not row["wheel_lift"]
        assert not row["unsafe_contact"]
