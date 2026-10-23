import numpy as np
import pytest

from emet.rl.manipulation.risk import TrajectoryRiskPredictor, fit_risk, trajectory_features


def test_trajectory_features_reject_nonfinite_and_invalid_shapes():
    history = np.zeros((20, 36))
    first = trajectory_features(history, np.zeros((4, 5)), 2)
    assert np.isfinite(first).all()
    assert not np.array_equal(first, trajectory_features(history, np.ones((4, 5)), 2))
    with pytest.raises(ValueError):
        trajectory_features(history, np.zeros((4, 5)), 0)
    history[0, 0] = np.nan
    with pytest.raises(ValueError):
        trajectory_features(history, np.zeros((4, 5)), 2)


def rows():
    values = np.linspace(-1, 1, 400)
    return [
        {
            "episode_id": str(i),
            "robot": "stretch",
            "fingerprint": "model",
            "controller_id": "controller",
            "risk_features": [float(x), float(x * x)],
            "is_success": bool(x > 0),
            "retention_lost": bool(x < 0),
            "unsafe_contact": False,
            "wheel_lift": None,
        }
        for i, x in enumerate(values)
    ]


def test_risk_abstains_on_missing_safety_evidence_and_different_models(tmp_path):
    manifest = fit_risk(rows(), tmp_path, robot="stretch", fingerprint="model", controller_id="controller", epochs=60)
    assert not set(manifest["training_episodes"]) & set(manifest["calibration_episodes"])
    assert not set(manifest["audit_episodes"]) & (
        set(manifest["training_episodes"]) | set(manifest["calibration_episodes"])
    )
    assert sum(len(manifest[key]) for key in ("training_episodes", "calibration_episodes", "audit_episodes")) == 400
    assert manifest["heads"]["success"]["metrics_split"] == "independent_audit"
    assert not manifest["heads"]["wheel_lift"]["supported"]
    policy = TrajectoryRiskPredictor(tmp_path)
    kwargs = {"robot": "stretch", "fingerprint": "model", "controller_id": "controller", "geometry_valid": True}
    result = policy.predict([0.8, 0.64], **kwargs)
    assert result["status"] == "unknown"
    assert result["probabilities"]["wheel_lift"] is None
    assert result["safety_certificate"] is False
    assert result["probabilities"]["success"] > 0.5
    assert policy.predict([10, 100], **kwargs)["reason"] == "out_of_distribution"
    assert policy.predict([0.8, 0.64], **{**kwargs, "controller_id": "new"})["status"] == "unknown"
    assert policy.predict([0.8, 0.64], **{**kwargs, "geometry_valid": False})["status"] == "geometry_rejected"


def test_risk_rejects_frozen_data_duplicate_episodes_and_wrong_controller(tmp_path):
    data = rows()
    with pytest.raises(ValueError, match="frozen"):
        fit_risk(
            [{**data[0], "split": "frozen_regression"}, *data[1:]],
            tmp_path,
            robot="stretch",
            fingerprint="model",
            controller_id="controller",
        )
    with pytest.raises(ValueError, match="duplicate"):
        fit_risk([data[0], *data], tmp_path, robot="stretch", fingerprint="model", controller_id="controller")
    with pytest.raises(ValueError, match="one robot"):
        fit_risk(data, tmp_path, robot="stretch", fingerprint="model", controller_id="changed")


@pytest.mark.parametrize('robot', ['stretch', 'innate_mars'])
def test_robot_specific_prefix_excludes_phase_and_retains_every_action(robot):
    from emet.rl.manipulation.observation_layout import proprioceptive_layout
    from emet.rl.manipulation.profiles import PROFILES

    profile = PROFILES[robot]
    fields, width = proprioceptive_layout(profile)
    token = np.zeros((profile.history_steps, width + 7))
    token[:, fields['previous_action']] = np.arange(profile.action_size) + 10
    token[:, width:] = 999  # phase must never become previous actions
    features = trajectory_features(token[:, :width], np.zeros((1, profile.arm_size)), 1, profile=profile)
    assert np.array_equal(features[:width], token[-1, :width])
    assert 999 not in features
    with pytest.raises(ValueError):
        trajectory_features(token, np.zeros((1, profile.arm_size)), 1, profile=profile)
