"""Spline geometry, physical execution, split isolation and bound preflight."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from emet.rl.manipulation.arm_spline_benchmark import episode, require_preflight
from emet.rl.manipulation.arm_splines import (
    ANCHORS,
    LEVELS,
    MIXTURE,
    ArmSplineReference,
    ArmSplineTrackingEnv,
    load_manifest,
    make_training_environment,
    manifest,
    spline_joint_extrema,
    training_mixture,
)
from emet.rl.manipulation.carry_benchmark import digest


def write(tmp_path, value, name="manifest.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_manifest_disjoint_coverage_and_weights(tmp_path):
    spec = manifest()
    assert load_manifest(write(tmp_path, spec)) == spec
    assert len(spec["cases"]) == 144
    assert sum(MIXTURE.values()) == pytest.approx(1)
    assert MIXTURE["tamp"] == 0.5 and MIXTURE["arm_splines"] == 0.2
    for split in ("train", "audit"):
        for level in range(3):
            assert {c["posture"] for c in spec["cases"] if c["level"] == level and c["split"] == split} == set(ANCHORS)


@pytest.mark.parametrize(
    "patch",
    [
        {"seed": True},
        {"seed": 470000000},
        {"split": "other"},
        {"level": True},
        {"posture": []},
        {"id": []},
        {"id": "unbound"},
        {"level": 3},
    ],
)
def test_manifest_invalid_cases_fail_before_physics(tmp_path, patch):
    spec = manifest(4, 4)
    spec["cases"][0].update(patch)
    with pytest.raises(ValueError):
        load_manifest(write(tmp_path, spec))


@pytest.mark.parametrize(
    "patch",
    [
        {"amplitude_rad": float("nan")},
        {"amplitude_rad": 0.71},
        {"speed_fraction": True},
        {"speed_fraction": 0},
        {"posture_jitter_rad": -0.01},
        {"knot_count": 2},
        {"knot_count": True},
        {"maximum_duration_s": 31},
        {"initial_hold_s": -1},
    ],
)
def test_invalid_spline_settings(patch):
    with pytest.raises(ValueError):
        replace(LEVELS[0], **patch)


def test_extrema_include_between_knot_overshoot():
    q = np.array([[0], [1], [1], [0]])
    spline = CubicSpline([0, 1, 2, 3], q, bc_type="natural")
    low, high = spline_joint_extrema(spline)
    assert high[0] > q.max() + 0.1
    dense = spline(np.linspace(0, 3, 10001))
    assert low[0] <= dense.min() and high[0] >= dense.max() - 1e-12


def test_training_mixture_rejects_audit_before_constructing_members():
    class Pool:
        cases = [c for c in manifest(4, 4)["cases"] if c["split"] == "audit"]

    with pytest.raises(ValueError, match="audit"):
        training_mixture(dict.fromkeys(("hand", "base", "goal_changes", "tamp")), Pool())


def test_preflight_requires_exact_sources_manifest_and_population(tmp_path, monkeypatch):
    import emet.rl.manipulation.arm_spline_benchmark as benchmark

    spec = manifest(4, 4)
    path = write(tmp_path, spec)
    monkeypatch.setattr(benchmark, "source_hashes", lambda: {"source": "one"})
    monkeypatch.setattr(benchmark, "source_state", lambda root: {"state": "one"})
    rows = [
        {
            "case": c,
            "admitted": True,
            "stop_confirmed": True,
            "success": True,
            "invalid_physics": False,
            "metrics": {"unsafe_contact": False, "wheel_lift": False},
        }
        for c in spec["cases"]
        if c["level"] == 0
    ]
    good = {
        "format": "arm-spline-preflight-v1",
        "complete": True,
        "manifest_sha256": digest(spec),
        "sources": {"source": "one"},
        "source_state": {"state": "one"},
        "level": 0,
        "results": {"teacher": rows},
    }
    assert require_preflight(write(tmp_path, good, "report.json"), path, 0) == good
    changes = [
        lambda x: x.update(sources={}),
        lambda x: x.update(source_state={}),
        lambda x: x.update(manifest_sha256="wrong"),
        lambda x: x["results"]["teacher"].pop(),
        lambda x: x["results"]["teacher"][0].update(success=False),
        lambda x: x["results"]["teacher"][0].update(stop_confirmed=False),
    ]
    for change in changes:
        bad = copy.deepcopy(good)
        change(bad)
        with pytest.raises(ValueError):
            require_preflight(write(tmp_path, bad, "report.json"), path, 0)


@pytest.mark.sim
def test_seeded_reference_has_continuous_motion_bounds_and_no_live_state_writes():
    cases = [c for c in manifest(4, 4)["cases"] if c["level"] == 0 and c["split"] == "train"]
    env = ArmSplineTrackingEnv(cases)
    try:
        observation, first = env.reset()
        e = env.unwrapped
        assert env.observation_space.contains(observation)
        before = {k: getattr(e.data, k).copy() for k in ("qpos", "qvel", "ctrl")}
        time = float(e.data.time)
        reference = ArmSplineReference(e.control, env.reference.certificate["joint_waypoints"], LEVELS[0])
        reference.sample(reference.end / 2)
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(e.data, key), value)
        assert float(e.data.time) == time
        assert np.linalg.norm(reference.joints(reference.joints.x[1], 1)) > 0.001
        for s in reference.joints.x[1:-1]:
            for derivative in (0, 1, 2):
                np.testing.assert_allclose(
                    reference.joints(s - 1e-9, derivative), reference.joints(s + 1e-9, derivative), atol=1e-6
                )
        certificate = reference.certificate
        assert np.all(
            np.asarray(certificate["derivative_bounds"])
            <= np.array([certificate[k] for k in ("speed", "acceleration", "jerk")]) * (1 + 1e-8)
        )
        assert np.asarray(reference.base_mask).all()
        assert e.model.jnt_type[e.model.joint("manipulation_base").id] == 0  # Free joint.
        assert not e.reach_hold_trial and e.control.pose_only_targets
        _, second = env.reset(options={"case_id": cases[0]["id"]})
        assert first["reference_sha256"] == second["reference_sha256"]
        assert first["spline_rejections"] == second["spline_rejections"]
        np.testing.assert_array_equal(e.data.qpos, before["qpos"])
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("posture", list(ANCHORS))
def test_small_spline_teacher_tracks_each_posture_and_confirms_stop(posture):
    case = next(
        c for c in manifest(4, 4)["cases"] if c["level"] == 0 and c["split"] == "train" and c["posture"] == posture
    )
    env = ArmSplineTrackingEnv([case])
    try:
        row = episode(env, case)
        assert row["success"] and row["stop_confirmed"]
        assert np.all(np.asarray(row["peak_path_error"]) <= [0.003, 0.02])
        assert not row["metrics"]["unsafe_contact"] and not row["metrics"]["wheel_lift"]
        assert row["joint_limit_push_ticks"] == 0
    finally:
        env.close()


@pytest.mark.sim
def test_deadline_remains_failure_and_latches_stop():
    case = manifest(4, 4)["cases"][0]
    env = ArmSplineTrackingEnv([case])
    try:
        env.reset()
        e = env.unwrapped
        e.config = replace(e.config, horizon=e.profile.dt)
        _, _, done, truncated, info = env.step(np.zeros(8))
        assert done and not truncated and e.control.stopped
        assert not info["primitive_success"] and not info["is_success"]
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(8))
    finally:
        env.close()


@pytest.mark.sim
def test_full_training_factory_keeps_only_training_cases_and_spaces():
    from emet.rl.manipulation.arm_spline_benchmark import DEFAULT

    env = make_training_environment(DEFAULT)
    try:
        assert dict(zip(env.names, env.weights, strict=True)) == MIXTURE
        for member in env.environments.values():
            assert all(c["split"] == "train" for c in getattr(member, "cases", []) if "split" in c)
        assert {c["split"] for c in env.environments["arm_splines"].cases} == {"train"}
        assert {c["level"] for c in env.environments["arm_splines"].cases} == {0}
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
    finally:
        env.close()


@pytest.mark.sim
def test_new_family_rollout_and_critic_warmup_use_existing_policy_contract():
    import torch

    from emet.rl.manipulation.arm_spline_training import critic_warmup
    from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig
    from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
    from emet.rl.manipulation.whole_body_training import collect_rollout

    torch.manual_seed(10)
    policy = MultiCriticPolicy()
    learner = MultiCriticPPO(policy, PPOConfig(epochs=1, critic_epochs=1, batch_size=4))
    env = ArmSplineTrackingEnv([manifest(4, 4)["cases"][0]])
    try:
        observation, _ = env.reset()
        rollout, _ = collect_rollout(env, policy, observation, 8)
        assert rollout["rewards"].shape == (8, 3)
        assert (rollout["ticks"] == 1).all()
        before = {k: v.clone() for k, v in policy.actor.state_dict().items()}
        std = policy.log_std.detach().clone()
        assert np.isfinite(critic_warmup(learner, rollout)["value_loss"])
        for key, value in policy.actor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        torch.testing.assert_close(policy.log_std, std, atol=0, rtol=0)
        stats = learner.update(rollout)
        assert stats and any(not torch.equal(v, before[k]) for k, v in policy.actor.state_dict().items())
    finally:
        env.close()
