# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

from emet.eval.benchmark_dynagraph import (
    DYNAMIC_EXPLORE_BACKENDS,
    apply_dynamic_explore_backend,
    resolve_dynamic_explore_profile,
)
from emet.eval.dynamic_exploration_config import (
    build_explore_run_matrix,
    filter_episodes,
    flatten_eval_metrics,
    load_dynamic_exploration_config,
    resolve_smoke_run_plan,
)


def test_load_dynamic_exploration_config():
    cfg = load_dynamic_exploration_config()
    assert len(cfg.episodes) >= 6
    assert cfg.explore_budgets == (8, 15, 30)
    assert "dynagraph" in cfg.profiles
    smoke = resolve_smoke_run_plan(cfg)
    assert smoke.episode_id == "robocasa_seed0"
    assert smoke.backend == "dynagraph"
    assert smoke.explore_max_iters == 3
    assert smoke.mapping_mode == "explore"


def test_filter_episodes_by_env():
    cfg = load_dynamic_exploration_config()
    rob = filter_episodes(cfg.episodes, env="robocasa")
    assert all(e.env == "robocasa" for e in rob)
    assert len(rob) == 3


def test_build_run_matrix_includes_rotate_only():
    cfg = load_dynamic_exploration_config()
    eps = filter_episodes(cfg.episodes, episode_ids=["robocasa_seed0"])
    runs = build_explore_run_matrix(
        cfg,
        eps,
        backends=["dynagraph"],
        explore_max_iters=[8],
        include_rotate_only=True,
    )
    modes = {r.mapping_mode for r in runs}
    assert "explore" in modes
    assert "rotate_only" in modes
    assert any(r.explore_max_iters == 0 for r in runs)


def test_flatten_eval_metrics():
    metrics = {
        "explore": {"explored_fraction": 0.5, "explored_area_m2": 12.0},
        "graph": {"node_count": 7, "edge_count": 3},
        "fusion": {"spatial_recall": 0.8, "label_recall": 0.6},
        "eqa": {"accuracy": 1.0},
    }
    row = flatten_eval_metrics(metrics, episode_wall_s=42.0)
    assert row["explored_fraction"] == 0.5
    assert row["eqa_accuracy"] == 1.0
    assert row["episode_wall_s"] == 42.0


def test_load_lifelong_episodes():
    cfg = load_dynamic_exploration_config()
    assert len(cfg.lifelong_episodes) >= 2
    by_id = {e.id: e for e in cfg.lifelong_episodes}
    rob = by_id["robocasa_seed0_lifelong"]
    assert rob.episode_id == "robocasa_seed0"
    assert rob.cycles == 3
    # Single question list repeats every cycle.
    assert rob.questions_for_cycle(0) == rob.questions_for_cycle(2)
    assert rob.questions_for_cycle(0)[0]["gt_body_key"] == "obj_main"
    # Changes apply between cycles only.
    assert rob.changes_after_cycle(0) is not None
    assert rob.changes_after_cycle(0)["moves"][0]["body"] == "obj_main"
    assert rob.changes_after_cycle(5) is None
    molmo = by_id["molmo_ithor0_lifelong"]
    assert molmo.changes_after_cycle(0)["doors"][0]["joint"].startswith("cabinet_")


def test_dynamic_explore_profiles():
    assert resolve_dynamic_explore_profile("dynagraph") == "interactive"
    assert resolve_dynamic_explore_profile("graph_eqa") == "graph_eqa_baseline"
    assert DYNAMIC_EXPLORE_BACKENDS == ("dynagraph", "graph_eqa")


def test_apply_dynamic_explore_backend():
    from emet.core.parameters import get_parameters

    params = get_parameters("dynav_config.yaml")
    out = apply_dynamic_explore_backend(params, "graph_eqa")
    assert float(out["dynagraph_merge_xy_m"]) == 0.0
    assert int(out["dynagraph_staleness_horizon"]) == 0


def test_dynagraph_subprocess_timeout_scales_with_explore_budget():
    from emet.eval.dynamic_exploration_runner import _dynagraph_subprocess_timeout_s

    short = _dynagraph_subprocess_timeout_s(explore_max_iters=3, sim_kind="robocasa", cpu_only=False)
    long = _dynagraph_subprocess_timeout_s(explore_max_iters=30, sim_kind="robocasa", cpu_only=False)
    assert short >= 3600.0
    assert long > short


def test_sim_set_body_pose_zmq_action():
    from emet.core.zmq_protocol import EMET_ACTION_SIM_SET_BODY_POSE_KEY, build_sim_set_body_pose_action

    act = build_sim_set_body_pose_action(3, "obj_main", [1.0, 2.0, 0.95])
    assert act["step"] == 3
    payload = act[EMET_ACTION_SIM_SET_BODY_POSE_KEY]
    assert payload["body"] == "obj_main"
    assert payload["pos"] == [1.0, 2.0, 0.95]


def test_sim_set_joint_qpos_zmq_action():
    from emet.core.zmq_protocol import (
        EMET_ACTION_SIM_SET_JOINT_QPOS_KEY,
        EMET_ZMQ_META_ACTION_KEYS,
        build_sim_set_joint_qpos_action,
    )
    from emet.simulation.sim_manipulation import parse_sim_set_joint_qpos_action

    act = build_sim_set_joint_qpos_action(5, "cab_main_group_leftdoorhinge", 1.2)
    assert act["step"] == 5
    assert EMET_ACTION_SIM_SET_JOINT_QPOS_KEY in EMET_ZMQ_META_ACTION_KEYS
    payload = act[EMET_ACTION_SIM_SET_JOINT_QPOS_KEY]
    joint, value = parse_sim_set_joint_qpos_action(payload)
    assert joint == "cab_main_group_leftdoorhinge"
    assert value == 1.2
    assert parse_sim_set_joint_qpos_action({"joint": ""}) == (None, None)


def test_set_named_joint_qpos_on_tiny_mjcf():
    import mujoco
    import numpy as np

    from emet.simulation.sim_manipulation import get_named_joint_qpos, set_named_joint_qpos

    xml = """
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="cabinet" pos="0 0 0.5">
          <geom type="box" size="0.2 0.2 0.2"/>
          <body name="door" pos="0.2 0 0">
            <joint name="cabinet_leftdoorhinge" type="hinge" axis="0 0 1" range="0 1.5"/>
            <geom type="box" size="0.01 0.2 0.2"/>
          </body>
        </body>
        <body name="ball" pos="1 0 0.1">
          <freejoint name="ball_free"/>
          <geom type="sphere" size="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert set_named_joint_qpos(model, data, "cabinet_leftdoorhinge", 1.2)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cabinet_leftdoorhinge")
    assert np.isclose(float(data.qpos[model.jnt_qposadr[jid]]), 1.2)
    assert get_named_joint_qpos(model, data, "cabinet_leftdoorhinge") == 1.2

    # Value clamps to the joint range.
    assert set_named_joint_qpos(model, data, "cabinet_leftdoorhinge", 9.0)
    assert np.isclose(float(data.qpos[model.jnt_qposadr[jid]]), 1.5)

    # Free joints and unknown joints are rejected.
    assert not set_named_joint_qpos(model, data, "ball_free", 0.5)
    assert not set_named_joint_qpos(model, data, "no_such_joint", 0.5)

