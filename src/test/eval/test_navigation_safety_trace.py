# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import itertools
import json
from types import SimpleNamespace

import pytest

from emet.eval import navigation_safety_trace as trace

mujoco = pytest.importorskip("mujoco")


def recording(tmp_path, monkeypatch, collision=False):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <geom name="wall" type="sphere" size=".1" pos="2 0 .15"/>
      <body name="robot" pos="0 0 .15"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    times = itertools.count(start=1, step=0.01)
    monkeypatch.setattr(trace.time, "time", lambda: next(times))
    path = tmp_path / "trace.jsonl"
    config = {"robot_body": "robot", "allowed_support_contacts": [["robot", "floor"]]}
    recorder = trace.NavigationSafetyTrace(model, config, path)
    for i in range(12):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0
        model.geom_pos[model.geom("wall").id, 0] = 0.18 if collision and i == 5 else 2
        mujoco.mj_step(model, data)
        recorder.record(model, data)
    recorder.close()
    return path, model


def test_physics_window_and_transient_contact(tmp_path, monkeypatch):
    path, _ = recording(tmp_path, monkeypatch, collision=True)
    result = trace.score_window(path, 1.025, 1.095)
    assert result["status"] == "failed"
    assert result["unexpected_contact_steps"] == 1
    assert result["first_contact"]["tick"] == 6


def test_clean_recording_does_not_certify_uncovered_intervals(tmp_path, monkeypatch):
    path, _ = recording(tmp_path, monkeypatch)
    assert trace.score_window(path, 1.025, 1.095)["status"] == "physics_contact_window_clear"
    assert trace.score_window(path, 0.5, 1.095)["reason"] == "uncovered_command_window"
    assert trace.score_window(path, 1.025, 2)["reason"] == "uncovered_command_window"


@pytest.mark.parametrize("failure", ["missing", "skipped_record", "clock_reset", "tipped", "targets"])
def test_incomplete_or_unsafe_data_never_passes(tmp_path, monkeypatch, failure):
    path, _ = recording(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "missing":
        del rows[6]
    elif failure == "skipped_record":
        rows[6]["sim_time"] += 0.0005
    elif failure == "clock_reset":
        rows[6]["wall_time"] = 0
    elif failure == "tipped":
        rows[6]["base_up_dot_world_z"] = 0
    else:
        rows[6]["actuator_targets"] = [1]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if failure == "targets":
        with pytest.raises(ValueError):
            trace.score_window(path, 1.025, 1.095)
    else:
        assert trace.score_window(path, 1.025, 1.095)["status"] != "physics_contact_window_clear"


def test_optional_capture_fails_closed_on_bad_configuration(tmp_path, monkeypatch):
    _, model = recording(tmp_path, monkeypatch)
    monkeypatch.delenv("EMET_NAVIGATION_TRACE_CONFIG", raising=False)
    monkeypatch.delenv("EMET_NAVIGATION_TRACE", raising=False)
    assert trace.from_environment(model) is None
    monkeypatch.setenv("EMET_NAVIGATION_TRACE", str(tmp_path / "out.jsonl"))
    with pytest.raises(ValueError, match="requires both"):
        trace.from_environment(model)
    with pytest.raises(ValueError, match="Robot root"):
        trace.NavigationSafetyTrace(model, {"robot_body": "world"}, tmp_path / "bad.jsonl")
    with pytest.raises(ValueError, match="Support exception"):
        trace.NavigationSafetyTrace(
            model, {"robot_body": "robot", "allowed_support_contacts": [["world", "floor"]]}, tmp_path / "bad.jsonl"
        )


def test_generic_physics_records_before_kinematic_attachment_snap(monkeypatch):
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    calls = []
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: calls.append("physics"))
    server = SimpleNamespace(
        _mjmodel=None,
        _mjdata=SimpleNamespace(time=0.002),
        _navigation_trace=SimpleNamespace(record=lambda *args: calls.append("trace")),
        _physics_fps_counter=SimpleNamespace(tick=lambda **kwargs: None),
        _snap_kinematic_attachments=lambda: calls.append("snap"),
    )
    RobosuiteZmqServer._mj_step_once(server)
    assert calls == ["physics", "trace", "snap"]
