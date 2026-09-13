"""Semantic scoring, trace integrity and CLI gates; no simulator or paid models."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from click.testing import CliRunner

from emet.cli_cmds.agent_tasks import agent_tasks_group
from emet.eval.agent_tasks.fixture import route_to
from emet.eval.agent_tasks.publication import export_paper
from emet.eval.agent_tasks.recording import Recorder, load_run, write_json
from emet.eval.agent_tasks.runner import preflight
from emet.eval.agent_tasks.spec import fingerprint, load_suite, policy_task, score, select_episode
from emet.eval.agent_tasks.visualize import export_html


@pytest.fixture
def suite():
    return load_suite()


def initial(suite):
    return {name: obj["position"][:] for name, obj in suite["scene"]["objects"].items()}


def deliver(suite, positions, goal):
    destination = suite["scene"]["objects"][goal["destination"]]["position"]
    positions[goal["object"]] = [destination[0], destination[1], destination[2] + 0.02]


def test_partial_and_wrong_destination_are_not_success(suite):
    episode = select_episode(suite, "two_object_collection")
    positions = initial(suite)
    assert not score(episode, positions)["success"]
    deliver(suite, positions, episode["goals"][0])
    assert score(episode, positions)["completed"] == 1
    assert not score(episode, positions)["success"]
    positions["blue_cube"] = positions["red_cylinder"][:]
    assert not score(episode, positions)["success"]
    deliver(suite, positions, episode["goals"][1])
    assert score(episode, positions)["success"]


def test_alternative_delivery_order_is_valid(suite):
    episode = select_episode(suite, "two_object_collection")
    positions = initial(suite)
    for goal in reversed(episode["goals"]):
        deliver(suite, positions, goal)
    assert score(episode, positions)["success"]


@pytest.mark.parametrize("position", [None, [float("nan"), 0, 0], [7.65, -1.2, 1.0], [7.65, -1.2, 0.1]])
def test_missing_nonfinite_or_wrong_height_is_not_success(suite, position):
    episode = suite["episodes"][0]
    positions = initial(suite)
    positions["red_cylinder"] = position
    assert not score(episode, positions)["success"]


def test_held_object_cannot_complete(suite):
    episode = suite["episodes"][0]
    positions = initial(suite)
    deliver(suite, positions, episode["goals"][0])
    assert not score(episode, positions, held=["red_cylinder"])["success"]


def test_task_input_excludes_solution_and_world_changes(suite):
    episode = suite["episodes"][2]
    task = policy_task(episode)
    assert set(task) == {"episode_id", "instruction"}
    assert "goals" not in task and "changes" not in task and "witness_order" not in task


def test_routes_pass_and_narrow_door_fails(suite):
    scene = suite["scene"]
    assert len(route_to(scene, [0, 0, 0], [8, -0.65, 0])) == 4
    broken = copy.deepcopy(scene)
    broken["doors"][0].update(y_min=-0.2, y_max=0.2)
    with pytest.raises(ValueError, match="collision"):
        route_to(broken, [0, 0, 0], [8, -0.65, 0])


def test_preflight_checks_changed_world_routes(suite):
    broken = copy.deepcopy(suite)
    broken["episodes"][2]["changes"][0]["position"] = [2, 1, 0.56]
    result = preflight(broken)
    assert not next(c for c in result["checks"] if c["name"] == "footprint_routes")["passed"]


def test_duplicate_episode_rejected(suite, tmp_path):
    import yaml

    suite["episodes"].append(suite["episodes"][0])
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(suite))
    with pytest.raises(ValueError, match="unique"):
        load_suite(path)


def record_run(tmp_path, suite, name="run"):
    path = tmp_path / name
    episode = suite["episodes"][0]
    manifest = {
        "suite": suite,
        "episode": episode,
        "suite_fingerprint": fingerprint(suite),
        "episode_fingerprint": fingerprint(episode),
        "robot": "rby1",
        "seed": 0,
        "assistance": {"manipulation": "teleport"},
        "mode": "assisted_fixture",
        "source_revision": "test",
    }
    rec = Recorder(path, manifest)
    rec.append(
        "start",
        policy={"memory": {"red": "old_location"}},
        evaluator={"positions": initial(suite), "robot_xyt": [0, 0, 0], "score": score(episode, initial(suite))},
    )
    rec.append("world_change", policy={"memory": {"red": "old_location"}}, evaluator={"moved": "red"})
    rec.append("inspection", policy={"memory": {"red": "new_location"}})
    rec.close()
    write_json(
        path / "metrics.json", {"event_hash": rec.last_hash, "status": "task_failed", "completed": 0, "total": 1}
    )
    return path


def test_recording_preserves_historical_policy_state(suite, tmp_path):
    run = record_run(tmp_path, suite)
    _, events, _ = load_run(run)
    assert events[0]["policy"]["memory"]["red"] == "old_location"
    assert events[1]["policy"]["memory"]["red"] == "old_location"
    assert events[2]["policy"]["memory"]["red"] == "new_location"
    assert "moved" not in events[1]["policy"]


def test_tampered_events_rejected(suite, tmp_path):
    run = record_run(tmp_path, suite)
    path = run / "events.jsonl"
    path.write_text(path.read_text().replace("old_location", "forged_location", 1))
    with pytest.raises(ValueError, match="hash"):
        load_run(run)


def test_missing_frame_rejected(suite, tmp_path):
    import numpy as np

    run = tmp_path / "frames"
    rec = Recorder(run, {})
    rec.append("observe", images={"head": np.zeros((10, 10, 3), dtype=np.uint8)})
    rec.close()
    (run / "images/00000_head.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_run(run)


def test_altered_image_rejected(tmp_path):
    import numpy as np

    run = tmp_path / "frames"
    rec = Recorder(run, {})
    rec.append("observe", images={"head": np.zeros((10, 10, 3), dtype=np.uint8)})
    rec.close()
    (run / "images/00000_head.png").write_bytes(b"altered")
    with pytest.raises(ValueError, match="image hash"):
        load_run(run)


def test_altered_manifest_rejected(suite, tmp_path):
    run = record_run(tmp_path, suite)
    path = run / "manifest.json"
    value = json.loads(path.read_text())
    value["seed"] = 87
    write_json(path, value)
    with pytest.raises(ValueError, match="manifest hash"):
        load_run(run)


def test_html_escapes_tool_content(suite, tmp_path):
    run = record_run(tmp_path, suite)
    manifest, events, metrics = load_run(run)
    events[0]["policy"]["result"] = "</script><script>alert(1)</script>"
    export_html(run, manifest, events, metrics)
    report = (run / "report.html").read_text()
    assert "</script><script>alert(1)" not in report
    assert "\\u003c/script>" in report
    assert "Evaluator" in report


def test_cli_comparison_rejects_unpaired_scenes(suite, tmp_path):
    a = record_run(tmp_path, suite, "a")
    changed = copy.deepcopy(suite)
    changed["seed"] = 3
    b = record_run(tmp_path, changed, "b")
    result = CliRunner().invoke(agent_tasks_group, ["compare", str(a), str(b)])
    assert result.exit_code != 0
    assert "unpaired" in result.output


def test_cli_preflight_and_help_are_model_free():
    runner = CliRunner()
    help_result = runner.invoke(agent_tasks_group, ["--help"])
    assert help_result.exit_code == 0
    for command in ("preflight", "run", "inspect", "replay", "compare", "export"):
        assert command in help_result.output
    result = runner.invoke(agent_tasks_group, ["preflight"])
    assert result.exit_code == 0
    assert json.loads(result.output)["paid_requests_enabled"] is False


def test_existing_output_cannot_be_overwritten(tmp_path):
    result = CliRunner().invoke(agent_tasks_group, ["run", "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "already exists" in result.output


@pytest.mark.parametrize("fault", ["altered_artifact", "changed_assistance"])
def test_paper_export_rejects_unmatched_evidence(suite, tmp_path, fault):
    episode = suite["episodes"][0]
    positions = initial(suite)
    deliver(suite, positions, episode["goals"][0])
    rows = []
    for repeat in range(3):
        root = tmp_path / f"run_{repeat}"
        manifest = {
            "suite_fingerprint": fingerprint(suite),
            "episode_fingerprint": fingerprint(episode),
            "episode": episode,
            "control": "witness",
            "robot": "rby1",
            "seed": 0,
            "mode": "assisted_fixture",
            "assistance": {"manipulation": "teleport"},
            "implementation_sha256": {"runner.py": "test"},
        }
        if fault == "changed_assistance" and repeat == 1:
            manifest["assistance"]["manipulation"] = "different"
        rec = Recorder(root, manifest)
        rec.append("final", evaluator={"positions": positions, "score": score(episode, positions)})
        rec.close()
        write_json(
            root / "metrics.json", {**score(episode, positions), "event_hash": rec.last_hash, "status": "completed"}
        )
        artifact = root / "overview.pdf"
        artifact.write_bytes(b"original artifact")
        write_json(
            root / "artifacts.json",
            {
                "complete": True,
                "source_event_hash": rec.last_hash,
                "artifacts": [artifact.name],
                "sha256": {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()},
            },
        )
        if fault == "altered_artifact" and repeat == 0:
            artifact.write_bytes(b"changed artifact")
        rows.append({"path": root.name})
    write_json(tmp_path / "certificate.json", {"rows": rows, "suite_fingerprint": fingerprint(suite)})
    with pytest.raises(ValueError, match="artifact hash|assistance changed"):
        export_paper(tmp_path, tmp_path / "paper")
    assert not (tmp_path / "paper").exists()
