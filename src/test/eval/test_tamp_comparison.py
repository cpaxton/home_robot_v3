import json

import pytest

from emet.eval.tamp_comparison import compare_runs


def write_run(root, success, *, registry="same", mode="kinematic_latch", pending=False):
    root.mkdir()
    cases = [{"id": str(i), "robot": "rby1", "execution_mode": mode,
              "stage": "evaluation", "registry_sha256": registry, "scene_index": i,
              "command": [str(root / "python"), str(root / "eval.py"), "--episodes", str(root / "cases.yaml")]}
             for i in range(len(success))]
    manifest = {"source_sha": root.name, "suite": "full", "robot_filter": "rby1",
                "episode_timeout_s": 100, "cases": cases, "search_protocol": {"seed": 0}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    ledger = {str(i): {"status": "completed" if value else "crashed", "task_success": value,
                      "exit_code": 0 if value else -11} for i, value in enumerate(success)}
    if pending:
        ledger["0"]["status"] = "running"
    (root / "ledger.json").write_text(json.dumps(ledger))


def test_paired_counts_include_crashes_and_keep_denominator(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, [True, False, False])
    write_run(b, [False, True, True])
    result = compare_runs(a, b)
    assert result["scheduled_pairs"] == 3
    assert result["gains"] == 2 and result["losses"] == 1
    assert result["success_rate_delta"] == pytest.approx(1 / 3)
    assert result["candidate_status_counts"] == {"crashed": 1, "completed": 2}
    assert result["scene_bootstrap_95_interval"] is None
    assert result["promotion"] == "requires_review"


@pytest.mark.parametrize("change", ["registry", "mode", "pending", "missing_case", "protocol"])
def test_rejects_unpaired_or_nonterminal_comparisons(tmp_path, change):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, [True, False])
    write_run(b, [True] if change == "missing_case" else [True, False],
              registry="different" if change == "registry" else "same",
              mode="oracle_teleport" if change == "mode" else "kinematic_latch",
              pending=change == "pending")
    if change == "protocol":
        path = b / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["search_protocol"] = {"seed": 1}
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        compare_runs(a, b)


def test_scene_bootstrap_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, [False] * 5)
    write_run(b, [True, False, True, False, True])
    assert compare_runs(a, b) == compare_runs(a, b)


def test_cli_exports_json_and_vector_figures(tmp_path, monkeypatch, capsys):
    import runpy
    from pathlib import Path

    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "comparison"
    write_run(a, [True, False])
    write_run(b, [True, True])
    script = Path(__file__).resolve().parents[3] / "scripts/compare_tamp_runs.py"
    monkeypatch.setattr("sys.argv", [str(script), str(a), str(b), "--output-dir", str(out), "--figure"])
    runpy.run_path(str(script), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["gains"] == 1
    assert json.loads((out / "comparison.json").read_text()) == result
    assert (out / "paired_success.pdf").read_bytes().startswith(b"%PDF")
    assert "<svg" in (out / "paired_success.svg").read_text()
