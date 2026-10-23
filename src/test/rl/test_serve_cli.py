import pytest
from click.testing import CliRunner

from emet.cli_cmds import serve as serve_module


@pytest.mark.parametrize("controller", ["baseline", "base"])
def test_reactive_cli_forwards_selection_and_preserves_legacy_default(monkeypatch, controller):
    calls = []
    monkeypatch.setattr(serve_module, "_run_module", lambda module, args: calls.append((module, args)) or 0)
    runner = CliRunner()
    result = runner.invoke(
        serve_module.serve, ["mujoco", "--robot", "innate_mars", "--controller", controller, "--headless"]
    )
    assert result.exit_code == 0, result.output
    assert "--controller" in calls[0][1]
    assert calls[0][1][calls[0][1].index("--controller") + 1] == controller
    result = runner.invoke(serve_module.serve, ["mujoco", "--robot", "innate_mars", "--headless"])
    assert result.exit_code == 0, result.output
    assert "--controller" not in calls[1][1]


def test_reactive_cli_rejects_unsupported_robot_and_missing_policy():
    runner = CliRunner()
    result = runner.invoke(serve_module.serve, ["mujoco", "--robot", "stretch", "--controller", "baseline"])
    assert result.exit_code != 0
    assert "innate_mars" in result.output
    result = runner.invoke(serve_module.serve, ["mujoco", "--robot", "innate_mars", "--controller", "rl"])
    assert result.exit_code != 0
    assert "--policy-path" in result.output
