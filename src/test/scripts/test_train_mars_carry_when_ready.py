"""The overnight launcher must fail before starting PPO on invalid evidence."""

import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def supervisor():
    return runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/train_mars_carry_when_ready.py"))


@pytest.mark.parametrize("failures,skipped", [(1, 0), (0, 1)])
def test_regressions_must_have_no_failures_or_skips(supervisor, tmp_path, failures, skipped):
    path = tmp_path / "checks.xml"
    path.write_text(
        f'<testsuites><testsuite tests="205" failures="{failures}" errors="0" skipped="{skipped}"/></testsuites>'
    )
    with pytest.raises(RuntimeError, match="regression"):
        supervisor["check_regressions"](path)


def test_dead_qualification_job_stops_without_training(supervisor, tmp_path, monkeypatch):
    args = supervisor["parse_args"](
        [
            "--output",
            str(tmp_path),
            "--qualification-pid-file",
            str(tmp_path / "pid"),
            "--carry-report",
            str(tmp_path / "carry.json"),
            "--pick-place-report",
            str(tmp_path / "pick.json"),
        ]
    )
    args.qualification_pid_file.write_text("123456789")
    start = Mock(side_effect=AssertionError("training must not start"))
    monkeypatch.setattr(supervisor["subprocess"], "run", start)
    monkeypatch.setattr(supervisor["psutil"], "pid_exists", lambda _: False)
    with pytest.raises(RuntimeError, match="qualification process exited"):
        supervisor["_supervise_locked"](args)
    start.assert_not_called()
    assert json.loads((tmp_path / "supervisor.json").read_text())["state"] == "stopped_without_retry"
