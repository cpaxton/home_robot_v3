"""Reject stale/unrelated test evidence and verify fresh canonical execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from emet.rl import regression_gate as gate


@pytest.fixture
def harness(tmp_path, monkeypatch):
    path = tmp_path / "src/test/example/test_gate.py"
    path.parent.mkdir(parents=True)
    path.write_text("def test_ok(): pass\n")
    monkeypatch.setattr(gate, "REGRESSION_TARGETS", ("src/test/example/test_gate.py",))
    state = {"source": "before"}
    monkeypatch.setattr(gate, "source_state", lambda root: dict(state))
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        generated = Path(next(c.split("=", 1)[1] for c in command if c.startswith("--junitxml=")))
        generated.write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="src.test.example.test_gate" name="test_ok"/>'
            "</testsuite></testsuites>"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", runner)
    return tmp_path, state, calls


def test_fresh_execution_receipt_binds_tests_sources_and_xml(harness):
    root, state, calls = harness
    xml = root / "fresh.xml"
    receipt = gate.run_regressions(xml, root=root)
    assert len(calls) == 1 and "-m" in calls[0] and "pytest" in calls[0]
    assert gate.verify_regressions(receipt, root=root)["test_ids"] == [["src.test.example.test_gate", "test_ok"]]
    state["source"] = "edited-after-tests"
    with pytest.raises(RuntimeError, match="stale"):
        gate.verify_regressions(receipt, root=root)


def test_existing_xml_is_not_a_launch_authorization(harness):
    root, _, calls = harness
    xml = root / "old.xml"
    xml.write_text('<testsuites><testsuite tests="999" failures="0"/></testsuites>')
    with pytest.raises(RuntimeError, match="stale XML"):
        gate.run_regressions(xml, root=root)
    assert not calls


def test_unrelated_junit_and_tampering_are_rejected(harness):
    root, _, _ = harness
    receipt = gate.run_regressions(root / "fresh.xml", root=root)
    (root / "fresh.xml").write_text(
        '<testsuites><testsuite tests="1"><testcase classname="other" name="test"/></testsuite></testsuites>'
    )
    with pytest.raises(RuntimeError, match="XML"):
        gate.verify_regressions(receipt, root=root)
    with pytest.raises(RuntimeError, match="expected test modules"):
        gate.check_xml(root / "fresh.xml", root)


def test_source_edits_during_pytest_invalidate_result(harness, monkeypatch):
    root, state, _ = harness
    original = gate.subprocess.run

    def edited(*args, **kwargs):
        result = original(*args, **kwargs)
        state["source"] = "changed-during-tests"
        return result

    monkeypatch.setattr(gate.subprocess, "run", edited)
    with pytest.raises(RuntimeError, match="changed during tests"):
        gate.run_regressions(root / "fresh.xml", root=root)
    assert not gate.receipt_path(root / "fresh.xml").exists()


def test_target_list_is_pinned_to_control_and_safety_suites():
    assert {
        "src/test/rl",
        "src/test/core",
        "src/test/robots/test_mars_motion_boundary.py",
        "src/test/robots/test_head_capability.py",
    } <= set(gate.REGRESSION_TARGETS)


@pytest.mark.parametrize('count', ['oops', '-1', 'NaN'])
def test_malformed_xml_counts_fail_closed(tmp_path, count):
    path = tmp_path / 'bad.xml'
    path.write_text(f'<testsuite tests="{count}"><testcase name="x"/></testsuite>')
    with pytest.raises(RuntimeError, match='malformed regression XML'):
        gate.check_xml(path, tmp_path)


def test_malformed_xml_syntax_fails_closed(tmp_path):
    path = tmp_path / 'bad.xml'
    path.write_text('<testsuite')
    with pytest.raises(RuntimeError, match='malformed regression XML'):
        gate.check_xml(path, tmp_path)
