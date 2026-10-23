"""Fresh, source-bound regression evidence for simulator training launches."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGRESSION_TARGETS = (
    "src/test/rl",
    "src/test/core",
    "src/test/robots/test_mars_motion_boundary.py",
    "src/test/robots/test_mars_control_session.py",
    "src/test/robots/test_head_capability.py",
    "src/test/scripts/test_train_mars_carry_when_ready.py",
    "src/test/scripts/test_trace_mars_carry.py",
)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_files(root):
    files = set()
    for target in REGRESSION_TARGETS:
        path = root / target
        files.update(path.rglob("test*.py")) if path.is_dir() else files.add(path)
    return sorted(files)


def source_state(root):
    from emet.rl.manipulation.carry_benchmark import source_hashes

    state = {"control/" + k: v for k, v in source_hashes().items()}
    paths = (
        list((root / "src/emet/core").rglob("*.py"))
        + list((root / "src/emet_core/emet/core").rglob("*.py"))
        + test_files(root)
        + [
            root / "src/test/conftest.py",
            root / "pyproject.toml",
            root / "uv.lock",
            root / "scripts/train_mars_carry_when_ready.py",
            root / "scripts/trace_mars_carry.py",
            root / "scripts/sync_core_runtime.py",
        ]
    )
    for path in paths:
        state[str(path.relative_to(root))] = file_digest(path)
    return state


def check_xml(path, root):
    try:
        document = ET.parse(path).getroot()
        suites = list(document.iter("testsuite"))
        cases = list(document.iter("testcase"))
        counts = [{key: int(s.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")} for s in suites]
        if any(value < 0 for count in counts for value in count.values()):
            raise ValueError("negative test count")
    except (ET.ParseError, ValueError, OSError) as exc:
        raise RuntimeError("malformed regression XML or count attributes") from exc
    if (
        not suites
        or not cases
        or any(c[key] for c in counts for key in ("failures", "errors", "skipped"))
        or any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
    ):
        raise RuntimeError("complete passing regression tests required")
    if sum(c["tests"] for c in counts) != len(cases):
        raise RuntimeError("regression test count mismatch")
    identities = [(c.get("classname"), c.get("name")) for c in cases]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate regression test identities")
    expected = {str(p.relative_to(root).with_suffix("")).replace("/", ".") for p in test_files(root)}
    actual = {c.get("classname", "") for c in cases}
    if any(not any(name == module or name.startswith(module + ".") for name in actual) for module in expected):
        raise RuntimeError("regression XML does not cover the expected test modules")
    if any(not any(name == module or name.startswith(module + ".") for module in expected) for name in actual):
        raise RuntimeError("regression XML contains unrelated test modules")
    return sorted([list(identity) for identity in identities])


def receipt_path(xml_path):
    return Path(xml_path).with_suffix(".receipt.json")


def run_regressions(xml_path, *, root=ROOT):
    """Run the canonical suites ourselves; an existing XML is never trusted."""
    xml_path = Path(xml_path).resolve()
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    if xml_path.exists() or receipt_path(xml_path).exists():
        raise RuntimeError("regression output exists; use a fresh path, not a stale XML")
    before = source_state(root)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="regression-", dir=xml_path.parent) as directory:
        generated = Path(directory) / "pytest.xml"
        command = [sys.executable, "-m", "pytest", *REGRESSION_TARGETS, "-q", "--tb=short", f"--junitxml={generated}"]
        environment = dict(
            os.environ,
            PYTEST_ADDOPTS="",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MPLCONFIGDIR="/tmp/mpl",
            MUJOCO_GL="egl",
        )
        with xml_path.with_suffix(".log").open("x") as log:
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        if result.returncode != 0 or not generated.exists():
            raise RuntimeError("regression subprocess failed; inspect the regression log")
        ids = check_xml(generated, root)
        if before != source_state(root):
            raise RuntimeError("regression sources changed during tests")
        os.replace(generated, xml_path)
    receipt = {
        "format": "emet-regression-v1",
        "source_state": before,
        "targets": list(REGRESSION_TARGETS),
        "test_ids": ids,
        "xml_sha256": file_digest(xml_path),
        "xml_name": xml_path.name,
        "started_unix": started,
        "finished_unix": time.time(),
        "returncode": 0,
    }
    receipt_path(xml_path).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt_path(xml_path)


def verify_regressions(path, *, root=ROOT):
    path = Path(path)
    evidence = json.loads(path.read_text())
    xml = path.parent / evidence.get("xml_name", "")
    if evidence.get("format") != "emet-regression-v1" or evidence.get("returncode") != 0:
        raise RuntimeError("invalid regression receipt")
    if evidence.get("source_state") != source_state(root) or evidence.get("targets") != list(REGRESSION_TARGETS):
        raise RuntimeError("stale or unrelated regression receipt")
    if evidence.get("xml_sha256") != file_digest(xml) or evidence.get("test_ids") != check_xml(xml, root):
        raise RuntimeError("regression XML does not match its receipt")
    return evidence
